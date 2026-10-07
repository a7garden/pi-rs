//! Port of `packages/durable/src/harness/generation.ts` (the built-in
//! `pi.generation` task).
//!
//! Erasure (D1 — see `tasks/mod.rs`): the checkpoint is raw JSON tagged by
//! `phase` ([`GenerationCheckpoint`]); the result is `{ entryId }`. D2 —
//! partial throttling uses the same at-most-one-commit machinery as
//! upstream (`Progress`-style in-flight tracking inline). D3 — hook
//! dispatch downcasts the erased handler table to [`GenerationHooks`].

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::types::{
	AssistantContent, AssistantMessage, DeferredHandle, Model, ModelThinkingLevel, SimpleStreamOptions,
	StopReason, ToolCall,
};
use pi_ai::utils::overflow::is_context_overflow;
use pi_ai::utils::retry::{is_retryable_assistant_error, retry_delay_ms, RetryPolicy};
use pi_ai::utils::transcript::get_current_tools;
use pi_chord::context::Context;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::entries::{ASSISTANT_ENTRY, RESET_ENTRY, SYSTEM_ENTRY, USER_ENTRY};
use crate::harness::types::coerce_hook_api;
use crate::harness::util::abort_reason_text;
use crate::harness::agent::add_tools;
use crate::harness::compaction::{create_compaction, estimate_context, select_cut, CompactionInput};
use crate::harness::inbox::{apply_boundary, prepare_boundary, QueueModes};
use crate::harness::json::assign_json;
use crate::harness::live::{end_run_on_tx, read_live, store_live, GenerationStatus, LiveState, RetryBackoff, ToolSlot, SlotStatus};
use crate::harness::prompt::{plan_system_entries, render_sections, replay_sections};
use crate::harness::provider::ensure_provider_session_id;
use crate::harness::tool::{append_tool_result, harness_error, tool_task};
use crate::harness::types::{
	BeforeRequestPayload, CompactionPolicy, CompactionReason, ConversationStreamOptions, EntryId,
	GenerationHooks, HookApi, ModelRef, PromptInput, SettledTask, SubmissionId, TaskId, TaskRecord,
	ToolControl, WhenBusy, OnYieldResult,
};
use crate::harness::usage::record_usage;
use crate::session::Tx;
use crate::tasks::{
	commit_change, define_task, BoxFuture, NextTaskState, PhaseHandler, TaskDefinition, TaskRuntime,
};
use crate::types::{SubmissionSettlement, TaskOutcome, TaskOutcomeError, TaskOutcomeStatus, TaskState};

/// Input of the generation task (an empty object).
pub type GenerationInput = Value;

/// Erased checkpoint of the generation task (`phase` names the phase).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum GenerationCheckpoint {
	Prepare {
		attempt: u32,
		/// The blocking compaction this generation waited for; it starts no
		/// other compaction (spec §8.3).
		#[serde(skip_serializing_if = "Option::is_none")]
		compacted: Option<TaskId>,
		/// Error text of the overflow that started `compacted`; checked once
		/// when `prepare` resumes.
		#[serde(skip_serializing_if = "Option::is_none")]
		overflow: Option<String>,
	},
	Request {
		attempt: u32,
		#[serde(skip_serializing_if = "Option::is_none")]
		compacted: Option<TaskId>,
		model: ModelRef,
		thinking_level: ModelThinkingLevel,
		/// The settings' request options when preparation committed; a
		/// resend after recovery uses them unchanged.
		stream_options: ConversationStreamOptions,
		/// Newest entry included in the request.
		cutoff: EntryId,
	},
	Retry {
		attempt: u32,
		#[serde(skip_serializing_if = "Option::is_none")]
		compacted: Option<TaskId>,
		until: u64,
	},
	Poll {
		attempt: u32,
		#[serde(skip_serializing_if = "Option::is_none")]
		compacted: Option<TaskId>,
		model: ModelRef,
		cutoff: EntryId,
		handle: DeferredHandle,
		poll_at: u64,
	},
	/// Waiting on the round's tool tasks, which the generation owns
	/// (spec §8.5).
	Tools {
		/// The tool-calling answer.
		assistant: EntryId,
		/// Tool tasks created so far, in call order; grows by one per
		/// started call of a sequential round.
		tools: Vec<TaskId>,
		/// Calls of a sequential round not started yet, in call order.
		pending: Vec<String>,
	},
}

/// Result of the generation task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationResult {
	pub entry_id: EntryId,
}

#[derive(Clone)]
struct Request {
	attempt: u32,
	compacted: Option<TaskId>,
	model: ModelRef,
	cutoff: EntryId,
	/// Committed model context through `cutoff`, when the phase already
	/// derived it.
	messages: Option<Vec<pi_ai::Message>>,
	/// Set when the message came from polling, so a still deferred result
	/// polls strictly later.
	poll_at: Option<u64>,
}

const DEFAULT_POLL_AFTER_MS: u64 = 5000;

fn generation_definition() -> TaskDefinition {
	let mut phases: BTreeMap<String, PhaseHandler> = BTreeMap::new();
	phases.insert("prepare".to_string(), Arc::new(|task, runtime, context| Box::pin(prepare_phase(task, runtime, context)) as BoxFuture<Result<(), String>>));
	phases.insert("request".to_string(), Arc::new(|task, runtime, context| Box::pin(request_phase(task, runtime, context)) as BoxFuture<Result<(), String>>));
	phases.insert("retry".to_string(), Arc::new(|task, runtime, context| Box::pin(retry_phase(task, runtime, context)) as BoxFuture<Result<(), String>>));
	phases.insert("poll".to_string(), Arc::new(|task, runtime, context| Box::pin(poll_phase(task, runtime, context)) as BoxFuture<Result<(), String>>));
	phases.insert("tools".to_string(), Arc::new(|task, runtime, context| Box::pin(tools_phase(task, runtime, context)) as BoxFuture<Result<(), String>>));
	TaskDefinition {
		name: "pi.generation".to_string(),
		version: 1,
		initial: Arc::new(|_| json!({ "phase": "prepare", "attempt": 1 })),
		phases: Arc::new(phases),
		abort: Arc::new(|task, runtime, context| Box::pin(abort_phase(task, runtime, context)) as BoxFuture<Result<(), String>>),
		migrate: None,
		hooks: None,
	}
}

static GENERATION_TASK: std::sync::LazyLock<crate::harness::types::AnyTask> =
	std::sync::LazyLock::new(|| crate::harness::types::AnyTask::new(generation_definition()));

/// Built-in generation task: prepares the positional system prompt and tool
/// loadout, requests or polls the model, retries, and classifies the
/// response. The run's inputs live in `pi.live.run`.
pub fn generation_task() -> crate::harness::types::AnyTask {
	GENERATION_TASK.clone()
}

fn checkpoint_of(record: &TaskRecord) -> Result<GenerationCheckpoint, String> {
	serde_json::from_value(record.state.checkpoint.clone().unwrap_or(Value::Null))
		.map_err(|error| format!("Generation checkpoint is invalid: {error}"))
}

/// The calls `call_ids` of the assistant entry, in the given order.
async fn read_calls(
	runtime: &Arc<dyn TaskRuntime>,
	assistant: EntryId,
	call_ids: &[String],
	context: &Context,
) -> Result<Vec<ToolCall>, String> {
	let entry = runtime.entry(Some(ASSISTANT_ENTRY.kind.as_str()), assistant, context).await?;
	let calls: Vec<ToolCall> = entry
		.and_then(|entry| entry.model)
		.and_then(|model| model.into_iter().next())
		.map(|message| match message {
			pi_ai::Message::Assistant(assistant) => assistant
				.content
				.into_iter()
				.filter_map(|block| match block {
					AssistantContent::ToolCall(call) => Some(call),
					_ => None,
				})
				.collect(),
			_ => Vec::new(),
		})
		.unwrap_or_default();
	Ok(call_ids
		.iter()
		.filter_map(|id| calls.iter().find(|call| &call.id == id).cloned())
		.collect())
}

/// A tool task for call `call_id`, owned by the generation.
fn create_tool_task(
	tx: &Tx<'_>,
	runtime: &Arc<dyn TaskRuntime>,
	assistant: EntryId,
	call_id: &str,
) -> Result<TaskId, String> {
	tx.create_task(
		&tool_task(),
		json!({ "assistant": assistant, "callId": call_id }),
		crate::types::TaskOptions {
			conversation_id: None,
			ownership: crate::types::TaskOwnership::Task { task_id: runtime.task_id() },
			background: None,
		},
	)
	.map_err(|e| e.to_string())
}

/// Which threshold compaction preparation starts before its request
/// (spec §8.3): `blocking` above `contextWindow - reserveTokens`,
/// `background` above the background threshold, and only when range
/// selection finds a cut. The caller starts a background one only while no
/// compaction is listed.
fn threshold_compaction(
	view: &crate::session::ContextView,
	planned_messages: Vec<pi_ai::Message>,
	context_window: u64,
	policy: &CompactionPolicy,
) -> Option<&'static str> {
	if !policy.enabled || context_window == 0 {
		return None;
	}
	let tokens = estimate_context(view, &planned_messages);
	#[cfg(test)]
	eprintln!("[estimate] tokens={tokens} blocking={} background={} entries={} messages={}", context_window.saturating_sub(policy.reserve_tokens), context_window.saturating_sub(policy.reserve_tokens).saturating_sub(policy.background_tokens), view.entries.len(), view.messages.len());
	let blocking = context_window.saturating_sub(policy.reserve_tokens);
	let background = blocking.saturating_sub(policy.background_tokens);
	let over = if tokens > blocking {
		"blocking"
	} else if policy.background_tokens > 0 && tokens > background {
		"background"
	} else {
		return None;
	};
	if select_cut(view, policy.keep_recent_tokens).is_none() {
		return None;
	}
	Some(over)
}

/// Settle the run's inputs `unanswered` with `model_error` and fail with
/// `text`.
async fn fail_model_error(runtime: &Arc<dyn TaskRuntime>, text: &str, context: &Context) -> Result<(), String> {
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	let text = text.to_string();
	runtime
		.commit(
			commit_change(move |tx, _current| {
				let text = text.clone();
					let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					end_run_on_tx(
						&mut live,
						tx,
						&task_id,
						SubmissionSettlement::Unanswered {
							reason: "model_error".to_string(),
							detail: Some(Value::String(text.clone())),
						},
					)?;
					store_live(&draft, &live)?;
					Ok(Some(NextTaskState::Terminal {
						outcome: TaskOutcome {
							status: TaskOutcomeStatus::Failed,
							result: None,
							error: Some(TaskOutcomeError {
								message: text,
								detail: Some(json!({ "reason": "model_error" })),
							}),
							reason: None,
						},
					}))
			}),
			context,
		)
		.await.map(|_| ())?;
	Ok(())
}

/// Settle the run's inputs `unanswered` with `no_model` and fail.
async fn fail_no_model(runtime: &Arc<dyn TaskRuntime>, reference: Option<ModelRef>, context: &Context) -> Result<(), String> {
	let message = match reference {
		None => "No model is configured".to_string(),
		Some(reference) => format!("Model {}/{} is not available", reference.provider, reference.model_id),
	};
	fail_model_error_reason(runtime, &message, "no_model", context).await
}

async fn fail_model_error_reason(
	runtime: &Arc<dyn TaskRuntime>,
	text: &str,
	reason: &str,
	context: &Context,
) -> Result<(), String> {
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	let text = text.to_string();
	let reason = reason.to_string();
	runtime
		.commit(
			commit_change(move |tx, _current| {
				let text = text.clone();
				let reason = reason.clone();
					let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					end_run_on_tx(
						&mut live,
						tx,
						&task_id,
						SubmissionSettlement::Unanswered {
							reason: reason.clone(),
							detail: Some(Value::String(text.clone())),
						},
					)?;
					store_live(&draft, &live)?;
					Ok(Some(NextTaskState::Terminal {
						outcome: TaskOutcome {
							status: TaskOutcomeStatus::Failed,
							result: None,
							error: Some(TaskOutcomeError {
								message: text,
								detail: Some(json!({ "reason": reason })),
							}),
							reason: None,
						},
					}))
			}),
			context,
		)
		.await.map(|_| ())?;
	Ok(())
}

/// Phase `prepare`.
async fn prepare_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let conversation_id = runtime.conversation_id();
	let agent = runtime.agent(&context).await?;
	let settings = runtime.settings();
	let (model_reference, thinking_level) = (agent.model.clone(), agent.thinking_level);
	let resolved = match &model_reference {
		Some(reference) => runtime.models().get_model(&reference.provider, &reference.model_id),
		None => None,
	};
	if model_reference.is_none() || resolved.is_none() {
		return Box::pin(fail_no_model(&runtime, model_reference, &context)).await;
	}
	let resolved = resolved.expect("checked");
	let checkpoint = checkpoint_of(&task)?;
	let GenerationCheckpoint::Prepare { attempt, compacted, overflow } = checkpoint else {
		return Err("generation prepare phase reached with another checkpoint".to_string());
	};
	if let Some(compacted) = compacted {
		if overflow.is_some() {
			let outcomes = runtime.outcomes(&[compacted], &context).await?;
			let valid = outcomes
				.first()
				.map(|outcome| {
					outcome.status == TaskOutcomeStatus::Completed
						&& outcome
							.result
							.as_ref()
							.and_then(|result| result.get("entryId"))
							.is_some_and(|entry| !entry.is_null())
				})
				.unwrap_or(false);
			if !valid {
				return Box::pin(fail_model_error(&runtime, overflow.as_deref().unwrap_or("Context overflow"), &context)).await;
			}
		}
	}
	let view = runtime.context_view(conversation_id, &context, None).await?;
	#[cfg(test)]
	eprintln!("[prepare-entry] attempt={attempt} compacted={compacted:?} overflow={overflow:?} entries={}", view.entries.len());
	let shown = replay_sections(&view.messages);
	let report_runtime = runtime.clone();
	let report = move |error: String| report_runtime.report(error);
	let mut env = None;
	match runtime.env(&context).await {
		Ok(resolved_env) => env = resolved_env,
		Err(error) => {
			if context.abort_signal().map(|signal| signal.aborted()).unwrap_or(false) {
				return Err(error);
			}
			report(error);
		}
	}
	let input = PromptInput {
		conversation_id,
		agent: agent.clone(),
		env,
		shown: shown.clone(),
		read: coerce_hook_api(runtime.clone()),
	};
	let mut desired = serde_json::Map::new();
	for section in &agent.sections {
		match (section.render)(input.clone(), context.clone()).await {
			Err(error) => {
				if context.abort_signal().map(|signal| signal.aborted()).unwrap_or(false) {
					return Err(error);
				}
				report(error);
				if let Some(kept) = shown.get(&section.key).and_then(Value::as_str) {
					desired.insert(section.key.clone(), Value::String(kept.to_string()));
				}
			}
			Ok(None) => {}
			Ok(Some(text)) => {
				let text = if section.tag == Some(false) { text } else { format!("<{}>\n{}\n</{}>", section.key, text, section.key) };
				desired.insert(section.key.clone(), Value::String(text));
			}
		}
	}
	let entries = plan_system_entries(&view, &desired, &agent.tool_declarations(), runtime.now() as i64);
	#[cfg(test)]
	eprintln!("[prep] tokens_check keep_recent={} background={}", runtime.settings().compaction.keep_recent_tokens, runtime.settings().compaction.background_tokens);
	// Upstream reads `settings.compaction` live at selection time (the runtime
	// exposes the harness settings object), so a policy changed while sections
	// rendered applies to the cut decision.
	let threshold = if compacted.is_none() {
		threshold_compaction(
			&view,
			entries.iter().filter_map(|entry| entry.model.clone()).flatten().collect(),
			resolved.context_window,
			&runtime.settings().compaction,
		)
	} else {
		None
	};
	#[cfg(test)]
	eprintln!("[prep] threshold={threshold:?} policy={:?}", runtime.settings().compaction);
	if threshold == Some("blocking") {
		// Compact first and prepare again; the transcript is unchanged until
		// the compaction appends.
		let task_id = runtime.task_id();
		let runtime_in = runtime.clone();
		let child = runtime
			.commit(
				commit_change(move |tx: &mut Tx<'_>, _current| {
					let _runtime = runtime_in;
					let child = create_compaction(
						tx,
						conversation_id,
						CompactionInput { reason: CompactionReason::Threshold, instructions: None },
						Some(task_id),
					)?;
						Ok(Some(NextTaskState::Waiting {
							checkpoint: json!({ "phase": "prepare", "attempt": attempt, "compacted": child }),
							on: vec![child],
							policy: crate::types::JoinPolicy::AllSettled,
						}))
				}),
				&context,
			)
			.await?;
		let _ = child;
		return Ok(());
	}
	let runtime2 = runtime.clone();
	runtime
		.commit(
			commit_change(move |tx, _current| {
				let runtime = runtime2.clone();
				let entries = entries.clone();
				let agent = agent.clone();
					let mut cutoff: Option<EntryId> = tx
						.scan_entries(
							&crate::types::EntryQuery {
								conversation_id,
								min_entry_id: None,
								max_entry_id: None,
							},
							1,
							None,
						)
						.map_err(|e| e.to_string())?
						.items
						.first()
						.map(|entry| entry.id);
					for entry in &entries {
						cutoff = Some(tx.append_entry(conversation_id, entry.clone()).map_err(|e| e.to_string())?.id);
					}
					let Some(cutoff) = cutoff else {
						return Err(format!("Conversation {conversation_id} has no entries to send"));
					};
					// Checked in this commit, so a compaction admitted during
					// preparation counts.
					if threshold == Some("background") {
						#[cfg(test)]
						eprintln!("[prep] starting background compaction");
						let draft = tx
							.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
							.map_err(|e| e.to_string())?;
						let live = read_live(&draft);
						if live.compactions.is_none() {
							create_compaction(
								tx,
								conversation_id,
								CompactionInput { reason: CompactionReason::Threshold, instructions: None },
								None,
							)?;
						}
					}
					let mut checkpoint = json!({
						"phase": "request",
						"attempt": attempt,
						"model": model_reference,
						"thinkingLevel": thinking_level,
						"streamOptions": stream_options_of(&settings.stream),
						"cutoff": cutoff,
					});
					if let Some(compacted) = compacted {
						checkpoint["compacted"] = json!(compacted);
					}
					let _ = &agent;
					Ok(Some(NextTaskState::Running { checkpoint }))
			}),
			&context,
		)
		.await.map(|_| ())?;
	Ok(())
}

fn stream_options_of(options: &ConversationStreamOptions) -> Value {
	serde_json::to_value(options).expect("ConversationStreamOptions serializes")
}
fn deferred_request(value: &Value) -> Option<pi_ai::types::DeferredRequest> {
	use pi_ai::types::{DeferredRequest, DeferredWindow};
	let window = match value {
		Value::Bool(enabled) => return Some(DeferredRequest::Bool(*enabled)),
		Value::Object(options) => match options.get("window") {
			None => None,
			Some(Value::String(window)) => Some(match window.as_str() {
				"15m" => DeferredWindow::M15,
				"1h" => DeferredWindow::H1,
				"24h" => DeferredWindow::H24,
				_ => return None,
			}),
			Some(_) => return None,
		},
		_ => return None,
	};
	Some(DeferredRequest::Window(window))
}

/// Phase `request`.
async fn request_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let GenerationCheckpoint::Request { attempt, compacted, model: reference, thinking_level, stream_options, cutoff } =
		checkpoint_of(&task)?
	else {
		return Err("generation request phase reached with another checkpoint".to_string());
	};
	let conversation_id = runtime.conversation_id();
	let attempt2 = attempt;
	runtime
		.commit(
			commit_change(move |tx, _current| {
					let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					convert_partial(tx, &mut live, conversation_id)?;
					live.generation = Some(GenerationStatus { attempt: attempt2 as u64, ..Default::default() });
					store_live(&draft, &live)?;
					Ok(None)
			}),
			&context,
		)
		.await?;
	let model = runtime.models().get_model(&reference.provider, &reference.model_id);
	let Some(model) = model else {
		return Box::pin(fail_no_model(&runtime, Some(reference), &context)).await;
	};
	let view = runtime.context_view(conversation_id, &context, Some(cutoff)).await?;
	let mut messages = view.messages.clone();
	// Hook: beforeRequest.
	{
		let hooks = runtime.hooks();
		let payload_messages = messages.clone();
		let runtime2 = runtime.clone();
		let context2 = context.clone();
		hooks
			.each(
				"beforeRequest",
				Arc::new(move |handler: &Arc<dyn crate::tasks::HookHandler>| {
					let payload = BeforeRequestPayload { messages: payload_messages.clone() };
					let runtime = runtime2.clone();
					let context = context2.clone();
					let hooks_table = handler.as_any().downcast_ref::<GenerationHooks>().cloned();
					let future: BoxFuture<Result<(), String>> = Box::pin(async move {
						let Some(hooks) = hooks_table else { return Ok(()) };
						let Some(hook) = &hooks.before_request else { return Ok(()) };
						let replaced = hook(payload, coerce_hook_api(runtime.clone()), context.clone()).await?;
						if let Some(replaced) = replaced {
							messages_replaced(&replaced.messages);
						}
						Ok(())
					});
					future
				}),
			)
			.await?;
	}
	let session_id = ensure_provider_session_id(runtime.as_ref(), &context).await?;
	let deferred = stream_options.deferred.as_ref().and_then(deferred_request);
	let mut options = SimpleStreamOptions {
		base: pi_ai::types::StreamOptions::from(stream_options),
		deferred,
		..Default::default()
	};
	options.base.signal = ai_signal_bridge(&runtime.signal());
	options.base.session_id = Some(session_id);
	if thinking_level != ModelThinkingLevel::Off {
		options.reasoning = Some(to_thinking_level(thinking_level));
	}
	let message = stream_response(&runtime, &model, &messages, options, attempt, &context).await?;
	let request = Request {
		attempt,
		compacted,
		model: reference,
		cutoff,
		messages: Some(view.messages.clone()),
		poll_at: None,
	};
	classify(&runtime, request, message, &context).await
}

thread_local! {
	/// Replacement messages from the last `beforeRequest` invocation (D3:
	/// the erased invoker cannot return values; the request phase reads the
	/// single replacement right after `hooks.each`).
	static REPLACED_MESSAGES: std::cell::RefCell<Option<Vec<pi_ai::Message>>> = const { std::cell::RefCell::new(None) };
}

fn messages_replaced(messages: &[pi_ai::Message]) {
	REPLACED_MESSAGES.with(|cell| *cell.borrow_mut() = Some(messages.to_vec()));
}

fn take_replaced_messages() -> Option<Vec<pi_ai::Message>> {
	REPLACED_MESSAGES.with(|cell| cell.borrow_mut().take())
}

/// Phase `retry`.
async fn retry_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let GenerationCheckpoint::Retry { attempt, compacted, until } = checkpoint_of(&task)? else {
		return Err("generation retry phase reached with another checkpoint".to_string());
	};
	runtime.sleep(until, &context).await?;
	let conversation_id = runtime.conversation_id();
	runtime
		.commit(
			commit_change(move |tx, _current| {
					let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					live.generation = Some(GenerationStatus {
						attempt: (attempt + 1) as u64,
						..Default::default()
					});
					store_live(&draft, &live)?;
					let mut checkpoint = json!({ "phase": "prepare", "attempt": attempt + 1 });
					if let Some(compacted) = compacted {
						checkpoint["compacted"] = json!(compacted);
					}
					Ok(Some(NextTaskState::Running { checkpoint }))
			}),
			&context,
		)
		.await.map(|_| ())?;
	Ok(())
}

/// Phase `poll`.
async fn poll_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let GenerationCheckpoint::Poll { attempt, compacted, model: reference, cutoff, handle, poll_at } =
		checkpoint_of(&task)?
	else {
		return Err("generation poll phase reached with another checkpoint".to_string());
	};
	let model = runtime.models().get_model(&reference.provider, &reference.model_id);
	let Some(model) = model else {
		return Box::pin(fail_no_model(&runtime, Some(reference), &context)).await;
	};
	runtime.sleep(poll_at, &context).await?;
	let message = runtime.models().fetch_deferred(&model, &handle, None).await;
	let request = Request { attempt, compacted, model: reference, cutoff, messages: None, poll_at: Some(poll_at) };
	classify(&runtime, request, message, &context).await
}

/// Phase `tools`.
async fn tools_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let GenerationCheckpoint::Tools { assistant, tools, pending } = checkpoint_of(&task)? else {
		return Err("generation tools phase reached with another checkpoint".to_string());
	};
	let Some(next) = pending.first().cloned() else {
		return Box::pin(finish_tool_round(&runtime, assistant, tools, &context)).await;
	};
	// Sequential round: start the next call and wait for it.
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	let next2 = next.clone();
	let runtime_in = runtime.clone();
	runtime
		.commit(
			commit_change(move |tx: &mut Tx<'_>, _current| {
				let next = next2.clone();
				let runtime = runtime_in.clone();
					let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					let new_task = create_tool_task(tx, &runtime, assistant, &next)?;
					if let Some(slot) = live
						.tools
						.as_mut()
						.and_then(|slots| slots.iter_mut().find(|slot| slot.call_id == next && slot.task_id.is_none()))
					{
						slot.task_id = Some(new_task);
					}
					store_live(&draft, &live)?;
					Ok(Some(NextTaskState::Waiting {
						checkpoint: json!({
							"phase": "tools",
							"assistant": assistant,
							"tools": tools.iter().chain(std::iter::once(&new_task)).collect::<Vec<_>>(),
							"pending": pending[1..],
						}),
						on: vec![new_task],
						policy: crate::types::JoinPolicy::AllSettled,
					}))
			}),
			&context,
		)
		.await.map(|_| ())?;
	Ok(())
}

/// Abort phase.
async fn abort_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let checkpoint = checkpoint_of(&task)?;
	if let GenerationCheckpoint::Poll { model: reference, handle, .. } = &checkpoint {
		if let Some(model) = runtime.models().get_model(&reference.provider, &reference.model_id) {
			if let Err(error) = runtime.models().cancel_deferred(&model, handle, None).await {
				runtime.report(thrown_text(&error));
			}
		}
	}
	let conversation_id = runtime.conversation_id();
	// Runs after the round's tool tasks are terminal; calls never started
	// get `aborted` results (spec §8.5).
	let unstarted = match &checkpoint {
		GenerationCheckpoint::Tools { assistant, pending, .. } => {
			read_calls(&runtime, *assistant, pending, &context).await?
		}
		_ => Vec::new(),
	};
	let task_id = runtime.task_id();
	runtime
		.commit(
			commit_change(move |tx, _current| {
				let unstarted = unstarted.clone();
					let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					convert_partial(tx, &mut live, conversation_id)?;
					for call in &unstarted {
						let result = harness_error("aborted", &format!("Tool {} was aborted", call.name));
						append_tool_result(tx, conversation_id, call, &result, 0)?;
					}
					end_run_on_tx(
						&mut live,
						tx,
						&task_id,
						SubmissionSettlement::Unanswered { reason: "aborted".to_string(), detail: None },
					)?;
					store_live(&draft, &live)?;
					Ok(Some(NextTaskState::Terminal {
						outcome: TaskOutcome {
							status: TaskOutcomeStatus::Aborted,
							result: None,
							error: None,
							reason: Some("aborted".to_string()),
						},
					}))
			}),
			&context,
		)
		.await.map(|_| ())?;
	Ok(())
}

/// Append a committed partial left by an interrupted, aborted, faulted, or
/// orphaned attempt as an aborted assistant entry; the caller replaces or
/// removes `generation` (sync on the Session line; tasks D3).
pub fn convert_partial(
	tx: &Tx<'_>,
	live: &mut LiveState,
	conversation_id: crate::harness::types::ConversationId,
) -> Result<(), String> {
	let Some(partial) = live.generation.as_ref().and_then(|generation| generation.message.clone()) else {
		return Ok(());
	};
	let mut message: AssistantMessage = serde_json::from_value(partial)
		.map_err(|error| format!("Stored partial is not an AssistantMessage: {error}"))?;
	message.stop_reason = StopReason::Aborted;
	append_assistant(tx, conversation_id, &message).map(|_| ())
}

/// Stream one request and return the terminal message. Partials commit as
/// trailing writes at most every `progress.partialIntervalMs` (default
/// 100 ms) with one commit in flight; the end of the stream stops the
/// throttle and awaits that commit, so no stale partial lands after the
/// outcome.
async fn stream_response(
	runtime: &Arc<dyn TaskRuntime>,
	model: &Model,
	messages: &[pi_ai::Message],
	options: SimpleStreamOptions,
	attempt: u32,
	context: &Context,
) -> Result<AssistantMessage, String> {
	let interval = std::time::Duration::from_millis(runtime.settings().progress.partial_interval_ms);
	let request_context = pi_ai::types::Context {
		system_prompt: None,
		messages: messages.to_vec(),
		tools: None,
	};
	let events = runtime.models().stream_simple(model, &request_context, Some(options.into()));
	use futures_util::{stream::FuturesUnordered, StreamExt};
	let mut stream = events;
	let mut pending = None;
	let mut timer = tokio::time::sleep(interval);
	tokio::pin!(timer);
	let mut timer_active = false;
	let mut in_flight = FuturesUnordered::new();
	let mut in_flight_active = false;
	loop {
		tokio::select! {
			event = stream.next() => {
				let Some(event) = event else { break };
				let partial = match &event {
					pi_ai::AssistantMessageEvent::Start { partial }
					| pi_ai::AssistantMessageEvent::TextStart { partial, .. }
					| pi_ai::AssistantMessageEvent::TextDelta { partial, .. }
					| pi_ai::AssistantMessageEvent::TextEnd { partial, .. }
					| pi_ai::AssistantMessageEvent::ThinkingStart { partial, .. }
					| pi_ai::AssistantMessageEvent::ThinkingDelta { partial, .. }
					| pi_ai::AssistantMessageEvent::ThinkingEnd { partial, .. }
					| pi_ai::AssistantMessageEvent::ToolcallStart { partial, .. }
					| pi_ai::AssistantMessageEvent::ToolcallDelta { partial, .. }
					| pi_ai::AssistantMessageEvent::ToolcallEnd { partial, .. } => partial,
					_ => continue,
				};
				if partial.content.is_empty() {
					continue;
				}
				pending = Some(serde_json::to_value(partial).expect("AssistantMessage serializes"));
				if !timer_active && !in_flight_active {
					timer.as_mut().reset(tokio::time::Instant::now() + interval);
					timer_active = true;
				}
			}
			() = &mut timer, if timer_active => {
				timer_active = false;
				if let Some(message) = pending.take() {
					let runtime = runtime.clone();
					let conversation_id = runtime.conversation_id();
					let context = context.clone();
					let attempt = attempt;
					in_flight.push(async move {
						// Rejections after an abort mark or close are expected; the committed state stays consistent.
						let _committed = runtime
							.commit(
								commit_change(move |tx: &mut Tx<'_>, _current| {
									let message = message.clone();
									let draft = tx
										.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
										.map_err(|e| e.to_string())?;
									let mut live = read_live(&draft);
									let generation = live.generation.get_or_insert_with(|| GenerationStatus {
										attempt: attempt as u64,
										..Default::default()
									});
									let _ = generation;
									let mut value = serde_json::to_value(&live).expect("LiveState serializes");
									assign_json(&mut value["generation"], "message", message);
									let replaced: LiveState = serde_json::from_value(value).unwrap_or(live.clone());
									live = replaced;
									store_live(&draft, &live)?;
									Ok(None)
								}),
								&context,
							)
							.await;
					});
					in_flight_active = true;
				}
			}
			_ = in_flight.next(), if in_flight_active => {
				in_flight_active = false;
				if pending.is_some() {
					timer.as_mut().reset(tokio::time::Instant::now() + interval);
					timer_active = true;
				}
			}
		}
	}
	let result = stream.result().await;
	if in_flight_active {
		let _ = in_flight.next().await;
	}
	Ok(result)
}

/// Classify a terminal provider message in one commit that also clears the
/// partial.
async fn classify(
	runtime: &Arc<dyn TaskRuntime>,
	request: Request,
	message: AssistantMessage,
	context: &Context,
) -> Result<(), String> {
	// An abort mark or close: the abort invocation or the reopened run
	// handles the committed state.
	if let Some(signal) = context.abort_signal() {
		if signal.aborted() {
			return Err(abort_reason_text(&signal));
		}
	}
	let conversation_id = runtime.conversation_id();
	let Request { attempt, compacted, model: reference, cutoff, poll_at, .. } = request.clone();
	if message.stop_reason == StopReason::Deferred {
		if let Some(handle) = &message.deferred {
			let poll_at = std::cmp::max(
				runtime.now() as u64 + handle.poll_after_ms.unwrap_or(DEFAULT_POLL_AFTER_MS),
				poll_at.map_or(0, |previous| previous + 1),
			);
			runtime
				.commit(
					commit_change(move |tx, _current| {
							let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
							let mut live = read_live(&draft);
							live.generation = Some(GenerationStatus {
								attempt: attempt as u64,
								deferred: Some(crate::harness::live::DeferredPoll { poll_at }),
								..Default::default()
							});
							store_live(&draft, &live)?;
							Ok(Some(NextTaskState::Running {
								checkpoint: json!({
									"phase": "poll",
									"attempt": attempt,
									"compacted": compacted,
									"model": reference,
									"cutoff": cutoff,
									"handle": handle,
									"pollAt": poll_at,
								}),
							}))
					}),
					context,
				)
				.await?;
			return Ok(());
		}
	}
	// Hook: afterResponse.
	{
		let hooks = runtime.hooks();
		let message2 = message.clone();
		let runtime2 = runtime.clone();
		let context2 = context.clone();
		hooks
			.each(
				"afterResponse",
				Arc::new(move |handler: &Arc<dyn crate::tasks::HookHandler>| {
					let message = message2.clone();
					let runtime = runtime2.clone();
					let context = context2.clone();
					let hooks_table = handler.as_any().downcast_ref::<GenerationHooks>().cloned();
						Box::pin(async move {
						let Some(hooks) = hooks_table else { return Ok(()) };
						if let Some(hook) = &hooks.after_response {
							hook(message, coerce_hook_api(runtime.clone()), context).await?;
						}
						Ok(())
						})
				}),
			)
			.await?;
	}
	let calls: Vec<ToolCall> = message
		.content
		.iter()
		.filter_map(|block| match block {
			AssistantContent::ToolCall(call) => Some(call.clone()),
			_ => None,
		})
		.collect();
	if message.stop_reason == StopReason::ToolUse && !calls.is_empty() {
		return Box::pin(start_tool_round(runtime, request, message, calls, context)).await;
	}
	if matches!(message.stop_reason, StopReason::Stop | StopReason::Length | StopReason::ToolUse) {
		return answer(runtime, message, context).await;
	}
	// The retry and compaction policies govern the next attempt, so they are
	// read now rather than pinned at preparation.
	let settings = runtime.settings();
	let overflow = message.stop_reason == StopReason::Error && is_context_overflow(&message, None);
	if overflow && compacted.is_none() && settings.compaction.enabled {
		let policy = settings.compaction;
		let view = runtime.context_view(conversation_id, context, Some(cutoff)).await?;
		if select_cut(&view, policy.keep_recent_tokens).is_some() {
			let text = message.error_message.clone().unwrap_or_else(|| "Context overflow".to_string());
			let attempt2 = attempt;
			runtime
				.commit(
					commit_change(move |tx, _current| {
						let message = message.clone();
						let text = text.clone();
							let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
							let mut live = read_live(&draft);
							append_assistant(tx, conversation_id, &message)?;
							live.generation = None;
							store_live(&draft, &live)?;
							let child = create_compaction(
								tx,
								conversation_id,
								CompactionInput { reason: CompactionReason::Overflow, instructions: None },
								Some(runtime.task_id()),
							)?;
							Ok(Some(NextTaskState::Waiting {
								checkpoint: json!({
									"phase": "prepare",
									"attempt": attempt2,
									"compacted": child,
									"overflow": text,
								}),
								on: vec![child],
								policy: crate::types::JoinPolicy::AllSettled,
							}))
					}),
					context,
				)
				.await?;
			return Ok(());
		}
	}
	let policy: RetryPolicy = to_retry_policy(&settings.retry);
	// An overflow is never retried: only a compaction can make the next
	// request fit.
	let retry = message.stop_reason == StopReason::Error
		&& !overflow
		&& is_retryable_assistant_error(&message)
		&& policy.enabled
		&& attempt <= policy.max_retries;
	let until = if retry { runtime.now() as u64 + retry_delay_ms(&policy, attempt) } else { 0 };
	let attempt2 = attempt;
	let compacted2 = compacted;
	runtime
		.commit(
			commit_change(move |tx, _current| {
				let message = message.clone();
					let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					append_assistant(tx, conversation_id, &message)?;
					if retry {
						live.generation = Some(GenerationStatus {
							attempt: attempt2 as u64,
							retry: Some(RetryBackoff {
								at: until,
								error: message.error_message.clone().unwrap_or_default(),
							}),
							..Default::default()
						});
						store_live(&draft, &live)?;
						let mut checkpoint = json!({ "phase": "retry", "attempt": attempt2, "until": until });
						if let Some(compacted) = compacted2 {
							checkpoint["compacted"] = json!(compacted);
						}
						return Ok(Some(NextTaskState::Running { checkpoint }));
					}
					let text = message.error_message.clone().unwrap_or_else(|| {
						format!("Model response ended with stop reason {}", stop_reason_str(message.stop_reason))
					});
					end_run_on_tx(
						&mut live,
						tx,
						&runtime.task_id(),
						SubmissionSettlement::Unanswered {
							reason: "model_error".to_string(),
							detail: Some(Value::String(text.clone())),
						},
					)?;
					store_live(&draft, &live)?;
					Ok(Some(NextTaskState::Terminal {
						outcome: TaskOutcome {
							status: TaskOutcomeStatus::Failed,
							result: None,
							error: Some(TaskOutcomeError {
								message: text,
								detail: Some(json!({ "reason": "model_error" })),
							}),
							reason: None,
						},
					}))
			}),
			context,
		)
		.await.map(|_| ())?;
	Ok(())
}


/// Bridge a chord abort signal into the pi-ai request signal: the returned
/// controller aborts when (and only while) the chord signal is live (D:
/// pi-ai request options carry their own `AbortSignal` type; request
/// cancellation follows the invocation's chord signal through this bridge).
fn ai_signal_bridge(signal: &pi_chord::context::AbortSignal) -> Option<pi_ai::utils::abort::AbortSignal> {
	let controller = pi_ai::utils::abort::AbortController::new();
	let forwarded = controller.signal();
	let detach = signal.add_listener(move || {
		pi_ai::utils::abort::AbortSignal::abort_with(&forwarded, pi_ai::utils::abort::AbortReason::Aborted);
	});
	// The bridge lives for the request; detach is best-effort and leaks one
	// listener entry per request otherwise.
	std::mem::forget(detach);
	Some(controller.signal())
}

fn to_thinking_level(level: ModelThinkingLevel) -> pi_ai::types::ThinkingLevel {
	use pi_ai::types::ThinkingLevel;
	match level {
		ModelThinkingLevel::Off => ThinkingLevel::Minimal,
		ModelThinkingLevel::Minimal => ThinkingLevel::Minimal,
		ModelThinkingLevel::Low => ThinkingLevel::Low,
		ModelThinkingLevel::Medium => ThinkingLevel::Medium,
		ModelThinkingLevel::High => ThinkingLevel::High,
		ModelThinkingLevel::Xhigh => ThinkingLevel::Xhigh,
		ModelThinkingLevel::Max => ThinkingLevel::Max,
	}
}

fn thrown_text(error: &pi_ai::utils::thrown::ThrownValue) -> String {
	format!("{error:?}")
}

fn stop_reason_str(reason: StopReason) -> &'static str {
	match reason {
		StopReason::Pending => "pending",
		StopReason::Stop => "stop",
		StopReason::Length => "length",
		StopReason::ToolUse => "toolUse",
		StopReason::Error => "error",
		StopReason::Aborted => "aborted",
		StopReason::Deferred => "deferred",
	}
}

fn to_retry_policy(policy: &crate::harness::types::ConversationRetryPolicy) -> RetryPolicy {
	RetryPolicy {
		enabled: policy.enabled,
		max_retries: policy.max_retries,
		base_delay_ms: policy.base_delay_ms,
		max_agent_delay_ms: policy.max_agent_delay_ms,
	}
}

/// A final answer; the final boundary places queued items (spec §6). The
/// first `onYield` continuation appends a user message and hands the run to
/// a successor generation, but only when the boundary selected no user item
/// and no reset. Otherwise the run's inputs settle `done`, and selected user
/// items start the next run.
async fn answer(runtime: &Arc<dyn TaskRuntime>, message: AssistantMessage, context: &Context) -> Result<(), String> {
	let mut continuation: Option<crate::harness::types::UserInput> = None;
	{
		let hooks = runtime.hooks();
		let message2 = message.clone();
		let runtime2 = runtime.clone();
		let context2 = context.clone();
		hooks
			.each(
				"onYield",
				Arc::new(move |handler: &Arc<dyn crate::tasks::HookHandler>| {
					let message = message2.clone();
					let runtime = runtime2.clone();
					let context = context2.clone();
					let hooks_table = handler.as_any().downcast_ref::<GenerationHooks>().cloned();
						Box::pin(async move {
						let Some(hooks) = hooks_table else { return Ok(()) };
						let Some(hook) = &hooks.on_yield else { return Ok(()) };
						let result = hook(message, coerce_hook_api(runtime.clone()), context).await?;
						if let Some(result) = result {
							set_continuation(result.continue_);
						}
						Ok(())
						})
				}),
			)
			.await?;
	}
	let continuation = take_continuation();
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	let modes = QueueModes {
		steering_mode: runtime.settings().steering_mode,
		follow_up_mode: runtime.settings().follow_up_mode,
	};
	let message2 = message.clone();
	runtime
		.commit(
			commit_change(move |tx, _current| {
				let message = message2.clone();
					// Queue modes are read on the Session line, when the
					// boundary is decided.
					let mut boundary = prepare_boundary(tx, conversation_id, modes)?;
					let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					let entry = append_assistant(tx, conversation_id, &message)?;
					let result = NextTaskState::Terminal {
						outcome: TaskOutcome {
							status: TaskOutcomeStatus::Completed,
							result: Some(json!({ "entryId": entry.id })),
							error: None,
							reason: None,
						},
					};
					let boundary_result = apply_boundary(tx, &mut boundary, crate::harness::inbox::BoundaryAt::Final, runtime.now() as i64)?;
					store_live(&draft, &live)?; // boundary stored its own inbox
					let live = read_live(&tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?);
					let mut live = live;
					if let Some(continuation) = &continuation {
						if boundary_result.users.is_empty() && !boundary_result.reset {
							let user = pi_ai::types::UserMessage {
								content: continuation.clone(),
								timestamp: runtime.now() as i64,
							};
							let mut entry_draft = crate::types::EntryDraft {
								kind: Some(USER_ENTRY.kind.clone()),
								..Default::default()
							};
							entry_draft.model = Some(vec![pi_ai::Message::User(user)]);
							tx.append_entry(conversation_id, entry_draft).map_err(|e| e.to_string())?;
							let successor = create_generation(tx, conversation_id)?;
							hand_over(&mut live, task_id, successor);
							live.generation = None;
							store_live(&draft, &live)?;
							return Ok(Some(result));
						}
					}
					end_run_on_tx(
						&mut live,
						tx,
						&task_id,
						SubmissionSettlement::Done { answer: entry.id },
					)?;
					if !boundary_result.users.is_empty() {
						start_run(tx, conversation_id, &mut live, boundary_result.users)?;
					}
					store_live(&draft, &live)?;
					Ok(Some(result))
			}),
			context,
		)
		.await?;
	Ok(())
}

thread_local! {
	static CONTINUATION: std::cell::RefCell<Option<crate::harness::types::UserInput>> = const { std::cell::RefCell::new(None) };
}

fn set_continuation(content: crate::harness::types::UserInput) {
	CONTINUATION.with(|cell| *cell.borrow_mut() = Some(content));
}

fn take_continuation() -> Option<crate::harness::types::UserInput> {
	CONTINUATION.with(|cell| cell.borrow_mut().take())
}

/// Append the tool-calling answer and start its tool round in one commit
/// (spec §8.3). A call to a tool the request did not offer gets its
/// `tool_unavailable` result here; every other call gets a tool task owned
/// by the generation, only the first one now when the round is sequential.
/// The generation then waits for them in its `tools` phase, keeping the run.
async fn start_tool_round(
	runtime: &Arc<dyn TaskRuntime>,
	request: Request,
	message: AssistantMessage,
	calls: Vec<ToolCall>,
	context: &Context,
) -> Result<(), String> {
	let conversation_id = runtime.conversation_id();
	let messages = match request.messages {
		Some(messages) => messages,
		None => runtime.context_view(conversation_id, context, Some(request.cutoff)).await?.messages,
	};
	let offered: std::collections::BTreeSet<String> =
		get_current_tools(&messages).into_iter().map(|tool| tool.name).collect();
	// Read as the round starts; a tool is resolved as its tool task resolves
	// it.
	let tools = runtime.agent(context).await?.tools;
	let sequential = runtime.settings().tool_execution == crate::harness::api::ToolExecutionMode::Sequential
		|| calls.iter().any(|call| {
			offered.contains(&call.name)
				&& tools
					.iter()
					.find(|tool| tool.tool.name == call.name)
					.map(|tool| tool.execution_mode == Some(crate::harness::api::ToolExecutionMode::Sequential))
					.unwrap_or(false)
		});
	let task_id = runtime.task_id();
	runtime
		.commit(
			commit_change(move |tx, _current| {
				let calls = calls.clone();
				let offered = offered.clone();
					let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					let entry = append_assistant(tx, conversation_id, &message)?;
					let mut slots: Vec<ToolSlot> = Vec::new();
					let mut tool_tasks: Vec<TaskId> = Vec::new();
					let mut pending: Vec<String> = Vec::new();
					for call in &calls {
						if !offered.contains(&call.name) {
							let unavailable = harness_error("tool_unavailable", &format!("Tool {} is not available", call.name));
							let result = append_tool_result(tx, conversation_id, call, &unavailable, runtime.now() as i64)?;
							slots.push(ToolSlot {
								call_id: call.id.clone(),
								name: call.name.clone(),
								task_id: None,
								status: SlotStatus::Done,
								entry: Some(result.id),
								..Default::default()
							});
							continue;
						}
						if sequential && !tool_tasks.is_empty() {
							pending.push(call.id.clone());
							slots.push(ToolSlot {
								call_id: call.id.clone(),
								name: call.name.clone(),
								status: SlotStatus::Pending,
								..Default::default()
							});
							continue;
						}
						let child = create_tool_task(tx, &runtime, entry.id, &call.id)?;
						tool_tasks.push(child);
						slots.push(ToolSlot {
							call_id: call.id.clone(),
							name: call.name.clone(),
							task_id: Some(child),
							status: SlotStatus::Pending,
							..Default::default()
						});
					}
					live.generation = None;
					live.tools = Some(slots);
					store_live(&draft, &live)?;
					Ok(Some(NextTaskState::Waiting {
						checkpoint: json!({
							"phase": "tools",
							"assistant": entry.id,
							"tools": tool_tasks,
							"pending": pending,
						}),
						on: tool_tasks,
						policy: crate::types::JoinPolicy::AllSettled,
					}))
			}),
			context,
		)
		.await?;
	let _ = task_id;
	Ok(())
}

/// The round's tools are terminal: apply their controls and either end the
/// run at the final boundary (`terminate`, `handoff`, or a queued reset) or
/// hand it to the next generation at the `postTools` boundary (spec §8.5).
async fn finish_tool_round(
	runtime: &Arc<dyn TaskRuntime>,
	assistant: EntryId,
	tools: Vec<TaskId>,
	context: &Context,
) -> Result<(), String> {
	let conversation_id = runtime.conversation_id();
	let mut controls: BTreeMap<TaskId, Option<ToolControl>> = BTreeMap::new();
	let outcomes = runtime.outcomes(&tools, context).await?;
	for (id, outcome) in tools.iter().zip(outcomes.iter()) {
		let control = if outcome.status == TaskOutcomeStatus::Completed {
			outcome
				.result
				.as_ref()
				.and_then(|result| serde_json::from_value::<ToolControl>(result.get("control").cloned().unwrap_or(Value::Null)).ok())
		} else {
			None
		};
		controls.insert(*id, control);
	}
	let snapshot = runtime
		.snapshot(&crate::harness::live::live_doc(), conversation_id, context)
		.await?;
	let slots: Vec<ToolSlot> = snapshot
		.map(|value| serde_json::from_value::<LiveState>(Value::Object(value)).ok())
		.flatten()
		.and_then(|live| live.tools)
		.unwrap_or_default();
	let results: Vec<EntryId> = slots.iter().filter_map(|slot| slot.entry).collect();
	let runtime2 = runtime.clone();
	// Hook: afterTools.
	{
		let hooks = runtime.hooks();
		let results2 = results.clone();
		let context2 = context.clone();
		hooks
			.each(
				"afterTools",
				Arc::new(move |handler: &Arc<dyn crate::tasks::HookHandler>| {
					let runtime = runtime2.clone();
					let context = context2.clone();
					let hooks_table = handler.as_any().downcast_ref::<GenerationHooks>().cloned();
					let results = results2.clone();
					Box::pin(async move {
							let Some(hooks) = hooks_table else { return Ok(()) };
							if let Some(hook) = &hooks.after_tools {
								hook(assistant, results, coerce_hook_api(runtime.clone()), context).await?;
							}
							Ok(())
						})
				}),
			)
			.await?;
	}
	// Every call of the round, including those answered without a task, must
	// ask to terminate.
	let terminate = !slots.is_empty()
		&& slots.iter().all(|slot| {
			slot.task_id
				.and_then(|id| controls.get(&id).cloned().flatten())
				.map(|control| control.terminate == Some(true))
				.unwrap_or(false)
		});
	let added: Vec<String> = controls.values().flatten().filter_map(|control| control.add_tools.clone()).flatten().collect();
	// The last handoff in call order wins.
	let handoff = controls
		.values()
		.rev()
		.find_map(|control| control.as_ref().and_then(|control| control.handoff.clone()));
	let task_id = runtime.task_id();
	let modes = QueueModes {
		steering_mode: runtime.settings().steering_mode,
		follow_up_mode: runtime.settings().follow_up_mode,
	};
	runtime
		.commit(
			commit_change(move |tx, _current| {
				let added = added.clone();
				let handoff = handoff.clone();
					let mut boundary = prepare_boundary(tx, conversation_id, modes)?;
					if !added.is_empty() {
						add_tools(tx, conversation_id, &added)?;
					}
					let draft = tx.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					let now = runtime.now() as i64;
					if terminate || handoff.is_some() {
						if let Some(handoff) = handoff {
							let message = pi_ai::types::UserMessage {
								content: pi_ai::types::UserContentField::Text(handoff),
								timestamp: now,
							};
							let mut entry_draft = crate::types::EntryDraft {
								kind: Some(RESET_ENTRY.kind.clone()),
								..Default::default()
							};
							entry_draft.model = Some(vec![pi_ai::Message::User(message)]);
							entry_draft.head = Some(crate::types::EntryHead::SelfMarker(crate::types::SelfMarker));
							let entry = tx.append_entry(conversation_id, entry_draft).map_err(|e| e.to_string())?;
							boundary.head = Some(entry.id);
						}
						let boundary_result =
							apply_boundary(tx, &mut boundary, crate::harness::inbox::BoundaryAt::Final, now)?;
						end_run_on_tx(&mut live, tx, &task_id, SubmissionSettlement::Done { answer: assistant })?;
						if !boundary_result.users.is_empty() {
							start_run(tx, conversation_id, &mut live, boundary_result.users)?;
						}
					} else {
						let boundary_result =
							apply_boundary(tx, &mut boundary, crate::harness::inbox::BoundaryAt::PostTools, now)?;
						if boundary_result.reset {
							// The queued reset cut the run's context before an
							// answer.
							end_run_on_tx(
								&mut live,
								tx,
								&task_id,
								SubmissionSettlement::Unanswered { reason: "reset".to_string(), detail: None },
							)?;
							if !boundary_result.users.is_empty() {
								start_run(tx, conversation_id, &mut live, boundary_result.users)?;
							}
						} else {
							live.tools = None;
							if let Some(run) = &mut live.run {
								if run.task_id == task_id {
									run.inputs.extend(boundary_result.users);
								}
							}
							let successor = create_generation(tx, conversation_id)?;
							hand_over(&mut live, task_id, successor);
						}
					}
					store_live(&draft, &live)?;
					Ok(Some(NextTaskState::Terminal {
						outcome: TaskOutcome {
							status: TaskOutcomeStatus::Completed,
							result: Some(json!({ "entryId": assistant })),
							error: None,
							reason: None,
						},
					}))
			}),
			context,
		)
		.await?;
	Ok(())
}

/// Append a provider result and add its usage to `pi.usage` in the same
/// commit.
///
/// Every built-in writer of assistant entries goes through here, so the
/// usage ledger stays complete.
fn append_assistant(
	tx: &Tx<'_>,
	conversation_id: crate::harness::types::ConversationId,
	message: &AssistantMessage,
) -> Result<crate::types::EntryRecord, String> {
	record_usage(
		tx,
		conversation_id,
		crate::harness::usage::UsageBucket::Models,
		&format!("{}/{}", message.provider, message.model),
		&message.usage,
	)?;
	let mut draft = crate::types::EntryDraft {
		kind: Some(ASSISTANT_ENTRY.kind.clone()),
		..Default::default()
	};
	draft.model = Some(vec![pi_ai::Message::Assistant(message.clone())]);
	tx.append_entry(conversation_id, draft).map_err(|e| e.to_string())
}

/// Start a run for `inputs`, placed input submissions: a new generation
/// takes `pi.live.run`.
pub fn start_run(
	tx: &Tx<'_>,
	conversation_id: crate::harness::types::ConversationId,
	live: &mut LiveState,
	inputs: Vec<SubmissionId>,
) -> Result<(), String> {
	let task_id = create_generation(tx, conversation_id)?;
	live.run = Some(crate::harness::live::RunState { task_id, inputs });
	Ok(())
}

/// A generation owned by its conversation.
fn create_generation(
	tx: &Tx<'_>,
	conversation_id: crate::harness::types::ConversationId,
) -> Result<TaskId, String> {
	tx.create_task(
		&generation_task(),
		json!({}),
		crate::types::TaskOptions {
			conversation_id: Some(conversation_id),
			ownership: crate::types::TaskOwnership::Conversation,
			background: None,
		},
	)
	.map_err(|e| e.to_string())
}

/// Hand run control from `from` to `to`; the run's inputs move with it.
pub fn hand_over(live: &mut LiveState, from: TaskId, to: TaskId) {
	if let Some(run) = &mut live.run {
		if run.task_id == from {
			run.task_id = to;
		}
	}
}

// Unused references kept for signature parity with upstream.
#[allow(unused)]
fn _markers(_t: Option<(WhenBusy, SettledTask, TaskState)>) {}
