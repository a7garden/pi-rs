//! Port of `packages/durable/src/harness/compaction.ts` (the built-in
//! `pi.compaction` task, spec §8.7).
//!
//! Erasure (D1 — see `tasks/mod.rs`): checkpoint JSON tagged `phase`
//! (`select`/`summarize`/`retry`); result is the erased
//! [`crate::harness::types::CompactionResult`]. D2 — `complete_simple`
//! replaces upstream `completeSimple`; options carry the same fields.

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::types::{AssistantContent, AssistantMessage, ModelThinkingLevel, SimpleStreamOptions, StopReason, UserContent};
use pi_ai::utils::estimate::{calculate_context_tokens, estimate_message_tokens};
use pi_ai::utils::retry::{is_retryable_assistant_error, retry_delay_ms, RetryPolicy};
use pi_chord::context::Context;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::entries::COMPACTION_ENTRY;
use crate::harness::types::coerce_hook_api;
use crate::harness::util::abort_reason_text;
use crate::harness::context::order_tool_results;
use crate::harness::live::{LiveState,
	add_compaction_status, compaction_status, read_live, remove_compaction_status, store_live, CompactionStatus,
};
use crate::harness::provider::ensure_provider_session_id;
use crate::harness::submissions::admit_submission;
use crate::harness::types::{
	CompactionHooks, CompactionReason, CompactionResult, ConversationId, EntryId, HookApi, ModelRef,
	TaskId, TaskRecord, WhenBusy,
};
use crate::harness::usage::record_usage;
use crate::session::Tx;
use crate::tasks::{commit_change, define_task, BoxFuture, NextTaskState, PhaseHandler, TaskDefinition, TaskRuntime};
use crate::types::{EntryDraft, EntryHead, TaskOutcome, TaskOutcomeError, TaskOutcomeStatus, TaskOptions};

/// Input of the compaction task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionInput {
	pub reason: CompactionReason,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub instructions: Option<String>,
}

/// The pinned summarization request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryRequest {
	pub attempt: u32,
	pub model: ModelRef,
	pub thinking_level: ModelThinkingLevel,
	pub stream_options: Value,
	pub max_tokens: u64,
	/// Newest entry of the context the range was selected from.
	pub tail: EntryId,
	/// First entry kept verbatim; the summary's `head`.
	pub first_kept: EntryId,
}

/// Erased checkpoint of the compaction task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
pub enum CompactionCheckpoint {
	Select,
	Summarize { #[serde(flatten)] request: SummaryRequest },
	Retry { until: u64, #[serde(flatten)] request: SummaryRequest },
}

/// Longest tool result text a serialized summary source keeps.
const TOOL_RESULT_MAX_CHARS: usize = 2000;

const SUMMARY_PREFIX: &str =
	"The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
const SUMMARY_SUFFIX: &str = "\n</summary>";

const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

const SUMMARIZATION_PROMPT: &str = r#"The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work. If the conversation starts with an earlier summary, preserve its information and fold the newer messages into it.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."#;

fn compaction_definition() -> TaskDefinition {
	let mut phases: BTreeMap<String, PhaseHandler> = BTreeMap::new();
	phases.insert("select".to_string(), Arc::new(|task, runtime, context| Box::pin(select_phase(task, runtime, context)) as BoxFuture<Result<(), String>>));
	phases.insert("summarize".to_string(), Arc::new(|task, runtime, context| Box::pin(summarize_phase(task, runtime, context)) as BoxFuture<Result<(), String>>));
	phases.insert("retry".to_string(), Arc::new(|task, runtime, context| Box::pin(retry_phase(task, runtime, context)) as BoxFuture<Result<(), String>>));
	TaskDefinition {
		name: "pi.compaction".to_string(),
		version: 1,
		initial: Arc::new(|_| json!({ "phase": "select" })),
		phases: Arc::new(phases),
		abort: Arc::new(|task, runtime, context| Box::pin(abort_phase(task, runtime, context)) as BoxFuture<Result<(), String>>),
		migrate: None,
		hooks: None,
	}
}

static COMPACTION_TASK_STATIC: std::sync::LazyLock<crate::harness::types::AnyTask> =
	std::sync::LazyLock::new(|| crate::harness::types::AnyTask::new(compaction_definition()));

/// Built-in compaction task: select an old prefix of the model context,
/// summarize it, and place a summary entry whose `head` is the first kept
/// entry. A compaction the generation owns blocks it and appends directly; a
/// conversation-owned one places its summary through a write submission.
pub fn compaction_task() -> crate::harness::types::AnyTask {
	COMPACTION_TASK_STATIC.clone()
}

fn compaction_checkpoint(record: &TaskRecord) -> Result<CompactionCheckpoint, String> {
	serde_json::from_value(record.state.checkpoint.clone().unwrap_or(Value::Null))
		.map_err(|error| format!("Compaction checkpoint is invalid: {error}"))
}

/// Create a compaction task with its status in this commit. `owner` is the
/// generation that waits for it (a blocking compaction); without one it is
/// conversation-owned, and `background` unless it is manual.
pub fn create_compaction(
	tx: &Tx<'_>,
	conversation_id: ConversationId,
	input: CompactionInput,
	owner: Option<TaskId>,
) -> Result<TaskId, String> {
	let reason = input.reason;
	let ownership = match owner {
		None => crate::types::TaskOwnership::Conversation,
		Some(owner) => crate::types::TaskOwnership::Task { task_id: owner },
	};
	let background = owner.is_none() && reason != CompactionReason::Manual;
	let task_id = tx.create_task(
		&compaction_task(),
		serde_json::to_value(&input).expect("CompactionInput serializes"),
		TaskOptions {
			ownership,
			conversation_id: Some(conversation_id),
			background: Some(background),
		},
	)
	.map_err(|e| e.to_string())?;
	let draft = tx
		.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
		.map_err(|e| e.to_string())?;
	let mut live = read_live(&draft);
	add_compaction_status(
		&mut live,
		CompactionStatus {
			task_id,
			reason,
			blocking: owner.is_some(),
			attempt: 1,
			retry: None,
		},
	);
	store_live(&draft, &live)?;
	Ok(task_id)
}

/// Index in `view.entries` of the first entry a summary keeps, or `None`
/// when there is nothing to compact (spec §8.7). Walks back from the tail
/// until `keep_recent_tokens` are kept, then cuts at the first candidate at
/// or after that entry: an entry whose contribution starts with a user or
/// assistant message, never a tool result, and never a user entry that a
/// result of the preceding assistant's calls still follows.
pub fn select_cut(view: &crate::session::ContextView, keep_recent_tokens: u64) -> Option<usize> {
	let contributions = &view.contributions;
	let start = if view.head.is_none() { 0 } else { 1 };
	let mut candidates: Vec<usize> = Vec::new();
	for index in start..contributions.len() {
		if is_candidate(contributions, index) {
			candidates.push(index);
		}
	}
	let mut kept: u64 = 0;
	let mut cut: Option<usize> = None;
	for index in (start..contributions.len()).rev() {
		for message in &contributions[index] {
			kept += estimate_message_tokens(message);
		}
		if kept < keep_recent_tokens {
			continue;
		}
		cut = candidates
			.iter()
			.find(|candidate| **candidate >= index)
			.or_else(|| candidates.last())
			.copied();
		break;
	}
	let cut = cut?;
	for index in start..cut {
		if !contributions[index].is_empty() {
			return Some(cut);
		}
	}
	None
}

fn is_candidate(contributions: &[Vec<pi_ai::Message>], index: usize) -> bool {
	let first = contributions[index].first();
	match first {
		Some(pi_ai::Message::Assistant(_)) => return true,
		Some(pi_ai::Message::User(_)) => {}
		_ => return false,
	}
	// A result of the preceding assistant's calls that follows this entry,
	// before the next assistant, belongs before it.
	let mut calls: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
	for before in (0..index).rev() {
		let Some(assistant) = contributions[before].iter().rev().find_map(|message| match message {
			pi_ai::Message::Assistant(assistant) => Some(assistant),
			_ => None,
		}) else {
			continue;
		};
		calls = assistant
			.content
			.iter()
			.filter_map(|content| match content {
				AssistantContent::ToolCall(call) => Some(call.id.clone()),
				_ => None,
			})
			.collect();
		break;
	}
	if calls.is_empty() {
		return true;
	}
	for (after, contribution) in contributions.iter().enumerate().skip(index) {
		for (position, message) in contribution.iter().enumerate() {
			match message {
				pi_ai::Message::Assistant(_) if after > index || position > 0 => return true,
				pi_ai::Message::ToolResult(result) if calls.contains(&result.tool_call_id) => return false,
				_ => {}
			}
		}
	}
	true
}

/// Model messages of the entries before `cut`: the head marker first,
/// ordered like model context (spec §2.1).
pub fn summarized_messages(view: &crate::session::ContextView, cut: usize) -> Vec<pi_ai::Message> {
	let flat: Vec<pi_ai::Message> = view.contributions.iter().take(cut).flatten().cloned().collect();
	order_tool_results(flat)
}

/// Size of a request over `view` followed by `extra` (spec §8.3): the usage
/// of the newest assistant appended after the head marker, whose request
/// included the marker, plus estimates of the messages after it; without
/// one, estimates of every message.
pub fn estimate_context(view: &crate::session::ContextView, extra: &[pi_ai::Message]) -> u64 {
	let mut measured: Option<AssistantMessage> = None;
	let after = view.head.as_ref().map(|head| head.id);
	for index in (0..view.entries.len()).rev() {
		if measured.is_some() {
			break;
		}
		if after.is_some_and(|after| view.entries[index].id <= after) {
			continue;
		}
		measured = view.contributions[index].iter().rev().find_map(|message| match message {
			pi_ai::Message::Assistant(assistant) if calculate_context_tokens(&assistant.usage) > 0 => {
				Some(assistant.clone())
			}
			_ => None,
		});
	}
	let from = match &measured {
		None => 0,
		Some(measured) => view
			.messages
			.iter()
			.rposition(|message| message == &pi_ai::Message::Assistant(measured.clone()))
			.map_or(0, |at| at + 1),
	};
	let mut tokens = measured.as_ref().map(|measured| calculate_context_tokens(&measured.usage)).unwrap_or(0);
	for message in &view.messages[from.min(view.messages.len())..] {
		tokens += estimate_message_tokens(message);
	}
	for message in extra {
		tokens += estimate_message_tokens(message);
	}
	tokens
}

/// The summary of a clean `stop` with text and no tool call; anything else
/// is not a summary.
fn summary_text(message: &AssistantMessage) -> Option<String> {
	if message.stop_reason != StopReason::Stop
		|| message.content.iter().any(|content| matches!(content, AssistantContent::ToolCall(_)))
	{
		return None;
	}
	let text = message
		.content
		.iter()
		.filter_map(|content| match content {
			AssistantContent::Text(text) => Some(text.text.as_str()),
			_ => None,
		})
		.collect::<Vec<_>>()
		.join("\n")
		.trim()
		.to_string();
	if text.is_empty() {
		None
	} else {
		Some(text)
	}
}

fn summary_failure(message: &AssistantMessage) -> String {
	if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
		return format!(
			"Summarization failed: {}",
			message
				.error_message
				.clone()
				.unwrap_or_else(|| stop_reason_str(message.stop_reason).to_string())
		);
	}
	if message.stop_reason == StopReason::Length {
		return "Summarization hit the token limit; the summary is incomplete".to_string();
	}
	if message.content.iter().any(|content| matches!(content, AssistantContent::ToolCall(_))) {
		return "Summarization attempted to call a tool".to_string();
	}
	"Summarization produced no text".to_string()
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

/// The summarizer's user message: the serialized conversation, the prompt,
/// and any instructions.
fn summary_prompt(messages: &[pi_ai::Message], instructions: Option<&str>) -> String {
	let focus = instructions.map_or(String::new(), |instructions| format!("\n\nAdditional focus: {instructions}"));
	format!(
		"<conversation>\n{}\n</conversation>\n\n{SUMMARIZATION_PROMPT}{focus}",
		serialize_conversation(messages)
	)
}

/// Messages as plain text, so the summarizer reads a transcript instead of
/// continuing it. System messages are omitted.
pub fn serialize_conversation(messages: &[pi_ai::Message]) -> String {
	let mut parts: Vec<String> = Vec::new();
	for message in messages {
		match message {
			pi_ai::Message::User(user) => {
				let text = content_text(&user.content);
				if !text.is_empty() {
					parts.push(format!("[User]: {text}"));
				}
			}
			pi_ai::Message::Assistant(assistant) => {
				let thinking: Vec<&String> = assistant
					.content
					.iter()
					.filter_map(|content| match content {
						AssistantContent::Thinking(thinking) => Some(&thinking.thinking),
						_ => None,
					})
					.collect();
				let text: Vec<&String> = assistant
					.content
					.iter()
					.filter_map(|content| match content {
						AssistantContent::Text(text) => Some(&text.text),
						_ => None,
					})
					.collect();
				let calls: Vec<String> = assistant
					.content
					.iter()
					.filter_map(|content| match content {
						AssistantContent::ToolCall(call) => {
							let args = call
								.arguments
								.iter()
								.map(|(key, value)| format!("{key}={}", serde_json::to_string(value).unwrap_or_default()))
								.collect::<Vec<_>>()
								.join(", ");
							Some(format!("{}({args})", call.name))
						}
						_ => None,
					})
					.collect();
				if !thinking.is_empty() {
					parts.push(format!("[Assistant thinking]: {}", thinking.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")));
				}
				if !text.is_empty() {
					parts.push(format!("[Assistant]: {}", text.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")));
				}
				if !calls.is_empty() {
					parts.push(format!("[Assistant tool calls]: {}", calls.join("; ")));
				}
			}
			pi_ai::Message::ToolResult(result) => {
				let text = content_text_of_blocks(&result.content);
				if !text.is_empty() {
					parts.push(format!("[Tool result]: {}", truncate(&text, TOOL_RESULT_MAX_CHARS)));
				}
			}
			pi_ai::Message::System(_) => {}
		}
	}
	parts.join("\n\n")
}

fn content_text(content: &pi_ai::types::UserContentField) -> String {
	match content {
		pi_ai::types::UserContentField::Text(text) => text.clone(),
		pi_ai::types::UserContentField::Blocks(blocks) => content_text_of_blocks(blocks),
	}
}

fn content_text_of_blocks(content: &[UserContent]) -> String {
	content
		.iter()
		.filter_map(|block| match block {
			UserContent::Text(text) => Some(text.text.clone()),
			_ => None,
		})
		.collect::<Vec<_>>()
		.join("\n")
}

fn truncate(text: &str, max_chars: usize) -> String {
	if text.chars().count() <= max_chars {
		return text.to_string();
	}
	let cut: String = text.chars().take(max_chars).collect();
	format!("{cut}\n\n[... {} more characters truncated]", text.chars().count() - max_chars)
}

/// Phase `select`.
async fn select_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let conversation_id = runtime.conversation_id();
	let agent = runtime.agent(&context).await?;
	let settings = runtime.settings();
	let reference = agent.model.clone();
	let model = match &reference {
		Some(reference) => runtime.models().get_model(&reference.provider, &reference.model_id),
		None => None,
	};
	if reference.is_none() || model.is_none() {
		return Box::pin(fail_no_model(&runtime, reference, &context)).await;
	}
	let policy = settings.compaction;
	let view = runtime.context_view(conversation_id, &context, None).await?;
	let Some(cut) = select_cut(&view, policy.keep_recent_tokens) else {
		return Box::pin(complete(&runtime, &context)).await;
	};
	let first_kept = view.entries[cut].id;
	let input: CompactionInput = serde_json::from_value(task.input.clone())
		.map_err(|error| format!("Compaction input is invalid: {error}"))?;
	let summarized = summarized_messages(&view, cut);
	// Hook: beforeCompact.
	let mut decision: Option<crate::harness::types::BeforeCompactDecision> = None;
	{
		let hooks = runtime.hooks();
		let payload = crate::harness::types::BeforeCompactPayload {
			reason: input.reason,
			entries: view.entries[..cut].to_vec(),
			messages: summarized.clone(),
			first_kept,
			instructions: input.instructions.clone(),
		};
		let runtime2 = runtime.clone();
		let context2 = context.clone();
		hooks
			.each(
				"beforeCompact",
				Arc::new(move |handler: &Arc<dyn crate::tasks::HookHandler>| {
					let payload = payload.clone();
					let runtime = runtime2.clone();
					let context = context2.clone();
					let hooks_table = handler.as_any().downcast_ref::<CompactionHooks>().cloned();
						Box::pin(async move {
						let Some(hooks) = hooks_table else { return Ok(()) };
						let Some(hook) = &hooks.before_compact else { return Ok(()) };
						if get_decision().is_none() {
							if let Some(decision) =
								hook(payload, coerce_hook_api(runtime.clone()), context).await?
							{
								set_decision(decision);
							}
						}
						Ok(())
						})
				}),
			)
			.await?;
	}
	if let Some(decision) = take_decision() {
		return match decision {
			crate::harness::types::BeforeCompactDecision::Decline => Box::pin(complete(&runtime, &context)).await,
			crate::harness::types::BeforeCompactDecision::Summary(summary) => {
				Box::pin(place(&runtime, first_kept, &summary, &context)).await
			}
		};
	}
	let model = model.expect("checked");
	let request = SummaryRequest {
		attempt: 1,
		model: reference.expect("checked"),
		thinking_level: agent.thinking_level,
		stream_options: serde_json::to_value(&settings.stream).expect("stream options serialize"),
		max_tokens: std::cmp::min(
			(8 * policy.reserve_tokens / 10),
			if model.max_tokens > 0 { model.max_tokens } else { u64::MAX },
		),
		tail: view.entries.iter().map(|entry| entry.id).fold(first_kept, |tail, id| if id > tail { id } else { tail }),
		first_kept,
	};
	let _ = input;
	runtime
		.commit(
			commit_change(move |_tx, _current| {
					Ok::<_, String>(Some(NextTaskState::Running {
						checkpoint: serde_json::to_value(CompactionCheckpoint::Summarize { request }).expect("summarize checkpoint serializes"),
					}))
			}),
			&context,
		)
		.await.map(|_| ())?;
	Ok(())
}

thread_local! {
	static COMPACTION_DECISION: std::cell::RefCell<Option<crate::harness::types::BeforeCompactDecision>> =
		const { std::cell::RefCell::new(None) };
}

fn set_decision(decision: crate::harness::types::BeforeCompactDecision) {
	COMPACTION_DECISION.with(|cell| *cell.borrow_mut() = Some(decision));
}

fn take_decision() -> Option<crate::harness::types::BeforeCompactDecision> {
	COMPACTION_DECISION.with(|cell| cell.borrow_mut().take())
}

fn get_decision() -> Option<crate::harness::types::BeforeCompactDecision> {
	COMPACTION_DECISION.with(|cell| cell.borrow().clone())
}

/// Phase `summarize`.
async fn summarize_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let CompactionCheckpoint::Summarize { request } = compaction_checkpoint(&task)? else {
		return Err("compaction summarize phase reached with another checkpoint".to_string());
	};
	let SummaryRequest { attempt, model: reference, thinking_level, stream_options, max_tokens, tail, first_kept } =
		request;
	let model = runtime.models().get_model(&reference.provider, &reference.model_id);
	let Some(model) = model else {
		return Box::pin(fail_no_model(&runtime, Some(reference), &context)).await;
	};
	// The context at `tail` is immutable, so this is the range `select`
	// chose.
	let view = runtime.context_view(runtime.conversation_id(), &context, Some(tail)).await?;
	let cut = view.entries.iter().position(|entry| entry.id == first_kept);
	let now = runtime.now() as i64;
	let input: CompactionInput = serde_json::from_value(task.input.clone())
		.map_err(|error| format!("Compaction input is invalid: {error}"))?;
	let messages: Vec<pi_ai::Message> = vec![
		pi_ai::Message::System(pi_ai::types::SystemMessage {
			content: pi_ai::types::SystemContent::Text(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
			sections: None,
			tools_added: None,
			tools_removed: None,
			timestamp: now,
		}),
		pi_ai::Message::User(pi_ai::types::UserMessage {
			content: pi_ai::types::UserContentField::Text(summary_prompt(
				&summarized_messages(&view, cut.unwrap_or(0)),
				input.instructions.as_deref(),
			)),
			timestamp: now,
		}),
	];
	let forwarded: crate::harness::types::ConversationStreamOptions =
		serde_json::from_value(stream_options.clone()).unwrap_or_default();
	let mut options: SimpleStreamOptions = SimpleStreamOptions {
		base: forwarded.into(),
		..Default::default()
	};
	options.base.cache_retention = Some(pi_ai::types::CacheRetention::None);
	options.base.max_tokens = Some(max_tokens);
	options.base.signal = ai_signal_bridge(&runtime.signal());
	options.base.session_id = Some(ensure_provider_session_id(runtime.as_ref(), &context).await?);
	if thinking_level != ModelThinkingLevel::Off {
		options.reasoning = Some(match thinking_level {
		ModelThinkingLevel::Off => pi_ai::types::ThinkingLevel::Minimal,
		ModelThinkingLevel::Minimal => pi_ai::types::ThinkingLevel::Minimal,
		ModelThinkingLevel::Low => pi_ai::types::ThinkingLevel::Low,
		ModelThinkingLevel::Medium => pi_ai::types::ThinkingLevel::Medium,
		ModelThinkingLevel::High => pi_ai::types::ThinkingLevel::High,
		ModelThinkingLevel::Xhigh => pi_ai::types::ThinkingLevel::Xhigh,
		ModelThinkingLevel::Max => pi_ai::types::ThinkingLevel::Max,
	});
	}
	let request_context = pi_ai::types::Context {
		system_prompt: None,
		messages,
		tools: None,
	};
	let message = runtime
		.models()
		.complete_simple(&model, &request_context, Some(options.into()))
		.await;
	// An abort mark or close: the abort invocation or the reopened task
	// handles the committed state.
	if let Some(signal) = context.abort_signal() {
		if signal.aborted() {
			return Err(abort_reason_text(&signal));
		}
	}
	let summary = summary_text(&message);
	let policy: RetryPolicy = to_retry_policy(&runtime.settings().retry);
	let retry = message.stop_reason == StopReason::Error
		&& is_retryable_assistant_error(&message)
		&& policy.enabled
		&& attempt <= policy.max_retries;
	let until = if retry { runtime.now() as u64 + retry_delay_ms(&policy, attempt) } else { 0 };
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	let owner = task.owner;
	let modes = modes_of(&runtime);
	let message2 = message.clone();
	let runtime_in = runtime.clone();
	runtime
		.commit(
			commit_change(move |tx: &mut crate::session::Tx<'_>, current| {
				let runtime = runtime_in.clone();
				let message = message2.clone();
					record_usage(
						tx,
						conversation_id,
						crate::harness::usage::UsageBucket::Models,
						&format!("{}/{}", message.provider, message.model),
						&message.usage,
					)?;
					let draft = tx
						.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
						.map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					if let Some(summary) = &summary {
						return place_summary(
							tx,
							conversation_id,
							owner,
							current,
							&mut live,
							&draft,
							first_kept,
							summary,
							runtime.now() as i64,
							modes,
						);
					}
					if retry {
						if let Some(status) = compaction_status(&mut live, &task_id) {
							status.retry = Some(crate::harness::live::RetryBackoff {
								at: until,
								error: message.error_message.clone().unwrap_or_default(),
							});
						}
						store_live(&draft, &live)?;
						let request = SummaryRequest {
							attempt,
							model: reference,
							thinking_level,
							stream_options,
							max_tokens,
							tail,
							first_kept,
						};
						return Ok(Some(NextTaskState::Running {
							checkpoint: serde_json::to_value(CompactionCheckpoint::Retry { until, request })
								.expect("retry checkpoint serializes"),
						}));
					}
					remove_compaction_status(&mut live, &task_id);
					store_live(&draft, &live)?;
					let text = summary_failure(&message);
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
			&context,
		)
		.await.map(|_| ())?;
	Ok(())
}

fn to_retry_policy(policy: &crate::harness::types::ConversationRetryPolicy) -> RetryPolicy {
	RetryPolicy {
		enabled: policy.enabled,
		max_retries: policy.max_retries,
		base_delay_ms: policy.base_delay_ms,
		max_agent_delay_ms: policy.max_agent_delay_ms,
	}
}

/// Phase `retry`.
async fn retry_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let CompactionCheckpoint::Retry { until, request } = compaction_checkpoint(&task)? else {
		return Err("compaction retry phase reached with another checkpoint".to_string());
	};
	runtime.sleep(until, &context).await?;
	let SummaryRequest { attempt, model, thinking_level, stream_options, max_tokens, tail, first_kept } = request;
	let attempt = attempt + 1;
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	runtime
		.commit(
			commit_change(move |tx, _current| {
					let draft = tx
						.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
						.map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					if let Some(status) = compaction_status(&mut live, &task_id) {
						status.attempt = attempt as u64;
						status.retry = None;
					}
					store_live(&draft, &live)?;
					let request = SummaryRequest {
						attempt,
						model,
						thinking_level,
						stream_options,
						max_tokens,
						tail,
						first_kept,
					};
					Ok(Some(NextTaskState::Running {
						checkpoint: serde_json::to_value(CompactionCheckpoint::Summarize { request })
							.expect("summarize checkpoint serializes"),
					}))
			}),
			&context,
		)
		.await.map(|_| ())?;
	Ok(())
}

/// Abort phase.
async fn abort_phase(_task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	runtime
		.commit(
			commit_change(move |tx, _current| {
					let draft = tx
						.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
						.map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					remove_compaction_status(&mut live, &task_id);
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

/// Place a summary supplied by a hook in its own commit.
async fn place(
	runtime: &Arc<dyn TaskRuntime>,
	first_kept: EntryId,
	summary: &str,
	context: &Context,
) -> Result<(), String> {
	let conversation_id = runtime.conversation_id();
	let owner = owner_task_of(runtime).await;
	let summary = summary.to_string();
	let modes = modes_of(runtime);
	runtime
		.commit(
			commit_change(move |tx, current| {
				let summary = summary.clone();
					let draft = tx
						.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
						.map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					place_summary(
						tx,
						conversation_id,
						owner,
						current,
						&mut live,
						&draft,
						first_kept,
						&summary,
						runtime.now() as i64,
						modes_of(runtime),
					)
			}),
			context,
		)
		.await.map(|_| ())?;
	Ok(())
}

fn modes_of(runtime: &Arc<dyn TaskRuntime>) -> crate::harness::inbox::QueueModes {
	let settings = runtime.settings();
	crate::harness::inbox::QueueModes {
		steering_mode: settings.steering_mode,
		follow_up_mode: settings.follow_up_mode,
	}
}

/// The compaction task's owner (absent for a conversation-owned task).
async fn owner_task_of(runtime: &Arc<dyn TaskRuntime>) -> Option<TaskId> {
	runtime.get_task(runtime.task_id(), &pi_chord::context::BACKGROUND_CONTEXT.clone()).await.ok().flatten().and_then(|record| record.owner)
}

/// Place the summary entry and complete (spec §8.7). A blocking compaction,
/// owned by its generation, appends it: the generation holds the run and
/// waits. A conversation-owned one admits it as a write submission, placed
/// at once when idle, otherwise at the next boundary, or settled `stale`.
///
/// Nothing else may append to a busy conversation, so every non-blocking
/// summary goes through admission.
#[allow(clippy::too_many_arguments)]
fn place_summary(
	tx: &mut Tx<'_>,
	conversation_id: ConversationId,
	owner: Option<TaskId>,
	current: &TaskRecord,
	live: &mut LiveState,
	draft: &crate::session::DocDraft,
	first_kept: EntryId,
	summary: &str,
	now: i64,
	modes: crate::harness::inbox::QueueModes,
) -> Result<Option<NextTaskState>, String> {
	remove_compaction_status(live, &current.id);
	let text = format!("{SUMMARY_PREFIX}{summary}{SUMMARY_SUFFIX}");
	let mut entry = EntryDraft {
		kind: Some(COMPACTION_ENTRY.kind.clone()),
		..Default::default()
	};
	entry.head = Some(EntryHead::Entry(first_kept));
	entry.model = Some(vec![pi_ai::Message::User(pi_ai::types::UserMessage {
		content: pi_ai::types::UserContentField::Blocks(vec![UserContent::Text(pi_ai::types::TextContent::new(text))]),
		timestamp: now,
	})]);
	let input: CompactionInput = serde_json::from_value(current.input.clone())
		.map_err(|error| format!("Compaction input is invalid: {error}"))?;
	entry.data = Some(json!({ "reason": input.reason }));
	let outcome = match owner {
		None => {
			let submission_id = admit_submission(
				tx,
				conversation_id,
				&crate::harness::types::SubmissionDraft::Write(crate::harness::types::WriteSubmissionDraft {
					request_id: Some(format!("compaction:{}", current.id)),
					entry,
				}),
				now,
				modes,
			)
			?;
			// Admission commits `live` itself (a queued summary that starts a
			// run writes `run`); adopt its state or the store below would
			// revert it, and re-apply the status removal admission read past.
			*live = read_live(draft);
			remove_compaction_status(live, &current.id);
			CompactionResult { entry_id: None, submission_id: Some(submission_id) }
		}
		Some(_) => {
			let record = tx.append_entry(conversation_id, entry).map_err(|e| e.to_string())?;
			CompactionResult { entry_id: Some(record.id), submission_id: None }
		}
	};
	store_live(draft, live)?;
	Ok(Some(NextTaskState::Terminal {
		outcome: TaskOutcome {
			status: TaskOutcomeStatus::Completed,
			result: Some(serde_json::to_value(&outcome).expect("CompactionResult serializes")),
			error: None,
			reason: None,
		},
	}))
}

/// Remove the status and complete without a summary.
async fn complete(runtime: &Arc<dyn TaskRuntime>, context: &Context) -> Result<(), String> {
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	runtime
		.commit(
			commit_change(move |tx, _current| {
					let draft = tx
						.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
						.map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					remove_compaction_status(&mut live, &task_id);
					store_live(&draft, &live)?;
					Ok(Some(NextTaskState::Terminal {
						outcome: TaskOutcome {
							status: TaskOutcomeStatus::Completed,
							result: Some(json!({})),
							error: None,
							reason: None,
						},
					}))
			}),
			context,
		)
		.await.map(|_| ())?;
	Ok(())
}

async fn fail_no_model(runtime: &Arc<dyn TaskRuntime>, reference: Option<ModelRef>, context: &Context) -> Result<(), String> {
	let message = match reference {
		None => "No model is configured".to_string(),
		Some(reference) => format!("Model {}/{} is not available", reference.provider, reference.model_id),
	};
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	runtime
		.commit(
			commit_change(move |tx, _current| {
					let draft = tx
						.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
						.map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					remove_compaction_status(&mut live, &task_id);
					store_live(&draft, &live)?;
					Ok(Some(NextTaskState::Terminal {
						outcome: TaskOutcome {
							status: TaskOutcomeStatus::Failed,
							result: None,
							error: Some(TaskOutcomeError {
								message,
								detail: Some(json!({ "reason": "no_model" })),
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

// Unused references kept for signature parity with upstream.
#[allow(unused)]
fn _markers(_w: Option<WhenBusy>) {}
