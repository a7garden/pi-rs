//! Port of `packages/coding-agent/src/core/compaction/compaction.ts` —
//! context compaction for long sessions: pure functions for compaction
//! logic. The session manager handles I/O, and after compaction the session
//! is reloaded.
//!
//! LLM calls go through the provider-neutral seams ([`pi_agent::StreamFn`] /
//! `completeSimple`) shared with the rest of the port; retry behavior reuses
//! [`pi_ai::utils::retry::retry_assistant_call`].

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use pi_agent::{AgentMessage, AgentThinkingLevel, StreamFn};
use pi_ai::types::{
	AssistantContent, AssistantMessage, CacheRetention, Context, Message, Model, SimpleStreamOptions, StopReason,
	SystemContent, SystemBlock, TextContent, TranscriptContext, Usage, UserContent, UserContentField, UserMessage,
};
use pi_ai::utils::abort::AbortSignal;
use pi_ai::utils::retry::{retry_assistant_call, RetryCallbacks, RetryPolicy};
use pi_ai::utils::text::content_blocks_text;
use pi_ai::utils::transcript::{get_current_system_message, normalize_context};
use pi_ai::utils::uuid::uuidv7_now;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::compaction::utils::{
	compute_file_lists, create_file_ops, extract_file_ops_from_message, serialize_conversation, format_file_operations,
	FileOperations, SUMMARIZATION_SYSTEM_PROMPT,
};
use crate::core::messages::convert_to_llm;
use crate::core::session_manager::{
	build_session_projection, session_entry_to_context_messages, ContextEditReplacement, Leaf, ProjectedSessionEntry,
	SessionEntry, SessionProjection,
};
use crate::core::usage_totals::combine_usage;
use crate::utils::time::now_ms;

// ============================================================================
// File Operation Tracking
// ============================================================================

/// Details stored in `CompactionEntry.details` for file tracking
/// (`CompactionDetails`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionDetails {
	pub read_files: Vec<String>,
	pub modified_files: Vec<String>,
}

/// Extract file operations from messages and previous compaction entries
/// (`extractFileOperations`). `prev_compaction_index` indexes `entries` (the
/// projected source-entry list).
fn extract_file_operations(
	messages: &[AgentMessage],
	entries: &[&SessionEntry],
	prev_compaction_index: Option<usize>,
) -> FileOperations {
	let mut file_ops = create_file_ops();

	// Collect from previous compaction's details (if pi-generated)
	if let Some(index) = prev_compaction_index {
		let Some(SessionEntry::Compaction(prev_compaction)) = entries.get(index) else { return file_ops };
		if prev_compaction.from_hook != Some(true) {
			if let Some(details) = &prev_compaction.details {
				// fromHook field kept for session file compatibility
				if let Some(read_files) = details.get("readFiles").and_then(Value::as_array) {
					for file in read_files.iter().filter_map(Value::as_str) {
						file_ops.read.insert(file.to_string());
					}
				}
				if let Some(modified_files) = details.get("modifiedFiles").and_then(Value::as_array) {
					for file in modified_files.iter().filter_map(Value::as_str) {
						file_ops.edited.insert(file.to_string());
					}
				}
			}
		}
	}

	// Extract from tool calls in messages
	for message in messages {
		extract_file_ops_from_message(message, &mut file_ops);
	}

	file_ops
}

// ============================================================================
// Message Extraction
// ============================================================================

/// Extract AgentMessages from a projected compaction input entry
/// (`getMessagesFromProjectedEntryForCompaction`).
fn get_messages_from_projected_entry_for_compaction(entry: &ProjectedSessionEntry) -> Vec<AgentMessage> {
	if matches!(entry.source_entry, SessionEntry::Compaction(_)) {
		return Vec::new();
	}
	// System messages are prompt state, not conversation; the compaction entry
	// carries their replay.
	entry
		.messages
		.iter()
		.filter(|message| message.role() != "system")
		.cloned()
		.collect()
}

/// Result from `compact()` - SessionManager adds uuid/parentUuid when saving
/// (`CompactionResult`). Upstream's generic `details` payload becomes the
/// concrete [`CompactionDetails`] shape (crate docs D23).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompactionResult {
	pub summary: String,
	pub first_kept_entry_id: String,
	pub tokens_before: u64,
	pub estimated_tokens_after: Option<u64>,
	/// Usage from the LLM call(s) that generated this summary, if available
	pub usage: Option<Usage>,
	/// Extension-specific data (e.g., ArtifactIndex, version markers for
	/// structured compaction)
	pub details: Option<CompactionDetails>,
}

// ============================================================================
// Types
// ============================================================================

/// Compaction thresholds (`CompactionSettings`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSettings {
	pub enabled: bool,
	pub reserve_tokens: u64,
	pub keep_recent_tokens: u64,
}

pub const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = CompactionSettings {
	enabled: true,
	reserve_tokens: 16384,
	keep_recent_tokens: 20000,
};

// ============================================================================
// Token calculation
// ============================================================================

/// Calculate total context tokens from usage (`calculateContextTokens`). Uses
/// the native totalTokens field when available, falls back to computing from
/// components.
pub fn calculate_context_tokens(usage: &Usage) -> u64 {
	if usage.total_tokens != 0 {
		usage.total_tokens
	} else {
		usage.input + usage.output + usage.cache_read + usage.cache_write
	}
}

/// Get usage from an assistant message if available (`getAssistantUsage`).
/// Skips aborted, error, and all-zero usage messages as they don't have valid
/// usage data.
fn get_assistant_usage(message: &AgentMessage) -> Option<&Usage> {
	if let AgentMessage::Llm(Message::Assistant(assistant_message)) = message {
		if assistant_message.stop_reason != StopReason::Aborted
			&& assistant_message.stop_reason != StopReason::Error
			&& calculate_context_tokens(&assistant_message.usage) > 0
		{
			return Some(&assistant_message.usage);
		}
	}
	None
}

/// Find the last valid assistant message usage from session entries
/// (`getLastAssistantUsage`).
pub fn get_last_assistant_usage(entries: &[SessionEntry]) -> Option<&Usage> {
	for entry in entries.iter().rev() {
		if let SessionEntry::Message(message) = entry {
			if let Some(usage) = get_assistant_usage(&message.message) {
				return Some(usage);
			}
		}
	}
	None
}

/// Context-token estimate breakdown (`ContextUsageEstimate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ContextUsageEstimate {
	pub tokens: u64,
	pub usage_tokens: u64,
	pub trailing_tokens: u64,
	pub last_usage_index: Option<usize>,
}

fn get_last_assistant_usage_info(messages: &[AgentMessage]) -> Option<(&Usage, usize)> {
	for (index, message) in messages.iter().enumerate().rev() {
		if let Some(usage) = get_assistant_usage(message) {
			return Some((usage, index));
		}
	}
	None
}

/// Estimate context tokens from messages, using the last assistant usage when
/// available (`estimateContextTokens`). If there are messages after the last
/// usage, estimate their tokens with [`estimate_tokens`].
pub fn estimate_context_tokens(messages: &[AgentMessage]) -> ContextUsageEstimate {
	let Some((usage, index)) = get_last_assistant_usage_info(messages) else {
		let mut estimated = 0u64;
		for message in messages {
			estimated += estimate_tokens(message) as u64;
		}
		return ContextUsageEstimate { tokens: estimated, usage_tokens: 0, trailing_tokens: estimated, last_usage_index: None };
	};

	let usage_tokens = calculate_context_tokens(usage);
	let mut trailing_tokens = 0u64;
	for message in &messages[index + 1..] {
		trailing_tokens += estimate_tokens(message) as u64;
	}

	ContextUsageEstimate {
		tokens: usage_tokens + trailing_tokens,
		usage_tokens,
		trailing_tokens,
		last_usage_index: Some(index),
	}
}

/// Estimate projected context without trusting usage captured before a later
/// edit or compaction (`estimateProjectedContextTokens`).
pub fn estimate_projected_context_tokens(
	projection: &SessionProjection,
	branch_entries: &[SessionEntry],
) -> ContextUsageEstimate {
	let estimate = estimate_context_tokens(&projection.messages);
	if let Some(last_usage_index) = estimate.last_usage_index {
		let mut projected_message_index = 0usize;
		let mut usage_entry_id: Option<String> = None;
		for entry in &projection.entries {
			let next_message_index = projected_message_index + entry.messages.len();
			if last_usage_index < next_message_index {
				usage_entry_id = Some(entry.source_entry.id().to_string());
				break;
			}
			projected_message_index = next_message_index;
		}

		let usage_entry_index = usage_entry_id
			.as_deref()
			.and_then(|id| branch_entries.iter().position(|entry| entry.id() == id));
		let latest_invalidating_entry_index = branch_entries
			.iter()
			.rposition(|entry| matches!(entry, SessionEntry::ContextEdit(_) | SessionEntry::Compaction(_)));
		if let Some(usage_entry_index) = usage_entry_index {
			if let Some(latest_invalidating_entry_index) = latest_invalidating_entry_index {
				if usage_entry_index > latest_invalidating_entry_index {
					return estimate;
				}
			}
		}
	}

	let llm_messages: Vec<Message> = projection.messages.iter().filter_map(AgentMessage::as_llm).cloned().collect();
	let current_system = get_current_system_message(&llm_messages);
	let mut tokens = current_system.map_or(0, |system| estimate_tokens(&AgentMessage::system(system)) as u64);
	for message in &projection.messages {
		if message.role() != "system" {
			tokens += estimate_tokens(message) as u64;
		}
	}
	ContextUsageEstimate { tokens, usage_tokens: 0, trailing_tokens: tokens, last_usage_index: None }
}

/// Check if compaction should trigger based on context usage
/// (`shouldCompact`).
pub fn should_compact(context_tokens: u64, context_window: u64, settings: &CompactionSettings) -> bool {
	if !settings.enabled {
		return false;
	}
	context_tokens > context_window.saturating_sub(settings.reserve_tokens)
}

// ============================================================================
// Cut point detection
// ============================================================================

const ESTIMATED_IMAGE_CHARS: usize = 4800;

fn estimate_user_blocks_chars(blocks: &[UserContent]) -> usize {
	blocks
		.iter()
		.map(|block| match block {
			UserContent::Text(text) => text.text.chars().count(),
			UserContent::Image(_) => ESTIMATED_IMAGE_CHARS,
		})
		.sum()
}

fn estimate_user_content_chars(content: &UserContentField) -> usize {
	match content {
		UserContentField::Text(text) => text.chars().count(),
		UserContentField::Blocks(blocks) => estimate_user_blocks_chars(blocks),
	}
}

fn estimate_system_content_chars(content: &SystemContent) -> usize {
	match content {
		SystemContent::Text(text) => text.chars().count(),
		SystemContent::Blocks(blocks) => blocks
			.iter()
			.map(|block| match block {
				SystemBlock::Text(text) => text.text.chars().count(),
			})
			.sum(),
	}
}

/// Estimated character length of an opaque custom message's content payload.
fn estimate_custom_content_chars(content: Option<&Value>) -> usize {
	match content {
		Some(Value::String(text)) => text.chars().count(),
		Some(Value::Array(blocks)) => blocks
			.iter()
			.map(|block| match block.get("type").and_then(Value::as_str) {
				Some("text") => block
					.get("text")
					.and_then(Value::as_str)
					.map_or(0, |text| text.chars().count()),
				Some("image") => ESTIMATED_IMAGE_CHARS,
				_ => 0,
			})
			.sum(),
		_ => 0,
	}
}

/// Estimate token count for a message using chars/4 heuristic
/// (`estimateTokens`). This is conservative (overestimates tokens).
pub fn estimate_tokens(message: &AgentMessage) -> usize {
	match message {
		AgentMessage::Llm(Message::System(system)) => {
			let mut chars = estimate_system_content_chars(&system.content);
			if let Some(sections) = &system.sections {
				for text in sections.values().flatten() {
					chars += text.chars().count();
				}
			}
			if let Some(tools_added) = &system.tools_added {
				chars += serde_json::to_string(tools_added).map_or(0, |json| json.chars().count());
			}
			chars.div_ceil(4)
		}
		AgentMessage::Llm(Message::User(user)) => estimate_user_content_chars(&user.content).div_ceil(4),
		AgentMessage::Llm(Message::Assistant(assistant)) => {
			let mut chars = 0usize;
			for block in &assistant.content {
				match block {
					AssistantContent::Text(text) => chars += text.text.chars().count(),
					AssistantContent::Thinking(thinking) => chars += thinking.thinking.chars().count(),
					AssistantContent::ToolCall(call) => {
						chars += call.name.chars().count()
							+ serde_json::to_string(&call.arguments).map_or(0, |json| json.chars().count());
					}
				}
			}
			chars.div_ceil(4)
		}
		AgentMessage::Llm(Message::ToolResult(result)) => estimate_user_blocks_chars(&result.content).div_ceil(4),
		AgentMessage::Custom(value) => {
			let role = value.get("role").and_then(Value::as_str).unwrap_or("custom");
			match role {
				"bashExecution" => {
					let command = value.get("command").and_then(Value::as_str).unwrap_or("");
					let output = value.get("output").and_then(Value::as_str).unwrap_or("");
					(command.chars().count() + output.chars().count()).div_ceil(4)
				}
				"branchSummary" | "compactionSummary" => {
					value
						.get("summary")
						.and_then(Value::as_str)
						.map_or(0, |summary| summary.chars().count())
						.div_ceil(4)
				}
				"custom" | "toolResult" => estimate_custom_content_chars(value.get("content")).div_ceil(4),
				_ => 0,
			}
		}
	}
}

fn is_cut_point_message(message: &AgentMessage) -> bool {
	match message.role() {
		"user" | "assistant" | "bashExecution" | "custom" | "branchSummary" | "compactionSummary" => true,
		_ => false,
	}
}

fn is_turn_start_message(message: &AgentMessage) -> bool {
	match message.role() {
		"user" | "bashExecution" | "custom" | "branchSummary" | "compactionSummary" => true,
		_ => false,
	}
}

fn is_turn_start_entry(entry: &SessionEntry) -> bool {
	if matches!(entry, SessionEntry::Compaction(_)) {
		return false;
	}
	session_entry_to_context_messages(entry).iter().any(is_turn_start_message)
}

/// Find valid cut points: indices of context-visible user-like or assistant
/// messages (`findValidCutPoints`). Never cut at tool results (they must
/// follow their tool call). When we cut at an assistant message with tool
/// calls, its tool results follow it and will be kept.
fn find_valid_cut_points(entries: &[SessionEntry], start_index: usize, end_index: usize) -> Vec<usize> {
	let mut cut_points = Vec::new();
	for index in start_index..end_index {
		let entry = &entries[index];
		if matches!(entry, SessionEntry::Compaction(_)) {
			continue;
		}
		if session_entry_to_context_messages(entry).iter().any(is_cut_point_message) {
			cut_points.push(index);
		}
	}
	cut_points
}

/// Find the context-visible user-role message that starts the turn containing
/// the given entry index (`findTurnStartIndex`). Returns `None` if no turn
/// start found before the index.
pub fn find_turn_start_index(entries: &[SessionEntry], entry_index: usize, start_index: usize) -> Option<usize> {
	for index in (start_index..=entry_index).rev() {
		if is_turn_start_entry(&entries[index]) {
			return Some(index);
		}
	}
	None
}

/// Cut-point selection (`CutPointResult`). `turn_start_index` mirrors
/// upstream's `-1` sentinel with `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutPointResult {
	/// Index of first entry to keep
	pub first_kept_entry_index: usize,
	/// Index of user message that starts the turn being split, or `None` if
	/// not splitting
	pub turn_start_index: Option<usize>,
	/// Whether this cut splits a turn (cut point is not a user message)
	pub is_split_turn: bool,
}

/// Find the cut point in session entries that keeps approximately
/// `keepRecentTokens` (`findCutPoint`).
///
/// Algorithm: Walk backwards from newest, accumulating estimated message
/// sizes. Stop when we've accumulated >= keepRecentTokens. Cut at that point.
///
/// Can cut at user OR assistant messages (never tool results). When cutting at
/// an assistant message with tool calls, its tool results come after and will
/// be kept.
///
/// Only considers entries between `startIndex` and `endIndex` (exclusive).
pub fn find_cut_point(
	entries: &[SessionEntry],
	start_index: usize,
	end_index: usize,
	keep_recent_tokens: u64,
) -> CutPointResult {
	let cut_points = find_valid_cut_points(entries, start_index, end_index);

	if cut_points.is_empty() {
		return CutPointResult { first_kept_entry_index: start_index, turn_start_index: None, is_split_turn: false };
	}

	// Walk backwards from newest, accumulating estimated message sizes
	let mut accumulated_tokens = 0u64;
	// Default: keep from first message (not header)
	let mut cut_index = cut_points[0];

	for index in (start_index..end_index).rev() {
		let entry = &entries[index];
		let message_tokens = session_entry_to_context_messages(entry)
			.iter()
			.map(|message| estimate_tokens(message) as u64)
			.sum::<u64>();
		if message_tokens == 0 {
			continue;
		}
		accumulated_tokens += message_tokens;

		// Check if we've exceeded the budget
		if accumulated_tokens >= keep_recent_tokens {
			// Prefer the closest valid cut point at or after this entry. If
			// trailing tool results exceed the budget by themselves, keep their
			// preceding assistant tool call instead of falling back to the
			// first message.
			cut_index = cut_points
				.iter()
				.find(|candidate| **candidate >= index)
				.copied()
				.unwrap_or(cut_points[cut_points.len() - 1]);
			break;
		}
	}

	// Scan backwards from cutIndex to include adjacent metadata entries that do
	// not affect context.
	while cut_index > start_index {
		let previous_entry = &entries[cut_index - 1];
		// Stop at compaction boundaries or context-visible entries.
		if matches!(previous_entry, SessionEntry::Compaction(_))
			|| !session_entry_to_context_messages(previous_entry).is_empty()
		{
			break;
		}
		cut_index -= 1;
	}

	// Determine if this is a split turn
	let starts_turn = is_turn_start_entry(&entries[cut_index]);
	let turn_start_index = if starts_turn { None } else { find_turn_start_index(entries, cut_index, start_index) };

	CutPointResult {
		first_kept_entry_index: cut_index,
		turn_start_index,
		is_split_turn: !starts_turn && turn_start_index.is_some(),
	}
}

// ============================================================================
// Summarization
// ============================================================================

const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.\n\nUse this EXACT format:\n\n## Goal\n[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned by user]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [Ordered list of what should happen next]\n\n## Critical Context\n- [Any data, examples, or references needed to continue]\n- [Or \"(none)\" if not applicable]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

const UPDATE_SUMMARIZATION_INSTRUCTIONS: &str = "Update the existing structured summary with new information. RULES:\n- PRESERVE all existing information from the previous summary\n- ADD new progress, decisions, and context from the new messages\n- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed\n- UPDATE \"Next Steps\" based on what was accomplished\n- PRESERVE exact file paths, function names, and error messages\n- If something is no longer relevant, you may remove it\n\nUse this EXACT format:\n\n## Goal\n[Preserve existing goals, add new ones if the task expanded]\n\n## Constraints & Preferences\n- [Preserve existing, add new ones discovered]\n\n## Progress\n### Done\n- [x] [Include previously done items AND newly completed items]\n\n### In Progress\n- [ ] [Current work - update based on progress]\n\n### Blocked\n- [Current blockers - remove if resolved]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n\n## Next Steps\n1. [Update based on current state]\n\n## Critical Context\n- [Preserve important context, add new if needed]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

/// `UPDATE_SUMMARIZATION_PROMPT` — the update instructions framed as the
/// user-message tail.
fn update_summarization_prompt() -> String {
	format!(
		"The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.\n\n{UPDATE_SUMMARIZATION_INSTRUCTIONS}"
	)
}

/// Returns an error message when a summarization response cannot safely be
/// persisted (`getSummarizationFailure`). A length stop contains partial text
/// and must not become a session checkpoint.
pub fn get_summarization_failure(response: &AssistantMessage, label: &str) -> Option<String> {
	if response.stop_reason == StopReason::Error {
		let error_message = response
			.error_message
			.as_deref()
			.filter(|message| !message.is_empty())
			.unwrap_or("Unknown error");
		return Some(format!("{label} failed: {error_message}"));
	}
	if response.stop_reason == StopReason::Length {
		return Some(format!("{label} failed: generation hit the token cap and the summary is incomplete"));
	}
	None
}

fn create_summarization_options(
	model: &Model,
	max_tokens: u64,
	api_key: Option<&str>,
	headers: Option<&pi_ai::types::ProviderHeaders>,
	env: Option<&pi_ai::types::ProviderEnv>,
	signal: Option<&AbortSignal>,
	thinking_level: Option<AgentThinkingLevel>,
	session_id: Option<&str>,
) -> SimpleStreamOptions {
	let mut options = SimpleStreamOptions::default();
	options.base.max_tokens = Some(max_tokens);
	options.base.session_id = session_id.map(str::to_string);
	options.base.base.api_key = api_key.map(str::to_string);
	options.base.base.headers = headers.cloned();
	options.base.base.env = env.cloned();
	options.base.base.signal = signal.cloned();
	if model.reasoning && matches!(thinking_level, Some(level) if level != AgentThinkingLevel::Off) {
		options.reasoning = thinking_level.and_then(AgentThinkingLevel::to_reasoning);
	}
	options
}

/// Terminal error message standing in for a `streamFn` synchronous throw
/// (crate docs D2: upstream `StreamFn` never throws; the Rust seam's `Err`
/// short-circuits without retries). The message intentionally matches none of
/// the retryable provider error patterns.
const STREAM_FN_SYNC_ERROR: &str = "stream function failed synchronously";

fn stream_fn_error_message(message: String) -> AssistantMessage {
	AssistantMessage {
		content: Vec::new(),
		api: String::new(),
		provider: String::new(),
		model: String::new(),
		response_model: None,
		response_id: None,
		provider_thinking_level: None,
		thinking_level: None,
		diagnostics: None,
		usage: Default::default(),
		stop_reason: StopReason::Error,
		deferred: None,
		error_message: Some(message),
		raw_stop_reason: None,
		end_turn: None,
		timestamp: now_ms(),
	}
}

/// Shared choke point for every compaction/branch-summary summarization call
/// (`completeSummarization`). Wraps the single LLM call in
/// [`retry_assistant_call`] so transient stream drops (e.g. `terminated`,
/// socket close) honor the configured retry policy instead of failing the
/// whole compaction on the first attempt. Deterministic errors and aborts
/// return immediately (see [`retry_assistant_call`]).
///
/// Upstream lets the underlying call throw synchronously; the port surfaces
/// that path as `Err` (crate docs D2/D23) instead of a rejected promise.
pub async fn complete_summarization(
	model: &Model,
	context: TranscriptContext,
	options: SimpleStreamOptions,
	stream_fn: Option<StreamFn>,
	retry: Option<&RetryPolicy>,
	callbacks: Option<&RetryCallbacks>,
) -> Result<AssistantMessage, String> {
	// Avoid cache writes for one-off summaries. Reuse caller-supplied routing
	// when available; callers without a session ID, including branch
	// summaries, receive a fresh routing ID.
	let mut request_options = options;
	request_options.base.cache_retention = Some(CacheRetention::None);
	if request_options.base.session_id.is_none() {
		request_options.base.session_id = Some(uuidv7_now().expect("uuidv7 clock"));
	}
	let signal = request_options.base.base.signal.clone();
	let sync_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

	let produce = {
		let model = model.clone();
		let sync_error = sync_error.clone();
		move || {
			let model = model.clone();
			let context = context.clone();
			let options = request_options.clone();
			let stream_fn = stream_fn.clone();
			let sync_error = sync_error.clone();
			async move {
				match stream_fn.as_ref() {
					Some(stream_fn) => match (stream_fn)(model.clone(), context.clone(), Some(options)).await {
						Ok(stream) => stream.result().await,
						Err(error) => {
							*sync_error.lock().expect("sync error lock") = Some(error);
							stream_fn_error_message(STREAM_FN_SYNC_ERROR.to_string())
						}
					},
					None => {
						let plain = Context { system_prompt: None, messages: context.messages().to_vec(), tools: None };
						pi_ai::compat::complete_simple(&model, &plain, Some(options)).await
					}
				}
			}
		}
	};

	let response = retry_assistant_call(produce, retry, signal.as_ref(), callbacks).await;
	if let Some(error) = sync_error.lock().expect("sync error lock").take() {
		return Err(error);
	}
	Ok(response)
}

/// Generated summary plus its provider usage (`generateSummaryWithUsage`
/// return shape).
#[derive(Debug, Clone, PartialEq)]
pub struct SummaryWithUsage {
	pub text: String,
	pub usage: Usage,
}

/// Generate a summary of the conversation using the LLM (`generateSummary`).
/// If previousSummary is provided, uses the update prompt to merge.
#[allow(clippy::too_many_arguments)]
pub async fn generate_summary(
	current_messages: &[AgentMessage],
	model: &Model,
	reserve_tokens: u64,
	api_key: Option<&str>,
	headers: Option<&pi_ai::types::ProviderHeaders>,
	signal: Option<&AbortSignal>,
	custom_instructions: Option<&str>,
	previous_summary: Option<&str>,
	thinking_level: Option<AgentThinkingLevel>,
	stream_fn: Option<StreamFn>,
	env: Option<&pi_ai::types::ProviderEnv>,
	retry: Option<&RetryPolicy>,
	callbacks: Option<&RetryCallbacks>,
	session_id: Option<&str>,
) -> Result<String, String> {
	Ok(generate_summary_with_usage(
		current_messages,
		model,
		reserve_tokens,
		api_key,
		headers,
		signal,
		custom_instructions,
		previous_summary,
		thinking_level,
		stream_fn,
		env,
		retry,
		callbacks,
		session_id,
	)
	.await?
	.text)
}

/// Build the provider context for a standalone summary request
/// (`buildSummarizationContext`).
fn build_summarization_context(prompt_text: String) -> TranscriptContext {
	normalize_context(&Context {
		system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
		messages: vec![Message::User(UserMessage {
			content: UserContentField::Blocks(vec![UserContent::Text(TextContent::new(prompt_text))]),
			timestamp: now_ms(),
		})],
		tools: None,
	})
}

/// Generate or update a conversation summary and return its provider usage
/// (`generateSummaryWithUsage`).
#[allow(clippy::too_many_arguments)]
pub async fn generate_summary_with_usage(
	current_messages: &[AgentMessage],
	model: &Model,
	reserve_tokens: u64,
	api_key: Option<&str>,
	headers: Option<&pi_ai::types::ProviderHeaders>,
	signal: Option<&AbortSignal>,
	custom_instructions: Option<&str>,
	previous_summary: Option<&str>,
	thinking_level: Option<AgentThinkingLevel>,
	stream_fn: Option<StreamFn>,
	env: Option<&pi_ai::types::ProviderEnv>,
	retry: Option<&RetryPolicy>,
	callbacks: Option<&RetryCallbacks>,
	session_id: Option<&str>,
) -> Result<SummaryWithUsage, String> {
	let base_cap = (0.8 * reserve_tokens as f64).floor() as u64;
	let max_tokens = if model.max_tokens > 0 { base_cap.min(model.max_tokens) } else { base_cap };

	// Use update prompt if we have a previous summary, otherwise initial prompt
	let mut base_prompt =
		if previous_summary.is_some() { update_summarization_prompt() } else { SUMMARIZATION_PROMPT.to_string() };
	if let Some(custom_instructions) = custom_instructions {
		base_prompt = format!("{base_prompt}\n\nAdditional focus: {custom_instructions}");
	}

	// Serialize conversation to text so model doesn't try to continue it
	// Convert to LLM messages first (handles custom types like bashExecution,
	// custom, etc.)
	let llm_messages = convert_to_llm(current_messages);
	let conversation_text = serialize_conversation(&llm_messages);

	// Build the prompt with conversation wrapped in tags
	let mut prompt_text = format!("<conversation>\n{conversation_text}\n</conversation>\n\n");
	if let Some(previous_summary) = previous_summary {
		prompt_text += &format!("<previous-summary>\n{previous_summary}\n</previous-summary>\n\n");
	}
	prompt_text += &base_prompt;

	let completion_options =
		create_summarization_options(model, max_tokens, api_key, headers, env, signal, thinking_level, session_id);

	let response = complete_summarization(
		model,
		build_summarization_context(prompt_text),
		completion_options,
		stream_fn,
		retry,
		callbacks,
	)
	.await?;

	if let Some(failure) = get_summarization_failure(&response, "Summarization") {
		return Err(failure);
	}
	if response.content.iter().any(|block| matches!(block, AssistantContent::ToolCall(_))) {
		return Err("Summarization attempted to call a tool".to_string());
	}

	let text_content = content_blocks_text(&response.content, "\n");

	Ok(SummaryWithUsage { text: text_content, usage: response.usage })
}

// ============================================================================
// Compaction Preparation (for extensions)
// ============================================================================

/// Pre-calculated compaction inputs (`CompactionPreparation`).
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPreparation {
	/// UUID of first entry to keep
	pub first_kept_entry_id: String,
	/// Messages that will be summarized and discarded
	pub messages_to_summarize: Vec<AgentMessage>,
	/// Messages that will be turned into turn prefix summary (if splitting)
	pub turn_prefix_messages: Vec<AgentMessage>,
	/// Whether this is a split turn (cut point in middle of turn)
	pub is_split_turn: bool,
	pub tokens_before: u64,
	/// Summary from previous compaction, for iterative update
	pub previous_summary: Option<String>,
	/// File operations extracted from messagesToSummarize
	pub file_ops: FileOperations,
	/// Compaction settings from settings.jsonl
	pub settings: CompactionSettings,
}

fn is_projected_turn_start(entry: &ProjectedSessionEntry) -> bool {
	if matches!(entry.source_entry, SessionEntry::Compaction(_)) {
		return false;
	}
	entry.messages.iter().any(is_turn_start_message)
}

fn find_projected_turn_start_index(
	entries: &[ProjectedSessionEntry],
	entry_index: usize,
	start_index: usize,
) -> Option<usize> {
	for index in (start_index..=entry_index).rev() {
		if is_projected_turn_start(&entries[index]) {
			return Some(index);
		}
	}
	None
}

fn find_projected_cut_point(
	entries: &[ProjectedSessionEntry],
	start_index: usize,
	end_index: usize,
	keep_recent_tokens: u64,
) -> CutPointResult {
	let mut cut_points: Vec<usize> = Vec::new();
	for index in start_index..end_index {
		let entry = &entries[index];
		if !matches!(entry.source_entry, SessionEntry::Compaction(_)) && entry.messages.iter().any(is_cut_point_message)
		{
			cut_points.push(index);
		}
	}
	if cut_points.is_empty() {
		return CutPointResult { first_kept_entry_index: start_index, turn_start_index: None, is_split_turn: false };
	}

	let mut accumulated_tokens = 0u64;
	let mut exceeded_budget = false;
	let mut cut_index = cut_points[0];
	for index in (start_index..end_index).rev() {
		let message_tokens = entries[index]
			.messages
			.iter()
			.map(|message| estimate_tokens(message) as u64)
			.sum::<u64>();
		if message_tokens == 0 {
			continue;
		}
		accumulated_tokens += message_tokens;
		if accumulated_tokens >= keep_recent_tokens {
			exceeded_budget = true;
			cut_index = cut_points
				.iter()
				.find(|candidate| **candidate >= index)
				.copied()
				.unwrap_or(cut_points[cut_points.len() - 1]);
			break;
		}
	}

	// A recovery attempt and its omission edits are context-invisible after the
	// last visible input. Advance only for a closed suffix containing an
	// omitted assistant attempt; arbitrary metadata must not move the cut past
	// unsent input.
	let suffix = &entries[cut_index + 1..end_index];
	let is_intrinsically_visible = |entry: &ProjectedSessionEntry| {
		!matches!(entry.source_entry, SessionEntry::ContextEdit(_))
			&& !session_entry_to_context_messages(&entry.source_entry).is_empty()
	};
	let is_omitted = |entry: &ProjectedSessionEntry| is_intrinsically_visible(entry) && entry.messages.is_empty();
	let omitted_suffix_ids: HashSet<&str> = suffix
		.iter()
		.filter(|entry| is_omitted(entry))
		.map(|entry| entry.source_entry.id())
		.collect();
	let has_external_replacement = suffix.iter().any(|entry| match &entry.source_entry {
		SessionEntry::ContextEdit(edit) => {
			edit.replacement != ContextEditReplacement::Null && !omitted_suffix_ids.contains(edit.target_id.as_str())
		}
		_ => false,
	});
	let is_recovery_omission_suffix = exceeded_budget
		&& !has_external_replacement
		&& suffix.iter().any(|entry| {
			matches!(&entry.source_entry, SessionEntry::Message(message) if message.message.role() == "assistant")
				&& is_omitted(entry)
		})
		&& suffix.iter().all(|entry| {
			!matches!(entry.source_entry, SessionEntry::Compaction(_))
				&& (!is_intrinsically_visible(entry) || is_omitted(entry))
		});
	if is_recovery_omission_suffix {
		cut_index += 1;
	}

	while cut_index > start_index {
		let previous = &entries[cut_index - 1];
		if matches!(previous.source_entry, SessionEntry::Compaction(_)) || !previous.messages.is_empty() {
			break;
		}
		cut_index -= 1;
	}
	let starts_turn = is_projected_turn_start(&entries[cut_index]);
	let turn_start_index =
		if starts_turn { None } else { find_projected_turn_start_index(entries, cut_index, start_index) };
	CutPointResult {
		first_kept_entry_index: cut_index,
		turn_start_index,
		is_split_turn: !starts_turn && turn_start_index.is_some(),
	}
}

/// Compute the compaction inputs for a session path (`prepareCompaction`).
/// Returns `None` when there is nothing to compact.
pub fn prepare_compaction(path_entries: &[SessionEntry], settings: &CompactionSettings) -> Option<CompactionPreparation> {
	if matches!(path_entries.last(), Some(SessionEntry::Compaction(_))) {
		return None;
	}

	let projection = build_session_projection(path_entries, Leaf::Undefined, None);
	let projected_entries = &projection.entries;
	let source_entries: Vec<&SessionEntry> = projected_entries.iter().map(|entry| &entry.source_entry).collect();
	// The newest compaction is projected first. Older compaction entries can
	// still occur in its retained raw range, but their projected contribution
	// is empty.
	let prev_compaction_index =
		projected_entries.iter().position(|entry| {
			matches!(entry.source_entry, SessionEntry::Compaction(_)) && !entry.messages.is_empty()
		});

	let mut previous_summary: Option<String> = None;
	let mut boundary_start = 0usize;
	if let Some(index) = prev_compaction_index {
		let SessionEntry::Compaction(compaction) = &projected_entries[index].source_entry else { return None };
		previous_summary = Some(compaction.summary.clone());
		// The canonical projection has already selected the previous
		// compaction's retained tail.
		boundary_start = index + 1;
	}
	let boundary_end = projected_entries.len();
	let tokens_before = estimate_projected_context_tokens(&projection, path_entries).tokens;
	let cut_point = find_projected_cut_point(projected_entries, boundary_start, boundary_end, settings.keep_recent_tokens);

	let first_kept_entry = projected_entries
		.get(cut_point.first_kept_entry_index)
		.map(|entry| &entry.source_entry)
		.filter(|entry| !entry.id().is_empty())?;
	let first_kept_entry_id = first_kept_entry.id().to_string();
	let history_end = if cut_point.is_split_turn {
		cut_point.turn_start_index.unwrap_or(cut_point.first_kept_entry_index)
	} else {
		cut_point.first_kept_entry_index
	};

	let messages_to_summarize: Vec<AgentMessage> = projected_entries[boundary_start..history_end]
		.iter()
		.flat_map(get_messages_from_projected_entry_for_compaction)
		.collect();
	let turn_prefix_messages: Vec<AgentMessage> = if cut_point.is_split_turn {
		projected_entries[cut_point.turn_start_index.unwrap_or(cut_point.first_kept_entry_index)
			..cut_point.first_kept_entry_index]
			.iter()
			.flat_map(get_messages_from_projected_entry_for_compaction)
			.collect()
	} else {
		Vec::new()
	};

	if messages_to_summarize.is_empty() && turn_prefix_messages.is_empty() {
		return None;
	}

	// Extract file operations from edited model-visible messages and the
	// previous compaction.
	let mut file_ops = extract_file_operations(&messages_to_summarize, &source_entries, prev_compaction_index);

	// Also extract file ops from turn prefix if splitting
	if cut_point.is_split_turn {
		for message in &turn_prefix_messages {
			extract_file_ops_from_message(message, &mut file_ops);
		}
	}

	Some(CompactionPreparation {
		first_kept_entry_id,
		messages_to_summarize,
		turn_prefix_messages,
		is_split_turn: cut_point.is_split_turn,
		tokens_before,
		previous_summary,
		file_ops,
		settings: settings.clone(),
	})
}

// ============================================================================
// Main compaction function
// ============================================================================

const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "The messages above are earlier context from an ongoing conversation. Later messages are stored separately and do not need to be reconstructed.\n\nCreate a concise checkpoint of the user's request and the progress shown above. This checkpoint will be placed before the later messages so the conversation can continue with the necessary context.\n\n## Original Request\n[What did the user ask for?]\n\n## Progress So Far\n- [Key decisions and work completed in these messages]\n\n## Context Needed to Continue\n- [Information from these messages needed to understand the later work]\n\nOnly summarize information explicitly present above. Do not infer or recreate later messages.";

/// Generate summaries for compaction using prepared data (`compact`).
/// Returns [`CompactionResult`] - SessionManager adds uuid/parentUuid when
/// saving.
///
/// Upstream throws on summarization failures; the port returns the exact
/// error strings as `Err` (crate docs D2/D23).
#[allow(clippy::too_many_arguments)]
pub async fn compact(
	preparation: &CompactionPreparation,
	model: &Model,
	api_key: Option<&str>,
	headers: Option<&pi_ai::types::ProviderHeaders>,
	custom_instructions: Option<&str>,
	signal: Option<&AbortSignal>,
	thinking_level: Option<AgentThinkingLevel>,
	stream_fn: Option<StreamFn>,
	env: Option<&pi_ai::types::ProviderEnv>,
	retry: Option<&RetryPolicy>,
	callbacks: Option<&RetryCallbacks>,
	session_id: Option<&str>,
) -> Result<CompactionResult, String> {
	let CompactionPreparation {
		first_kept_entry_id,
		messages_to_summarize,
		turn_prefix_messages,
		is_split_turn,
		tokens_before,
		previous_summary,
		file_ops,
		settings,
	} = preparation;

	// Generate summaries and merge into one
	let summary: String;
	let summary_usage: Usage;

	if *is_split_turn && !turn_prefix_messages.is_empty() {
		let mut history_text = previous_summary.clone().unwrap_or_else(|| "No prior history.".to_string());
		let mut history_usage: Option<Usage> = None;
		if !messages_to_summarize.is_empty() {
			let history_result = generate_summary_with_usage(
				messages_to_summarize,
				model,
				settings.reserve_tokens,
				api_key,
				headers,
				signal,
				custom_instructions,
				previous_summary.as_deref(),
				thinking_level,
				stream_fn.clone(),
				env,
				retry,
				callbacks,
				session_id,
			)
			.await?;
			history_text = history_result.text;
			history_usage = Some(history_result.usage);
		}
		let turn_prefix_result = generate_turn_prefix_summary(
			turn_prefix_messages,
			model,
			settings.reserve_tokens,
			api_key,
			headers,
			env,
			signal,
			thinking_level,
			stream_fn,
			retry,
			callbacks,
			session_id,
		)
		.await;
		let turn_prefix_result = turn_prefix_result?;
		// Merge into single summary
		summary =
			format!("{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{}", turn_prefix_result.text);
		summary_usage = match history_usage {
			Some(history_usage) => combine_usage(&history_usage, &turn_prefix_result.usage),
			None => turn_prefix_result.usage,
		};
	} else {
		// Just generate history summary
		let result = generate_summary_with_usage(
			messages_to_summarize,
			model,
			settings.reserve_tokens,
			api_key,
			headers,
			signal,
			custom_instructions,
			previous_summary.as_deref(),
			thinking_level,
			stream_fn.clone(),
			env,
			retry,
			callbacks,
			session_id,
		)
		.await?;
		summary = result.text;
		summary_usage = result.usage;
	}

	// Compute file lists and append to summary
	let lists = compute_file_lists(file_ops);
	let summary = summary + &format_file_operations(&lists.read_files, &lists.modified_files);

	if first_kept_entry_id.is_empty() {
		return Err("First kept entry has no UUID - session may need migration".to_string());
	}

	Ok(CompactionResult {
		summary,
		first_kept_entry_id: first_kept_entry_id.clone(),
		tokens_before: *tokens_before,
		estimated_tokens_after: None,
		usage: Some(summary_usage),
		details: Some(CompactionDetails {
			read_files: lists.read_files,
			modified_files: lists.modified_files,
		}),
	})
}

/// Generate a summary for a turn prefix (when splitting a turn)
/// (`generateTurnPrefixSummary`).
#[allow(clippy::too_many_arguments)]
async fn generate_turn_prefix_summary(
	messages: &[AgentMessage],
	model: &Model,
	reserve_tokens: u64,
	api_key: Option<&str>,
	headers: Option<&pi_ai::types::ProviderHeaders>,
	env: Option<&pi_ai::types::ProviderEnv>,
	signal: Option<&AbortSignal>,
	thinking_level: Option<AgentThinkingLevel>,
	stream_fn: Option<StreamFn>,
	retry: Option<&RetryPolicy>,
	callbacks: Option<&RetryCallbacks>,
	session_id: Option<&str>,
) -> Result<SummaryWithUsage, String> {
	// Smaller budget for turn prefix
	let base_cap = (0.5 * reserve_tokens as f64).floor() as u64;
	let max_tokens = if model.max_tokens > 0 { base_cap.min(model.max_tokens) } else { base_cap };
	let llm_messages = convert_to_llm(messages);
	let conversation_text = serialize_conversation(&llm_messages);
	let prompt_text = format!("# Conversation\n{conversation_text}\n\n# Instructions\n{TURN_PREFIX_SUMMARIZATION_PROMPT}");

	let response = complete_summarization(
		model,
		build_summarization_context(prompt_text),
		create_summarization_options(model, max_tokens, api_key, headers, env, signal, thinking_level, session_id),
		stream_fn,
		retry,
		callbacks,
	)
	.await?;

	if let Some(failure) = get_summarization_failure(&response, "Turn prefix summarization") {
		return Err(failure);
	}
	if response.content.iter().any(|block| matches!(block, AssistantContent::ToolCall(_))) {
		return Err("Turn prefix summarization attempted to call a tool".to_string());
	}

	Ok(SummaryWithUsage { text: content_blocks_text(&response.content, "\n"), usage: response.usage })
}

#[cfg(test)]
#[path = "compaction_tests.rs"]
mod compaction_tests;
