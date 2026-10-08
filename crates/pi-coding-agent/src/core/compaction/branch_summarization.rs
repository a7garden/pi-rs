//! Port of
//! `packages/coding-agent/src/core/compaction/branch-summarization.ts` —
//! branch summarization for tree navigation. When navigating to a different
//! point in the session tree, this generates a summary of the branch being
//! left so context isn't lost.

use std::collections::BTreeSet;

use pi_agent::{AgentMessage, StreamFn};
use pi_ai::types::{AssistantContent, Model, ProviderEnv, ProviderHeaders, SimpleStreamOptions, Usage};
use pi_ai::utils::abort::AbortSignal;
use pi_ai::utils::retry::{RetryCallbacks, RetryPolicy};
use pi_ai::utils::text::content_blocks_text;
use pi_ai::utils::transcript::normalize_context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::compaction::compaction::{complete_summarization, estimate_tokens, get_summarization_failure};
use crate::core::compaction::utils::{
	compute_file_lists, create_file_ops, extract_file_ops_from_message, format_file_operations,
	serialize_conversation, FileOperations, SUMMARIZATION_SYSTEM_PROMPT,
};
use crate::core::messages::{
	convert_to_llm, create_branch_summary_message, create_compaction_summary_message, create_custom_message,
	custom_message_to_agent,
};
use crate::core::session_manager::{SessionEntry, SessionManager};
use crate::utils::time::now_ms;

/// Upstream `export type { FileOperations } from "./utils.ts"`.
pub use crate::core::compaction::utils::FileOperations as BranchSummaryFileOperations;

// ============================================================================
// Types
// ============================================================================

/// Result of [`generate_branch_summary`] (`BranchSummaryResult`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BranchSummaryResult {
	pub summary: Option<String>,
	pub usage: Option<Usage>,
	pub read_files: Vec<String>,
	pub modified_files: Vec<String>,
	pub aborted: bool,
	pub error: Option<String>,
}

/// Details stored in `BranchSummaryEntry.details` for file tracking
/// (`BranchSummaryDetails`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryDetails {
	pub read_files: Vec<String>,
	pub modified_files: Vec<String>,
}

/// Entries prepared for summarization (`BranchPreparation`).
#[derive(Debug, Clone, PartialEq)]
pub struct BranchPreparation {
	/// Messages extracted for summarization, in chronological order
	pub messages: Vec<AgentMessage>,
	/// File operations extracted from tool calls
	pub file_ops: FileOperations,
	/// Total estimated tokens in messages
	pub total_tokens: u64,
}

/// Entries collected for branch summarization (`CollectEntriesResult`).
#[derive(Debug, Clone, PartialEq)]
pub struct CollectEntriesResult {
	/// Entries to summarize, in chronological order
	pub entries: Vec<SessionEntry>,
	/// Common ancestor between old and new position, if any
	pub common_ancestor_id: Option<String>,
}

/// Options for [`generate_branch_summary`] (`GenerateBranchSummaryOptions`).
pub struct GenerateBranchSummaryOptions<'a> {
	/// Model to use for summarization
	pub model: Model,
	/// API key for the model
	pub api_key: Option<&'a str>,
	/// Request headers for the model
	pub headers: Option<&'a ProviderHeaders>,
	/// Provider-scoped environment values for the model
	pub env: Option<&'a ProviderEnv>,
	/// Abort signal for cancellation
	pub signal: AbortSignal,
	/// Optional custom instructions for summarization
	pub custom_instructions: Option<&'a str>,
	/// If true, customInstructions replaces the default prompt instead of
	/// being appended
	pub replace_instructions: bool,
	/// Tokens reserved when selecting branch history (default 16384)
	pub reserve_tokens: Option<u64>,
	/// Optional session stream function. Used to preserve SDK request
	/// behavior without mutating agent state.
	pub stream_fn: Option<StreamFn>,
	/// Retry policy for transient summarization errors. Reuses coding-agent's
	/// `settings.retry`.
	pub retry: Option<&'a RetryPolicy>,
	/// Optional callbacks for retry reporting (e.g. TUI retry indicators).
	pub callbacks: Option<&'a RetryCallbacks>,
}

// ============================================================================
// Entry Collection
// ============================================================================

/// Collect entries that should be summarized when navigating from one
/// position to another (`collectEntriesForBranchSummary`).
///
/// Walks from oldLeafId back to the common ancestor with targetId, collecting
/// entries along the way. Does NOT stop at compaction boundaries - those are
/// included and their summaries become context.
///
/// - `session` - Session manager (read-only access)
/// - `old_leaf_id` - Current position (where we're navigating from)
/// - `target_id` - Target position (where we're navigating to)
///
/// Returns entries to summarize and the common ancestor.
pub fn collect_entries_for_branch_summary(
	session: &SessionManager,
	old_leaf_id: Option<&str>,
	target_id: &str,
) -> CollectEntriesResult {
	// If no old position, nothing to summarize
	let Some(old_leaf_id) = old_leaf_id else {
		return CollectEntriesResult { entries: Vec::new(), common_ancestor_id: None };
	};

	// Find common ancestor (deepest node that's on both paths)
	let old_path: BTreeSet<String> = session.get_branch_from(old_leaf_id).iter().map(|entry| entry.id().to_string()).collect();
	let target_path = session.get_branch_from(target_id);

	// targetPath is root-first, so iterate backwards to find deepest common
	// ancestor
	let mut common_ancestor_id: Option<String> = None;
	for entry in target_path.iter().rev() {
		if old_path.contains(entry.id()) {
			common_ancestor_id = Some(entry.id().to_string());
			break;
		}
	}

	// Collect entries from old leaf back to common ancestor
	let mut entries: Vec<SessionEntry> = Vec::new();
	let mut current: Option<String> = Some(old_leaf_id.to_string());

	while let Some(id) =
		current.filter(|id| common_ancestor_id.as_deref() != Some(id.as_str()))
	{
		let Some(entry) = session.get_entry(&id) else { break };
		entries.push(entry.clone());
		current = entry.parent_id().map(str::to_string);
	}

	// Reverse to get chronological order
	entries.reverse();

	CollectEntriesResult { entries, common_ancestor_id }
}

// ============================================================================
// Entry to Message Conversion
// ============================================================================

/// Extract AgentMessage from a session entry (`getMessageFromEntry`). Similar
/// to the compaction message extraction but also handles compaction entries.
fn get_message_from_entry(entry: &SessionEntry) -> Option<AgentMessage> {
	match entry {
		SessionEntry::Message(message_entry) => {
			// Skip tool results - context is in assistant's tool call
			if message_entry.message.role() == "toolResult" {
				return None;
			}
			Some(message_entry.message.clone())
		}
		SessionEntry::CustomMessage(custom) => Some(custom_message_to_agent(&create_custom_message(
			&custom.custom_type,
			custom.content.clone(),
			custom.display,
			custom.details.clone(),
			&custom.base.timestamp,
		))),
		SessionEntry::BranchSummary(summary) => Some(custom_message_to_agent(&create_branch_summary_message(
			&summary.summary,
			&summary.from_id,
			&summary.base.timestamp,
		))),
		SessionEntry::Compaction(compaction) => Some(custom_message_to_agent(&create_compaction_summary_message(
			&compaction.summary,
			compaction.tokens_before,
			&compaction.base.timestamp,
		))),
		// These don't contribute to conversation content
		SessionEntry::ThinkingLevelChange(_)
		| SessionEntry::ModelChange(_)
		| SessionEntry::Custom(_)
		| SessionEntry::Label(_)
		| SessionEntry::SessionInfo(_) => None,
		_ => None,
	}
}

/// Prepare entries for summarization with token budget
/// (`prepareBranchEntries`).
///
/// Walks entries from NEWEST to OLDEST, adding messages until we hit the
/// token budget. This ensures we keep the most recent context when the branch
/// is too long.
///
/// Also collects file operations from:
/// - Tool calls in assistant messages
/// - Existing branch_summary entries' details (for cumulative tracking)
///
/// - `entries` - Entries in chronological order
/// - `token_budget` - Maximum tokens to include (0 = no limit)
pub fn prepare_branch_entries(entries: &[SessionEntry], token_budget: u64) -> BranchPreparation {
	let mut messages: Vec<AgentMessage> = Vec::new();
	let mut file_ops = create_file_ops();
	let mut total_tokens = 0u64;

	// First pass: collect file ops from ALL entries (even if they don't fit in
	// token budget). This ensures we capture cumulative file tracking from
	// nested branch summaries. Only extract from pi-generated summaries
	// (fromHook !== true), not extension-generated ones.
	for entry in entries {
		if let SessionEntry::BranchSummary(summary) = entry {
			if summary.from_hook == Some(true) {
				continue;
			}
			if let Some(details) = &summary.details {
				// Upstream duck-types each field (`Array.isArray` checks), so
				// malformed payloads skip only the offending field.
				if let Some(read_files) = details.get("readFiles").and_then(Value::as_array) {
					for file in read_files.iter().filter_map(Value::as_str) {
						file_ops.read.insert(file.to_string());
					}
				}
				// Modified files go into both edited and written for proper
				// deduplication
				if let Some(modified_files) = details.get("modifiedFiles").and_then(Value::as_array) {
					for file in modified_files.iter().filter_map(Value::as_str) {
						file_ops.edited.insert(file.to_string());
					}
				}
			}
		}
	}

	// Second pass: walk from newest to oldest, adding messages until token
	// budget
	for entry in entries.iter().rev() {
		let Some(message) = get_message_from_entry(entry) else { continue };

		// Extract file ops from assistant messages (tool calls)
		extract_file_ops_from_message(&message, &mut file_ops);

		let tokens = estimate_tokens(&message) as u64;

		// Check budget before adding
		if token_budget > 0 && total_tokens + tokens > token_budget {
			// If this is a summary entry, try to fit it anyway as it's
			// important context
			if matches!(entry, SessionEntry::Compaction(_) | SessionEntry::BranchSummary(_)) {
				if (total_tokens as f64) < token_budget as f64 * 0.9 {
					messages.insert(0, message);
					total_tokens += tokens;
				}
			}
			// Stop - we've hit the budget
			break;
		}

		messages.insert(0, message);
		total_tokens += tokens;
	}

	BranchPreparation { messages, file_ops, total_tokens }
}

// ============================================================================
// Summary Generation
// ============================================================================

const BRANCH_SUMMARY_PREAMBLE: &str = "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n";

const BRANCH_SUMMARY_PROMPT: &str = "Create a structured summary of this conversation branch for context when returning later.\n\nUse this EXACT format:\n\n## Goal\n[What was the user trying to accomplish in this branch?]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Work that was started but not finished]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [What should happen next to continue this work]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

/// Generate a summary of abandoned branch entries (`generateBranchSummary`).
///
/// - `entries` - Session entries to summarize (chronological order)
/// - `options` - Generation options
///
/// Upstream throws on transport-level failures (rejected promise); the port
/// surfaces them through [`BranchSummaryResult::error`] (crate docs D23).
pub async fn generate_branch_summary(
	entries: &[SessionEntry],
	options: GenerateBranchSummaryOptions<'_>,
) -> BranchSummaryResult {
	let GenerateBranchSummaryOptions {
		model,
		api_key,
		headers,
		env,
		signal,
		custom_instructions,
		replace_instructions,
		reserve_tokens,
		stream_fn,
		retry,
		callbacks,
	} = options;

	// Token budget = context window minus reserved space for prompt + response
	let context_window = if model.context_window != 0 { model.context_window } else { 128_000 };
	let token_budget = context_window.saturating_sub(reserve_tokens.unwrap_or(16384));

	let preparation = prepare_branch_entries(entries, token_budget);
	let file_ops = preparation.file_ops;
	let messages = preparation.messages;

	if messages.is_empty() {
		return BranchSummaryResult { summary: Some("No content to summarize".to_string()), ..Default::default() };
	}

	// Transform to LLM-compatible messages, then serialize to text.
	// Serialization prevents the model from treating it as a conversation to
	// continue
	let llm_messages = convert_to_llm(&messages);
	let conversation_text = serialize_conversation(&llm_messages);

	// Build prompt
	let instructions = if replace_instructions && custom_instructions.is_some() {
		custom_instructions.unwrap_or("").to_string()
	} else if let Some(custom_instructions) = custom_instructions {
		format!("{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom_instructions}")
	} else {
		BRANCH_SUMMARY_PROMPT.to_string()
	};
	let prompt_text = format!("<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}");

	let summarization_messages = vec![pi_ai::types::Message::User(pi_ai::types::UserMessage {
		content: pi_ai::types::UserContentField::Blocks(vec![pi_ai::types::UserContent::Text(
			pi_ai::types::TextContent::new(prompt_text),
		)]),
		timestamp: now_ms(),
	})];

	let max_tokens = if model.max_tokens > 0 { model.max_tokens.min(4096) } else { 4096 };

	// Call LLM for summarization. Prefer the session stream function so SDK
	// request behavior (timeouts, retries, attribution headers) stays
	// consistent without running through agent state/events. Retried via
	// completeSummarization so transient stream drops reuse the configured
	// retry policy.
	let context = normalize_context(&pi_ai::types::Context {
		system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
		messages: summarization_messages,
		tools: None,
	});
	let mut request_options = SimpleStreamOptions::default();
	request_options.base.max_tokens = Some(max_tokens);
	request_options.base.base.api_key = api_key.map(str::to_string);
	request_options.base.base.headers = headers.cloned();
	request_options.base.base.env = env.cloned();
	request_options.base.base.signal = Some(signal.clone());
	let response = complete_summarization(&model, context, request_options, stream_fn, retry, callbacks).await;
	let response = match response {
		Ok(response) => response,
		Err(error) => return BranchSummaryResult { error: Some(error), ..Default::default() },
	};

	// Check if aborted or errored
	if response.stop_reason == pi_ai::types::StopReason::Aborted {
		return BranchSummaryResult { aborted: true, ..Default::default() };
	}
	if let Some(failure) = get_summarization_failure(&response, "Branch summarization") {
		return BranchSummaryResult { error: Some(failure), ..Default::default() };
	}
	if response.content.iter().any(|block| matches!(block, AssistantContent::ToolCall(_))) {
		return BranchSummaryResult {
			error: Some("Branch summarization attempted to call a tool".to_string()),
			..Default::default()
		};
	}

	let summary = content_blocks_text(&response.content, "\n");

	// Prepend preamble to provide context about the branch summary
	let mut summary = format!("{BRANCH_SUMMARY_PREAMBLE}{summary}");

	// Compute file lists and append to summary
	let lists = compute_file_lists(&file_ops);
	summary += &format_file_operations(&lists.read_files, &lists.modified_files);

	let summary = if summary.is_empty() { "No summary generated".to_string() } else { summary };

	BranchSummaryResult {
		summary: Some(summary),
		usage: Some(response.usage),
		read_files: lists.read_files,
		modified_files: lists.modified_files,
		aborted: false,
		error: None,
	}
}
