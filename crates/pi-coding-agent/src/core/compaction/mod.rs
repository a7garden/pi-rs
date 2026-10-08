//! Port of `packages/coding-agent/src/core/compaction/` — compaction and
//! summarization utilities ([`compaction`], [`branch_summarization`],
//! [`utils`]).
//!
//! Upstream `index.ts` re-exports the three modules; this module mirrors that
//! surface.
//!
//! LLM-dependent functions take the crate-shared seams: a
//! [`pi_agent::StreamFn`] override or the default `completeSimple` transport,
//! with [`pi_ai::utils::retry::RetryPolicy`] handling transient stream drops
//! (crate docs D23).

pub mod branch_summarization;
#[path = "compaction.rs"]
pub mod compaction;
pub mod utils;

pub use branch_summarization::{
	collect_entries_for_branch_summary, generate_branch_summary, prepare_branch_entries, BranchPreparation,
	BranchSummaryDetails, BranchSummaryFileOperations, BranchSummaryResult, CollectEntriesResult,
	GenerateBranchSummaryOptions,
};
pub use compaction::{
	calculate_context_tokens, compact, complete_summarization, estimate_context_tokens,
	estimate_projected_context_tokens, estimate_tokens, find_cut_point, find_turn_start_index, generate_summary,
	generate_summary_with_usage, get_last_assistant_usage, get_summarization_failure, prepare_compaction,
	should_compact, CompactionDetails, CompactionPreparation, CompactionResult, CompactionSettings,
	ContextUsageEstimate, CutPointResult, SummaryWithUsage, DEFAULT_COMPACTION_SETTINGS,
};
pub use utils::{
	compute_file_lists, create_file_ops, extract_file_ops_from_message, format_file_operations,
	serialize_conversation, FileLists, FileOperations, SUMMARIZATION_SYSTEM_PROMPT,
};
