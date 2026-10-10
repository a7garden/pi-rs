//! Port of `packages/durable/src/harness` (the durable agent harness).
//!
//! Module ledgers live at the top of each file. Upstream `index.ts`
//! re-exports are mirrored below; the integrator-owned tool seam
//! ([`api`]) is re-exported verbatim, and the session-owned conversation
//! surfaces are re-exported from [`crate::session`].
//!
//! Test modules under `tests.rs` port the upstream harness suites; open
//! reconciliation items are tracked in `PORT_MAP.md`.

pub mod agent;
pub mod api;
pub mod compaction;
pub mod context;
pub mod define;
pub mod events;
pub mod generation;
pub mod harness;
pub mod inbox;
pub mod json;
pub mod live;
pub mod output;
pub mod prompt;
pub mod provider;
pub mod registry;
pub mod scheduler;
pub mod submissions;
pub mod task_graph;
pub mod tool;
pub mod types;
pub mod usage;
pub mod util;
pub mod view;

pub use crate::harness::api::{
    define_tool, OutputRetain, QueueMode, ToolControl, ToolDiagnostic, ToolDiagnosticSeverity,
    ToolExecute, ToolExecutionApi, ToolExecutionMode, ToolExecutionResult, ToolOutputLimits,
    ToolRegistration, ToolReplay,
};
pub use crate::session::{ContextView, ConversationAbortOptions, ConversationHandle, SettledTask};
pub use agent::{
    add_tools, agent_doc, agent_hooks, configure, create_agent, resolve_agent, resolve_settings,
    DEFAULT_COMPACTION_POLICY, DEFAULT_CONTEXT_RETENTION_MS, DEFAULT_PROGRESS_POLICY,
    DEFAULT_RETRY_POLICY, INSTRUCTIONS_KEY,
};
pub use compaction::{
    compaction_task, create_compaction, estimate_context, select_cut, serialize_conversation,
    summarized_messages, CompactionCheckpoint, CompactionInput, SummaryRequest,
};
pub use context::{
    active_entries, capture_context_bounds, derive_context, order_tool_results, read_context_from,
    replay_sections, ContextBounds, ContextRange,
};
pub use define::{define_extension, hook, section, wrap_section, wrap_tool};
pub use events::{watch_events, AgentEvent, AgentEventStream, MessageChange, SnapshotEvent};
pub use generation::{
    convert_partial, generation_task, hand_over, start_run, GenerationCheckpoint, GenerationInput,
    GenerationResult,
};
pub use harness::{open_harness, Conversation, ConversationWatch, Harness};
pub use inbox::{
    apply_boundary, inbox_doc, is_stale, prepare_boundary, remove_inbox_item,
    withdraw_queued_inputs, Boundary, BoundaryAt, BoundaryResult, InboxItem, InboxState,
    QueueModes,
};
pub use live::{
    add_compaction_status, clear_progress, compaction_status, end_run, finish_slot, live_doc,
    read_live, remove_compaction_status, settle_scheduler_outcome, store_live, tool_slot,
    CompactionStatus, DeferredPoll, GenerationStatus, LiveState, SlotStatus, ToolSlot,
};
pub use output::{
    bound_output, character_end, sanitize_output, BoundedOutput, Chunk, OutputBuffer, OutputLimits,
    OutputSlice, Progress, Retain, PROGRESS_BYTES_PER_SECOND,
};
pub use prompt::{plan_system_entries, render_sections};
pub use provider::{ensure_provider_session_id, provider_doc, ProviderState};
pub use registry::{create_registry, RegistryImpl, RegistryState};
pub use task_graph::{TaskGraph, TaskGraphNode, TaskGraphState, TaskGraphView, TaskGraphWatch};
pub use tool::{
    append_tool_result, harness_error, tool_task, ToolTaskCheckpoint, ToolTaskInput, ToolTaskResult,
};
pub use types::*;
pub use usage::{add_usage, add_usage_state, record_usage, usage_doc, UsageBucket, UsageState};
pub use util::{closed_error, flush, scan_all, Waiters};
pub use view::{ConversationView, ConversationViews, ViewObserver};

#[cfg(test)]
mod tests;
