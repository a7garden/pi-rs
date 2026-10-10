//! Port of `packages/durable/src/harness/live.ts`.
//!
//! Deviations: D1 — `LiveState` and its parts are typed serde structs; the
//! commit-path glue converts the erased conversation document
//! (`JsonObject`) to/from [`LiveState`] around mutation (upstream mutates a
//! chord `Draft` in place). D2 — `settleSchedulerOutcome` takes the erased
//! `Tx`; the partial-message conversion calls
//! [`crate::harness::generation::convert_partial`].

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::documents::{DocDefinition, DocToken};
use crate::harness::types::{CompactionReason, EntryId, SubmissionId, TaskId, ToolDiagnostic};
use crate::session::Tx;
use crate::types::{DocumentFork, DocumentHistory};
use crate::types::{SubmissionSettlement, TaskRecord};

/// Presentation of one tool call of the current round.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSlot {
    pub call_id: String,
    pub name: String,
    /// Absent for a call not started yet (sequential round) and for a call
    /// its request did not offer, which starts `done` with the `entry`
    /// generation wrote.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    pub status: SlotStatus,
    /// Retained running output and what the bounds dropped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dropped_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dropped_lines: Option<u64>,
    /// Last `details()` value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Diagnostics recorded through `api.diagnostic()`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<ToolDiagnostic>>,
    /// Result entry once done; absent when the tool task faulted or was
    /// orphaned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry: Option<EntryId>,
}

/// Slot lifecycle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotStatus {
    #[default]
    Pending,
    Running,
    Done,
}

/// Presentation of one live compaction task (spec §8.7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionStatus {
    pub task_id: TaskId,
    pub reason: CompactionReason,
    /// Whether a generation waits for it: a compaction the generation owns.
    pub blocking: bool,
    pub attempt: u64,
    /// Durable backoff before the next summarization attempt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryBackoff>,
}

/// Durable backoff marker shared by generation and compaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetryBackoff {
    pub at: u64,
    pub error: String,
}

/// Presentation of the current generation attempt.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationStatus {
    pub attempt: u64,
    /// Committed throttled partial of the in-flight response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Value>,
    /// Durable backoff before the next attempt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryBackoff>,
    /// Provider-side deferred response being polled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredPoll>,
}

/// Provider-side deferred response being polled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredPoll {
    pub poll_at: u64,
}

/// Run control: the task that settles the run's inputs, and those inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunState {
    pub task_id: TaskId,
    pub inputs: Vec<SubmissionId>,
}

/// Built-in live conversation state: run control and presentation of the
/// current generation and tool round.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveState {
    /// Present exactly while busy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<RunState>,
    /// Presentation of the current generation attempt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationStatus>,
    /// The current tool round in call order, from the tool-calling answer
    /// until the generation's `tools` phase ends it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSlot>>,
    /// Live compaction tasks in task ID order; absent when none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compactions: Option<Vec<CompactionStatus>>,
}

fn live_definition() -> DocDefinition {
    DocDefinition::conversation(
        "pi.live",
        1,
        DocumentHistory::Latest,
        DocumentFork::Initial,
        || {
            serde_json::json!({})
                .as_object()
                .cloned()
                .unwrap_or_default()
        },
    )
    .with_checkpoint_when(|value, _, _| {
        // A complete base whenever nothing runs (spec §8.2): no generation
        // and no running tool slot. That holds while idle, in the commit
        // handing a generation over to its tool round, and between tools, so
        // the delta chain spans at most one generation or the overlapping
        // execution of one round's tools. A slot holds output only while
        // running, so every base is small. Do not add a delta-count bound;
        // the tool output benchmark checks this rule.
        let state: Result<LiveState, _> = serde_json::from_value(Value::Object(value.clone()));
        match state {
            Ok(state) => {
                state.generation.is_none()
                    && !state
                        .tools
                        .unwrap_or_default()
                        .iter()
                        .any(|slot| slot.status == SlotStatus::Running)
            }
            Err(_) => true,
        }
    })
}

/// Built-in live presentation document of a conversation.
static LIVE_DOC: std::sync::LazyLock<DocToken> =
    std::sync::LazyLock::new(|| DocToken::new(live_definition()));

pub fn live_doc() -> DocToken {
    LIVE_DOC.clone()
}

/// Built-in task kinds that can own `pi.live.run`.
fn is_run_task_kind(kind: &str) -> bool {
    kind == "pi.generation"
}
const TOOL_TASK_KIND: &str = "pi.tool";
const COMPACTION_TASK_KIND: &str = "pi.compaction";

/// End the run owned by `task_id`: settle each of its inputs and remove
/// `run`. Always removes `generation` and `tools`, whose presentation
/// belongs to the ending run.
pub fn end_run(
    live: &mut LiveState,
    tx: &Tx<'_>,
    task_id: &TaskId,
    settlement: SubmissionSettlement,
) -> Result<(), String> {
    end_run_on_tx(live, tx, task_id, settlement)
}

/// Settle each input of `run` through `tx` (the direct form used where a
/// `Tx` is at hand).
pub fn end_run_on_tx(
    live: &mut LiveState,
    tx: &Tx<'_>,
    task_id: &TaskId,
    settlement: SubmissionSettlement,
) -> Result<(), String> {
    let inputs: Vec<SubmissionId> = live
        .run
        .as_ref()
        .filter(|run| &run.task_id == task_id)
        .map(|run| run.inputs.clone())
        .unwrap_or_default();
    if !inputs.is_empty() {
        live.run = None;
    }
    live.generation = None;
    live.tools = None;
    for id in inputs {
        tx.settle_submission(id, settlement.clone())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Add the status of a compaction task created in this commit; statuses stay
/// in task ID order.
pub fn add_compaction_status(live: &mut LiveState, status: CompactionStatus) {
    live.compactions.get_or_insert_with(Vec::new).push(status);
}

/// The status of compaction task `task_id`, if listed.
pub fn compaction_status<'a>(
    live: &'a mut LiveState,
    task_id: &TaskId,
) -> Option<&'a mut CompactionStatus> {
    live.compactions
        .as_mut()?
        .iter_mut()
        .find(|status| &status.task_id == task_id)
}

/// Remove the status of compaction task `task_id`, and the list once empty.
pub fn remove_compaction_status(live: &mut LiveState, task_id: &TaskId) {
    let Some(statuses) = live.compactions.as_mut() else {
        return;
    };
    if let Some(index) = statuses
        .iter()
        .position(|status| &status.task_id == task_id)
    {
        statuses.remove(index);
    }
    if statuses.is_empty() {
        live.compactions = None;
    }
}

/// The slot of tool task `task_id` in the current round, if the round still
/// lists it.
pub fn tool_slot<'a>(live: &'a mut LiveState, task_id: &TaskId) -> Option<&'a mut ToolSlot> {
    live.tools
        .as_mut()?
        .iter_mut()
        .find(|slot| slot.task_id.as_ref() == Some(task_id))
}

/// Mark a slot done: the result entry, if any, now carries its running
/// output, details, and diagnostics.
pub fn finish_slot(slot: &mut ToolSlot, entry: Option<EntryId>) {
    slot.status = SlotStatus::Done;
    if let Some(entry) = entry {
        slot.entry = Some(entry);
    }
    clear_progress(slot);
}

/// Remove what a tool published while running; its result entry or a rerun
/// replaces it.
pub fn clear_progress(slot: &mut ToolSlot) {
    slot.output = None;
    slot.dropped_bytes = None;
    slot.dropped_lines = None;
    slot.details = None;
    slot.diagnostics = None;
}

/// Scheduler outcome reason for an unanswered settlement (upstream
/// `SchedulerOutcome` status part lives in `scheduler.rs`; this carries the
/// subset `settleSchedulerOutcome` reads).
pub enum SchedulerFault {
    Faulted { message: String },
    Aborted { reason: String },
    Orphaned { reason: String },
}

/// Harness cleanup for a terminal outcome the scheduler writes itself
/// (`faulted` or `orphaned`). A run task ends its run; a tool task's slot is
/// marked done without an entry, and context derivation synthesizes the
/// missing result; a compaction task's status is removed.
///
/// Ignores other kinds so it never creates `pi.live` elsewhere. The
/// scheduler calls this without knowing task kinds; the Harness passes it in
/// (spec §5.4).
///
/// A committed generation partial becomes an aborted assistant entry here,
/// exactly as in the generation abort handler, so the transcript keeps what
/// the model produced and `pi.usage` counts its spend. The scheduler's
/// commit has no task scope, so that entry has no `byTaskId`.
pub fn settle_scheduler_outcome(
    tx: &Tx<'_>,
    record: &TaskRecord,
    outcome: &SchedulerFault,
) -> Result<(), String> {
    if record.kind == TOOL_TASK_KIND {
        let draft = tx
            .doc(&live_doc(), Some(record.conversation_id.0), None, None)
            .map_err(|e| e.to_string())?;
        let mut live = read_live(&draft);
        if let Some(slot) = tool_slot(&mut live, &record.id) {
            finish_slot(slot, None);
        }
        store_live(&draft, &live)?;
        return Ok(());
    }
    if record.kind == COMPACTION_TASK_KIND {
        let draft = tx
            .doc(&live_doc(), Some(record.conversation_id.0), None, None)
            .map_err(|e| e.to_string())?;
        let mut live = read_live(&draft);
        remove_compaction_status(&mut live, &record.id);
        store_live(&draft, &live)?;
        return Ok(());
    }
    if !is_run_task_kind(&record.kind) {
        return Ok(());
    }
    let draft = tx
        .doc(&live_doc(), Some(record.conversation_id.0), None, None)
        .map_err(|e| e.to_string())?;
    let mut live = read_live(&draft);
    if live.run.as_ref().map(|run| &run.task_id) != Some(&record.id) {
        return Ok(());
    }
    crate::harness::generation::convert_partial(tx, &mut live, record.conversation_id)?;
    let settlement = match outcome {
        SchedulerFault::Faulted { message } => SubmissionSettlement::Unanswered {
            reason: "faulted".to_string(),
            detail: Some(Value::String(message.clone())),
        },
        SchedulerFault::Aborted { reason } => SubmissionSettlement::Unanswered {
            reason: reason.clone(),
            detail: None,
        },
        SchedulerFault::Orphaned { reason } => SubmissionSettlement::Unanswered {
            reason: reason.clone(),
            detail: None,
        },
    };
    end_run_on_tx(&mut live, tx, &record.id, settlement)?;
    store_live(&draft, &live)?;
    Ok(())
}

/// Read the typed state from an erased document draft (missing or invalid
/// shapes read as the empty state).
pub fn read_live(draft: &crate::session::DocDraft) -> LiveState {
    serde_json::from_value(Value::Object(draft.value().unwrap_or_default())).unwrap_or_default()
}

/// Store the typed state through an erased document draft. Recursive writes
/// preserve the nested deltas consumed by event translation.
pub fn store_live(draft: &crate::session::DocDraft, live: &LiveState) -> Result<(), String> {
    use pi_chord::delta::Seg;
    let value = serde_json::to_value(live).expect("LiveState serializes");
    let obj = value.as_object().expect("LiveState is an object");
    for key in ["run", "generation", "tools", "compactions"] {
        let mut path = vec![Seg::key(key)];
        let before = draft.get(&path)?;
        store_live_value(draft, &mut path, before.as_ref(), obj.get(key))?;
    }
    Ok(())
}

fn store_live_value(
    draft: &crate::session::DocDraft,
    path: &mut Vec<pi_chord::delta::Seg>,
    before: Option<&Value>,
    after: Option<&Value>,
) -> Result<(), String> {
    use pi_chord::delta::Seg;
    if before == after {
        return Ok(());
    }
    match (before, after) {
        (Some(_), None) => draft.delete(path),
        (None, Some(value)) => draft.set(path, value.clone()),
        (Some(Value::Object(before)), Some(Value::Object(after))) => {
            for (key, old) in before {
                path.push(Seg::key(key));
                let result = store_live_value(draft, path, Some(old), after.get(key));
                path.pop();
                result?;
            }
            for (key, value) in after {
                if before.contains_key(key) {
                    continue;
                }
                path.push(Seg::key(key));
                let result = store_live_value(draft, path, None, Some(value));
                path.pop();
                result?;
            }
            Ok(())
        }
        (Some(Value::Array(before)), Some(Value::Array(after))) => {
            for (index, (old, new)) in before.iter().zip(after).enumerate() {
                path.push(Seg::Index(index));
                let result = store_live_value(draft, path, Some(old), Some(new));
                path.pop();
                result?;
            }
            if after.len() > before.len() {
                draft.push(path, after[before.len()..].to_vec()).map(|_| ())
            } else if after.len() < before.len() {
                draft.set_length(path, after.len())
            } else {
                Ok(())
            }
        }
        (_, Some(value)) => draft.set(path, value.clone()),
        (None, None) => Ok(()),
    }
}
