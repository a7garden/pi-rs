//! Port of `packages/durable/src/tasks.ts` (`defineTask`) plus the erased task
//! machine types upstream declares in `types.ts` (`TaskDefinition`,
//! `TaskRuntime`, `RunningTask`, `NextTaskState`, `HookRunner`,
//! `PhaseHandler`). Owned by the harness slice (integrator assignment).
//!
//! Erasure (D1 — see `harness/types.rs`): upstream `Task<I, S extends {phase},
//! R, H>` becomes one [`TaskDefinition`] whose checkpoints are
//! `serde_json::Value` (`checkpoint["phase"]` names the phase), whose inputs
//! and results are `Value`/[`TaskOutcome`], and whose phase map is a
//! `BTreeMap<String, PhaseHandler>`. Hooks become erased named-handler tables:
//! a [`HookRegistration`](crate::harness::types::HookRegistration) carries an
//! `Arc<dyn HookHandler>` and the scheduler dispatches by name
//! ([`HookRunner::each`]); the built-in hook payloads live in
//! `crate::harness::types` as concrete handler structs (D2).
//!
//! `Transaction` operations are synchronous on the Session line (upstream
//! awaits resolve in order on one line), so runtime/commit callbacks are sync
//! closures `FnMut(&mut Tx, &TaskRecord) -> Result<Option<NextTaskState>,
//! String>` (D3); async work (providers, tools) happens between commits.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use pi_chord::context::Context;
use serde_json::Value;

use crate::harness::types::RegistrySnapshot;
use crate::session::Tx;
use crate::types::{
    ConversationId, EntryId, EntryRecord, JoinPolicy, JsonObject, TaskId, TaskOutcome, TaskRecord,
};

/// Boxed sendable future.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Erased hook handler object (upstream: the `handlers` object of a hook
/// registration). Dispatch is by handler *name* through
/// [`HookRunner::each`]; the invoker downcasts to the concrete hooks struct it
/// expects (D2).
///
/// **Calling `as_any`/`as_any_arc`:** the blanket impl below covers every
/// sized `Send + Sync` type, including `Arc<dyn HookHandler>` and references
/// to it. Method-call sugar therefore picks the wrapper's own impl, not the
/// inner handler's (`arc.as_any()` yields the `Arc` itself as the `Any`;
/// even a `&dyn HookHandler` receiver resolves to the blanket impl for the
/// reference type). The only correct form is the unsized place expression:
/// `(*handler).as_any()` for an owned `Arc` binding, `(**handler).as_any()`
/// for `&Arc<dyn HookHandler>` (e.g. a [`HookInvokeFn`] parameter). These
/// hit the trait-object vtable and produce the concrete handler as `Any`.
pub trait HookHandler: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync>;
}

impl<T: Send + Sync + 'static> HookHandler for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync> {
        self
    }
}

/// Invoker of one hook dispatch step: receives a handler table and returns the
/// outcome of invoking the named handlers on it. Composition happens inside.
pub type HookInvokeFn =
    Arc<dyn Fn(&Arc<dyn HookHandler>) -> BoxFuture<Result<(), String>> + Send + Sync>;

/// Dispatches one hook of a task to every matching registered handler, in
/// registry order of the phase snapshot (upstream `HookRunner`).
#[derive(Clone)]
pub struct HookRunner {
    dispatch: Arc<dyn Fn(&str, HookInvokeFn) -> BoxFuture<Result<(), String>> + Send + Sync>,
}

impl HookRunner {
    /// Build a runner over a dispatch closure (the scheduler supplies one).
    pub fn new(
        dispatch: impl Fn(&str, HookInvokeFn) -> BoxFuture<Result<(), String>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            dispatch: Arc::new(dispatch),
        }
    }

    /// A runner with no hooks (host objects outside an invocation).
    pub fn empty() -> Self {
        Self::new(|_name, _invoke| Box::pin(async { Ok(()) }))
    }

    /// Call `invoke` with each matching handler named `name`. An ordinary
    /// error from `invoke` is reported and the next handler runs; once the
    /// invocation is signalled, the error propagates (the scheduler's
    /// dispatch closure implements that rule).
    pub async fn each(&self, name: &str, invoke: HookInvokeFn) -> Result<(), String> {
        (self.dispatch)(name, invoke).await
    }
}

impl std::fmt::Debug for HookRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HookRunner")
    }
}

/// Live task record reserved by one invocation (upstream `RunningTask`): the
/// erased [`TaskRecord`] while `state.status == running`.
pub type RunningTask = TaskRecord;

/// Next state a task commits for itself: a replacement checkpoint, a wait, or
/// its outcome (upstream `NextTaskState`). A returned `terminal` state is
/// stored as `completing` while ordinary owned work below the task is live.
#[derive(Debug, Clone, PartialEq)]
pub enum NextTaskState {
    /// Continue with a replacement checkpoint.
    Running { checkpoint: Value },
    /// Park until every task in `on` is terminal.
    Waiting {
        checkpoint: Value,
        on: Vec<TaskId>,
        policy: JoinPolicy,
    },
    /// Decide the outcome; the scheduler holds it as `completing` when needed.
    Terminal { outcome: TaskOutcome },
}

impl NextTaskState {
    /// Terminal `completed` receipt (test helper upstream `completed()`).
    pub fn completed(result: Value) -> Self {
        NextTaskState::Terminal {
            outcome: TaskOutcome::completed(result),
        }
    }

    /// Terminal `aborted` receipt with a reason (upstream `abortedWith()`).
    pub fn aborted_with(reason: impl Into<String>) -> Self {
        NextTaskState::Terminal {
            outcome: TaskOutcome::aborted(Some(reason.into())),
        }
    }
}

/// Result of a migration from an older stored version (upstream `migrate`).
#[derive(Debug, Clone, PartialEq)]
pub struct TaskMigration {
    pub input: Value,
    pub checkpoint: Value,
}

/// One phase handler: runs the phase named by the task's checkpoint. It must
/// commit a changed checkpoint or a terminal outcome through
/// [`TaskRuntime::commit`]; returning without durable progress faults the
/// task. Erased form of upstream `PhaseHandler<I, P, S, R, H>` (D1): the task
/// arrives as its full [`TaskRecord`] (`state.status == running`) and the
/// checkpoint is raw JSON.
pub type PhaseHandler = Arc<
    dyn Fn(TaskRecord, Arc<dyn TaskRuntime>, Context) -> BoxFuture<Result<(), String>>
        + Send
        + Sync,
>;

/// Executable durable state machine definition, registered in the registry by
/// `name` (erased `TaskDefinition`).
#[derive(Clone)]
pub struct TaskDefinition {
    /// Registered task kind persisted in `TaskRecord.kind`.
    pub name: String,
    /// Definition version persisted with live input and checkpoints.
    pub version: u64,
    /// First durable checkpoint for a newly created task (`initial(input)`).
    pub initial: Arc<dyn Fn(&Value) -> Value + Send + Sync>,
    /// Exhaustive phase map keyed by the checkpoint's `phase` string.
    pub phases: Arc<BTreeMap<String, PhaseHandler>>,
    /// Runs in a fresh invocation after an abort mark and must commit a
    /// terminal outcome.
    pub abort: PhaseHandler,
    /// Convert a record stored by any older supported version; runs at
    /// reservation.
    pub migrate:
        Option<Arc<dyn Fn(&Value, &Value, u64) -> Result<TaskMigration, String> + Send + Sync>>,
    /// Hook handler table for this task's own hooks (upstream `hooks?: H`);
    /// extension hooks are registered separately.
    pub hooks: Option<Arc<dyn HookHandler>>,
}

impl std::fmt::Debug for TaskDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskDefinition")
            .field("name", &self.name)
            .field("version", &self.version)
            .field("phases", &self.phases.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

/// Typed executable task (upstream `Task`): `define_task` wraps a definition.
#[derive(Clone, Debug)]
pub struct Task {
    pub definition: Arc<TaskDefinition>,
}

/// Define an executable task. Register it in the registry so a Harness can
/// run tasks of its kind.
pub fn define_task(definition: TaskDefinition) -> Task {
    Task {
        definition: Arc::new(definition),
    }
}

/// Committed document watch returned by a task invocation (upstream `DocumentWatch`).
pub type DocumentWatch = crate::session::observation::DocumentWatch;

/// Callback of [`TaskRuntime::commit`]: runs inside the task's commit with
/// the transaction and the freshly reread running record; returns the next
/// state or `None` to leave it unchanged. Boxed-future form because `dyn
/// AsyncFnOnce` is not stable.
pub type CommitChange<'a> =
    Box<dyn FnOnce(&mut Tx<'_>, &RunningTask) -> Result<Option<NextTaskState>, String> + Send + 'a>;

/// Wrap a synchronous commit closure into a [`CommitChange`] (tasks D3: Tx
/// operations are sync on the Session line).
pub fn commit_change<'a>(
    f: impl FnOnce(&mut Tx<'_>, &RunningTask) -> Result<Option<NextTaskState>, String> + Send + 'a,
) -> CommitChange<'a> {
    Box::new(f)
}

/// Operations of one task invocation (erased upstream `TaskRuntime`). Every
/// operation rejects after the invocation ends; watches acquired through it
/// stop at invocation end. Built-in task helpers are conversation-scoped;
/// document-address methods expose the full reader surface to adapters.
#[async_trait::async_trait]
pub trait TaskRuntime: Send + Sync {
    /// The invoked task.
    fn task_id(&self) -> TaskId;
    /// The task's conversation.
    fn conversation_id(&self) -> ConversationId;
    /// Aborted when the run is signalled by `abort_task()`, the Harness
    /// closes, or the invocation ends.
    fn signal(&self) -> pi_chord::context::AbortSignal;
    /// Registry snapshot of the current phase; refreshed at every phase
    /// boundary.
    fn registry(&self) -> Arc<dyn RegistrySnapshot>;
    /// The task's conversation's agent, resolved at most once per phase, at
    /// first use, and fixed for the phase.
    async fn agent(&self, context: &Context) -> Result<crate::harness::types::Agent, String>;
    /// `HarnessOptions.settings`, resolved at each access.
    fn settings(&self) -> crate::harness::types::Settings;
    /// pi-ai model access.
    fn models(&self) -> Arc<dyn pi_ai::Models>;
    /// Calls `HarnessOptions.env` for the task's conversation; rejects with
    /// its error.
    async fn env(
        &self,
        context: &Context,
    ) -> Result<Option<Arc<dyn crate::env::ExecutionEnv>>, String>;
    /// Handlers of this task's name from the extensions its conversation
    /// selects, in extension order.
    fn hooks(&self) -> HookRunner;

    /// Commit on the Session line after rereading the task. Rejects when the
    /// task is terminal, the invocation ended, the Harness is closing, or, in
    /// a run invocation, the task carries an abort mark. A returned state
    /// replaces the task's state in the same commit; `None` leaves it
    /// unchanged. `Tx::create_task` defaults to the task's conversation. The
    /// change closure is synchronous on the line (D3).
    async fn commit(
        &self,
        change: CommitChange<'_>,
        context: &Context,
    ) -> Result<Option<NextTaskState>, String>;
    /// Commit a borrowed tool callback under this task's running-state guards.
    async fn commit_tool(
        &self,
        change: &mut crate::session::CommitValueCallback<'_>,
        context: &Context,
    ) -> Result<Value, String>;

    /// Read a durable memo of this task.
    async fn memo(&self, name: &str, context: &Context) -> Result<Option<Value>, String>;
    /// Store `candidate` unless a memo already exists; return the durable
    /// winner.
    async fn memo_put(
        &self,
        name: &str,
        candidate: Value,
        context: &Context,
    ) -> Result<Value, String>;
    /// Committed task record.
    async fn get_task(&self, id: TaskId, context: &Context) -> Result<Option<TaskRecord>, String>;
    /// Resolve with the task's terminal receipt; rejects when the invocation
    /// ends.
    async fn wait_for_task(&self, id: TaskId, context: &Context) -> Result<TaskRecord, String>;
    /// Outcomes of terminal tasks, in order; rejects when one is missing or
    /// not terminal. Used after a wait.
    async fn outcomes(&self, ids: &[TaskId], context: &Context)
        -> Result<Vec<TaskOutcome>, String>;
    /// Invocation-bound handle of an existing conversation, for example one
    /// this task owns; `None` when absent.
    async fn conversation(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Result<Option<Arc<dyn crate::session::ConversationHandle>>, String>;
    /// Committed entry visible from the task's conversation; `None` when
    /// absent, not visible, or of another kind than `kind` (upstream's two
    /// `entry` overloads collapse behind `kind`).
    async fn entry(
        &self,
        kind: Option<&str>,
        id: EntryId,
        context: &Context,
    ) -> Result<Option<EntryRecord>, String>;
    /// Committed raw active transcript and model context, optionally cut off
    /// at the visible entry `at`.
    async fn context_view(
        &self,
        conversation_id: ConversationId,
        context: &Context,
        at: Option<EntryId>,
    ) -> Result<crate::session::ContextView, String>;
    /// The Harness clock.
    fn now(&self) -> u64;
    /// Forward a non-fatal failure to `HarnessOptions.onReport`.
    fn report(&self, error: String);
    /// Resolve once the Harness clock reaches `until`; rejects when the
    /// invocation or `context` is cancelled.
    async fn sleep(&self, until: u64, context: &Context) -> Result<(), String>;

    /// Committed document state (upstream `runtime.snapshot`).
    async fn snapshot(
        &self,
        token: &crate::documents::DocToken,
        conversation_id: ConversationId,
        context: &Context,
    ) -> Result<Option<JsonObject>, String>;
    /// Committed document state as of an entry (upstream
    /// `runtime.snapshotAsOf`).
    async fn snapshot_as_of(
        &self,
        token: &crate::documents::DocToken,
        conversation_id: ConversationId,
        at: EntryId,
        context: &Context,
    ) -> Result<Option<JsonObject>, String>;
    /// Watch a conversation document (upstream `runtime.watchDoc`).
    async fn watch_doc(
        &self,
        token: &crate::documents::DocToken,
        conversation_id: ConversationId,
        context: &Context,
    ) -> Result<Option<DocumentWatch>, String>;
    /// Read a document by its full address (upstream `runtime.snapshot`).
    async fn document_snapshot(
        &self,
        token: &crate::documents::DocToken,
        owner: Option<u64>,
        key: Option<&str>,
        context: &Context,
    ) -> Result<Option<JsonObject>, String>;
    /// Read a conversation document at one visible entry (upstream
    /// `runtime.snapshotAsOf`).
    async fn document_snapshot_as_of(
        &self,
        token: &crate::documents::DocToken,
        conversation_id: ConversationId,
        key: Option<&str>,
        at: EntryId,
        context: &Context,
    ) -> Result<Option<JsonObject>, String>;
    /// Watch a document by its full address (upstream `runtime.watchDoc`).
    async fn document_watch_doc(
        &self,
        token: &crate::documents::DocToken,
        owner: Option<u64>,
        key: Option<&str>,
        context: &Context,
    ) -> Result<Option<DocumentWatch>, String>;
}
