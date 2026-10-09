//! Port of `packages/durable/src/harness/scheduler.ts` (the durable task
//! scheduler, spec §5).
//!
//! Deviations: D1 — `queueMicrotask` scheduling becomes `tokio::spawn`
//! Wake-driven passes (`#scheduleReconcile`/`#kick`); the serialize-on-the-
//! Session-line invariant is unchanged (decisions run inside commits).
//! D2 — `AbortController` per invocation becomes a chord `AbortSignal`
//! controller pair; `done` is a oneshot notify. D3 — `setTimeout` delays
//! become `tokio::time::sleep` (no 32-bit ms cap, so `MAX_TIMER_DELAY`
//! stepping is unnecessary). D4 — JSON equality for progress detection is
//! `serde_json` value equality (map order preserved by `preserve_order`,
//! matching upstream write order). D5 — runtime `now`/`report` surface
//! "invocation has ended" as a panic (their signatures are infallible;
//! upstream throws), and `step` reports a rejected commit itself and returns
//! `Ok(None)` where upstream catches and returns `undefined` — `#start`'s
//! catch never fires, so no second report.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use pi_chord::context::{AbortReason, AbortSignal, Context};
use parking_lot::Mutex;
use serde_json::Value;

use crate::harness::agent::agent_hooks;
use crate::harness::live::settle_scheduler_outcome;
use crate::harness::types::{
	Agent, AnyTask, ConversationHandle, ConversationId, EntryId, EntryRecord, HarnessInspection, RegistryReader,
	RegistrySnapshot, SettledTask, Settings, SchedulingState, TaskId, TaskInspection, TaskInspectionState,
	Unsubscribe,
};
use crate::harness::util::{abort_reason_text, closed_error, scan_all, Waiters};
use crate::session::{Session, Tx};
use crate::tasks::{BoxFuture, NextTaskState, TaskDefinition, TaskRuntime};
use crate::types::{
	CommitPublication, DocumentId, JoinPolicy, JsonObject, Page, Storage, SubmissionStatus, SubmissionType,
	TaskOutcome, TaskRecord, TaskState, TaskStateStatus,
};

const SCAN_PAGE_SIZE: u64 = 256;
const LIVE_STATUSES: [TaskStateStatus; 4] = [
	TaskStateStatus::Pending,
	TaskStateStatus::Running,
	TaskStateStatus::Waiting,
	TaskStateStatus::Completing,
];

/// Why a pending task cannot be reserved under a registry snapshot.
/// Derived, never persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockedReason {
	MissingTask,
	TaskTooOld,
	MigrationFailed,
}

impl BlockedReason {
	pub fn as_str(self) -> &'static str {
		match self {
			BlockedReason::MissingTask => "missing_task",
			BlockedReason::TaskTooOld => "task_too_old",
			BlockedReason::MigrationFailed => "migration_failed",
		}
	}
}

/// A record that can still run code: pending, running, or waiting.
fn is_runnable(record: &TaskRecord) -> bool {
	matches!(record.state.status, TaskStateStatus::Pending | TaskStateStatus::Running | TaskStateStatus::Waiting)
}

/// The immutable ownership fields of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TaskNode {
	conversation_id: ConversationId,
	owner: Option<TaskId>,
	background: bool,
}

/// Where a walk up the ownership tree continues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Up {
	Task(TaskId),
	Conversation(ConversationId),
}

/// One step of a walk up.
enum Step {
	Task(TaskId, TaskNode),
	Conversation(ConversationId),
	Unknown,
}

/// Candidate records a commit staged; they override committed records in
/// ownership walks.
struct Overlay {
	tasks: HashMap<TaskId, TaskRecord>,
	edges: HashMap<ConversationId, Option<TaskId>>,
}

/// Where ordinary ownership traversal starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
	Conversation(ConversationId),
	Roots,
}

/// Terminal outcomes the scheduler writes without running task code.
pub type SchedulerOutcome = TaskOutcome;

/// An invocation a conversation handle is bound to: its signal, and a check
/// that fails once it ended.
#[derive(Clone)]
pub struct InvocationBinding {
	pub signal: AbortSignal,
	check: Arc<dyn Fn() -> Result<(), String> + Send + Sync>,
}

impl InvocationBinding {
	pub fn check(&self) -> Result<(), String> {
		(self.check)()
	}

	/// The raw check closure, for handles that re-check per operation.
	pub fn check_fn(&self) -> Arc<dyn Fn() -> Result<(), String> + Send + Sync> {
		self.check.clone()
	}
}

/// One in-memory execution of a task in run or abort mode.
struct Invocation {
	task_id: TaskId,
	conversation_id: ConversationId,
	mode: Mode,
	ended: Arc<std::sync::atomic::AtomicBool>,
	/// Context passed to handlers; cancelled at invocation end.
	context: Context,
	/// Fired when the invocation ends.
	controller: pi_chord::context::AbortController,
	/// Watches the runtime acquired (upstream `invocation.watches`); stopped
	/// when the invocation ends, before the controller aborts.
	watches: Arc<parking_lot::Mutex<Vec<Arc<crate::tasks::DocumentWatch>>>>,
	done: tokio::sync::watch::Receiver<bool>,
	done_tx: tokio::sync::watch::Sender<bool>,
}

impl Invocation {
	fn signal(&self) -> AbortSignal {
		self.controller.signal()
	}

	fn abort_invocation(&self, reason: String) {
		self.controller.abort(pi_chord::context::AbortReason::Message(reason));
	}

	/// Track an acquired watch; dropped from the set once it terminates
	/// (upstream `watch.closed.then(() => invocation.watches.delete(watch))`).
	fn register_watch(self: &Arc<Self>, watch: crate::tasks::DocumentWatch) {
		let watch = Arc::new(watch);
		if self.ended.load(std::sync::atomic::Ordering::SeqCst) {
			watch.stop_detached();
			return;
		}
		self.watches.lock().push(watch.clone());
		let tracked = self.clone();
		tokio::spawn(async move {
			watch.closed().await;
			tracked.watches.lock().retain(|current| !Arc::ptr_eq(current, &watch));
		});
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
	Run,
	Abort,
}

/// What a runtime reads for the phase handler it serves.
struct PhaseState {
	snapshot: Mutex<Arc<dyn RegistrySnapshot>>,
	task: Mutex<AnyTask>,
	agent: Mutex<Option<Agent>>,
}

/// Reservation produced by one reserve pass.
struct Reservation {
	invocation: Arc<Invocation>,
	task: AnyTask,
	snapshot: Arc<dyn RegistrySnapshot>,
	record: TaskRecord,
}

/// Decision of the step callback.
enum Decision {
	Continue,
	End,
	Fault(String),
}

/// Options of [`TaskScheduler`].
pub struct TaskSchedulerOptions {
	pub session: Arc<Session>,
	pub storage: Arc<dyn Storage>,
	pub registry: Arc<dyn RegistryReader>,
	pub models: Arc<dyn pi_ai::Models>,
	/// Resolve a conversation's agent against a snapshot; the runtime calls
	/// it at most once per phase.
	pub agent: Box<dyn Fn(ConversationId, &dyn RegistrySnapshot, &Context) -> BoxFuture<Result<Agent, String>> + Send + Sync>,
	/// Resolve the settings; read at each access.
	pub settings: Box<dyn Fn() -> Settings + Send + Sync>,
	/// Build a conversation's environment with `HarnessOptions.env`.
	pub env: crate::harness::types::EnvBuilder,
	pub now: crate::harness::types::NowFn,
	pub report: crate::harness::types::ReportHook,
	/// Harness cleanup staged in the commit that makes an outcome the
	/// scheduler wrote itself terminal.
	pub settle_outcome: SettleOutcomeFn,
	/// Withdraw a conversation's queued inputs, for conversation abort and
	/// abort cascades.
	pub withdraw_inputs: WithdrawInputsFn,
	/// Invocation-bound handle of an existing conversation, for task
	/// runtimes and tools.
	pub conversation: ConversationResolverFn,
	/// Context for scheduler commits and invocations; carries no caller
	/// cancellation.
	pub context: Context,
}

pub type SettleOutcomeFn = Arc<dyn Fn(&Tx, &TaskRecord, &SchedulerOutcome) -> Result<(), String> + Send + Sync>;
pub type WithdrawInputsFn = Arc<dyn Fn(&Tx, ConversationId) -> Result<(), String> + Send + Sync>;
pub type ConversationResolverFn = Arc<
	dyn Fn(ConversationId, InvocationBinding, &Context) -> BoxFuture<Result<Option<Arc<dyn ConversationHandle>>, String>>
		+ Send
		+ Sync,
>;

/// Durable task scheduler of one Harness (upstream `TaskScheduler`).
pub struct TaskScheduler {
	session: Arc<Session>,
	storage: Arc<dyn Storage>,
	registry: Arc<dyn RegistryReader>,
	models: Arc<dyn pi_ai::Models>,
	agent: Box<dyn Fn(ConversationId, &dyn RegistrySnapshot, &Context) -> BoxFuture<Result<Agent, String>> + Send + Sync>,
	settings: Box<dyn Fn() -> Settings + Send + Sync>,
	env: crate::harness::types::EnvBuilder,
	now: crate::harness::types::NowFn,
	report: crate::harness::types::ReportHook,
	settle_outcome: SettleOutcomeFn,
	withdraw_inputs: WithdrawInputsFn,
	conversation: ConversationResolverFn,
	context: Context,

	inner: Mutex<SchedulerState>,
	wake: tokio::sync::Notify,
	/// The runtime the scheduler was opened on; publications that fire from
	/// the storage-bridge thread spawn scheduler work through this handle so
	/// they never land on (and deadlock against) the bridge's own runtime.
	runtime_handle: tokio::runtime::Handle,
}

struct SchedulerState {
	/// Every committed non-terminal task record.
	live: BTreeMap<TaskId, TaskRecord>,
	invocations: HashMap<TaskId, Arc<Invocation>>,
	task_waiters: Waiters<TaskId, SettledTask>,
	/// Idle waiters by conversation; `None` waits for the whole Harness.
	idle_waiters: Waiters<Option<ConversationId>, ()>,
	/// Definition whose migration failed per task; retried only once the
	/// registry resolves another definition.
	failed_migrations: HashMap<TaskId, AnyTask>,
	/// Owner task of each loaded conversation, `None` when ownerless.
	edges: HashMap<ConversationId, Option<TaskId>>,
	/// Tasks that own a loaded conversation; their nodes stay in `settled`
	/// once terminal.
	conversation_owners: BTreeSet<TaskId>,
	/// Ownership fields of terminal tasks that walks pass through.
	settled: HashMap<TaskId, TaskNode>,
	/// `failFast` waiters the next reconcile checks for a failed task in
	/// `on`.
	fail_fast_checks: BTreeSet<TaskId>,
	reconcile_scheduled: bool,
	cascade_pending: bool,
	_unsubscribe: Option<Unsubscribe>,
	enabled: bool,
	closing: bool,
	dirty: bool,
	draining: bool,
}

impl TaskScheduler {
	pub fn new(options: TaskSchedulerOptions) -> Arc<Self> {
		Arc::new(Self {
			session: options.session,
			storage: options.storage,
			registry: options.registry,
			models: options.models,
			agent: options.agent,
			settings: options.settings,
			env: options.env,
			now: options.now,
			report: options.report,
			settle_outcome: options.settle_outcome,
			withdraw_inputs: options.withdraw_inputs,
			conversation: options.conversation,
			context: options.context,
			runtime_handle: tokio::runtime::Handle::current(),
			inner: Mutex::new(SchedulerState {
				live: BTreeMap::new(),
				invocations: HashMap::new(),
				task_waiters: Waiters::default(),
				idle_waiters: Waiters::default(),
				failed_migrations: HashMap::new(),
				edges: HashMap::new(),
				conversation_owners: BTreeSet::new(),
				settled: HashMap::new(),
				fail_fast_checks: BTreeSet::new(),
				reconcile_scheduled: false,
				cascade_pending: false,
				_unsubscribe: None,
				enabled: false,
				closing: false,
				dirty: false,
				draining: false,
			}),
			wake: tokio::sync::Notify::new(),
		})
	}

	/// Load live tasks and change surviving `running` tasks back to
	/// `pending`. Dispatches nothing.
	pub async fn open(self: &Arc<Self>, context: &Context) -> Result<(), String> {
		let observer = self.clone();
		self.session.subscribe_commits(Arc::new(move |publication: &CommitPublication, _context: &Context| {
			observer.observe(publication);
		}));
		let closer = self.clone();
		self.session.subscribe_close(Arc::new(move || closer.seal()));
		let unsubscribe = self.registry.subscribe({
			let scheduler = self.clone();
			Arc::new(move || scheduler.kick())
		});
		self.inner.lock()._unsubscribe = Some(unsubscribe);
		let scheduler = self.clone();
		let scheduler_in = scheduler.clone();
		self.session
			.commit(
				move |tx| {
					let scheduler = scheduler_in;
					// Every table read before the first write: collect the
					// scans for all live statuses, then apply the resets.
					let mut scanned: Vec<TaskRecord> = Vec::new();
					for status in LIVE_STATUSES {
						let query = crate::types::TaskQuery {
							conversation_id: None,
							status: Some(status),
							..Default::default()
						};
						scanned.extend(scan_tasks_on_tx(tx, &query)?);
					}
					for record in scanned {
						let pending = record.state.status == TaskStateStatus::Running;
						scheduler.inner.lock().live.insert(record.id, record.clone());
						if pending {
							let mut reset = record.clone();
							reset.state = TaskState {
								status: TaskStateStatus::Pending,
								..record.state.clone()
							};
							tx.set_task(reset).map_err(|e| e.to_string())?;
						}
						if record.state.status == TaskStateStatus::Waiting
							&& record.state.policy == Some(JoinPolicy::FailFast)
						{
							scheduler.inner.lock().fail_fast_checks.insert(record.id);
						}
					}
					Ok(())
				},
				context,
			)
			.await
			.map_err(|e| e.to_string())?;
		// Derive abort marks a crash left unapplied below cancelled owners,
		// and finalize held outcomes.
		self.inner.lock().cascade_pending = true;
		self.schedule_reconcile();
		Ok(())
	}

	/// Enable scheduling. Idempotent; the kick does nothing once closing.
	pub fn resume(self: &Arc<Self>) {
		self.inner.lock().enabled = true;
		self.kick();
	}

	/// Wait for every invocation signalled by [`Self::seal`]. Writes
	/// nothing.
	pub async fn join(&self) {
		let dones: Vec<tokio::sync::watch::Receiver<bool>> = {
			let inner = self.inner.lock();
			inner.invocations.values().map(|invocation| invocation.done.clone()).collect()
		};
		for mut done in dones {
			while !*done.borrow_and_update() {
				if done.changed().await.is_err() {
					break;
				}
			}
		}
	}

	/// Commit the abort mark, or settle a task that no registered definition
	/// can take as `orphaned` when nothing it owns is live, then join the
	/// run invocation seen on the line; the commit listener signalled it.
	/// The abort invocation starts once the task's ordinary owned work is
	/// gone. A `completing` task is only marked.
	pub async fn abort(self: &Arc<Self>, id: TaskId, context: &Context) -> Result<&'static str, String> {
		let scheduler_in = self.clone();
		let id_in = id;
		let marked = self
			.session
			.commit(
				move |tx| {
					let scheduler = scheduler_in;
					let id = id_in;
					let current = tx.task(id).map_err(|e| e.to_string())?;
					let Some(current) = current else {
						return Err(format!("Task {id} does not exist"));
					};
					if current.state.status == TaskStateStatus::Terminal {
						return Ok(("terminal", None));
					}
					let invocation = scheduler.inner.lock().invocations.get(&id).cloned();
					if invocation.is_none() && current.state.status != TaskStateStatus::Completing {
						scheduler.load_scopes_on(tx, Vec::new(), None)?;
						if !scheduler.owned_live_on(tx).contains_key(&id) {
							let snapshot = scheduler.registry.snapshot();
							let resolution = scheduler.resolve(&current, snapshot.as_ref());
							if let Some(reason) = resolution.blocked() {
								scheduler.terminate_on(tx, &current, orphaned_outcome(reason))?;
								return Ok(("marked", None));
							}
						}
					}
					if !current.abort_requested {
						let mut marked = current.clone();
						marked.abort_requested = true;
						tx.set_task(marked).map_err(|e| e.to_string())?;
					}
					// Only the run invocation seen on the line is joined
					// (upstream `marked.run`); a live abort handler is never
					// waited for.
					let run_done = invocation
						.filter(|invocation| invocation.mode == Mode::Run)
						.map(|invocation| invocation.done.clone());
					Ok(("marked", run_done))
				},
				context,
			)
			.await
			.map_err(|e| e.to_string())?;
		// The commit listener signalled the run; join it, but release the
		// wait when the caller's context is cancelled (upstream
		// `awaitWithContext`).
		if let Some(mut done) = marked.1 {
			let join = async move {
				while !*done.borrow_and_update() {
					if done.changed().await.is_err() {
						break;
					}
				}
			};
			pi_chord::context::await_with_context(join, context).await.map_err(|error| error.to_string())?;
		}
		Ok(marked.0)
	}

	pub async fn wait_for_task(self: &Arc<Self>, id: TaskId, context: &Context) -> Result<SettledTask, String> {
		let waiter = {
			let inner = self.inner.lock();
			if inner.closing { return Err(closed_error()); }
			if inner.live.contains_key(&id) {
				Some(inner.task_waiters.clone().add(id, context))
			} else {
				None
			}
		};
		if let Some(waiter) = waiter { return waiter.await; }
		self.storage.task(id, context).await.map(SettledTask::new).ok_or_else(|| format!("Task {id} does not exist"))
	}

	/// Resolve when ordinary traversal from the conversation, or from every
	/// ownerless conversation, reaches no live non-background task.
	pub async fn wait_for_idle(
		self: &Arc<Self>,
		conversation_id: Option<ConversationId>,
		context: &Context,
	) -> Result<(), String> {
		if self.inner.lock().closing {
			return Err(closed_error());
		}
		if self.idle(conversation_id.clone()) {
			return Ok(());
		}
		self.schedule_reconcile();
		let waiter = { self.inner.lock().idle_waiters.clone().add(conversation_id, context) };
		waiter.await
	}

	/// `Conversation.abort()`: in one commit, withdraw the queued inputs and
	/// mark every live non-background task that ordinary traversal from the
	/// conversation reaches; resolves once the scope is idle. With
	/// `background`, traversal crosses background boundaries, and the wait
	/// also covers every task it reached.
	pub async fn abort_conversation(
		self: &Arc<Self>,
		conversation_id: ConversationId,
		background: bool,
		context: &Context,
	) -> Result<(), String> {
		let scheduler = self.clone();
		let queued = self.clone().queued_conversations().await;
		let reached = self
			.session
			.commit(
				move |tx| {
					let scheduler = scheduler;
					let background = background;
					let conversation_id = conversation_id;
					scheduler.load_scopes_on(tx, queued.clone(), None)?;
					let mut reached = Vec::new();
					for record in scheduler.live_snapshot() {
						if record.background && !background {
							continue;
						}
						if scheduler.in_scope(parent_of(&record), Scope::Conversation(conversation_id), background) != Some(true)
						{
							continue;
						}
						reached.push(record.id);
						if !record.abort_requested {
							let mut marked = record.clone();
							marked.abort_requested = true;
							tx.set_task(marked).map_err(|e| e.to_string())?;
						}
					}
					for id in queued {
						if scheduler.in_scope(Up::Conversation(id), Scope::Conversation(conversation_id), background)
							== Some(true)
						{
							(scheduler.withdraw_inputs)(tx, id)?;
						}
					}
					Ok(reached)
				},
				context,
			)
			.await
			.map_err(|e| e.to_string())?;
		if background {
			for id in reached {
				self.wait_for_task(id, context).await?;
			}
		}
		self.wait_for_idle(Some(conversation_id), context).await
	}

	fn live_snapshot(&self) -> Vec<TaskRecord> {
		self.inner.lock().live.values().cloned().collect()
	}

	// ─── Scheduling ──────────────────────────────────────────────────────

	fn observe(self: &Arc<Self>, publication: &CommitPublication) {
		let mut updated: Vec<TaskRecord> = Vec::new();
		let mut failed: Vec<TaskId> = Vec::new();
		let mut changed = false;
		{
			let mut inner = self.inner.lock();
			for change in &publication.changes {
				let crate::types::CommitChange::Table(crate::types::TableCommitChange::Task { value: record }) = change
				else {
					continue;
				};
				changed = true;
				let previous = inner.live.get(&record.id).cloned();
				if failed_outcome(record)
					&& previous.as_ref().map(|previous| !failed_outcome(previous)).unwrap_or(true)
				{
					failed.push(record.id);
				}
				if record.state.status == TaskStateStatus::Terminal {
					inner.live.remove(&record.id);
					inner.failed_migrations.remove(&record.id);
					inner.fail_fast_checks.remove(&record.id);
					if inner.conversation_owners.contains(&record.id) {
						inner.settled.insert(record.id, node_of(record));
					}
					inner.task_waiters.resolve(&record.id, SettledTask::new(record.clone()));
					// Its owner may finalize now.
					drop(inner);
					self.schedule_reconcile();
					inner = self.inner.lock();
					continue;
				}
				if record.abort_requested && previous.as_ref().map(|previous| previous.abort_requested) != Some(true) {
					inner.cascade_pending = true;
					// live must be updated before the wake: the signalled
					// step reads `inner.live` and must see the mark, or
					// decide rule 6 miswrites faulted→terminal (upstream's
					// synchronous listener ordering guarantees this).
					inner.live.insert(record.id, record.clone());
					// Signal a run invocation of the newly marked task; its
					// next step ends it.
					if let Some(invocation) = inner.invocations.get(&record.id) {
						if invocation.mode == Mode::Run {
							invocation.abort_invocation("abort mark".to_string());
						}
					}
				}
				let status = record.state.status;
				if status == TaskStateStatus::Completing
					&& previous.as_ref().map(|previous| previous.state.status) != Some(TaskStateStatus::Completing)
				{
					if cancellation_intent(record) {
						inner.cascade_pending = true;
					}
					drop(inner);
					self.schedule_reconcile();
					inner = self.inner.lock();
				}
				if status == TaskStateStatus::Waiting
					&& record.state.policy == Some(JoinPolicy::FailFast)
					&& previous.as_ref().map(|previous| previous.state.status) != Some(TaskStateStatus::Waiting)
				{
					inner.fail_fast_checks.insert(record.id);
					drop(inner);
					self.schedule_reconcile();
					inner = self.inner.lock();
				}
				inner.live.insert(record.id, record.clone());
				updated.push(record.clone());
			}
			for id in &failed {
				let live_records: Vec<TaskRecord> = inner.live.values().cloned().collect();
				for record in live_records {
					let state = &record.state;
					if state.status == TaskStateStatus::Waiting
						&& state.policy == Some(JoinPolicy::FailFast)
						&& state.on.as_ref().is_some_and(|on| on.contains(id))
					{
						// Upstream adds the WAITER's id (record.id), not the
						// failed task's: the reconcile re-checks the waiter
						// when its dependency's failure lands.
						inner.fail_fast_checks.insert(record.id);
						drop(inner);
						self.schedule_reconcile();
						inner = self.inner.lock();
					}
				}
			}
			for change in &publication.changes {
				if let crate::types::CommitChange::Table(crate::types::TableCommitChange::Conversation { value }) = change
				{
					if !inner.edges.contains_key(&value.id) {
						let owner = value.owner.as_ref().map(|owner| owner.task_id);
						set_edge(&mut inner, value.id, owner);
					}
				}
			}
			for change in &publication.changes {
				// A queued input below a cancelled owner is withdrawn, even
				// after its cascade.
				let crate::types::CommitChange::Table(crate::types::TableCommitChange::Submission { value }) = change
				else {
					continue;
				};
				if value.status != SubmissionStatus::Queued || value.submission_type != SubmissionType::Input
				{
					continue;
				}
				let up = Up::Conversation(value.conversation_id);
				if !chain_known(&*inner, up, None) || below_cancelled(&*inner, up, None) {
					inner.cascade_pending = true;
				}
			}
			for record in &updated {
				// Work created below a cancelled owner, even after its
				// cascade, is aborted too.
				if !chain_known(&*inner, parent_of(record), None) {
					drop(inner);
					self.schedule_reconcile();
					inner = self.inner.lock();
				} else if !record.background
					&& !record.abort_requested
					&& below_cancelled(&*inner, parent_of(record), None)
				{
					inner.cascade_pending = true;
				}
			}
			// Also retries, with the next commit of any kind, a cascade whose
			// commit failed.
			if inner.cascade_pending {
				drop(inner);
				self.schedule_reconcile();
				inner = self.inner.lock();
			}
		}
		if !changed {
			return;
		}
		self.resolve_idle_waiters();
		self.kick();
	}

	fn resolve_idle_waiters(self: &Arc<Self>) {
		let keys: Vec<Option<ConversationId>> = self.inner.lock().idle_waiters.keys();
		for conversation_id in keys {
			if self.idle(conversation_id.clone()) {
				self.inner.lock().idle_waiters.resolve(&conversation_id, ());
			}
		}
	}

	// ─── Ownership ───────────────────────────────────────────────────────

	fn schedule_reconcile(self: &Arc<Self>) {
		{
			let mut inner = self.inner.lock();
			if inner.reconcile_scheduled || inner.closing {
				return;
			}
			inner.reconcile_scheduled = true;
		}
		let scheduler = self.clone();
		self.runtime_handle.spawn(async move {
			tokio::task::yield_now().await;
			scheduler.reconcile().await;
		});
	}

	/// One commit that applies what committed records imply: abort marks
	/// below live owners with cancellation intent (spec §5.4), `failFast`
	/// marks (spec §5.5), withdrawn queued inputs below cancelled owners,
	/// and the final terminal record of every `completing` task whose
	/// ordinary owned work is gone. The durable records are the intent, so
	/// this also repairs whatever a crash left unapplied. Resolves idle
	/// waiters that the loaded edges decide.
	async fn reconcile(self: &Arc<Self>) {
		let (cascade, checks) = {
			let mut inner = self.inner.lock();
			inner.reconcile_scheduled = false;
			let cascade = inner.cascade_pending;
			inner.cascade_pending = false;
			(cascade, std::mem::take(&mut inner.fail_fast_checks))
		};
		let scheduler = self.clone();
		let checks2 = checks.clone();
		let cascade_queued = if cascade { self.clone().queued_conversations().await } else { Vec::new() };
		let result = self
			.session
			.commit(
				move |tx| {
					let scheduler = scheduler;
					let queued = cascade_queued.clone();
					if scheduler.inner.lock().closing {
						return Ok(());
					}
					scheduler.load_scopes_on(tx, if cascade { cascade_queued.clone() } else { Vec::new() }, None)?;
					let mut marked: BTreeSet<TaskId> = BTreeSet::new();
					for record in scheduler.live_snapshot() {
						if !record.background && below_cancelled(&*scheduler.inner.lock(), parent_of(&record), None) {
							if record.abort_requested || !marked.insert(record.id) {
								continue;
							}
							let mut m = record.clone();
							m.abort_requested = true;
							tx.set_task(m).map_err(|e| e.to_string())?;
						}
					}
					for id in checks2 {
						let waiter = scheduler.inner.lock().live.get(&id).cloned();
						let Some(waiter) = waiter else { continue };
						if waiter.state.status != TaskStateStatus::Waiting
							|| !scheduler.any_failed_on(tx, waiter.state.on.clone().unwrap_or_default())?
						{
							continue;
						}
						// Every other live task: the failed one keeps its own
						// outcome.
						for member in waiter.state.on.clone().unwrap_or_default() {
							let record = scheduler.inner.lock().live.get(&member).cloned();
							if let Some(record) = record {
								if !failed_outcome(&record) && marked.insert(record.id) {
									let mut m = record.clone();
									m.abort_requested = true;
									tx.set_task(m).map_err(|e| e.to_string())?;
								}
							}
						}
					}
					for id in queued {
						if below_cancelled(&*scheduler.inner.lock(), Up::Conversation(id), None) {
							(scheduler.withdraw_inputs)(tx, id)?;
						}
					}
					scheduler.finalize_on(tx)?;
					Ok(())
				},
				&self.context,
			)
			.await;
		if let Err(error) = result {
			// Any pass may have staged marks, so a failed one is retried with
			// the next commit.
			let mut inner = self.inner.lock();
			inner.cascade_pending = true;
			inner.fail_fast_checks.extend(checks);
			if !inner.closing {
				drop(inner);
				(self.report)(error.to_string());
			}
		}
		self.resolve_idle_waiters();
	}

	/// Whether any of `ids` holds or ended with an outcome other than
	/// `completed`.
	fn any_failed_on(&self, tx: &Tx<'_>, ids: Vec<TaskId>) -> Result<bool, String> {
		for id in ids {
			let record = self
				.inner
				.lock()
				.live
				.get(&id)
				.cloned()
				.or_else(|| tx.task(id).map_err(|e| e.to_string()).ok().flatten());
			if record.as_ref().is_some_and(|record| failed_outcome(record)) {
				return Ok(true);
			}
		}
		Ok(false)
	}

	/// Write the terminal record of every `completing` task without live
	/// ordinary owned work. Finalizing one can free its owner, so this
	/// repeats over the commit's candidates until nothing changes. A held
	/// scheduler outcome gets its Harness cleanup here.
	fn finalize_on(&self, tx: &Tx<'_>) -> Result<(), String> {
		loop {
			let overlay = overlay_of(tx);
			let owned = self.owned_live_overlay(&overlay);
			let done: Vec<TaskRecord> = self
				.live_records_overlay(&overlay)
				.into_iter()
				.filter(|record| record.state.status == TaskStateStatus::Completing && !owned.contains_key(&record.id))
				.collect();
			if done.is_empty() {
				return Ok(());
			}
			for record in done {
				let Some(outcome) = record.state.outcome.clone() else { continue };
				tx.set_task(with_state(&record, TaskState {
					status: TaskStateStatus::Terminal,
					outcome: Some(outcome.clone()),
					..record.state.clone()
				}))
				.map_err(|e| e.to_string())?;
				// Only the scheduler writes `faulted` and `orphaned`
				// (spec §5.4); their cleanup waits for this commit. The
				// settle hook is sync on the Session line (tasks D3).
				if matches!(outcome.status, crate::types::TaskOutcomeStatus::Faulted | crate::types::TaskOutcomeStatus::Orphaned)
				{
					(self.settle_outcome)(tx, &record, &outcome)?;
				}
			}
		}
	}

	/// Load the owner chains of every live task and, with `queued`, the
	/// pre-captured queued conversations, on the Session line; returns the
	/// queued list handed in (D5: captured by an async storage scan before
	/// the commit, since the erased `Tx` has no submission scan). With the
	/// commit's overlay (upstream `#commitState`'s terminal branch), also
	/// loads the candidates' chains against the overlay, so same-commit work
	/// counts for the owned-live decision.
	fn load_scopes_on(
		&self,
		tx: &Tx<'_>,
		queued: Vec<ConversationId>,
		overlay: Option<&Overlay>,
	) -> Result<Vec<ConversationId>, String> {
		for record in self.live_snapshot() {
			if !chain_known(&*self.inner.lock(), parent_of(&record), None) {
				self.load_chain_on(tx, parent_of(&record), None)?;
			}
		}
		if let Some(overlay) = overlay {
			for record in overlay.tasks.values() {
				if !chain_known(&*self.inner.lock(), parent_of(record), Some(overlay)) {
					self.load_chain_on(tx, parent_of(record), Some(overlay))?;
				}
			}
		}
		for id in &queued {
			self.load_chain_on(tx, Up::Conversation(*id), None)?;
		}
		Ok(queued)
	}

	/// Conversations with queued submissions (async storage scan; call
	/// before the commit that needs the scope).
	async fn queued_conversations(self: &Arc<Self>) -> Vec<ConversationId> {
		let submissions = {
			let context = self.context.clone();
			scan_all(|cursor| {
				let storage = self.storage.clone();
				let context = context.clone();
				let query = crate::types::SubmissionQuery {
					conversation_id: None,
					status: Some(SubmissionStatus::Queued),
				};
				async move {
					storage
						.scan_submissions(&query, SCAN_PAGE_SIZE, cursor.as_ref(), &context)
						.await
				}
			})
			.await
		};
		let mut conversations: Vec<ConversationId> = Vec::new();
		for record in submissions {
			if !conversations.contains(&record.conversation_id) {
				conversations.push(record.conversation_id);
			}
		}
		conversations
	}

	/// Load the owner edges and task nodes from `start` up to its ownerless
	/// root.
	fn load_chain_on(&self, tx: &Tx<'_>, start: Up, overlay: Option<&Overlay>) -> Result<(), String> {
		let mut at = Some(start);
		while let Some(current) = at {
			match current {
				Up::Task(id) => {
					let node = self.node(id, overlay);
					let node = match node {
						Some(node) => node,
						None => {
							let record = tx.task(id).map_err(|e| e.to_string())?;
							let Some(record) = record else { return Ok(()) };
							let node = node_of(&record);
							if record.state.status == TaskStateStatus::Terminal {
								self.inner.lock().settled.insert(record.id, node);
							}
							node
						}
					};
					at = Some(parent_of_node(&node));
				}
				Up::Conversation(id) => {
					let edge = self.edge(id, overlay);
					let edge = match edge {
						Some(edge) => edge,
						None => {
							let record = tx.conversation(id).map_err(|e| e.to_string())?;
							let edge = record.and_then(|record| record.owner.as_ref().map(|owner| owner.task_id));
							let mut inner = self.inner.lock();
							set_edge(&mut inner, id, edge);
							edge
						}
					};
					at = edge.map(Up::Task);
				}
			}
		}
		Ok(())
	}

	fn edge(&self, id: ConversationId, overlay: Option<&Overlay>) -> Option<Option<TaskId>> {
		match overlay {
			Some(overlay) if overlay.edges.contains_key(&id) => Some(overlay.edges[&id]),
			_ => self.inner.lock().edges.get(&id).copied(),
		}
	}

	fn node(&self, id: TaskId, overlay: Option<&Overlay>) -> Option<TaskNode> {
		if let Some(overlay) = overlay {
			if let Some(record) = overlay.tasks.get(&id) {
				return Some(node_of(record));
			}
		}
		let inner = self.inner.lock();
		inner
			.live
			.get(&id)
			.map(|record| node_of(record))
			.or_else(|| inner.settled.get(&id).copied())
	}

	/// Live records, with the overlay's candidates replacing committed ones;
	/// terminal candidates are gone.
	fn live_records_overlay(&self, overlay: &Overlay) -> Vec<TaskRecord> {
		let mut records: Vec<TaskRecord> = Vec::new();
		for record in self.inner.lock().live.values() {
			let candidate = overlay.tasks.get(&record.id).cloned().unwrap_or_else(|| record.clone());
			if candidate.state.status != TaskStateStatus::Terminal {
				records.push(candidate);
			}
		}
		for record in overlay.tasks.values() {
			if !self.inner.lock().live.contains_key(&record.id) && record.state.status != TaskStateStatus::Terminal {
				records.push(record.clone());
			}
		}
		records
	}

	/// Every task with live ordinary owned work (spec §5.5), mapped to that
	/// work: each live non-background task counts for every owner task above
	/// it up to and including the first background one. Owner chains must be
	/// loaded.
	fn owned_live_on(&self, tx: &Tx<'_>) -> HashMap<TaskId, Vec<TaskId>> {
		self.owned_live_overlay(&overlay_of(tx))
	}

	fn owned_live_overlay(&self, overlay: &Overlay) -> HashMap<TaskId, Vec<TaskId>> {
		let mut owned: HashMap<TaskId, Vec<TaskId>> = HashMap::new();
		for record in self.live_records_overlay(overlay) {
			if record.background {
				continue;
			}
			for step in self.above(parent_of(&record), Some(overlay)) {
				let step = match step {
					Step::Unknown => break,
					Step::Conversation(_) => continue,
					Step::Task(id, node) => (id, node),
				};
				owned.entry(step.0).or_default().push(record.id);
				if step.1.background {
					break;
				}
			}
		}
		owned
	}

	/// Walk up from `start`: owner tasks and conversations, ending at an
	/// ownerless root or an edge not loaded yet.
	fn above(&self, start: Up, overlay: Option<&Overlay>) -> Vec<Step> {
		let mut steps = Vec::new();
		let mut at = Some(start);
		while let Some(current) = at {
			match current {
				Up::Task(id) => match self.node(id, overlay) {
					None => {
						steps.push(Step::Unknown);
						return steps;
					}
					Some(node) => {
						steps.push(Step::Task(id, node));
						at = Some(parent_of_node(&node));
					}
				},
				Up::Conversation(id) => {
					steps.push(Step::Conversation(id));
					match self.edge(id, overlay) {
						None => {
							steps.push(Step::Unknown);
							return steps;
						}
						Some(None) => return steps,
						Some(Some(owner)) => at = Some(Up::Task(owner)),
					}
				}
			}
		}
		steps
	}

	/// Whether ordinary traversal from `scope` reaches `start`: walking up
	/// reaches the scope's conversation, or an ownerless one for `Roots`,
	/// without crossing a background owner, unless `cross_background`.
	/// `None` while an edge is not loaded.
	fn in_scope(&self, start: Up, scope: Scope, cross_background: bool) -> Option<bool> {
		for step in self.above(start, None) {
			match step {
				Step::Unknown => return None,
				Step::Conversation(conversation) => {
					if let Scope::Conversation(scope_conversation) = scope {
						if conversation == scope_conversation {
							return Some(true);
						}
					}
				}
				Step::Task(_, node) => {
					if node.background && !cross_background {
						return Some(false);
					}
				}
			}
		}
		Some(scope == Scope::Roots)
	}

	/// Whether a live owner's cancellation intent reaches `start`: walking
	/// up finds an owner with intent before a background owner without it.
	/// Terminal owners never cascade (spec §5.4).
	fn below_cancelled(&self, start: Up) -> bool {
		below_cancelled(&*self.inner.lock(), start, None)
	}

	/// Close listener: runs synchronously once admission is sealed, before
	/// `join`.
	fn seal(self: &Arc<Self>) {
		let error = closed_error();
		let invocations: Vec<Arc<Invocation>>;
		{
			let mut inner = self.inner.lock();
			inner.closing = true;
			// Upstream `#unsubscribeRegistry()` invokes the disposer; dropping
			// a `Box<dyn FnOnce>` would leak the subscription.
			if let Some(unsubscribe) = inner._unsubscribe.take() {
				unsubscribe();
			}
			inner.task_waiters.reject_all(&error);
			inner.idle_waiters.reject_all(&error);
			invocations = inner.invocations.values().cloned().collect();
		}
		for invocation in invocations {
			invocation.abort_invocation(ended_message(invocation.task_id));
		}
	}

	fn kick(self: &Arc<Self>) {
		{
			let mut inner = self.inner.lock();
			inner.dirty = true;
			if inner.draining || !inner.enabled || inner.closing {
				return;
			}
			inner.draining = true;
		}
		let scheduler = self.clone();
		self.runtime_handle.spawn(async move {
			scheduler.drain().await;
		});
	}

	async fn drain(self: &Arc<Self>) {
		let result: Result<(), String> = async {
			// Like `schedule_reconcile`, never reserve synchronously after a
			// kick: a task end that resolves a joiner (upstream's
			// `invocation.finish()` microtask ordering) must let that joiner's
			// next commit land before a reservation reads the line.
			tokio::task::yield_now().await;
			loop {
				let proceed = {
					let mut inner = self.inner.lock();
					if !(inner.dirty && inner.enabled && !inner.closing) {
						break;
					}
					inner.dirty = false;
					true
				};
				if !proceed {
					break;
				}
				for reservation in self.reserve().await? {
					self.start(reservation);
				}
			}
			Ok(())
		}
		.await;
		if let Err(error) = result {
			if !self.inner.lock().closing {
				(self.report)(error);
			}
		}
		{
			let mut inner = self.inner.lock();
			inner.draining = false;
			// A wakeup that arrived during a failed pass still needs its
			// pass.
			let dirty = inner.dirty;
			if dirty {
				drop(inner);
				self.kick();
			}
		}
	}

	/// Reserve every eligible task in one commit; orphan abort-marked tasks
	/// no definition can take.
	async fn reserve(self: &Arc<Self>) -> Result<Vec<Reservation>, String> {
		let reservations: Arc<parking_lot::Mutex<Vec<Reservation>>> = Arc::default();
		let reservations_in = reservations.clone();
		let scheduler = self.clone();
		let reserve_result = self
			.session
			.commit(
				move |tx| {
					let scheduler = scheduler;
					let reservations = reservations_in;
					let enabled = {
						let inner = scheduler.inner.lock();
						inner.enabled && !inner.closing
					};
					if !enabled {
						return Ok(());
					}
					scheduler.load_scopes_on(tx, Vec::new(), None)?;
					#[cfg(test)]
					for record in scheduler.live_snapshot() {
						if record.kind == "pi.compaction" {
							eprintln!("[reserve-live] task={:?} status={:?}", record.id, record.state.status);
						}
					}
					let owned = scheduler.owned_live_on(tx);
					// Taken once per pass, and only when some task is a
					// candidate.
					let mut snapshot: Option<Arc<dyn RegistrySnapshot>> = None;
					for record in scheduler.live_snapshot() {
						let blocked = {
							let inner = scheduler.inner.lock();
							inner.invocations.contains_key(&record.id)
								|| !waiting_on_with(&record, &owned, &inner.live).is_empty()
						};
						if blocked {
							continue;
						}
						if record.state.status == TaskStateStatus::Completing {
							continue;
						}
						let mode = if record.abort_requested { Mode::Abort } else { Mode::Run };
						let taken = snapshot.get_or_insert_with(|| scheduler.registry.snapshot());
						let resolution = scheduler.resolve(&record, taken.as_ref());
						if let Some(reason) = resolution.blocked() {
							if mode == Mode::Abort {
								scheduler.terminate_on(
									tx,
									&record,
									orphaned_outcome(reason),
								)?;
							}
							continue;
						}
						let (task, resolved_record) = resolution.ready().expect("not blocked");
						if resolved_record.state.status != TaskStateStatus::Running || resolved_record != record {
							tx.set_task(with_state(&resolved_record, TaskState {
								status: TaskStateStatus::Running,
								..resolved_record.state.clone()
							}))
							.map_err(|e| e.to_string())?;
						}
						// Registered on the line, so marks and later
						let invocation = scheduler.create_invocation(&record, mode);
						reservations.lock().push(Reservation {
							invocation,
							task,
							snapshot: taken.clone(),
							record: resolved_record,
						});
					}
					Ok(())
				},
				&self.context,
			)
			.await;
		match reserve_result {
			Ok(()) => {
				let drained: Vec<Reservation> = std::mem::take(&mut *reservations.lock());
				Ok(drained)
			}
			Err(error) => {
				for reservation in reservations.lock().iter() {
					self.end(&reservation.invocation);
				}
				Err(error.to_string())
			}
		}
	}

	/// Resolve the record's definition by kind, migrating an older stored
	/// version.
	fn resolve(&self, record: &TaskRecord, snapshot: &dyn RegistrySnapshot) -> Resolution {
		let task = snapshot.task(&record.kind);
		let fit = self.fit(record, task);
		match fit {
			Fit::Blocked { reason } => Resolution::Blocked(reason),
			Fit::Ready { task, migrates } => {
				if !migrates {
					return Resolution::Ready(task, record.clone());
				}
				let definition = &task.definition;
				let Some(migrate) = &definition.migrate else {
					let error = missing_migration(record, definition);
					(self.report)(error.clone());
					return Resolution::Blocked(BlockedReason::MigrationFailed);
				};
				match migrate(
					&record.input,
					record.state.checkpoint.as_ref().unwrap_or(&Value::Null),
					record.version,
				) {
					Ok(migrated) => {
						let mut migrated_record = record.clone();
						migrated_record.version = definition.version;
						migrated_record.input = migrated.input;
						migrated_record.state.checkpoint = Some(migrated.checkpoint);
						Resolution::Ready(task, migrated_record)
					}
					Err(error) => {
						self.inner.lock().failed_migrations.insert(record.id, task.clone());
						(self.report)(error);
						Resolution::Blocked(BlockedReason::MigrationFailed)
					}
				}
			}
		}
	}

	fn fit(&self, record: &TaskRecord, task: Option<AnyTask>) -> Fit {
		let Some(task) = task else { return Fit::Blocked { reason: BlockedReason::MissingTask } };
		let version = task.definition.version;
		if version == record.version {
			return Fit::Ready { task, migrates: false };
		}
		if version < record.version {
			return Fit::Blocked { reason: BlockedReason::TaskTooOld };
		}
		// Upstream: a migration failure blocks only that definition
		// (`failed?.task === task`); a changed definition retries.
		let failed_same = self
			.inner
			.lock()
			.failed_migrations
			.get(&record.id)
			.map(|failed| Arc::ptr_eq(&failed.definition, &task.definition))
			.unwrap_or(false);
		if failed_same {
			return Fit::Blocked { reason: BlockedReason::MigrationFailed };
		}
		Fit::Ready { task, migrates: true }
	}

	/// [`Self::inspect`] against the scheduler's own registry snapshot.
	pub async fn inspect_snapshot(self: &Arc<Self>) -> Result<(SchedulingState, Vec<TaskInspection>), String> {
		let snapshot = self.registry.snapshot();
		self.inspect(snapshot).await
	}

	/// Scheduling state and every live task with its derived state, read on
	/// the Session line. Runs no task code: a pending migration shows as
	/// `ready` with `migrates`, and only a migration the scheduler already
	/// tried, or one that cannot exist, shows as failed.
	pub async fn inspect(
		self: &Arc<Self>,
		snapshot: Arc<dyn RegistrySnapshot>,
	) -> Result<(SchedulingState, Vec<TaskInspection>), String> {
		{
			let scheduler = self.clone();
			self.session
				.read_on_line(async move { scheduler.load_scopesstorage().await })
				.await?;
		}
		let owned = self.owned_live_overlay(&Overlay { tasks: HashMap::new(), edges: HashMap::new() });
		let mut tasks: Vec<TaskInspection> = Vec::new();
		for record in self.live_snapshot() {
			let state = self.inspect_task(&record, snapshot.as_ref(), &owned);
			tasks.push(TaskInspection { record, state });
		}
		let scheduling = {
			let inner = self.inner.lock();
			if inner.closing {
				SchedulingState::Closing
			} else if inner.enabled {
				SchedulingState::Running
			} else {
				SchedulingState::Paused
			}
		};
		Ok((scheduling, tasks))
	}

	async fn load_scopesstorage(&self) -> Result<(), String> {
		// The scheduler's chain loads are tx-scoped; on the read path only
		// loaded edges are consulted.
		Ok(())
	}

	fn inspect_task(
		&self,
		record: &TaskRecord,
		snapshot: &dyn RegistrySnapshot,
		owned: &HashMap<TaskId, Vec<TaskId>>,
	) -> TaskInspectionState {
		if self.inner.lock().invocations.contains_key(&record.id) {
			return TaskInspectionState::Running;
		}
		if record.state.status == TaskStateStatus::Completing {
			return TaskInspectionState::Completing;
		}
		let on = waiting_on_with(record, owned, &self.inner.lock().live);
		if !on.is_empty() {
			return TaskInspectionState::Waiting { on };
		}
		match self.fit(record, snapshot.task(&record.kind)) {
			Fit::Blocked { reason } => TaskInspectionState::Blocked {
				reason: match reason {
					BlockedReason::MissingTask => crate::harness::types::BlockedReason::MissingTask,
					BlockedReason::TaskTooOld => crate::harness::types::BlockedReason::TaskTooOld,
					BlockedReason::MigrationFailed => crate::harness::types::BlockedReason::MigrationFailed,
				},
				error: None,
			},
			Fit::Ready { task, migrates } => {
				if migrates && task.definition.migrate.is_none() {
					return TaskInspectionState::Blocked { reason: crate::harness::types::BlockedReason::MigrationFailed, error: None };
				}
				TaskInspectionState::Ready { migrates }
			}
		}
	}

	fn create_invocation(self: &Arc<Self>, record: &TaskRecord, mode: Mode) -> Arc<Invocation> {
		let controller = pi_chord::context::AbortController::new();
		let signal = controller.signal();
		let (done_tx, done) = tokio::sync::watch::channel(false);
		let base = pi_chord::context::without_abort_signal(self.context.clone());
		let invocation = Arc::new(Invocation {
			task_id: record.id,
			conversation_id: record.conversation_id,
			mode,
			ended: Arc::new(std::sync::atomic::AtomicBool::new(false)),
			context: pi_chord::context::with_abort_signal(signal, base.clone()),
			controller,
			watches: Arc::default(),
			done,
			done_tx,
		});
		self.inner.lock().invocations.insert(record.id, invocation.clone());
		invocation
	}

	fn start(self: &Arc<Self>, reservation: Reservation) {
		let scheduler = self.clone();
		self.runtime_handle.spawn(async move {
			let invocation = reservation.invocation.clone();
			let result: Result<(), String> = async {
				match invocation.mode {
					Mode::Run => scheduler.run(reservation).await,
					Mode::Abort => scheduler.run_abort(reservation).await,
				}
			}
			.await;
			if let Err(error) = result {
				(scheduler.report)(error);
			}
			scheduler.end(&invocation);
			let _ = invocation.done_tx.send(true);
			scheduler.kick();
		});
	}

	/// Run phase handlers, each preceded by a step that decides on the line
	/// whether the invocation continues.
	async fn run(self: &Arc<Self>, reservation: Reservation) -> Result<(), String> {
		let invocation = reservation.invocation.clone();
		let state = Arc::new(Mutex::new(RunState {
			task: reservation.task.clone(),
			snapshot: reservation.snapshot.clone(),
			reported: None,
			agent: Arc::new(tokio::sync::OnceCell::new()),
		}));
		let runtime = self.runtime(&invocation, state.clone());
		let mut previous: Option<PhaseResult> = None;
		loop {
			let scheduler = self.clone();
			let state_ref = state.clone();
			let previous_ref = previous.clone();
			let current = self
				.step(&invocation, move |tx, current| {
					scheduler.decide(tx, current, previous_ref.clone(), &state_ref)
				})
				.await?;
			// Close may seal between the decision and dispatch.
			let Some(current) = current else { return Ok(()) };
			if self.inner.lock().closing {
				return Ok(());
			}
			let checkpoint = current.state.checkpoint.clone().unwrap_or(Value::Null);
			let phase = checkpoint
				.get("phase")
				.and_then(|phase| phase.as_str())
				.unwrap_or_default()
				.to_string();
			// Each phase handler resolves its agent afresh, at first use;
			// the runtime resolves on demand below.
			let handler = state.lock().task.definition.phases.get(&phase).cloned();
			let result = match handler {
				None => Err(format!("Task {} has no phase {phase}", current.kind)),
				Some(handler) => handler(current.clone(), runtime.clone(), invocation.context.clone()).await,
			};
			previous = Some(match result {
				Ok(()) => PhaseResult { checkpoint, failure: None },
				Err(error) => PhaseResult { checkpoint, failure: Some(error) },
			});
		}
	}

	/// Precedence rules for a run invocation, on the line. Rules 1 (terminal,
	/// `completing`, or `waiting`) and 2 (closing) are applied by `step`.
	/// Returns whether the invocation continues with the next phase.
	fn decide(
		&self,
		tx: &Tx<'_>,
		current: &TaskRecord,
		previous: Option<PhaseResult>,
		state: &Arc<Mutex<RunState>>,
	) -> Decision {
		// 3. abort mark: end; a fresh abort invocation starts once the task's
		// ordinary owned work is gone.
		if current.abort_requested {
			return Decision::End;
		}
		let Some(previous) = previous.clone() else { return Decision::Continue };
		// 4. uncaught error.
		if let Some(failure) = previous.failure {
			return Decision::Fault(failure);
		}
		// 6. no durable progress.
		if current.state.checkpoint.as_ref() == Some(&previous.checkpoint) {
			let phase = previous
				.checkpoint
				.get("phase")
				.and_then(|phase| phase.as_str())
				.unwrap_or_default();
			let message = format!(
				"Task {} phase {phase} returned without durable progress",
				current.kind
			);
			return Decision::Fault(message);
		}
		// 5. progress: refresh the snapshot; hand over to a replacement
		// definition that can take the task.
		let snapshot = self.registry.snapshot();
		let next = snapshot.task(&current.kind);
		let mut state = state.lock();
		let same = state.task.definition.name == current.kind
			&& next.as_ref().map(|next| Arc::ptr_eq(&next.definition, &state.task.definition)) == Some(true)
			&& next.is_some();
		if !same {
			let replaceable = next
				.as_ref()
				.map(|next| can_reserve(next, current))
				.unwrap_or(false);
			if replaceable {
				tx.set_task(with_state(current, TaskState {
					status: TaskStateStatus::Pending,
					..current.state.clone()
				}))
				.map_err(|e| e.to_string())
				.expect("staged task write");
				return Decision::End;
			}
			let next_name = next.as_ref().map(|next| next.definition.name.clone());
			if state.reported.as_ref() != Some(&next_name) {
				state.reported = Some(next_name);
				let cause = if next.is_none() { "missing_task" } else { "incompatible_task" };
				(self.report)(format!(
					"Task {} keeps running under its old {} definition ({cause})",
					current.id, current.kind
				));
			}
			*state.snapshot_mut() = snapshot;
		} else {
			*state.snapshot_mut() = snapshot;
		}
		state.agent = Arc::new(tokio::sync::OnceCell::new());
		Decision::Continue
	}

	/// Run the abort handler once; rules 1, 2, and 4 apply, and returning
	/// without an outcome faults.
	async fn run_abort(self: &Arc<Self>, reservation: Reservation) -> Result<(), String> {
		let invocation = reservation.invocation.clone();
		let current = self.inner.lock().live.get(&invocation.task_id).cloned();
		let Some(current) = current else { return Ok(()) };
		if self.inner.lock().closing {
			return Ok(());
		}
		let state = Arc::new(Mutex::new(RunState {
			task: reservation.task.clone(),
			snapshot: reservation.snapshot.clone(),
			reported: None,
			agent: Arc::new(tokio::sync::OnceCell::new()),
		}));
		let runtime = self.runtime(&invocation, state.clone());
		let failure = (reservation.task.definition.abort)(current, runtime, invocation.context.clone())
			.await
			.err();
		let message = format!("Abort handler of task {} returned without a terminal outcome", invocation.task_id);
		self.step(&invocation, move |_tx, _current| {
			Decision::Fault(failure.clone().unwrap_or_else(|| message.clone()))
		})
		.await?;
		Ok(())
	}

	/// One synchronous decision on the Session line. A task that is no
	/// longer running (rule 1: terminal, `completing`, or `waiting`) or a
	/// closing Harness (rule 2) ends the invocation without a write;
	/// otherwise `decide` may stage a write and returns whether the
	/// invocation continues. Ending happens inside the callback, before a
	/// fault's Harness cleanup. A rejected step, such as admission after
	/// close, also ends the invocation.
	async fn step<F>(self: &Arc<Self>, invocation: &Arc<Invocation>, decide: F) -> Result<Option<TaskRecord>, String>
	where
		F: FnMut(&Tx, &TaskRecord) -> Decision + Send + 'static,
	{
		let scheduler = self.clone();
		let invocation2 = invocation.clone();
		let decide = Mutex::new(decide);
		let scheduler_in = scheduler.clone();
		let invocation_in = invocation2.clone();
		let result = self
			.session
			.commit(
				move |tx| {
					let scheduler = scheduler_in.clone();
					let invocation2 = invocation_in.clone();
					let found = scheduler.inner.lock().live.get(&invocation2.task_id).cloned();
					let current = found.filter(|found| found.state.status == TaskStateStatus::Running);
					// Upstream `#step` reads `#closing` (a plain field) before
					// `decide`; the guard must not span the callback —
					// `decide` may re-enter registry snapshots whose callers
					// block on `seal`, which itself takes `inner`.
					let closing = scheduler.inner.lock().closing;
					let decision = match (&current, closing) {
						(Some(current), false) => (decide.lock())(tx, current),
						_ => Decision::End,
					};
					match decision {
						Decision::Continue => Ok(current),
						Decision::End => {
							scheduler.end(&invocation2);
							Ok(None)
						}
						Decision::Fault(message) => {
							let current = current.expect("fault decision requires a running record");
							scheduler.end(&invocation2);
							scheduler.terminate_on(
								tx,
								&current,
								TaskOutcome {
									status: crate::types::TaskOutcomeStatus::Faulted,
									result: None,
									error: Some(crate::types::TaskOutcomeError { message, detail: None }),
									reason: None,
								},
							)?;
							Ok(None)
						}
					}
				},
				&scheduler.context,
			)
			.await;
		// Upstream `#step` catches a rejected commit itself: it ends the
		// invocation, reports once, and returns `undefined` — `#run` returns
		// without rejecting, so `#start`'s catch never reports a second time.
		Ok(match result {
			Ok(current) => current,
			Err(error) => {
				scheduler.end(invocation);
				if !scheduler.inner.lock().closing {
					(scheduler.report)(error.to_string());
				}
				None
			}
		})
	}

	/// Write an outcome the scheduler decided. While the task's ordinary
	/// owned work is live it holds as `completing` and its Harness cleanup
	/// waits for the final commit (spec §5.5, rule 4); otherwise it is
	/// terminal with its cleanup.
	fn terminate_on(&self, tx: &Tx<'_>, record: &TaskRecord, outcome: SchedulerOutcome) -> Result<(), String> {
		self.load_scopes_on(tx, Vec::new(), None)?;
		if self.owned_live_on(tx).contains_key(&record.id) {
			tx.set_task(with_state(record, TaskState {
				status: TaskStateStatus::Completing,
				outcome: Some(outcome),
				..record.state.clone()
			}))
			.map_err(|e| e.to_string())?;
			return Ok(());
		}
		tx.set_task(with_state(record, TaskState {
			status: TaskStateStatus::Terminal,
			outcome: Some(outcome.clone()),
			..record.state.clone()
		}))
		.map_err(|e| e.to_string())?;
		(self.settle_outcome)(tx, record, &outcome)
	}

	/// End an invocation: its runtime operations reject from now on, its
	/// watches stop, its signal aborts, and its task is free.
	fn end(&self, invocation: &Arc<Invocation>) {
		if invocation.ended.swap(true, std::sync::atomic::Ordering::SeqCst) {
			return;
		}
		{
			let mut inner = self.inner.lock();
			if inner.invocations.get(&invocation.task_id).map(|current| Arc::ptr_eq(current, invocation)) == Some(true)
			{
				inner.invocations.remove(&invocation.task_id);
			}
		}
		// Upstream `#end` stops the invocation's watches (synchronously,
		// so they settle `stopped`) before aborting its controller.
		for watch in invocation.watches.lock().drain(..) {
			watch.stop_detached();
		}
		// Pending waits bound to the invocation, such as a tool's
		// wait_for_task, reject with it.
		invocation.abort_invocation(ended_message(invocation.task_id));
	}

	/// No live non-background task in the scope; a task whose owner edges are
	/// not loaded yet counts as inside.
	fn idle(&self, conversation_id: Option<ConversationId>) -> bool {
		let scope = match conversation_id {
			None => Scope::Roots,
			Some(conversation_id) => Scope::Conversation(conversation_id),
		};
		for record in self.live_snapshot() {
			if !record.background && self.in_scope(parent_of(&record), scope, false) != Some(false) {
				return false;
			}
		}
		true
	}

	// ─── Invocation runtime ──────────────────────────────────────────────

	fn runtime(self: &Arc<Self>, invocation: &Arc<Invocation>, phase: Arc<Mutex<RunState>>) -> Arc<dyn TaskRuntime> {
		Arc::new(SchedulerRuntime {
			scheduler: self.clone(),
			invocation: invocation.clone(),
			phase,
		})
	}

	/// Run a committed-state read unless the invocation has ended.
	fn check_read(&self, invocation: &Invocation) -> Result<(), String> {
		if invocation.ended.load(std::sync::atomic::Ordering::SeqCst) {
			Err(ended_message(invocation.task_id))
		} else {
			Ok(())
		}
	}

	/// Commit after rereading the task on the line and gating the invocation.
	async fn gated(
		self: &Arc<Self>,
		invocation: &Arc<Invocation>,
		change: impl FnOnce(&mut Tx<'_>, &TaskRecord) -> Result<Option<NextTaskState>, String> + Send + 'static,
		context: &Context,
	) -> Result<Option<NextTaskState>, String> {
		if invocation.ended.load(std::sync::atomic::Ordering::SeqCst) {
			return Err(ended_message(invocation.task_id));
		}
		let scheduler = self.clone();
		let invocation2 = invocation.clone();
		let change = Mutex::new(Some(change));
		let scheduler_in = scheduler.clone();
		let invocation_in = invocation2.clone();
		self.session
			.commit_with(
				move |tx| {
					let scheduler = scheduler_in;
					let invocation2 = invocation_in;
					if invocation2.ended.load(std::sync::atomic::Ordering::SeqCst) {
						return Err(ended_message(invocation2.task_id));
					}
					if scheduler.inner.lock().closing {
						return Err(closed_error());
					}
					let found = scheduler.inner.lock().live.get(&invocation2.task_id).cloned();
					let Some(found) = found else {
						return Err(format!("Task {} is terminal", invocation2.task_id));
					};
					if found.state.status != TaskStateStatus::Running {
						return Err(format!(
							"Task {} is {}",
							invocation2.task_id,
							state_str(found.state.status)
						));
					}
					if invocation2.mode == Mode::Run && found.abort_requested {
						return Err(format!("Task {} has a durable abort mark", invocation2.task_id));
					}
					let change = change.lock().take().expect("change invoked once");
					change(tx, &found)
				},
				context,
				crate::session::TransactionScope {
					conversation_id: Some(invocation.conversation_id),
					task_id: Some(invocation.task_id),
				},
			)
			.await
			.map_err(|e| e.to_string())
	}

	/// Wait until the Harness clock reaches `until`, rechecking it after
	/// every timer.
	async fn sleep(&self, invocation: &Invocation, until: u64, context: &Context) -> Result<(), String> {
		if invocation.ended.load(std::sync::atomic::Ordering::SeqCst) {
			return Err(ended_message(invocation.task_id));
		}
		loop {
			if invocation.controller.signal().aborted() {
				return Err(ended_message(invocation.task_id));
			}
			if let Some(signal) = context.abort_signal() {
				if signal.aborted() {
					return Err(abort_reason_text(&signal));
				}
			}
			let now = (self.now)();
			let remaining = until as i64 - now;
			if remaining <= 0 {
				return Ok(());
			}
			tokio::select! {
				_ = tokio::time::sleep(std::time::Duration::from_millis(remaining as u64)) => {}
				_ = invocation.controller.signal().aborted_future() => {
					return Err(ended_message(invocation.task_id));
				}
			}
		}
	}
}

struct RunState {
	task: AnyTask,
	snapshot: Arc<dyn RegistrySnapshot>,
	/// The definition last reported as "keeps running under its old
	/// definition": `None` = a missing definition (upstream
	/// `state.reported.task`), so the first progress step always reports.
	reported: Option<Option<String>>,
	agent: Arc<tokio::sync::OnceCell<Result<Agent, String>>>,
}

impl RunState {
	fn snapshot_mut(&mut self) -> &mut Arc<dyn RegistrySnapshot> {
		&mut self.snapshot
	}
}

#[derive(Clone)]
struct PhaseResult {
	checkpoint: Value,
	failure: Option<String>,
}

enum Resolution {
	Ready(AnyTask, TaskRecord),
	Blocked(BlockedReason),
}

impl Resolution {
	fn blocked(&self) -> Option<BlockedReason> {
		match self {
			Resolution::Blocked(reason) => Some(*reason),
			Resolution::Ready(..) => None,
		}
	}

	fn ready(self) -> Option<(AnyTask, TaskRecord)> {
		match self {
			Resolution::Ready(task, record) => Some((task, record)),
			Resolution::Blocked(..) => None,
		}
	}
}

enum Fit {
	Ready { task: AnyTask, migrates: bool },
	Blocked { reason: BlockedReason },
}

fn state_str(status: TaskStateStatus) -> &'static str {
	match status {
		TaskStateStatus::Pending => "pending",
		TaskStateStatus::Running => "running",
		TaskStateStatus::Waiting => "waiting",
		TaskStateStatus::Completing => "completing",
		TaskStateStatus::Terminal => "terminal",
	}
}

/// The per-invocation [`TaskRuntime`] (upstream runtime object).
#[derive(Clone)]
struct SchedulerRuntime {
	scheduler: Arc<TaskScheduler>,
	invocation: Arc<Invocation>,
	phase: Arc<Mutex<RunState>>,
}

impl SchedulerRuntime {
	fn check(&self) -> Result<(), String> {
		self.scheduler.check_read(&self.invocation)
	}

	fn snapshot(&self) -> Arc<dyn RegistrySnapshot> {
		self.phase.lock().snapshot.clone()
	}

	fn task(&self) -> AnyTask {
		self.phase.lock().task.clone()
	}
}

fn validate_task_commit(
	tx: &mut Tx<'_>,
	invocation: &Invocation,
	scheduler: &TaskScheduler,
) -> Result<TaskRecord, String> {
	if invocation.ended.load(std::sync::atomic::Ordering::SeqCst) {
		return Err(ended_message(invocation.task_id));
	}
	if scheduler.inner.lock().closing {
		return Err(closed_error());
	}
	let current = tx
		.task(invocation.task_id)
		.map_err(|error| error.to_string())?
		.ok_or_else(|| format!("Task {} is terminal", invocation.task_id))?;
	if current.state.status != TaskStateStatus::Running {
		return Err(format!("Task {} is {}", invocation.task_id, state_str(current.state.status)));
	}
	if invocation.mode == Mode::Run && current.abort_requested {
		return Err(format!("Task {} has a durable abort mark", invocation.task_id));
	}
	Ok(current)
}

#[async_trait::async_trait]
impl TaskRuntime for SchedulerRuntime {
	fn task_id(&self) -> TaskId { self.invocation.task_id }
	fn conversation_id(&self) -> ConversationId { self.invocation.conversation_id }
	fn signal(&self) -> AbortSignal { self.invocation.signal() }
	fn registry(&self) -> Arc<dyn RegistrySnapshot> { self.snapshot() }

	async fn agent(&self, context: &Context) -> Result<Agent, String> {
		if self.invocation.ended.load(std::sync::atomic::Ordering::SeqCst) {
			return Err(ended_message(self.invocation.task_id));
		}
		let scheduler = self.scheduler.clone();
		let id = self.invocation.conversation_id;
		let snapshot = self.snapshot();
		let cell = self.phase.lock().agent.clone();
		// Upstream `agent: (context) => awaitWithContext(agent(), context)`:
		// the phase's shared resolution starts at first use with the
		// invocation's context (upstream passes `invocation.context` into
		// `#agent`), fixed for the phase; `awaitWithContext` races only the
		// caller's wait, so a cancelled caller rejects with its own reason
		// without observing — or poisoning — the shared resolution (whose
		// failure stays shared for later callers, upstream
		// `phase.agent.catch(() => {})`). Timing deviation: a pre-cancelled
		// caller rejects before polling the cell, so the shared resolution
		// starts at the next caller rather than eagerly with the rejected
		// one (upstream evaluates `agent()` eagerly); it runs at most once
		// per phase with the same invocation context, so the outcome is the
		// same.
		let invocation_context = self.invocation.context.clone();
		match pi_chord::context::await_with_context(
			cell.get_or_init(|| async move { (scheduler.agent)(id, snapshot.as_ref(), &invocation_context).await }),
			context,
		)
		.await
		{
			Ok(agent) => agent.clone(),
			Err(error) => Err(error.to_string()),
		}
	}

	fn settings(&self) -> Settings { (self.scheduler.settings)() }
	fn models(&self) -> Arc<dyn pi_ai::Models> { self.scheduler.models.clone() }

	async fn env(&self, context: &Context) -> Result<Option<Arc<dyn crate::env::ExecutionEnv>>, String> {
		self.check()?;
		let bound = pi_chord::context::with_abort_signal(self.invocation.signal(), context.clone());
		let state = self.scheduler.session.snapshot(&crate::harness::agent::agent_doc(), Some(self.conversation_id().0), None, &bound)
			.await.map_err(|error| error.to_string())?;
		let cwd = state.as_ref().and_then(|state| state.get("cwd")).and_then(Value::as_str).map(str::to_string);
		(self.scheduler.env)(crate::harness::types::EnvTarget {
			conversation_id: self.conversation_id(), cwd, read: self.scheduler.session.clone(),
		}, bound).await
	}

	fn hooks(&self) -> crate::tasks::HookRunner {
		let runtime = self.clone();
		crate::tasks::HookRunner::new(move |name, invoke| {
			let runtime = runtime.clone();
			Box::pin(async move {
				let agent = runtime.agent(&runtime.invocation.context).await?;
				for handler in crate::harness::agent::agent_hooks(&agent, &runtime.task().definition.name) {
					if runtime.invocation.ended.load(std::sync::atomic::Ordering::SeqCst) {
						return Err(ended_message(runtime.invocation.task_id));
					}
					if let Err(error) = invoke(&handler).await {
						if runtime.invocation.signal().aborted() { return Err(error); }
						(runtime.scheduler.report)(error);
					}
				}
				Ok(())
			})
		})
	}

	async fn commit(&self, change: crate::tasks::CommitChange<'_>, context: &Context) -> Result<Option<NextTaskState>, String> {
		let invocation = self.invocation.clone();
		let scheduler = self.scheduler.clone();
		let captured = Arc::new(Mutex::new(None));
		let captured_in = captured.clone();
		let mut change = Some(change);
		let mut callback = move |tx: &mut Tx<'_>| -> Result<Value, String> {
			let current = validate_task_commit(tx, &invocation, &scheduler)?;
			let next = change.take().expect("commit callback invoked once")(tx, &current)?;
			if let Some(state) = &next {
				let mut updated = current;
				updated.state = match state {
					NextTaskState::Running { checkpoint } => TaskState { status: TaskStateStatus::Running, checkpoint: Some(checkpoint.clone()), on: None, policy: None, outcome: None },
					NextTaskState::Waiting { checkpoint, on, policy } => TaskState { status: TaskStateStatus::Waiting, checkpoint: Some(checkpoint.clone()), on: Some(on.clone()), policy: Some(*policy), outcome: None },
					NextTaskState::Terminal { outcome } => TaskState { status: TaskStateStatus::Completing, checkpoint: None, on: None, policy: None, outcome: Some(outcome.clone()) },
				};
				tx.set_task(updated).map_err(|error| error.to_string())?;
			}
			// Upstream `#commitState` loads scopes in its terminal branch
			// before judging owned live work on the commit's candidates, so
			// work the same commit creates below the task counts even while
			// its owner chain is not loaded yet (spec §5.5). The port stages
			// `completing` and lets `finalize_on` settle it in the same
			// commit when nothing is owned-live (observably upstream's
			// direct `terminal` write); the load is chain-known-guarded, so
			// already-loaded scopes cost only in-memory walks.
			if matches!(&next, Some(NextTaskState::Terminal { .. })) {
				let overlay = overlay_of(tx);
				scheduler.load_scopes_on(tx, Vec::new(), Some(&overlay))?;
			}
			scheduler.finalize_on(tx)?;
			*captured_in.lock() = Some(next);
			Ok(Value::Null)
		};
		// Upstream passes the caller's context to `#gated` unchanged; the
		// invocation signal is not chained here (the phase handler's context
		// already carries it), so a closing harness rejects with its closed
		// error instead of a stale abort reason.
		let scope = crate::session::TransactionScope {
			conversation_id: Some(self.conversation_id()),
			task_id: Some(self.invocation.task_id),
		};
		self.scheduler
			.session
			.commit_dyn_with_scope(&mut callback, context, scope)
			.await
			.map_err(|error| error.to_string())?;
		let result = captured.lock().take().unwrap_or(None);
		Ok(result)
	}
	async fn commit_tool(
		&self,
		change: &mut crate::session::CommitValueCallback<'_>,
		context: &Context,
	) -> Result<Value, String> {
		let invocation = self.invocation.clone();
		let scheduler = self.scheduler.clone();
		let mut guard = move |tx: &mut Tx<'_>| {
			validate_task_commit(tx, &invocation, &scheduler).map(|_| ())
		};
		let scope = crate::session::TransactionScope {
			conversation_id: Some(self.conversation_id()),
			task_id: Some(self.invocation.task_id),
		};
		self.scheduler
			.session
			.commit_dyn_with_guard_and_scope(&mut guard, change, context, scope)
			.await
			.map_err(|error| error.to_string())
	}

	async fn memo(&self, name: &str, context: &Context) -> Result<Option<Value>, String> {
		self.check()?;
		let bound = pi_chord::context::with_abort_signal(self.invocation.signal(), context.clone());
		let record = self.scheduler.session.read_on_line_tx(|tx| tx.task(self.task_id()).map_err(|error| error.to_string()), &bound).await.map_err(|error| error.to_string())?;
		Ok(record.and_then(|record| record.memos.and_then(|memos| memos.get(name).cloned())))
	}

	async fn memo_put(&self, name: &str, candidate: Value, context: &Context) -> Result<Value, String> {
		let name = name.to_string();
		let invocation = self.invocation.clone();
		let scheduler = self.scheduler.clone();
		let task_id = self.task_id();
		let candidate = candidate.clone();
		self.scheduler.session.commit_with(move |tx| {
			if invocation.ended.load(std::sync::atomic::Ordering::SeqCst) {
				return Err(ended_message(task_id));
			}
			if scheduler.inner.lock().closing { return Err(closed_error()); }
			let mut record = tx.task(task_id).map_err(|error| error.to_string())?
				.ok_or_else(|| format!("Task {task_id} is terminal"))?;
			if record.state.status != TaskStateStatus::Running {
				return Err(format!("Task {task_id} is {}", state_str(record.state.status)));
			}
			if invocation.mode == Mode::Run && record.abort_requested {
				return Err(format!("Task {task_id} has a durable abort mark"));
			}
			let mut memos = record.memos.clone().unwrap_or_default();
			if let Some(value) = memos.get(&name) { return Ok(value.clone()); }
			memos.insert(name, candidate.clone());
			record.memos = Some(memos);
			tx.set_task(record).map_err(|error| error.to_string())?;
			Ok(candidate)
		}, context, crate::session::TransactionScope { conversation_id: Some(self.conversation_id()), task_id: Some(task_id) })
		.await.map_err(|error| error.to_string())
	}

	async fn get_task(&self, id: TaskId, context: &Context) -> Result<Option<TaskRecord>, String> {
		self.check()?;
		Ok(self.scheduler.session.read_on_line_tx(|tx| tx.task(id).map_err(|error| error.to_string()), context).await.map_err(|error| error.to_string())?)
	}
	async fn wait_for_task(&self, id: TaskId, context: &Context) -> Result<TaskRecord, String> {
		self.check()?;
		let bound = pi_chord::context::with_abort_signal(self.invocation.signal(), context.clone());
		self.scheduler.wait_for_task(id, &bound).await.map(|task| task.record)
	}
	async fn outcomes(&self, ids: &[TaskId], context: &Context) -> Result<Vec<TaskOutcome>, String> {
		let mut outcomes = Vec::with_capacity(ids.len());
		for id in ids {
			let record = self.get_task(*id, context).await?.ok_or_else(|| format!("Task {id} does not exist"))?;
			if record.state.status != TaskStateStatus::Terminal { return Err(format!("Task {id} is not terminal")); }
			outcomes.push(record.state.outcome.ok_or_else(|| format!("Task {id} has no terminal outcome"))?);
		}
		Ok(outcomes)
	}
	async fn conversation(&self, id: ConversationId, context: &Context) -> Result<Option<Arc<dyn crate::session::ConversationHandle>>, String> {
		self.check()?;
		(self.scheduler.conversation)(id, self.binding(), context).await
	}
	async fn entry(&self, kind: Option<&str>, id: EntryId, context: &Context) -> Result<Option<EntryRecord>, String> {
		self.check()?;
		Ok(self.scheduler.storage.entry_in(self.conversation_id(), id, context).await
			.map(|(entry, _)| entry).filter(|entry| kind.is_none_or(|kind| entry.kind == kind)))
	}
	async fn context_view(&self, conversation_id: ConversationId, context: &Context, at: Option<EntryId>) -> Result<crate::session::ContextView, String> {
		self.check()?;
		let bounds = crate::harness::context::capture_context_bounds(self.scheduler.storage.as_ref(), conversation_id, context, at).await?;
		crate::harness::context::derive_context(self.scheduler.storage.as_ref(), conversation_id, bounds, context).await
	}
	/// Upstream `now`/`report` throw once the invocation has ended; the
	/// non-Result signatures surface that as a panic carrying the same
	/// message.
	fn now(&self) -> u64 {
		if let Err(error) = self.check() {
			panic!("{error}");
		}
		(self.scheduler.now)().max(0) as u64
	}
	fn report(&self, error: String) {
		if let Err(ended) = self.check() {
			panic!("{ended}");
		}
		(self.scheduler.report)(error)
	}
	async fn sleep(&self, until: u64, context: &Context) -> Result<(), String> { self.scheduler.sleep(&self.invocation, until, context).await }
	async fn snapshot(&self, token: &crate::documents::DocToken, conversation_id: ConversationId, context: &Context) -> Result<Option<crate::types::JsonObject>, String> {
		self.check()?;
		let bound = pi_chord::context::with_abort_signal(self.invocation.signal(), context.clone());
		Ok(self.scheduler.session.snapshot(token, Some(conversation_id.0), None, &bound).await.map_err(|error| error.to_string())?.map(|value| (*value).clone()))
	}
	async fn snapshot_as_of(&self, token: &crate::documents::DocToken, conversation_id: ConversationId, at: EntryId, context: &Context) -> Result<Option<crate::types::JsonObject>, String> {
		self.check()?;
		let bound = pi_chord::context::with_abort_signal(self.invocation.signal(), context.clone());
		self.scheduler.session.snapshot_as_of(token, conversation_id, None, at, &bound).await.map_err(|error| error.to_string())
	}
	async fn watch_doc(&self, token: &crate::documents::DocToken, conversation_id: ConversationId, context: &Context) -> Result<Option<crate::tasks::DocumentWatch>, String> {
		self.check()?;
		let bound = pi_chord::context::with_abort_signal(self.invocation.signal(), context.clone());
		let watch = self
			.scheduler
			.session
			.watch_doc(token, Some(conversation_id.0), None, &bound)
			.await
			.map_err(|error| error.to_string())?;
		if let Some(watch) = watch {
			// Upstream `#watchDoc` stops and rejects when the invocation ended
			// during acquisition, and otherwise tracks the watch for `#end`.
			if self.invocation.ended.load(std::sync::atomic::Ordering::SeqCst) {
				watch.stop_detached();
				return Err(ended_message(self.invocation.task_id));
			}
			self.invocation.register_watch(watch.clone());
			return Ok(Some(watch));
		}
		Ok(None)
	}
	async fn document_snapshot(
		&self,
		token: &crate::documents::DocToken,
		owner: Option<u64>,
		key: Option<&str>,
		context: &Context,
	) -> Result<Option<crate::types::JsonObject>, String> {
		self.check()?;
		let bound = pi_chord::context::with_abort_signal(self.invocation.signal(), context.clone());
		let value = self
			.scheduler
			.session
			.snapshot(token, owner, key, &bound)
			.await
			.map_err(|error| error.to_string())?;
		Ok(value.map(|value| (*value).clone()))
	}

	async fn document_snapshot_as_of(
		&self,
		token: &crate::documents::DocToken,
		conversation_id: ConversationId,
		key: Option<&str>,
		at: EntryId,
		context: &Context,
	) -> Result<Option<crate::types::JsonObject>, String> {
		self.check()?;
		let bound = pi_chord::context::with_abort_signal(self.invocation.signal(), context.clone());
		self.scheduler
			.session
			.snapshot_as_of(token, conversation_id, key, at, &bound)
			.await
			.map_err(|error| error.to_string())
	}

	async fn document_watch_doc(
		&self,
		token: &crate::documents::DocToken,
		owner: Option<u64>,
		key: Option<&str>,
		context: &Context,
	) -> Result<Option<crate::tasks::DocumentWatch>, String> {
		self.check()?;
		let bound = pi_chord::context::with_abort_signal(self.invocation.signal(), context.clone());
		let watch = self
			.scheduler
			.session
			.watch_doc(token, owner, key, &bound)
			.await
			.map_err(|error| error.to_string())?;
		if let Some(watch) = watch {
			// Same acquisition race as `watch_doc` (upstream `#watchDoc`).
			if self.invocation.ended.load(std::sync::atomic::Ordering::SeqCst) {
				watch.stop_detached();
				return Err(ended_message(self.invocation.task_id));
			}
			self.invocation.register_watch(watch.clone());
			return Ok(Some(watch));
		}
		Ok(None)
	}
}

impl SchedulerRuntime {
	fn binding(&self) -> InvocationBinding {
		let invocation = self.invocation.clone();
		InvocationBinding {
			signal: invocation.signal(),
			check: Arc::new(move || if invocation.ended.load(std::sync::atomic::Ordering::SeqCst) { Err(ended_message(invocation.task_id)) } else { Ok(()) }),
		}
	}
}

fn ended_message(task_id: TaskId) -> String {
	format!("Task {task_id} invocation has ended")
}

fn missing_migration(record: &TaskRecord, definition: &TaskDefinition) -> String {
	format!(
		"Task {} version {} has no migration from {}",
		record.kind, definition.version, record.version
	)
}

fn can_reserve(task: &AnyTask, record: &TaskRecord) -> bool {
	let definition = &task.definition;
	definition.version == record.version
		|| (definition.version > record.version && definition.migrate.is_some())
}

/// Replace a live record's state; memos disappear once an outcome is
/// decided.
fn with_state(record: &TaskRecord, state: TaskState) -> TaskRecord {
	let mut replaced = record.clone();
	if state.status == TaskStateStatus::Terminal || state.status == TaskStateStatus::Completing {
		replaced.memos = None;
	}
	replaced.state = state;
	replaced
}

/// A live owner's durable cancellation intent: its abort mark, or a held
/// outcome other than `completed`.
fn cancellation_intent(record: &TaskRecord) -> bool {
	record.state.status != TaskStateStatus::Terminal
		&& (record.abort_requested || failed_outcome(record))
}

/// Whether the record holds or ends with an outcome other than `completed`.
fn failed_outcome(record: &TaskRecord) -> bool {
	let state = &record.state;
	matches!(state.status, TaskStateStatus::Completing | TaskStateStatus::Terminal)
		&& state.outcome.as_ref().map(|outcome| outcome.status) != Some(crate::types::TaskOutcomeStatus::Completed)
}

fn parent_of(record: &TaskRecord) -> Up {
	match record.owner {
		Some(owner) => Up::Task(owner),
		None => Up::Conversation(record.conversation_id),
	}
}

fn parent_of_node(node: &TaskNode) -> Up {
	match node.owner {
		Some(owner) => Up::Task(owner),
		None => Up::Conversation(node.conversation_id),
	}
}

fn node_of(record: &TaskRecord) -> TaskNode {
	TaskNode {
		conversation_id: record.conversation_id,
		owner: record.owner,
		background: record.background,
	}
}

fn overlay_of(tx: &Tx<'_>) -> Overlay {
	let mut tasks = HashMap::new();
	for record in tx.staged_tasks() {
		tasks.insert(record.id, record);
	}
	let mut edges = HashMap::new();
	for record in tx.staged_conversations() {
		edges.insert(record.id, record.owner.as_ref().map(|owner| owner.task_id));
	}
	Overlay { tasks, edges }
}

fn set_edge(inner: &mut SchedulerState, conversation_id: ConversationId, owner: Option<TaskId>) {
	inner.edges.insert(conversation_id, owner);
	if let Some(owner) = owner {
		inner.conversation_owners.insert(owner);
	}
}

fn chain_known(inner: &SchedulerState, start: Up, overlay: Option<&Overlay>) -> bool {
	let mut at = Some(start);
	while let Some(current) = at {
		match current {
			Up::Task(id) => {
				let node = if let Some(overlay) = overlay {
					overlay.tasks.get(&id).map(node_of).or_else(|| {
						inner.live.get(&id).map(node_of).or_else(|| inner.settled.get(&id).copied())
					})
				} else {
					inner.live.get(&id).map(node_of).or_else(|| inner.settled.get(&id).copied())
				};
				let Some(node) = node else { return false };
				at = Some(parent_of_node(&node));
			}
			Up::Conversation(id) => {
				let edge = if let Some(overlay) = overlay {
					if overlay.edges.contains_key(&id) {
						Some(overlay.edges[&id])
					} else {
						inner.edges.get(&id).copied()
					}
				} else {
					inner.edges.get(&id).copied()
				};
				match edge {
					None => return false,
					Some(None) => return true,
					Some(Some(owner)) => at = Some(Up::Task(owner)),
				}
			}
		}
	}
	true
}

fn below_cancelled(inner: &SchedulerState, start: Up, overlay: Option<&Overlay>) -> bool {
	let mut at = Some(start);
	while let Some(current) = at {
		match current {
			Up::Task(id) => {
				let node = if let Some(overlay) = overlay {
					overlay.tasks.get(&id).map(node_of).or_else(|| {
						inner.live.get(&id).map(node_of).or_else(|| inner.settled.get(&id).copied())
					})
				} else {
					inner.live.get(&id).map(node_of).or_else(|| inner.settled.get(&id).copied())
				};
				let Some(node) = node else { return false };
				if let Some(live) = inner.live.get(&id) {
					if cancellation_intent(live) {
						return true;
					}
				}
				if node.background {
					return false;
				}
				at = Some(parent_of_node(&node));
			}
			Up::Conversation(id) => {
				let edge = if let Some(overlay) = overlay {
					if overlay.edges.contains_key(&id) {
						Some(overlay.edges[&id])
					} else {
						inner.edges.get(&id).copied()
					}
				} else {
					inner.edges.get(&id).copied()
				};
				match edge {
					None => return false,
					Some(None) => return false,
					Some(Some(owner)) => at = Some(Up::Task(owner)),
				}
			}
		}
	}
	false
}

fn waiting_on_with(
	record: &TaskRecord,
	owned: &HashMap<TaskId, Vec<TaskId>>,
	live: &BTreeMap<TaskId, TaskRecord>,
) -> Vec<TaskId> {
	if record.abort_requested {
		return owned.get(&record.id).cloned().unwrap_or_default();
	}
	if record.state.status != TaskStateStatus::Waiting {
		return Vec::new();
	}
	record
		.state
		.on
		.clone()
		.unwrap_or_default()
		.into_iter()
		.filter(|id| live.contains_key(id))
		.collect()
}


fn orphaned_outcome(reason: BlockedReason) -> TaskOutcome {
	TaskOutcome {
		status: crate::types::TaskOutcomeStatus::Orphaned,
		result: None,
		error: None,
		reason: Some(reason.as_str().to_string()),
	}
}

fn scan_tasks_on_tx(tx: &Tx<'_>, query: &crate::types::TaskQuery) -> Result<Vec<TaskRecord>, String> {
	let mut records = Vec::new();
	let mut cursor: Option<crate::types::Cursor> = None;
	loop {
		let page: Page<TaskRecord> = tx.scan_tasks(query, SCAN_PAGE_SIZE, cursor.as_ref()).map_err(|e| e.to_string())?;
		let exhausted = page.next.is_none();
		records.extend(page.items);
		cursor = page.next;
		if exhausted {
			break;
		}
	}
	Ok(records)
}

