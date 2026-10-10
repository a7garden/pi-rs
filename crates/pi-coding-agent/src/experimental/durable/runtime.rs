//! Port of `experimental/durable/runtime.ts` — the headless durable runtime:
//! the plain-value view the TUI renders, the controller the TUI drives, and
//! `openDurable`.
//!
//! Deviations:
//! - **D174 — the view/controller machinery is shared with
//!   `vacation/runtime.ts` through [`open_harness_view`].** The two upstream
//!   runtimes are clones differing only in registry construction, the root
//!   agent shape, and the execution-environment wiring; the port
//!   parameterizes those three and both modules keep their upstream surface
//!   (`vacation::runtime` re-exports the shared types).
//! - [`DurableView::conversation`]/[`DurableView::tasks`] carry the erased
//!   `serde_json` forms of the upstream `ConversationView`/`TaskGraph`
//!   (plain values cross the view boundary; no Harness objects do).
//! - The exact-frame conversation/task watches (upstream
//!   `AttachedReplicatedState` objects) are
//!   [`pi_durable::session::observation::CommittedWatch`] values:
//!   `subscribe` → `start(listener)`, `dispose` → `stop_detached`,
//!   `.value` → `value()`.
//! - Upstream `BACKGROUND_CONTEXT` maps to [`pi_chord::context::TODO_CONTEXT`]
//!   (the crate's established background-context seam).
//! - Upstream serializes commands through a promise chain (`queue =
//!   queue.then(operation)`); the port holds a tokio mutex for the same
//!   one-at-a-time order, with failures reported instead of poisoning the
//!   queue. `setImmediate` notification coalescing becomes a spawned
//!   burst gate over the listener set.
//! - `closing ??=` memoization becomes a shared future.
//! - `process.cwd()` maps to [`std::env::current_dir`] with a `"."` fallback.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use futures_util::future::Shared;
use futures_util::FutureExt;
use pi_chord::context::TODO_CONTEXT;
use pi_chord::delta::Op;
use pi_durable::harness::harness::{open_harness, Conversation, ConversationWatch, Harness};
use pi_durable::harness::registry::RegistryImpl;
use pi_durable::harness::submissions::SubmissionHandle;
use pi_durable::harness::task_graph::{TaskGraph, TaskGraphWatch};
use pi_durable::harness::types::{
    AgentChange, AgentState, ModelRef, Registry, RegistryReader, RootConversationOptions,
    SetOrClear, Submission, Unsubscribe,
};
use pi_durable::harness::view::ConversationView;
use pi_durable::storage::{NodeSqliteStorageOptions, SqliteStorage};
use pi_durable::tasks::BoxFuture;
use pi_durable::types::{
    CommitChange, ConversationId, Cursor, EntryRecord, Page, TableCommitChange,
    ROOT_CONVERSATION_ID,
};
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;

use crate::core::model_resolver::ModelRuntimeSource;
use crate::core::model_runtime::ModelRuntime;
use crate::experimental::durable::harness_setup::{
    self, create_model_runtime, create_settings_manager, find_initial_agent_model, CliModel,
    ExecutionEnvs, SharedSettingsManager,
};
use crate::experimental::durable::sessions::{select_session, SessionLocation};
use crate::experimental::durable::subagent::subagent_extension;

/// Notification severity of a view notice (upstream `Notice.level`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeLevel {
    Info,
    Warning,
    Error,
}

/// One notice in the view (upstream `Notice`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notice {
    pub id: u64,
    pub level: NoticeLevel,
    pub message: String,
}

/// A model the conversation can switch to (upstream `ModelSummary`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSummary {
    pub provider: String,
    pub model_id: String,
    pub name: String,
    pub context_window: u64,
}

/// A conversation the user can switch to: the main one, or a subagent's
/// (upstream `ConversationSummary`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationSummary {
    pub id: ConversationId,
    pub label: String,
    /// The first user message, for a subagent its task.
    pub title: Option<String>,
}

/// The session half of the view (upstream the inline `session` object).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionSummary {
    pub id: String,
    pub directory: String,
    pub cwd: String,
}

/// Everything the TUI renders. Plain values; no Harness objects cross this
/// boundary (upstream `DurableView`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DurableView {
    pub session: SessionSummary,
    /// The conversation shown and talked to (erased `ConversationView`).
    pub conversation: Value,
    pub conversations: Vec<ConversationSummary>,
    pub models: Vec<ModelSummary>,
    pub notices: Vec<Notice>,
    /// The live task graph while the task panel is open (erased `TaskGraph`).
    pub tasks: Option<Value>,
}

/// Upstream `DurableViewSource`.
#[derive(Clone)]
pub struct DurableViewSource {
    inner: Arc<ViewStateInner>,
}

struct ViewStateInner {
    state: parking_lot::Mutex<DurableView>,
    listeners: parking_lot::Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>,
    notifying: AtomicBool,
}

impl DurableViewSource {
    /// The current view snapshot.
    pub fn current(&self) -> DurableView {
        self.inner.state.lock().clone()
    }

    /// Call `listener` once per update burst (upstream `subscribe`); returns
    /// the unsubscribe disposer.
    pub fn subscribe(&self, listener: Arc<dyn Fn() + Send + Sync>) -> Unsubscribe {
        self.inner.listeners.lock().push(Arc::clone(&listener));
        let listeners = Arc::downgrade(&self.inner);
        Box::new(move || {
            if let Some(inner) = listeners.upgrade() {
                inner
                    .listeners
                    .lock()
                    .retain(|registered| !Arc::ptr_eq(registered, &listener));
            }
        })
    }

    fn notify_once(&self) {
        if self.inner.notifying.swap(true, Ordering::SeqCst) {
            return;
        }
        let inner = Arc::clone(&self.inner);
        tokio::spawn(async move {
            // One notify per burst: the gate clears a tick later, so updates
            // that arrive meanwhile coalesce into this single fan-out.
            tokio::task::yield_now().await;
            inner.notifying.store(false, Ordering::SeqCst);
            for listener in inner.listeners.lock().iter() {
                listener();
            }
        });
    }
}

/// Upstream `OpenDurableOptions`.
#[derive(Debug, Clone, Default)]
pub struct OpenDurableOptions {
    pub cwd: Option<String>,
    pub continue_session: bool,
}

/// Memoized close sequence (upstream the `closing ??=` promise).
#[derive(Clone)]
pub struct Closing {
    inner: Arc<parking_lot::Mutex<Option<Shared<BoxFuture<Result<(), String>>>>>>,
}

impl Closing {
    pub fn new() -> Self {
        Self {
            inner: Arc::default(),
        }
    }

    /// Install the close sequence; the first call wins.
    fn install(&self, future: BoxFuture<Result<(), String>>) {
        let mut slot = self.inner.lock();
        if slot.is_none() {
            *slot = Some(future.shared());
        }
    }

    /// Run the close sequence once; later calls await the first (upstream
    /// `close()`).
    pub async fn close(&self) -> Result<(), String> {
        let shared = self
            .inner
            .lock()
            .as_ref()
            .map(Shared::clone)
            .ok_or_else(|| "Close sequence was never installed".to_string())?;
        shared.await
    }
}

impl Default for Closing {
    fn default() -> Self {
        Self::new()
    }
}

/// Upstream `DurableController`: what the TUI may ask for.
#[derive(Clone)]
pub struct DurableController {
    inner: Arc<ControllerInner>,
}

struct ControllerInner {
    source: DurableViewSource,
    harness: Arc<Harness>,
    model_runtime: Arc<ModelRuntime>,
    /// The conversation shown and talked to.
    current: AsyncMutex<Conversation>,
    /// The shown conversation's watch; replaced by `switch_conversation` and
    /// stopped on close.
    conversation: parking_lot::Mutex<Option<ConversationWatch>>,
    /// The live task-graph watch while the panel is open.
    tasks: parking_lot::Mutex<Option<TaskGraphWatch>>,
    /// Serializes controller commands (upstream the promise queue).
    command: AsyncMutex<()>,
    fail: Arc<dyn Fn(String) + Send + Sync>,
}

/// Upstream `OpenDurableResult` (`settings` keeps its upstream slot for the
/// TUI's theme and terminal capabilities; the TUI itself defers with
/// interactive mode).
pub struct OpenDurableResult {
    pub view: DurableViewSource,
    pub controller: DurableController,
    /// pi's settings.
    pub settings: SharedSettingsManager,
    pub close: Closing,
}

/// The agent document of a view; unknown shapes decode to the empty state
/// (upstream `agentOf`).
pub fn agent_of(view: &Value) -> AgentState {
    view.get("docs")
        .and_then(|docs| docs.get("pi.agent"))
        .cloned()
        .and_then(|agent| serde_json::from_value(agent).ok())
        .unwrap_or_default()
}

fn view_json(view: &ConversationView) -> Value {
    serde_json::to_value(view).unwrap_or(Value::Null)
}

/// `pi.user` — the erased user-entry kind.
const USER_ENTRY_KIND: &str = "pi.user";

/// A subagent's task: the oldest user message of its conversation. The main
/// conversation needs no title (upstream `firstInput`).
async fn first_input(harness: &Arc<Harness>, id: ConversationId) -> Option<String> {
    if id == ROOT_CONVERSATION_ID {
        return None;
    }
    let Some(conversation) = harness.conversation(id, &TODO_CONTEXT).await.ok().flatten() else {
        return None;
    };
    let mut first: Option<String> = None;
    let mut cursor: Option<Cursor> = None;
    loop {
        let page: Page<EntryRecord> = conversation
            .entries(Default::default(), 256, cursor.as_ref(), &TODO_CONTEXT)
            .await
            .unwrap_or_else(|_| Page::empty());
        // Newest-first pages; `findLast` keeps the oldest user message.
        if let Some(entry) = page
            .items
            .iter()
            .rev()
            .find(|entry| entry.kind == USER_ENTRY_KIND)
        {
            first = Some(title_text(entry));
        }
        cursor = page.next;
        if cursor.is_none() {
            break;
        }
    }
    first
}

/// The text of a user entry, as a one-line title (upstream `titleOf`).
fn title_text(entry: &EntryRecord) -> String {
    let Some(pi_ai::Message::User(message)) = entry.model.as_ref().and_then(|model| model.first())
    else {
        return String::new();
    };
    let text = match &message.content {
        pi_ai::types::UserContentField::Text(text) => text.clone(),
        pi_ai::types::UserContentField::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                pi_ai::types::UserContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
    };
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Everything the shared open body injects per cluster (the two upstream
/// `openDurable` bodies' differences).
pub struct ClusterOptions {
    /// The cluster's registry, fully built.
    pub registry: RegistryImpl,
    /// The root conversation's agent change, from the resolved initial model
    /// (absent while the session continues).
    pub root_agent: Arc<dyn Fn(Option<harness_setup::InitialModel>) -> AgentChange + Send + Sync>,
    /// `HarnessOptions.env` (durable only).
    pub env: Option<pi_durable::harness::types::EnvBuilder>,
    /// Runs after the Harness closed, before the session lock releases
    /// (durable: execution-env cleanup).
    pub cleanup: Option<Arc<dyn Fn() -> BoxFuture<Result<(), String>> + Send + Sync>>,
}

/// The durable cluster's `openDurable`.
pub async fn open_durable(options: OpenDurableOptions) -> Result<OpenDurableResult, String> {
    let cwd = match options.cwd {
        Some(cwd) => cwd,
        None => std::env::current_dir()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_else(|_| ".".to_string()),
    };
    let location = select_session(&cwd, options.continue_session).await?;
    let envs = Arc::new(ExecutionEnvs::new(location.cwd.clone()));
    let registry = harness_setup::create_coding_registry(
        create_settings_manager(&location.cwd),
        &location.cwd,
    );
    registry.install(subagent_extension());
    let session_cwd = location.cwd.clone();
    open_harness_view(
        location,
        ClusterOptions {
            registry,
            root_agent: Arc::new(move |initial: Option<harness_setup::InitialModel>| {
                let initial = initial.unwrap_or_default();
                AgentChange {
                    cwd: Some(SetOrClear::Set(session_cwd.clone())),
                    model: initial.model.map(SetOrClear::Set),
                    thinking_level: initial.thinking_level.map(SetOrClear::Set),
                    ..AgentChange::default()
                }
            }),
            env: Some(envs.builder()),
            cleanup: Some(Arc::new(move || {
                let envs = Arc::clone(&envs);
                Box::pin(async move {
                    envs.cleanup(&TODO_CONTEXT).await;
                    Ok(())
                })
            })),
        },
    )
    .await
}

/// The shared `openDurable` body (upstream the twin runtime bodies).
/// Owns the session location: on failure it releases the lock; on success
/// the returned [`Closing`] releases it.
pub async fn open_harness_view(
    location: SessionLocation,
    cluster: ClusterOptions,
) -> Result<OpenDurableResult, String> {
    let context = &*TODO_CONTEXT;
    let model_runtime = create_model_runtime().await?;
    let settings_manager = create_settings_manager(&location.cwd);
    harness_setup::configure_harness_http(&settings_manager.lock())?;
    let settings = harness_setup::create_harness_settings(Arc::clone(&settings_manager));

    // Extension reports that land before the notice wiring are buffered
    // (upstream `pendingReports`).
    let pending_reports: Arc<parking_lot::Mutex<Vec<String>>> = Arc::default();
    let report_sink = Arc::clone(&pending_reports);
    let harness = open_harness(
        Arc::new(
            SqliteStorage::open_path(&location.database, &NodeSqliteStorageOptions::default())
                .await
                .map_err(|error| error.to_string())?,
        ),
        pi_durable::harness::types::HarnessOptions {
            models: Arc::clone(&model_runtime) as Arc<dyn pi_ai::Models>,
            registry: Arc::new(cluster.registry) as Arc<dyn RegistryReader>,
            settings: None,
            settings_provider: Some(settings),
            env: cluster.env,
            conversation_created: None,
            now: None,
            on_report: Some(Arc::new(move |error: String| {
                report_sink.lock().push(error);
            })),
        },
        context,
    )
    .await?;

    let initial = if location.created {
        Some(find_initial_agent_model(
            &settings_manager.lock(),
            &model_runtime,
            None::<CliModel>,
        )?)
    } else {
        None
    };
    let root = harness
        .root(
            context,
            Some(RootConversationOptions {
                agent: Some((cluster.root_agent)(initial.clone())),
                init: None,
            }),
        )
        .await?;
    let root_id = root.id;

    // Scan every conversation once for the switcher (upstream the
    // `commit(tx => tx.scanConversations(...))` loop).
    let mut summaries: Vec<ConversationSummary> = Vec::new();
    let mut cursor: Option<Cursor> = None;
    loop {
        let page = harness
            .session
            .read_on_line_tx(
                |tx| {
                    tx.scan_conversations(
                        &pi_durable::types::ConversationQuery::default(),
                        256,
                        cursor.as_ref(),
                    )
                    .map_err(|error| error.to_string())
                },
                context,
            )
            .await
            .map_err(|error| error.to_string())?;
        for record in &page.items {
            summaries.push(ConversationSummary {
                id: record.id,
                label: conversation_label(record.id, root_id),
                title: first_input(&harness, record.id).await,
            });
        }
        cursor = page.next;
        if cursor.is_none() {
            break;
        }
    }

    let conversation_state = root.view_state(context).await.map_err(|error| error)?;
    let view_inner = Arc::new(ViewStateInner {
        state: parking_lot::Mutex::new(DurableView {
            session: SessionSummary {
                id: location.id.clone(),
                directory: location.directory.to_string_lossy().into_owned(),
                cwd: location.cwd.clone(),
            },
            conversation: conversation_state.value().clone(),
            conversations: summaries,
            models: model_summaries(&model_runtime),
            notices: Vec::new(),
            tasks: None,
        }),
        listeners: parking_lot::Mutex::new(Vec::new()),
        notifying: AtomicBool::new(false),
    });
    let source = DurableViewSource {
        inner: Arc::clone(&view_inner),
    };

    let next_notice = Arc::new(AtomicU64::new(1));
    let notice: Arc<dyn Fn(NoticeLevel, String) + Send + Sync> = {
        let source = source.clone();
        let next_notice = Arc::clone(&next_notice);
        Arc::new(move |level: NoticeLevel, message: String| {
            let id = next_notice.fetch_add(1, Ordering::SeqCst);
            {
                let mut state = source.inner.state.lock();
                state.notices.push(Notice { id, level, message });
                if state.notices.len() > 20 {
                    let drop = state.notices.len() - 20;
                    state.notices.drain(..drop);
                }
            }
            source.notify_once();
        })
    };
    let fail: Arc<dyn Fn(String) + Send + Sync> = {
        let notice = Arc::clone(&notice);
        Arc::new(move |message: String| notice(NoticeLevel::Error, message))
    };
    // Buffered extension reports surface as warnings once the notice wiring
    // exists (upstream `pendingReports`).
    for error in pending_reports.lock().drain(..) {
        notice(NoticeLevel::Warning, error);
    }

    // The shown conversation's watch drives view updates; acquired now,
    // installed once the controller owns its disposal.
    let root_watch = root.watch(context).await.map_err(|error| error)?;

    let controller = DurableController {
        inner: Arc::new(ControllerInner {
            source: source.clone(),
            harness: Arc::clone(&harness),
            model_runtime: Arc::clone(&model_runtime),
            current: AsyncMutex::new(root),
            conversation: parking_lot::Mutex::new(None),
            tasks: parking_lot::Mutex::new(None),
            command: AsyncMutex::new(()),
            fail: Arc::clone(&fail),
        }),
    };

    // The shown conversation's watch drives view updates; the controller
    // owns its disposal.
    {
        let watch = root_watch;
        let listener = {
            let source = source.clone();
            Arc::new(
                move |value: ConversationView,
                      _ops: Vec<Op>,
                      _context: pi_chord::context::Context|
                      -> BoxFuture<()> {
                    let source = source.clone();
                    Box::pin(async move {
                        source.inner.state.lock().conversation = view_json(&value);
                        source.notify_once();
                    })
                },
            )
        };
        watch.start(listener).map_err(|error| error.to_string())?;
        *controller.inner.conversation.lock() = Some(watch);
    }

    // Subagents appear as their conversations are created. A commit listener
    // only records; it calls no Session API (upstream `subscribeCommits`).
    {
        let source = source.clone();
        harness.session.subscribe_commits(Arc::new(
            move |publication: &pi_durable::types::CommitPublication,
                  _context: &pi_chord::context::Context| {
                let mut changed = false;
                {
                    let mut state = source.inner.state.lock();
                    for change in &publication.changes {
                        match change {
                            CommitChange::Table(TableCommitChange::Conversation { value }) => {
                                state.conversations.push(ConversationSummary {
                                    id: value.id,
                                    label: conversation_label(value.id, root_id),
                                    title: None,
                                });
                                changed = true;
                            }
                            CommitChange::Table(TableCommitChange::Entry { value })
                                if value.kind == USER_ENTRY_KIND =>
                            {
                                let id = value.conversation_id;
                                for summary in state.conversations.iter_mut() {
                                    if summary.id == id && summary.title.is_none() {
                                        summary.title = Some(title_text(value));
                                    }
                                }
                                changed = true;
                            }
                            _ => {}
                        }
                    }
                }
                if changed {
                    source.notify_once();
                }
            },
        ));
    }

    // Saved-model and initial-fallback notices (upstream the block before
    // `toggleTasks`).
    match agent_of(&controller.inner.source.current().conversation).model {
        None => notice(
            NoticeLevel::Warning,
            "No model configured; select one with /model.".to_string(),
        ),
        Some(saved) => {
            if model_runtime
                .get_model(&saved.provider, &saved.model_id)
                .is_none()
            {
                notice(
                    NoticeLevel::Warning,
                    format!(
                        "Saved model is unavailable: {}/{}",
                        saved.provider, saved.model_id
                    ),
                );
            }
        }
    }
    if let Some(fallback) = initial.and_then(|initial| initial.fallback_message) {
        notice(NoticeLevel::Info, fallback);
    }

    // The task panel starts open; /tasks hides it. Recovered work from an
    // interrupted turn continues now.
    controller.toggle_tasks().await?;
    harness.resume();

    let close = Closing::new();
    {
        let controller = controller.clone();
        close.install(Box::pin(async move {
            // Close writes no outcome: a running turn resumes with
            // `--continue` (upstream `close`).
            if let Some(previous) = controller.inner.conversation.lock().take() {
                previous.stop_detached();
            }
            if let Some(tasks) = controller.inner.tasks.lock().take() {
                tasks.stop_detached();
            }
            let close_error = harness.close(context).await.err();
            let cleanup_error = match &cluster.cleanup {
                Some(cleanup) => cleanup().await.err(),
                None => None,
            };
            location.release.release();
            match (close_error, cleanup_error) {
                (None, None) => Ok(()),
                (Some(error), None) | (None, Some(error)) => Err(error),
                (Some(close_error), Some(cleanup_error)) => {
                    Err(format!("{close_error}; {cleanup_error}"))
                }
            }
        }));
    }

    Ok(OpenDurableResult {
        view: source,
        controller,
        settings: settings_manager,
        close,
    })
}

fn conversation_label(id: ConversationId, root_id: ConversationId) -> String {
    if id == root_id {
        "main".to_string()
    } else {
        format!("subagent {id}")
    }
}

fn model_summaries(model_runtime: &ModelRuntime) -> Vec<ModelSummary> {
    model_runtime
        .get_available_snapshot()
        .iter()
        .map(|model| ModelSummary {
            provider: model.base.provider.clone(),
            model_id: model.base.id.clone(),
            name: model.base.name.clone(),
            context_window: model.context_window,
        })
        .collect()
}

fn outcome_status_text(status: pi_durable::types::TaskOutcomeStatus) -> &'static str {
    match status {
        pi_durable::types::TaskOutcomeStatus::Completed => "completed",
        pi_durable::types::TaskOutcomeStatus::Failed => "failed",
        pi_durable::types::TaskOutcomeStatus::Aborted => "aborted",
        pi_durable::types::TaskOutcomeStatus::Faulted => "faulted",
        pi_durable::types::TaskOutcomeStatus::Orphaned => "orphaned",
    }
}

impl DurableController {
    /// The conversation shown and talked to.
    async fn current(&self) -> Conversation {
        self.inner.current.lock().await.clone()
    }

    /// Prompt when idle; otherwise steer or queue a follow-up (upstream
    /// `submit`).
    pub async fn submit(
        &self,
        text: String,
        when_busy: pi_durable::harness::types::WhenBusy,
    ) -> Result<(), String> {
        let _queue = self.inner.command.lock().await;
        let submission = self
            .current()
            .await
            .submit(
                pi_durable::harness::types::SubmissionDraft::Input(
                    pi_durable::harness::types::InputSubmissionDraft {
                        request_id: None,
                        content: pi_durable::harness::types::UserInput::Text(text),
                        when_busy: Some(when_busy),
                    },
                ),
                &TODO_CONTEXT,
            )
            .await?;
        self.watch_answer(submission);
        Ok(())
    }

    /// Compact, then report the outcome once it is known; the status line
    /// shows the compaction meanwhile (upstream `compact`).
    pub async fn compact(&self, instructions: Option<String>) -> Result<(), String> {
        let _queue = self.inner.command.lock().await;
        let harness = Arc::clone(&self.inner.harness);
        let fail = Arc::clone(&self.inner.fail);
        let id = self
            .current()
            .await
            .compact(instructions, &TODO_CONTEXT)
            .await?;
        tokio::spawn(async move {
            let receipt = match harness.wait_for_task(id, &TODO_CONTEXT).await {
                Ok(receipt) => receipt,
                Err(error) => return fail(error),
            };
            let Some(outcome) = receipt.outcome() else {
                return;
            };
            match outcome.status {
                pi_durable::types::TaskOutcomeStatus::Completed => {
                    let result: pi_durable::harness::types::CompactionResult = outcome
                        .result
                        .clone()
                        .and_then(|value| serde_json::from_value(value).ok())
                        .unwrap_or_default();
                    // A summary written while busy is a submission: placed
                    // now, queued, or dropped as stale.
                    let status = match result.submission_id {
                        None => None,
                        Some(submission_id) => {
                            match harness.submission(submission_id, &TODO_CONTEXT).await {
                                Ok(Some(submission)) => submission
                                    .status(&TODO_CONTEXT)
                                    .await
                                    .ok()
                                    .map(|record| record.status),
                                _ => None,
                            }
                        }
                    };
                    let message = if result.entry_id.is_some()
                        || status == Some(pi_durable::types::SubmissionStatus::Done)
                    {
                        "Compacted."
                    } else if status == Some(pi_durable::types::SubmissionStatus::Queued) {
                        "Compaction summary queued; it is placed at the next turn boundary."
                    } else if status == Some(pi_durable::types::SubmissionStatus::Unanswered) {
                        "Compaction summary dropped: the context changed under it."
                    } else {
                        "Nothing to compact: the context fits in the recent window that is kept verbatim."
                    };
                    fail(message.to_string());
                }
                pi_durable::types::TaskOutcomeStatus::Aborted => {
                    fail("Compaction aborted.".to_string())
                }
                other => {
                    let detail = outcome
                        .error
                        .as_ref()
                        .map(|error| error.message.clone())
                        .or_else(|| outcome.reason.clone())
                        .unwrap_or_default();
                    fail(format!(
                        "Compaction {}: {detail}",
                        outcome_status_text(other)
                    ));
                }
            }
        });
        Ok(())
    }

    /// Not queued: it waits until the conversation is idle (upstream
    /// `abort`).
    pub async fn abort(&self) {
        if let Err(error) = self.current().await.abort(&TODO_CONTEXT, None).await {
            (self.inner.fail)(error);
        }
    }

    /// Cycle the shown conversation through its model's thinking levels
    /// (upstream `cycleThinking`).
    pub async fn cycle_thinking(&self) -> Result<(), String> {
        let _queue = self.inner.command.lock().await;
        let model = self.agent_model()?;
        if !model.reasoning {
            return Err("Current model does not support thinking".into());
        }
        let levels = pi_ai::get_supported_thinking_levels(&model);
        let level = agent_of(&self.inner.source.current().conversation)
            .thinking_level
            .unwrap_or(pi_ai::types::ModelThinkingLevel::Off);
        let index = levels
            .iter()
            .position(|candidate| *candidate == level)
            .map_or(0, |at| at + 1);
        let next = levels
            .get(index % levels.len().max(1))
            .copied()
            .unwrap_or(pi_ai::types::ModelThinkingLevel::Off);
        self.current()
            .await
            .configure(
                AgentChange {
                    thinking_level: Some(SetOrClear::Set(next)),
                    ..AgentChange::default()
                },
                &TODO_CONTEXT,
            )
            .await
    }

    /// Point the shown conversation at another model (upstream `setModel`).
    pub async fn set_model(&self, reference: ModelRef) -> Result<(), String> {
        let _queue = self.inner.command.lock().await;
        let Some(model) = self
            .inner
            .model_runtime
            .get_model(&reference.provider, &reference.model_id)
        else {
            return Err(format!(
                "Unknown model: {}/{}",
                reference.provider, reference.model_id
            ));
        };
        let thinking = agent_of(&self.inner.source.current().conversation)
            .thinking_level
            .unwrap_or(pi_ai::types::ModelThinkingLevel::Off);
        let level = pi_ai::clamp_thinking_level(&model, thinking);
        self.current()
            .await
            .configure(
                AgentChange {
                    model: Some(SetOrClear::Set(reference)),
                    thinking_level: Some(SetOrClear::Set(level)),
                    ..AgentChange::default()
                },
                &TODO_CONTEXT,
            )
            .await
    }

    /// Show or hide the live task panel (upstream `toggleTasks`).
    pub async fn toggle_tasks(&self) -> Result<(), String> {
        let _queue = self.inner.command.lock().await;
        if let Some(open) = self.inner.tasks.lock().take() {
            open.stop_detached();
            self.inner.source.inner.state.lock().tasks = None;
            return Ok(());
        }
        let watch = self.inner.harness.task_graph.watch(&TODO_CONTEXT).await?;
        let graph = watch.value();
        self.inner.source.inner.state.lock().tasks =
            Some(serde_json::to_value(&graph).unwrap_or(Value::Null));
        let source = self.inner.source.clone();
        let listener = Arc::new(
            move |value: TaskGraph,
                  _ops: Vec<Op>,
                  _context: pi_chord::context::Context|
                  -> BoxFuture<()> {
                let source = source.clone();
                Box::pin(async move {
                    source.inner.state.lock().tasks =
                        Some(serde_json::to_value(&value).unwrap_or(Value::Null));
                    source.notify_once();
                })
            },
        );
        watch.start(listener).map_err(|error| error.to_string())?;
        *self.inner.tasks.lock() = Some(watch);
        Ok(())
    }

    /// Show and talk to another conversation (upstream `switchConversation`).
    pub async fn switch_conversation(&self, id: ConversationId) -> Result<(), String> {
        let _queue = self.inner.command.lock().await;
        let next = self
            .inner
            .harness
            .conversation(id, &TODO_CONTEXT)
            .await?
            .ok_or_else(|| format!("Conversation {id} does not exist"))?;
        let next_watch = next.watch(&TODO_CONTEXT).await.map_err(|error| error)?;
        let value = view_json(&next_watch.value());
        if let Some(previous) = self.inner.conversation.lock().take() {
            previous.stop_detached();
        }
        *self.inner.current.lock().await = next;
        self.inner.source.inner.state.lock().conversation = value;
        let source = self.inner.source.clone();
        let listener = Arc::new(
            move |value: ConversationView,
                  _ops: Vec<Op>,
                  _context: pi_chord::context::Context|
                  -> BoxFuture<()> {
                let source = source.clone();
                Box::pin(async move {
                    source.inner.state.lock().conversation = view_json(&value);
                    source.notify_once();
                })
            },
        );
        next_watch
            .start(listener)
            .map_err(|error| error.to_string())?;
        *self.inner.conversation.lock() = Some(next_watch);
        Ok(())
    }

    /// Resolve the shown conversation's model (upstream `agentModel`).
    fn agent_model(&self) -> Result<pi_ai::Model, String> {
        let Some(reference) = agent_of(&self.inner.source.current().conversation).model else {
            return Err("No model selected".into());
        };
        self.inner
            .model_runtime
            .get_model(&reference.provider, &reference.model_id)
            .ok_or_else(|| "Current model is unavailable".to_string())
    }

    /// Report unanswered submissions (upstream `watchAnswer`).
    fn watch_answer(&self, submission: SubmissionHandle) {
        let fail = Arc::clone(&self.inner.fail);
        tokio::spawn(async move {
            match submission.wait(&TODO_CONTEXT).await {
                Err(error) => fail(error),
                Ok(settled) => {
                    if settled.status == pi_durable::types::SubmissionStatus::Unanswered
                        && settled.reason.as_deref() != Some("aborted")
                    {
                        let detail = settled
                            .detail
                            .as_ref()
                            .map(|detail| {
                                format!(" {}", serde_json::to_string(detail).unwrap_or_default())
                            })
                            .unwrap_or_default();
                        fail(format!(
                            "No answer: {}{detail}",
                            settled.reason.unwrap_or_else(|| "unknown".to_string())
                        ));
                    }
                }
            }
        });
    }
}
