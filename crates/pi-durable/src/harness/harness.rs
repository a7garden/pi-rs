//! Port of `packages/durable/src/harness/harness.ts`.
//!
//! Deviations: D1 — `HarnessImpl extends SessionImpl` becomes composition:
//! [`Harness`] owns the [`crate::session::Session`] (Rust has no `async`
//! trait inheritance). D2 — `ConversationHandle` (upstream interface) is the
//! session-owned trait; the harness provides the invocation-bound instance.
//! D3 — `ConversationWatch` is the session's `CommittedWatch<ConversationView>`
//! wrapper (session D-note).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::context::Context;

use crate::entries::RESET_ENTRY;
use crate::harness::agent::{configure, resolve_agent, resolve_settings};
use crate::harness::compaction::{create_compaction, CompactionInput};
use crate::harness::inbox::withdraw_queued_inputs;
use crate::harness::live::settle_scheduler_outcome;
use crate::harness::registry::builtin_tasks;
use crate::harness::scheduler::{InvocationBinding, TaskScheduler, TaskSchedulerOptions};
use crate::harness::submissions::Submissions;
use crate::harness::task_graph::TaskGraphView;
use crate::harness::types::InputSubmissionDraft;
use crate::harness::types::{
    Agent, AgentChange, CompactionReason, ConversationAbortOptions, ConversationCreateOptions,
    ConversationId, ConversationInit, EntryDraft, HarnessInspection, HarnessOptions,
    HarnessSettings, ReportHook, RootConversationOptions, SubmissionAbort, SubmissionDraft,
    SubmissionRecord, TaskId,
};
use crate::harness::usage::{add_usage_state, usage_doc, UsageState};
use crate::harness::util::{closed_error, scan_all};
use crate::harness::view::{ConversationView, ConversationViews};
use crate::session::observation::CommittedWatch;
use crate::session::{
    create_session, ContextView, ConversationDocumentSnapshot, ConversationHandle, Session,
    SessionError, SessionHooks, SettledTask, TransactionScope, TxError,
};
use crate::types::{
    ConversationOwnership, EntryQuery, EntryRecord, Page, Storage, SubmissionStatus,
    ROOT_CONVERSATION_ID,
};
use crate::types::{EntryHead, SelfMarker};

const SCAN_PAGE_SIZE: u64 = 256;

/// Where the creating conveniences apply (upstream `CreateTarget`).
enum CreateTarget {
    Root,
    Independent {
        ownership: ConversationOwnership,
    },
    Fork {
        parent_id: ConversationId,
        at: crate::harness::types::EntryId,
        ownership: ConversationOwnership,
    },
}

/// What the conveniences apply in the creating commit, after the creation
/// hook.
#[derive(Default)]
struct CreateOptions {
    agent: Option<AgentChange>,
    init: Option<ConversationInit>,
}

/// Serialized exact-frame watch of one conversation's view.
pub type ConversationWatch = CommittedWatch<ConversationView>;

/// Durable agent harness over one Session (upstream `HarnessImpl`).
pub struct Harness {
    pub session: Arc<Session>,
    storage: Arc<dyn Storage>,
    options: HarnessOptions,
    report: ReportHook,
    now: crate::harness::types::NowFn,
    settings: Arc<dyn Fn() -> HarnessSettings + Send + Sync>,
    pub tasks: Arc<TaskScheduler>,
    pub submissions: Arc<Submissions>,
    pub views: Arc<ConversationViews>,
    pub task_graph: Arc<TaskGraphView>,
    pub agent_resolutions: Arc<Mutex<()>>,
    closed: Arc<Mutex<bool>>,
}

/// A stateless handle for one conversation, bound to the Harness that
/// returned it. Compare handles by `id`.
#[derive(Clone)]
pub struct Conversation {
    pub id: ConversationId,
    harness: Arc<Harness>,
}

impl std::fmt::Debug for Conversation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Conversation")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

/// Open a Harness over storage. The registry may keep changing while the
/// Harness runs.
pub async fn open_harness(
    storage: Arc<dyn Storage>,
    options: HarnessOptions,
    context: &Context,
) -> Result<Arc<Harness>, String> {
    if let Some(signal) = context.abort_signal() {
        if signal.aborted() {
            return Err(abort_text(signal.reason()));
        }
    }
    let snapshot = options.registry.snapshot();
    let missing: Vec<String> = builtin_tasks()
        .iter()
        .filter(|task| snapshot.task(&task.definition.name).is_none())
        .map(|task| task.definition.name.clone())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "Registry lacks built-in tasks {}; create it with create_registry()",
            missing.join(", ")
        ));
    }
    let harness = Harness::new(storage, options, context);
    if let Err(error) = harness.open_tasks(context).await {
        // The caller's context may be what failed open: close without it, and
        // return the open error.
        let quiet = context.clone();
        if let Err(close_error) = harness.close(&quiet).await {
            (harness.report.clone())(close_error);
        }
        return Err(error);
    }
    Ok(harness)
}

fn abort_text(reason: Option<pi_chord::context::AbortReason>) -> String {
    reason
        .map(|reason| reason.to_string())
        .unwrap_or_else(|| "This operation was aborted".to_string())
}

impl Harness {
    fn new(storage: Arc<dyn Storage>, options: HarnessOptions, context: &Context) -> Arc<Self> {
        let report: ReportHook = options
            .on_report
            .clone()
            .unwrap_or_else(|| Arc::new(|_| {}));
        let now: crate::harness::types::NowFn = options.now.clone().unwrap_or_else(|| {
            Arc::new(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0)
            })
        });
        let settings: Arc<dyn Fn() -> HarnessSettings + Send + Sync> =
            match options.settings_provider.clone() {
                Some(provider) => {
                    // The provider is a live overlay (upstream: tests mutate one
                    // settings object): fields it leaves absent keep the static
                    // settings.
                    let static_settings = options.settings.clone();
                    Arc::new(move || {
                        let mut merged = static_settings.clone().unwrap_or_default();
                        let dynamic = provider();
                        merged.extensions = dynamic.extensions.or(merged.extensions);
                        merged.stream = dynamic.stream.or(merged.stream);
                        merged.retry = dynamic.retry.or(merged.retry);
                        merged.compaction = dynamic.compaction.or(merged.compaction);
                        merged.progress = dynamic.progress.or(merged.progress);
                        merged.tool_execution = dynamic.tool_execution.or(merged.tool_execution);
                        merged.steering_mode = dynamic.steering_mode.or(merged.steering_mode);
                        merged.follow_up_mode = dynamic.follow_up_mode.or(merged.follow_up_mode);
                        merged
                    }) as Arc<dyn Fn() -> HarnessSettings + Send + Sync>
                }
                None => {
                    let settings_value = options.settings.clone();
                    Arc::new(move || settings_value.clone().unwrap_or_default())
                }
            };
        let session = std::sync::Arc::new(create_session(storage.clone(), None));
        let conversation_created = options.conversation_created.clone();
        let harness_slot: Arc<Mutex<Option<Arc<Harness>>>> = Arc::default();
        let scheduler = TaskScheduler::new(TaskSchedulerOptions {
            session: session.clone(),
            storage: storage.clone(),
            registry: options.registry.clone(),
            models: options.models.clone(),
            agent: {
                let session = session.clone();
                let settings = settings.clone();
                let report = report.clone();
                let registry = options.registry.clone();
                Box::new(move |id, _snapshot, call_context| {
                    let session = session.clone();
                    let registry = registry.clone();
                    let settings = settings.clone();
                    let report = report.clone();
                    let call_context = call_context.clone();
                    Box::pin(async move {
                        let snapshot = registry.snapshot();
                        let agent_doc = crate::harness::agent::agent_doc();
                        let state = session
                            .snapshot(&agent_doc, Some(id.0), None, &call_context)
                            .await
                            .map_err(|e| e.to_string())?;
                        let state: Option<crate::harness::types::AgentState> = state.map(|value| {
                            serde_json::from_value(serde_json::Value::Object((*value).clone()))
                                .unwrap_or_default()
                        });
                        Ok(resolve_agent(
                            state.as_ref(),
                            snapshot.as_ref(),
                            &resolve_settings(Some(&settings())),
                            report.as_ref(),
                        ))
                    })
                })
            },
            settings: {
                let settings = settings.clone();
                Box::new(move || resolve_settings(Some(&settings())))
            },
            env: options
                .env
                .clone()
                .unwrap_or_else(|| Arc::new(|_, _| Box::pin(async { Ok(None) }))),
            now: now.clone(),
            report: report.clone(),
            settle_outcome: Arc::new(|tx, record, outcome| {
                let fault = match outcome.status {
                    crate::types::TaskOutcomeStatus::Faulted => {
                        crate::harness::live::SchedulerFault::Faulted {
                            message: outcome
                                .error
                                .as_ref()
                                .map(|error| error.message.clone())
                                .unwrap_or_default(),
                        }
                    }
                    crate::types::TaskOutcomeStatus::Aborted => {
                        crate::harness::live::SchedulerFault::Aborted {
                            reason: outcome
                                .reason
                                .clone()
                                .unwrap_or_else(|| "aborted".to_string()),
                        }
                    }
                    // The scheduler also writes `orphaned` (spec §5.4); its
                    // cleanup clears the compaction status and ends the run.
                    crate::types::TaskOutcomeStatus::Orphaned => {
                        crate::harness::live::SchedulerFault::Orphaned {
                            reason: outcome
                                .reason
                                .clone()
                                .unwrap_or_else(|| "orphaned".to_string()),
                        }
                    }
                    _ => return Ok(()),
                };
                settle_scheduler_outcome(tx, record, &fault)
            }),
            withdraw_inputs: Arc::new(withdraw_queued_inputs),
            conversation: {
                let session = session.clone();
                let harness_slot = harness_slot.clone();
                Arc::new(move |id, binding, call_context| {
                    let session = session.clone();
                    let harness_slot = harness_slot.clone();
                    let call_context = call_context.clone();
                    Box::pin(async move {
                        let record = session
                            .read_on_line_tx(
                                |tx| tx.conversation(id).map_err(|e| e.to_string()),
                                &call_context,
                            )
                            .await
                            .map_err(|e| e.to_string())?;
                        if record.is_none() {
                            return Ok(None);
                        }
                        let harness = { harness_slot.lock().clone() };
                        let Some(harness) = harness else {
                            return Err(closed_error());
                        };
                        Ok(Some(Arc::new(bound_conversation(
                            id,
                            binding,
                            harness.clone(),
                            harness.tasks.clone(),
                        ))
                            as Arc<dyn crate::session::ConversationHandle>))
                    })
                })
            },
            context: context.clone(),
        });
        let scheduler_for_close = scheduler.clone();
        session.install_hooks(SessionHooks {
            before_close: Some(Arc::new(move || {
                let scheduler = scheduler_for_close.clone();
                Box::pin(async move { scheduler.join().await })
                    as Pin<Box<dyn Future<Output = ()> + Send>>
            })),
            conversation_created: Some(Box::new(move |tx, record| {
                let _ = tx.doc(
                    &crate::harness::live::live_doc(),
                    Some(record.id.0),
                    None,
                    None,
                )?;
                let _ = tx.doc(
                    &crate::harness::inbox::inbox_doc(),
                    Some(record.id.0),
                    None,
                    None,
                )?;
                let _ = tx.doc(&usage_doc(), Some(record.id.0), None, None)?;
                let _ = tx.doc(
                    &crate::harness::provider::provider_doc(),
                    Some(record.id.0),
                    None,
                    None,
                )?;
                crate::harness::agent::create_agent(tx, record).map_err(TxError::message)?;
                if let Some(hook) = &conversation_created {
                    hook(tx, record).map_err(TxError::message)?;
                }
                Ok(())
            })),
        });
        let submissions = Submissions::new(
            session.clone(),
            storage.clone(),
            Box::new({
                let now = now.clone();
                move || now()
            }),
            Box::new({
                let settings = settings.clone();
                move || {
                    let settings = resolve_settings(Some(&settings()));
                    crate::harness::inbox::QueueModes {
                        steering_mode: settings.steering_mode,
                        follow_up_mode: settings.follow_up_mode,
                    }
                }
            }),
            {
                let scheduler = scheduler.clone();
                Box::new(move || scheduler.resume())
            },
        );
        let views = ConversationViews::new(session.clone(), storage.clone());
        let task_graph = TaskGraphView::new(session.clone(), storage.clone());
        let harness = Arc::new(Self {
            session: session.clone(),
            storage,
            report: report.clone(),
            now,
            settings,
            tasks: scheduler,
            submissions,
            views,
            task_graph,
            agent_resolutions: Arc::default(),
            closed: Arc::new(Mutex::new(false)),
            options,
        });
        *harness_slot.lock() = Some(harness.clone());
        harness
    }

    /// Resolve a conversation's committed `pi.agent` against `snapshot`, or
    /// the current one, and the current settings.
    pub async fn resolve_agent(
        &self,
        id: ConversationId,
        snapshot: Option<Arc<dyn crate::harness::types::RegistrySnapshot>>,
        context: &Context,
    ) -> Result<Agent, String> {
        let registry = snapshot.unwrap_or_else(|| self.options.registry.snapshot());
        let state = self
            .session
            .snapshot(
                &crate::harness::agent::agent_doc(),
                Some(id.0),
                None,
                context,
            )
            .await
            .map_err(|e| e.to_string())?;
        let state: Option<crate::harness::types::AgentState> = state.map(|value| {
            serde_json::from_value(serde_json::Value::Object((*value).clone())).unwrap_or_default()
        });
        Ok(resolve_agent(
            state.as_ref(),
            registry.as_ref(),
            &resolve_settings(Some(&(self.settings)())),
            self.report.as_ref(),
        ))
    }

    /// Build a conversation's environment from its current `cwd`; `None`
    /// without an `env` option.
    pub async fn build_env(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Result<Option<Arc<dyn crate::env::ExecutionEnv>>, String> {
        let Some(build) = &self.options.env else {
            return Ok(None);
        };
        let state = self
            .session
            .snapshot(
                &crate::harness::agent::agent_doc(),
                Some(id.0),
                None,
                context,
            )
            .await
            .map_err(|e| e.to_string())?;
        let cwd = state
            .as_ref()
            .and_then(|value| value.get("cwd"))
            .and_then(|cwd| cwd.as_str())
            .map(|cwd| cwd.to_string());
        build(
            crate::harness::types::EnvTarget {
                conversation_id: id,
                cwd,
                read: Arc::new(self.session.clone()),
            },
            context.clone(),
        )
        .await
    }

    /// Reconcile surviving `running` tasks to `pending`; part of open.
    pub async fn open_tasks(&self, context: &Context) -> Result<(), String> {
        self.tasks.open(context).await
    }

    pub fn resume(&self) {
        self.assert_open().unwrap_or_else(|error| panic!("{error}"));
        self.tasks.resume();
    }

    pub async fn get_task(
        &self,
        id: TaskId,
        context: &Context,
    ) -> Result<Option<crate::types::TaskRecord>, String> {
        self.session
            .read_on_line_tx(|tx| tx.task(id).map_err(|e| e.to_string()), context)
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn inspect(&self, context: &Context) -> Result<HarnessInspection, String> {
        let scheduler = self.tasks.clone();
        let storage = self.storage.clone();
        let context = context.clone();
        let (scheduling, tasks) = scheduler.inspect_snapshot().await?;
        let mut submissions = scan_all(|cursor| {
            let storage = storage.clone();
            let context = context.clone();
            let query = crate::types::SubmissionQuery {
                conversation_id: None,
                status: Some(SubmissionStatus::Queued),
                order: None,
            };
            async move {
                storage
                    .scan_submissions(&query, SCAN_PAGE_SIZE, cursor.as_ref(), &context)
                    .await
            }
        })
        .await;
        submissions.extend(
            scan_all(|cursor| {
                let storage = storage.clone();
                let context = context.clone();
                let query = crate::types::SubmissionQuery {
                    conversation_id: None,
                    status: Some(SubmissionStatus::Placed),
                    order: None,
                };
                async move {
                    storage
                        .scan_submissions(&query, SCAN_PAGE_SIZE, cursor.as_ref(), &context)
                        .await
                }
            })
            .await,
        );
        submissions.sort_by_key(|submission| submission.id);
        Ok(HarnessInspection {
            scheduling,
            tasks,
            submissions,
        })
    }

    pub async fn submission(
        self: &Arc<Self>,
        id: crate::harness::types::SubmissionId,
        context: &Context,
    ) -> Result<Option<crate::harness::submissions::SubmissionHandle>, String> {
        self.submissions.get(id, context).await
    }

    pub async fn abort_submission(
        self: &Arc<Self>,
        id: crate::harness::types::SubmissionId,
        context: &Context,
        conversation_id: Option<ConversationId>,
    ) -> Result<Option<SubmissionAbort>, String> {
        self.submissions.abort(id, context, conversation_id).await
    }

    pub async fn abort_task(&self, id: TaskId, context: &Context) -> Result<&'static str, String> {
        self.tasks.abort(id, context).await
    }

    pub async fn wait_for_task(
        self: &Arc<Self>,
        id: TaskId,
        context: &Context,
    ) -> Result<SettledTask, String> {
        self.tasks.resume();
        self.tasks.wait_for_task(id, context).await
    }

    pub async fn wait_for_idle(&self, context: &Context) -> Result<(), String> {
        self.tasks.resume();
        self.tasks.wait_for_idle(None, context).await
    }

    /// Sum every conversation's committed `pi.usage`. Each document is read
    /// at its own point; totals only grow.
    pub async fn usage(&self, context: &Context) -> Result<UsageState, String> {
        let storage = self.storage.clone();
        let conversations = self
            .session
            .read_on_line(async move {
                Ok(scan_all(|cursor| {
                    let storage = storage.clone();
                    let context = context.clone();
                    let query = crate::types::ConversationQuery::default();
                    async move {
                        storage
                            .scan_conversations(&query, SCAN_PAGE_SIZE, cursor.as_ref(), &context)
                            .await
                    }
                })
                .await)
            })
            .await?;
        let mut total = UsageState::default();
        for conversation in conversations {
            let state = self
                .session
                .snapshot(&usage_doc(), Some(conversation.id.0), None, context)
                .await
                .map_err(|e| e.to_string())?;
            if let Some(state) = state {
                if let Ok(state) = serde_json::from_value::<UsageState>(serde_json::Value::Object(
                    (*state).clone(),
                )) {
                    add_usage_state(&mut total, &state);
                }
            }
        }
        Ok(total)
    }

    pub async fn root(
        self: &Arc<Self>,
        context: &Context,
        options: Option<RootConversationOptions>,
    ) -> Result<Conversation, String> {
        let options = options.map(|options| CreateOptions {
            agent: options.agent,
            init: options.init,
        });
        self.create(CreateTarget::Root, options.unwrap_or_default(), context)
            .await
    }

    pub async fn conversation(
        self: &Arc<Self>,
        id: ConversationId,
        context: &Context,
    ) -> Result<Option<Conversation>, String> {
        self.assert_open()?;
        let record = self
            .session
            .read_on_line_tx(|tx| tx.conversation(id).map_err(|e| e.to_string()), context)
            .await
            .map_err(|e| e.to_string())?;
        Ok(record.map(|record| Conversation {
            id: record.id,
            harness: self.clone(),
        }))
    }

    pub async fn create_conversation(
        self: &Arc<Self>,
        options: ConversationCreateOptions,
        context: &Context,
    ) -> Result<Conversation, String> {
        self.create(
            CreateTarget::Independent {
                ownership: options.ownership,
            },
            CreateOptions {
                agent: options.agent,
                init: options.init,
            },
            context,
        )
        .await
    }

    pub async fn close(&self, context: &Context) -> Result<(), String> {
        *self.closed.lock() = true;
        self.session.close(context).await.map_err(|e| e.to_string())
    }

    async fn create(
        self: &Arc<Self>,
        target: CreateTarget,
        options: CreateOptions,
        context: &Context,
    ) -> Result<Conversation, String> {
        self.assert_open()?;
        let id = self
            .session
            .commit(
                move |tx| {
                    if let CreateTarget::Root = target {
                        if tx
                            .conversation(ROOT_CONVERSATION_ID)
                            .map_err(|e| e.to_string())?
                            .is_some()
                        {
                            return Ok(ROOT_CONVERSATION_ID);
                        }
                    }
                    let record = match target {
                        CreateTarget::Root => {
                            tx.create_root_conversation().map_err(|e| e.to_string())?
                        }
                        CreateTarget::Fork {
                            parent_id,
                            at,
                            ownership,
                        } => tx
                            .fork_conversation(parent_id, at, ownership)
                            .map_err(|e| e.to_string())?,
                        CreateTarget::Independent { ownership } => tx
                            .create_conversation(ownership)
                            .map_err(|e| e.to_string())?,
                    };
                    if let Some(agent) = &options.agent {
                        configure(tx, record.id, agent)?;
                    }
                    if let Some(init) = &options.init {
                        init(tx, record.id)?;
                    }
                    Ok(record.id)
                },
                context,
            )
            .await
            .map_err(|e| e.to_string())?;
        Ok(Conversation {
            id,
            harness: self.clone(),
        })
    }

    fn assert_open(&self) -> Result<(), String> {
        if *self.closed.lock() {
            Err(closed_error())
        } else {
            Ok(())
        }
    }
}

impl Conversation {
    pub async fn agent(&self, context: &Context) -> Result<Agent, String> {
        self.harness.resolve_agent(self.id, None, context).await
    }

    pub async fn configure(&self, change: AgentChange, context: &Context) -> Result<(), String> {
        let id = self.id;
        self.harness
            .session
            .commit(
                move |tx| crate::harness::agent::configure(tx, id, &change),
                context,
            )
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn submit(
        &self,
        submission: SubmissionDraft,
        context: &Context,
    ) -> Result<crate::harness::submissions::SubmissionHandle, String> {
        self.harness
            .submissions
            .submit(self.id, submission, context)
            .await
    }

    pub async fn compact(
        &self,
        instructions: Option<String>,
        context: &Context,
    ) -> Result<TaskId, String> {
        self.harness.tasks.resume();
        let input = CompactionInput {
            reason: CompactionReason::Manual,
            instructions,
        };
        let conversation_id = self.id;
        let id = self
            .harness
            .session
            .commit(
                move |tx| create_compaction(tx, conversation_id, input, None),
                context,
            )
            .await
            .map_err(|e| e.to_string())?;
        // The creation publication can land inside a drain pass whose reserve
        // already ran; kick again so the new task dispatches immediately.
        self.harness.tasks.resume();
        Ok(id)
    }

    pub async fn reset(&self, handoff: Option<String>, context: &Context) -> Result<(), String> {
        let mut entry = EntryDraft {
            kind: Some(RESET_ENTRY.kind.clone()),
            ..Default::default()
        };
        entry.head = Some(EntryHead::SelfMarker(crate::types::SelfMarker));
        if let Some(handoff) = handoff {
            entry.model = Some(vec![pi_ai::Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContentField::Text(handoff),
                timestamp: (self.harness.now)(),
            })]);
        }
        self.submit(
            SubmissionDraft::Write(crate::harness::types::WriteSubmissionDraft {
                request_id: None,
                entry,
            }),
            context,
        )
        .await
        .map(|_| ())
    }

    pub async fn commit<T>(
        &self,
        change: impl FnOnce(&mut crate::session::Tx<'_>) -> Result<T, String> + Send + 'static,
        context: &Context,
    ) -> Result<T, String>
    where
        T: Send + 'static,
    {
        self.harness
            .session
            .commit_with(
                change,
                context,
                TransactionScope {
                    conversation_id: Some(self.id),
                    task_id: None,
                },
            )
            .await
            .map_err(|e| e.to_string())
    }

    /// Committed raw active transcript and model context. With `at`, the
    /// context as of that visible entry: the same view `fork(at)` would start
    /// with, without creating a conversation.
    pub async fn context(
        &self,
        context: &Context,
        at: Option<crate::harness::types::EntryId>,
    ) -> Result<ContextView, String> {
        let bounds = crate::harness::context::capture_context_bounds(
            self.harness.storage.as_ref(),
            self.id,
            context,
            at,
        )
        .await?;
        crate::harness::context::derive_context(
            self.harness.storage.as_ref(),
            self.id,
            bounds,
            context,
        )
        .await
    }

    /// Fork-aware history of this conversation, newest first unless
    /// `query.order` is ascending.
    pub async fn entries(
        &self,
        query: crate::harness::types::ConversationEntryQuery,
        limit: u64,
        cursor: Option<&crate::types::Cursor>,
        context: &Context,
    ) -> Result<Page<EntryRecord>, String> {
        let bounded = EntryQuery {
            conversation_id: self.id,
            min_entry_id: query.min_entry_id,
            max_entry_id: query.max_entry_id,
            order: query.order,
        };
        self.harness
            .session
            .read_on_line_tx(
                |tx| {
                    tx.scan_entries(&bounded, limit, cursor)
                        .map_err(|e| e.to_string())
                },
                context,
            )
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn fork(
        &self,
        at: crate::harness::types::EntryId,
        options: ConversationCreateOptions,
        context: &Context,
    ) -> Result<Conversation, String> {
        self.harness
            .create(
                CreateTarget::Fork {
                    parent_id: self.id,
                    at,
                    ownership: options.ownership,
                },
                CreateOptions {
                    agent: options.agent,
                    init: options.init,
                },
                context,
            )
            .await
    }

    pub async fn abort(
        &self,
        context: &Context,
        options: Option<ConversationAbortOptions>,
    ) -> Result<(), String> {
        self.harness.tasks.resume();
        self.harness
            .tasks
            .abort_conversation(
                self.id,
                options
                    .and_then(|options| options.background)
                    .unwrap_or(false),
                context,
            )
            .await
    }

    pub async fn wait_for_idle(&self, context: &Context) -> Result<(), String> {
        self.harness.tasks.resume();
        self.harness
            .tasks
            .wait_for_idle(Some(self.id), context)
            .await
    }

    pub async fn view_state(
        &self,
        context: &Context,
    ) -> Result<crate::session::observation::DocumentStateView, String> {
        self.harness.views.state(self.id, context).await
    }

    pub async fn watch(&self, context: &Context) -> Result<ConversationWatch, String> {
        self.harness.views.watch(self.id, context).await
    }
}

/// Invocation-bound handle for tasks and tools (upstream `boundConversation`).
fn bound_conversation(
    id: ConversationId,
    binding: InvocationBinding,
    harness: Arc<Harness>,
    tasks: Arc<TaskScheduler>,
) -> impl ConversationHandle {
    BoundConversation {
        id,
        binding,
        harness,
        tasks,
    }
}

struct BoundConversation {
    id: ConversationId,
    binding: InvocationBinding,
    harness: Arc<Harness>,
    tasks: Arc<TaskScheduler>,
}

#[async_trait::async_trait]
impl ConversationHandle for BoundConversation {
    fn id(&self) -> ConversationId {
        self.id
    }

    async fn submit(
        &self,
        submission: InputSubmissionDraft,
        context: &Context,
    ) -> Result<Arc<dyn crate::harness::types::Submission>, String> {
        self.binding.check()?;
        let bound =
            pi_chord::context::with_abort_signal(self.binding.signal.clone(), context.clone());
        let submission = self
            .harness
            .submissions
            .submit(self.id, SubmissionDraft::Input(submission), &bound)
            .await?;
        Ok(Arc::new(BoundSubmission {
            id: submission.id,
            submissions: self.harness.submissions.clone(),
            signal: self.binding.signal.clone(),
            check: self.binding.check_fn(),
        }) as Arc<dyn crate::harness::types::Submission>)
    }

    async fn abort(
        &self,
        context: &Context,
        options: Option<ConversationAbortOptions>,
    ) -> Result<(), String> {
        self.binding.check()?;
        let bound =
            pi_chord::context::with_abort_signal(self.binding.signal.clone(), context.clone());
        self.tasks
            .abort_conversation(
                self.id,
                options
                    .and_then(|options| options.background)
                    .unwrap_or(false),
                &bound,
            )
            .await
    }

    async fn wait_for_idle(&self, context: &Context) -> Result<(), String> {
        self.binding.check()?;
        let bound =
            pi_chord::context::with_abort_signal(self.binding.signal.clone(), context.clone());
        self.tasks.wait_for_idle(Some(self.id), &bound).await
    }
}

/// The bound submission returned by `ConversationHandle::submit`.
struct BoundSubmission {
    id: crate::harness::types::SubmissionId,
    submissions: Arc<Submissions>,
    signal: pi_chord::context::AbortSignal,
    check: Arc<dyn Fn() -> Result<(), String> + Send + Sync>,
}

#[async_trait::async_trait]
impl crate::harness::types::Submission for BoundSubmission {
    fn id(&self) -> crate::harness::types::SubmissionId {
        self.id
    }

    async fn status(&self, context: &Context) -> Result<SubmissionRecord, String> {
        (self.check)()?;
        let bound = pi_chord::context::with_abort_signal(self.signal.clone(), context.clone());
        self.submissions.status(self.id, &bound).await
    }

    async fn wait(
        &self,
        context: &Context,
    ) -> Result<crate::harness::types::SettledSubmissionRecord, String> {
        (self.check)()?;
        let bound = pi_chord::context::with_abort_signal(self.signal.clone(), context.clone());
        self.submissions.wait(self.id, &bound).await
    }

    async fn abort(&self, context: &Context) -> Result<SubmissionAbort, String> {
        (self.check)()?;
        let bound = pi_chord::context::with_abort_signal(self.signal.clone(), context.clone());
        self.submissions
            .abort(self.id, &bound, None)
            .await?
            .ok_or_else(|| format!("Submission {} does not exist", self.id))
    }
}
