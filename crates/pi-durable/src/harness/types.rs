//! Port of `packages/durable/src/harness/types.ts` (all harness types) on the
//! erased runtime forms of `crate::types` / `crate::tasks`.
//!
//! Erasure decisions (D1, per slice assignment): upstream `Tool`/`Task`
//! generics collapse to `serde_json::Value`-based shapes — tool arguments and
//! details are `Value` (see the integrator-owned `api.rs` seam, re-exported
//! below); phase maps are `BTreeMap<String, PhaseHandler>`; hooks become
//! erased named-handler tables (`HookRegistration` carries an
//! `Arc<dyn HookHandler>`; the built-in hook surfaces are concrete structs of
//! optional `Arc` closures — D2). `Conversation`, `Harness`, and the registry
//! interfaces become concrete types / object-safe traits as noted per type.
//! `TaskId<R>` branding is erased (`TaskId`); `SettledTask`,
//! `ContextView`, `ConversationHandle`, and `ConversationAbortOptions` live
//! in [`crate::session`] and are re-exported here (integrator decision).
//!
//! Deviations: D3 — `commit` conveniences on `Conversation`/`Harness` are
//! erased to `Value` results (upstream generic `commit<T>`). D4 — hook
//! handler payloads/results are structs (`OnYieldResult.continue_` for the
//! `continue` keyword). D5 — `HarnessOptions.on_report` receives the error
//! text (`String`), not an arbitrary thrown value.

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::types::{Message, ModelThinkingLevel, Transport, UserContentField};
use pi_chord::context::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::documents::DocToken;
use crate::env::ExecutionEnv;
use crate::session::Tx;
use crate::tasks::{BoxFuture, HookHandler};
pub use crate::types::{
    ConversationId, ConversationOwnership, ConversationRecord, Cursor, EntryDraft, EntryHead,
    EntryId, EntryQuery, EntryRecord, JoinPolicy, JsonObject, Page, SelfMarker, Seq, SubmissionId,
    SubmissionRecord, SubmissionStatus, SubmissionType, TaskId, TaskOutcome, TaskRecord,
    WatchHandle,
};

pub use crate::harness::api::{
    define_tool, OutputRetain, QueueMode, ToolControl, ToolDiagnostic, ToolDiagnosticSeverity,
    ToolExecute, ToolExecutionApi, ToolExecutionMode, ToolExecutionResult, ToolOutputLimits,
    ToolRegistration, ToolReplay,
};
pub use crate::session::{ContextView, ConversationAbortOptions, ConversationHandle, SettledTask};

/// Provider and model ID resolved through pi-ai `Models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef {
    pub provider: String,
    pub model_id: String,
}

/// Host submission content: user input that may start a run.
pub type UserInput = UserContentField;

/// Busy behavior of an input submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WhenBusy {
    Steer,
    FollowUp,
    Reject,
}

/// Host submission: user input that may start a run (upstream `type:
/// "input"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InputSubmissionDraft {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub content: UserInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub when_busy: Option<WhenBusy>,
}

/// Host submission: a passive entry write (upstream `type: "write"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WriteSubmissionDraft {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub entry: EntryDraft,
}

/// Host submission: user input that may start a run, or a passive entry write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SubmissionDraft {
    #[serde(rename = "input")]
    Input(InputSubmissionDraft),
    #[serde(rename = "write")]
    Write(WriteSubmissionDraft),
}

impl SubmissionDraft {
    /// `"input"` or `"write"`.
    pub fn submission_type(&self) -> &'static str {
        match self {
            SubmissionDraft::Input(_) => "input",
            SubmissionDraft::Write(_) => "write",
        }
    }

    /// The draft's deduplication key, whichever variant it is.
    pub fn request_id(&self) -> Option<&str> {
        match self {
            SubmissionDraft::Input(draft) => draft.request_id.as_deref(),
            SubmissionDraft::Write(draft) => draft.request_id.as_deref(),
        }
    }
}

/// `SubmissionRecord` whose status is settled (`done` or `unanswered`); the
/// erased upstream narrowing is contractual, the record is the same shape.
pub type SettledSubmissionRecord = SubmissionRecord;

/// Whether a submission record has settled.
pub fn is_settled(record: &SubmissionRecord) -> bool {
    matches!(
        record.status,
        crate::types::SubmissionStatus::Done | crate::types::SubmissionStatus::Unanswered
    )
}

/// Result of aborting a submission (upstream literal union).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmissionAbort {
    Aborted,
    AlreadyPlaced,
    Settled,
}

impl SubmissionAbort {
    /// Upstream discriminator string.
    pub fn as_str(&self) -> &'static str {
        match self {
            SubmissionAbort::Aborted => "aborted",
            SubmissionAbort::AlreadyPlaced => "already_placed",
            SubmissionAbort::Settled => "settled",
        }
    }
}

/// Result of `Harness::abort_submission`, which also reports `not_found`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortSubmissionResult {
    Abort(SubmissionAbort),
    NotFound,
}

/// Result of `Harness::abort_task`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbortTaskResult {
    Marked,
    Terminal,
}

/// Awaitable host object for one durably admitted submission.
#[async_trait::async_trait]
pub trait Submission: Send + Sync {
    /// The admitted submission's ID.
    fn id(&self) -> SubmissionId;
    /// The submission's current record.
    async fn status(&self, context: &Context) -> Result<SubmissionRecord, String>;
    /// Resolve with the settled record.
    async fn wait(&self, context: &Context) -> Result<SettledSubmissionRecord, String>;
    /// Withdraw a queued submission; placed inputs and settled submissions
    /// are reported.
    async fn abort(&self, context: &Context) -> Result<SubmissionAbort, String>;
}

/// Erased executable task definition stored in the registry (upstream
/// `AnyTask = { definition }`).
#[derive(Clone, Debug)]
pub struct AnyTask {
    pub definition: Arc<crate::tasks::TaskDefinition>,
}

impl AnyTask {
    /// Wrap a definition.
    pub fn new(definition: crate::tasks::TaskDefinition) -> Self {
        Self {
            definition: Arc::new(definition),
        }
    }

    /// Wrap an already-shared definition.
    pub fn shared(definition: Arc<crate::tasks::TaskDefinition>) -> Self {
        Self { definition }
    }

    /// The registered task kind.
    pub fn name(&self) -> &str {
        &self.definition.name
    }
}

impl std::ops::Deref for AnyTask {
    type Target = crate::tasks::TaskDefinition;

    fn deref(&self) -> &Self::Target {
        &self.definition
    }
}

/// Why a pending task cannot be reserved under a registry snapshot. Derived,
/// never persisted (upstream `BlockedReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockedReason {
    MissingTask,
    TaskTooOld,
    MigrationFailed,
}

impl BlockedReason {
    /// Upstream discriminator string.
    pub fn as_str(&self) -> &'static str {
        match self {
            BlockedReason::MissingTask => "missing_task",
            BlockedReason::TaskTooOld => "task_too_old",
            BlockedReason::MigrationFailed => "migration_failed",
        }
    }
}

/// Live task and what the scheduler would do with it under the current
/// registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskInspection {
    pub record: TaskRecord,
    pub state: TaskInspectionState,
}

/// Derived scheduling state of one live task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum TaskInspectionState {
    /// An invocation is active.
    Running,
    /// The next scheduling pass reserves it; `migrates` when its definition
    /// is newer and has `migrate`.
    Ready { migrates: bool },
    /// Waits for these live tasks: the live part of its `on`, or, when
    /// abort-marked, its live ordinary owned work.
    Waiting { on: Vec<TaskId> },
    /// Outcome held until its ordinary owned work drains.
    Completing,
    /// No registered definition can take it; aborting it settles it as
    /// `orphaned`.
    Blocked {
        reason: BlockedReason,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

/// Scheduling state of a Harness (upstream literal union).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulingState {
    Paused,
    Running,
    Closing,
}

/// Point-in-time view of live work: unfinished tasks and submissions, read on
/// the Session line.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessInspection {
    pub scheduling: SchedulingState,
    pub tasks: Vec<TaskInspection>,
    /// Queued and placed submissions, in ID order.
    pub submissions: Vec<SubmissionRecord>,
}

/// Committed document reads (upstream `DocumentReader`); the Session
/// implements it and every prompt/environment target reads through it.
#[async_trait::async_trait]
pub trait DocumentReader: Send + Sync {
    /// Committed state of one document at its address.
    async fn snapshot(
        &self,
        token: &DocToken,
        owner: Option<u64>,
        key: Option<&str>,
        context: &Context,
    ) -> Result<Option<Arc<JsonObject>>, String>;
    /// Rewound state of a conversation document as of an entry.
    async fn snapshot_as_of(
        &self,
        token: &DocToken,
        conversation_id: ConversationId,
        key: Option<&str>,
        at: EntryId,
        context: &Context,
    ) -> Result<Option<JsonObject>, String>;
}

/// Input to system prompt section rendering for one request preparation.
#[derive(Clone)]
pub struct PromptInput {
    pub conversation_id: ConversationId,
    /// The request's resolution; `agent.tools` are the tools offered in this
    /// request.
    pub agent: Agent,
    /// Built by `HarnessOptions.env` for this preparation; `None` without an
    /// environment.
    pub env: Option<Arc<dyn ExecutionEnv>>,
    /// Sections already in effect after replaying the active transcript.
    /// Sections already in effect (values are `Value::String`).
    pub shown: serde_json::Map<String, Value>,
    /// Committed document reads.
    pub read: Arc<dyn DocumentReader>,
}

/// Section render of one request preparation (upstream `PromptSection.render`).
pub type SectionRender =
    Arc<dyn Fn(PromptInput, Context) -> BoxFuture<Result<Option<String>, String>> + Send + Sync>;

/// One system prompt section; the agent's sections render in order before
/// each request.
#[derive(Clone)]
pub struct PromptSection {
    pub key: String,
    pub render: SectionRender,
    /// Default true: wrap the text as `<key>\n...\n</key>`.
    pub tag: Option<bool>,
}

impl std::fmt::Debug for PromptSection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptSection")
            .field("key", &self.key)
            .field("tag", &self.tag)
            .finish_non_exhaustive()
    }
}

/// Built by `hook()`; matches tasks by name.
#[derive(Clone)]
pub struct HookRegistration {
    pub task: String,
    pub handlers: Arc<dyn HookHandler>,
}

impl std::fmt::Debug for HookRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookRegistration")
            .field("task", &self.task)
            .finish_non_exhaustive()
    }
}

/// Wrap of one tool (upstream `{ tool, wrap }`).
#[derive(Clone)]
pub struct ToolWrap {
    pub tool: String,
    pub wrap: Arc<dyn Fn(Arc<ToolRegistration>) -> Arc<ToolRegistration> + Send + Sync>,
}

/// Wrap of one section (upstream `{ section, wrap }`).
#[derive(Clone)]
pub struct SectionWrap {
    pub section: String,
    pub wrap: Arc<dyn Fn(PromptSection) -> PromptSection + Send + Sync>,
}

/// Built by `wrap_tool()` and `wrap_section()`; targets a tool name or a
/// section key. Wrappers are pure.
#[derive(Clone)]
pub enum Wrap {
    Tool(ToolWrap),
    Section(SectionWrap),
}

impl std::fmt::Debug for Wrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Wrap::Tool(wrap) => f
                .debug_struct("Wrap::Tool")
                .field("tool", &wrap.tool)
                .finish_non_exhaustive(),
            Wrap::Section(wrap) => f
                .debug_struct("Wrap::Section")
                .field("section", &wrap.section)
                .finish_non_exhaustive(),
        }
    }
}

/// Named bundle of code; installed in a registry and selected by conversations
/// by name.
#[derive(Clone, Default)]
pub struct Extension {
    pub name: String,
    pub tools: Vec<Arc<ToolRegistration>>,
    pub sections: Vec<PromptSection>,
    pub hooks: Vec<HookRegistration>,
    /// Applied where this extension is selected, in order.
    pub wraps: Vec<Wrap>,
    /// Resolved by name for every task, whichever conversations select this
    /// extension.
    pub tasks: Vec<AnyTask>,
}

impl std::fmt::Debug for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extension")
            .field("name", &self.name)
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.tool.name.clone())
                    .collect::<Vec<_>>(),
            )
            .field(
                "sections",
                &self
                    .sections
                    .iter()
                    .map(|section| section.key.clone())
                    .collect::<Vec<_>>(),
            )
            .field(
                "tasks",
                &self
                    .tasks
                    .iter()
                    .map(|task| task.name().to_string())
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl Extension {
    /// An extension with only a name; builders attach members.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Attach tools (builder).
    pub fn with_tools(mut self, tools: Vec<Arc<ToolRegistration>>) -> Self {
        self.tools = tools;
        self
    }

    /// Attach sections (builder).
    pub fn with_sections(mut self, sections: Vec<PromptSection>) -> Self {
        self.sections = sections;
        self
    }

    /// Attach hooks (builder).
    pub fn with_hooks(mut self, hooks: Vec<HookRegistration>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Attach wraps (builder).
    pub fn with_wraps(mut self, wraps: Vec<Wrap>) -> Self {
        self.wraps = wraps;
        self
    }

    /// Attach tasks (builder).
    pub fn with_tasks(mut self, tasks: Vec<AnyTask>) -> Self {
        self.tasks = tasks;
        self
    }
}

/// Immutable view of one published registry state (upstream interface; Rust
/// object-safe trait, implemented by `harness::registry::RegistryState`).
pub trait RegistrySnapshot: Send + Sync {
    /// Every installed extension, in install order.
    fn installed(&self) -> Vec<Arc<Extension>>;
    /// One installed extension by name.
    fn extension(&self, name: &str) -> Option<Arc<Extension>>;
    /// Every installed tool with its extension, in install order. Names may
    /// repeat across extensions.
    fn tools(&self) -> Vec<(Arc<Extension>, Arc<ToolRegistration>)>;
    /// Every installed section with its extension, in install order.
    fn sections(&self) -> Vec<(Arc<Extension>, PromptSection)>;
    /// Built-in and installed task definitions.
    fn tasks(&self) -> Vec<AnyTask>;
    /// One task definition by name.
    fn task(&self, name: &str) -> Option<AnyTask>;
}

/// Registry change listener (upstream `() => void`).
pub type RegistryListener = Arc<dyn Fn() + Send + Sync>;
/// Unsubscribe handle (upstream `() => void` disposer).
pub type Unsubscribe = Box<dyn FnOnce() + Send>;

/// Read side of a registry consumed by a Harness.
pub trait RegistryReader: Send + Sync {
    /// Immutable view of the whole current registry.
    fn snapshot(&self) -> Arc<dyn RegistrySnapshot>;
    /// Called synchronously after every publication; wakes the scheduler to
    /// reconsider blocked tasks.
    fn subscribe(&self, listener: RegistryListener) -> Unsubscribe;
}

/// Application-owned registry of extensions (implemented by
/// `harness::registry::RegistryImpl`).
pub trait Registry: RegistryReader {
    /// Install `extension`, or replace the installed extension with its name
    /// in place. Publishes at once.
    fn install(&self, extension: Extension);
    /// Remove the installed extension with `extension.name`, whichever object
    /// it is. A later install appends.
    fn uninstall(&self, extension: &Extension);
}

/// Selected or edited extension list of a stored agent (upstream
/// `string[] | { add?, remove? }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExtensionSelection {
    /// Selects exactly these extensions, in order.
    List(Vec<String>),
    /// Edits the host default selection.
    Edit {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        add: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        remove: Option<Vec<String>>,
    },
}

/// Tool filter of a stored agent (upstream `string[] | { remove }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolsSelection {
    /// Offers exactly these, in order.
    List(Vec<String>),
    /// Filters the selected extensions' tools.
    Remove { remove: Vec<String> },
}

/// Stored choices of one conversation; names, not objects. Unset fields
/// follow the host (upstream `AgentState`, the value of the `pi.agent` doc).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ModelThinkingLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<ExtensionSelection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsSelection>,
    /// Rendered after every extension section, as the section `instructions`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Directory within the environment's file system, passed to
    /// `HarnessOptions.env`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

impl AgentState {
    /// Decode a stored `pi.agent` value; unknown shapes decode to the empty
    /// state (upstream documents are trusted JSON objects).
    pub fn from_json(value: &JsonObject) -> Self {
        serde_json::from_value(Value::Object(value.clone())).unwrap_or_default()
    }

    /// Encode for document storage.
    pub fn to_json(&self) -> JsonObject {
        match serde_json::to_value(self) {
            Ok(Value::Object(map)) => map,
            _ => serde_json::Map::new(),
        }
    }
}

/// One field of an [`AgentChange`]: a given value replaces the stored one,
/// [`SetOrClear::Clear`] removes it, `None` changes nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum SetOrClear<T> {
    Set(T),
    Clear,
}

/// Extension list of an [`AgentChange`] (upstream `readonly Extension[] | {
/// add?, remove? } | null`).
#[derive(Debug, Clone)]
pub enum ExtensionsChange {
    List(Vec<Extension>),
    Edit {
        add: Vec<Extension>,
        remove: Vec<Extension>,
    },
}

/// Tool list of an [`AgentChange`] (upstream `readonly ToolRegistration[] | {
/// remove } | null`).
#[derive(Debug, Clone)]
pub enum ToolsChange {
    List(Vec<ToolRegistration>),
    Remove(Vec<ToolRegistration>),
}

/// A change to `pi.agent` (upstream `AgentChange`); every field is `None`
/// (unchanged), [`SetOrClear::Set`] (replace), or [`SetOrClear::Clear`].
#[derive(Debug, Clone, Default)]
pub struct AgentChange {
    pub model: Option<SetOrClear<ModelRef>>,
    pub thinking_level: Option<SetOrClear<ModelThinkingLevel>>,
    pub extensions: Option<SetOrClear<ExtensionsChange>>,
    pub tools: Option<SetOrClear<ToolsChange>>,
    pub instructions: Option<SetOrClear<String>>,
    pub cwd: Option<SetOrClear<String>>,
}

/// A conversation's agent resolved against a registry snapshot and the
/// settings.
#[derive(Clone)]
pub struct Agent {
    pub model: Option<ModelRef>,
    pub thinking_level: ModelThinkingLevel,
    pub extensions: Vec<Arc<Extension>>,
    /// The tools a request offers, in order (upstream `Tool` registrations;
    /// the pi-ai declaration fields travel in `ToolRegistration::tool`).
    pub tools: Vec<Arc<ToolRegistration>>,
    /// Extension sections, then `instructions` when set.
    pub sections: Vec<PromptSection>,
    pub instructions: Option<String>,
    pub cwd: Option<String>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field(
                "extensions",
                &self
                    .extensions
                    .iter()
                    .map(|extension| extension.name.clone())
                    .collect::<Vec<_>>(),
            )
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.tool.name.clone())
                    .collect::<Vec<_>>(),
            )
            .field(
                "sections",
                &self
                    .sections
                    .iter()
                    .map(|section| section.key.clone())
                    .collect::<Vec<_>>(),
            )
            .field("instructions", &self.instructions)
            .field("cwd", &self.cwd)
            .finish()
    }
}

/// Runs inside the creating commit, after the creation hook and the `agent`
/// change (upstream `ConversationInit`; sync because `Tx` operations are sync
/// on the Session line — see `crate::tasks` D3).
pub type ConversationInit = Arc<dyn Fn(&Tx, ConversationId) -> Result<(), String> + Send + Sync>;

/// Options of creating or forking a conversation (upstream
/// `ConversationCreateOptions`).
#[derive(Clone)]
pub struct ConversationCreateOptions {
    pub ownership: ConversationOwnership,
    /// Applied in the creating commit after the creation hook's copy, before
    /// `init`.
    pub agent: Option<AgentChange>,
    pub init: Option<ConversationInit>,
}

/// Options of `Harness::root`.
#[derive(Clone, Default)]
pub struct RootConversationOptions {
    pub agent: Option<AgentChange>,
    pub init: Option<ConversationInit>,
}

/// Curated pi-ai request options; absent fields use pi-ai defaults (upstream
/// `ConversationStreamOptions`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationStreamOptions {
    pub transport: Option<Transport>,
    pub timeout_ms: Option<u64>,
    /// Provider/SDK retries inside one request attempt.
    pub max_retries: Option<u32>,
    pub max_retry_delay_ms: Option<u64>,
    pub headers: Option<BTreeMap<String, String>>,
    pub metadata: Option<JsonObject>,
    pub cache_retention: Option<pi_ai::types::CacheRetention>,
    /// Deferred switch (upstream `boolean | { window? }`).
    /// Erased deferred switch (upstream `boolean | { window? }`); the
    /// pi-ai request type has no serde derives, so the durable form is JSON.
    pub deferred: Option<Value>,
}

/// Durable generation attempt retries; the JSON shape of pi-ai `RetryPolicy`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationRetryPolicy {
    pub enabled: bool,
    pub max_retries: u32,
    pub base_delay_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_agent_delay_ms: Option<u64>,
}

/// Overrides of [`ConversationRetryPolicy`] in `HarnessSettings` (upstream
/// `Partial<ConversationRetryPolicy>`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetryPolicyOverrides {
    pub enabled: Option<bool>,
    pub max_retries: Option<u32>,
    pub base_delay_ms: Option<u64>,
    pub max_agent_delay_ms: Option<u64>,
}

/// Automatic compaction thresholds (spec §8.7); manual compaction ignores
/// `enabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionPolicy {
    pub enabled: bool,
    /// Room kept free for the answer: generation blocks to compact above
    /// `contextWindow - reserveTokens`.
    pub reserve_tokens: u64,
    /// Approximate size of the recent context a summary keeps verbatim.
    pub keep_recent_tokens: u64,
    /// Background compaction starts `backgroundTokens` below the blocking
    /// threshold; `0` disables it.
    pub background_tokens: u64,
}

/// Overrides of [`CompactionPolicy`] (upstream `Partial<CompactionPolicy>`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactionPolicyOverrides {
    pub enabled: Option<bool>,
    pub reserve_tokens: Option<u64>,
    pub keep_recent_tokens: Option<u64>,
    pub background_tokens: Option<u64>,
}

/// How often running progress is committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressPolicy {
    /// Minimum pause between commits of the answer being generated.
    pub partial_interval_ms: u64,
    /// Minimum pause between commits of running tool output; large commits
    /// also pause in proportion to their size.
    pub output_interval_ms: u64,
}

/// Overrides of [`ProgressPolicy`] (upstream `Partial<ProgressPolicy>`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProgressPolicyOverrides {
    pub partial_interval_ms: Option<u64>,
    pub output_interval_ms: Option<u64>,
}

/// Why a compaction runs: `compact()`, a threshold in generation preparation,
/// or a context overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CompactionReason {
    Manual,
    Threshold,
    Overflow,
}

impl CompactionReason {
    /// Upstream discriminator string.
    pub fn as_str(&self) -> &'static str {
        match self {
            CompactionReason::Manual => "manual",
            CompactionReason::Threshold => "threshold",
            CompactionReason::Overflow => "overflow",
        }
    }
}

/// `entryId` of a blocking compaction's summary, or the `submissionId` of a
/// conversation-owned compaction's summary write; both absent when nothing
/// was compacted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_id: Option<EntryId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submission_id: Option<SubmissionId>,
}

impl CompactionResult {
    /// Encode as the task result JSON.
    pub fn to_json(self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// Harness-wide run policy, resolved at every access. Static values are
/// snapshots; use [`HarnessSettingsProvider`] for upstream's live getters.
#[derive(Debug, Clone, Default)]
pub struct HarnessSettings {
    /// Default extension selection; absent: every installed extension, in
    /// install order.
    pub extensions: Option<Vec<Arc<Extension>>>,
    pub stream: Option<ConversationStreamOptions>,
    pub retry: Option<RetryPolicyOverrides>,
    pub compaction: Option<CompactionPolicyOverrides>,
    pub progress: Option<ProgressPolicyOverrides>,
    pub tool_execution: Option<ToolExecutionMode>,
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
    /// How long an idle conversation keeps its last context read in memory, so
    /// its next run reads only newer entries. Busy conversations always keep
    /// it; `Some(0)` drops it once the conversation is idle.
    pub context_retention_ms: Option<u64>,
}
/// Synchronous source of current Harness settings (upstream getters).
pub type HarnessSettingsProvider = Arc<dyn Fn() -> HarnessSettings + Send + Sync>;

/// Resolved settings: every field over its built-in default, object fields
/// merged (see `harness::agent::resolve_settings`).
#[derive(Debug, Clone)]
pub struct Settings {
    /// Absent: every installed extension, in install order.
    pub extensions: Option<Vec<Arc<Extension>>>,
    pub stream: ConversationStreamOptions,
    pub retry: ConversationRetryPolicy,
    pub compaction: CompactionPolicy,
    pub progress: ProgressPolicy,
    pub tool_execution: ToolExecutionMode,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    /// How long an idle conversation keeps its last context read (default ten
    /// minutes; `0` drops it once idle).
    pub context_retention_ms: u64,
}

/// What `HarnessOptions.env` builds an environment for.
pub struct EnvTarget {
    pub conversation_id: ConversationId,
    /// The conversation's agent `cwd`.
    pub cwd: Option<String>,
    /// Committed document reads.
    pub read: Arc<dyn DocumentReader>,
}

/// Environment builder (upstream `HarnessOptions.env`; async upstream, boxed
/// future here).
pub type EnvBuilder = Arc<
    dyn Fn(EnvTarget, Context) -> BoxFuture<Result<Option<Arc<dyn ExecutionEnv>>, String>>
        + Send
        + Sync,
>;

/// Built-in creation hook (upstream `HarnessOptions.conversationCreated`;
/// sync per `tasks` D3).
pub type ConversationCreatedHook =
    Arc<dyn Fn(&Tx, &ConversationRecord) -> Result<(), String> + Send + Sync>;

/// Failure reporter (upstream `HarnessOptions.on_report`; D5 — receives the
/// error text).
pub type ReportHook = Arc<dyn Fn(String) + Send + Sync>;

/// Harness clock (upstream `HarnessOptions.now`).
pub type NowFn = Arc<dyn Fn() -> i64 + Send + Sync>;

/// Options for opening a Harness.
#[derive(Clone)]
pub struct HarnessOptions {
    /// pi-ai model access used by generation.
    pub models: Arc<dyn pi_ai::Models>,
    pub registry: Arc<dyn RegistryReader>,
    pub settings: Option<HarnessSettings>,
    /// Live settings source; when present, it supersedes `settings`.
    pub settings_provider: Option<HarnessSettingsProvider>,
    /// Builds a conversation's environment at each use. Never called on the
    /// Session line; may be async.
    pub env: Option<EnvBuilder>,
    /// Runs in every commit that creates or forks a conversation, after the
    /// built-in `pi.*` documents and before the conveniences apply `agent`
    /// and run `init`.
    pub conversation_created: Option<ConversationCreatedHook>,
    pub now: Option<NowFn>,
    /// Receives extension failures that do not fail the calling operation.
    pub on_report: Option<ReportHook>,
}

impl Default for HarnessOptions {
    fn default() -> Self {
        Self {
            models: pi_ai::create_models(None),
            registry: Arc::new(crate::harness::registry::create_registry()),
            settings: None,
            settings_provider: None,
            env: None,
            conversation_created: None,
            now: None,
            on_report: None,
        }
    }
}

/// What the conveniences apply in the creating commit (harness-internal
/// upstream `CreateTarget`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateTarget {
    Root,
    Independent,
    Fork {
        parent_id: ConversationId,
        at: EntryId,
    },
}

/// The structural view of one conversation (upstream `view.ts`
/// `ConversationView`); entries are raw active entries and docs the built-in
/// conversation documents keyed by kind.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltinConversationView {
    pub conversation: ConversationRecord,
    pub entries: Vec<EntryRecord>,
    pub docs: BTreeMap<String, JsonObject>,
}

/// Serialized exact-frame watch of one conversation's view.
pub type ConversationWatch = WatchHandle<BuiltinConversationView>;

/// What a hook may use: committed reads and the asking task's memos, which
/// hooks and the task share (upstream `HookApi`).
#[async_trait::async_trait]
pub trait HookApi: DocumentReader {
    fn task_id(&self) -> TaskId;
    fn conversation_id(&self) -> ConversationId;
    /// `HarnessOptions.models`: catalog, credentials, and request transforms.
    fn models(&self) -> Arc<dyn pi_ai::Models>;
    async fn memo(&self, name: &str, context: &Context) -> Result<Option<Value>, String>;
    async fn memo_put(
        &self,
        name: &str,
        candidate: Value,
        context: &Context,
    ) -> Result<Value, String>;
}

/// Payload of `GenerationHooks::before_request`.
#[derive(Debug, Clone)]
pub struct BeforeRequestPayload {
    pub messages: Vec<Message>,
}

/// Result of `GenerationHooks::before_request` (upstream `{ messages }`).
#[derive(Debug, Clone)]
pub struct BeforeRequestResult {
    pub messages: Vec<Message>,
}

/// Result of `GenerationHooks::on_yield` (upstream `{ continue }`; D4).
#[derive(Debug, Clone)]
pub struct OnYieldResult {
    pub continue_: UserInput,
}

/// Generation hook handlers (upstream `GenerationHooks`; each method is
/// optional — an absent closure is skipped).
#[derive(Clone, Default)]
pub struct GenerationHooks {
    /// Before every request attempt, including recovery; the result is used
    /// for that request only.
    pub before_request: Option<
        Arc<
            dyn Fn(
                    BeforeRequestPayload,
                    Arc<dyn HookApi>,
                    Context,
                ) -> BoxFuture<Result<Option<BeforeRequestResult>, String>>
                + Send
                + Sync,
        >,
    >,
    /// Every terminal provider message, before classification.
    pub after_response: Option<
        Arc<
            dyn Fn(
                    pi_ai::AssistantMessage,
                    Arc<dyn HookApi>,
                    Context,
                ) -> BoxFuture<Result<(), String>>
                + Send
                + Sync,
        >,
    >,
    /// A final answer; the first `continue` appends a user message and
    /// continues the run.
    pub on_yield: Option<
        Arc<
            dyn Fn(
                    pi_ai::AssistantMessage,
                    Arc<dyn HookApi>,
                    Context,
                ) -> BoxFuture<Result<Option<OnYieldResult>, String>>
                + Send
                + Sync,
        >,
    >,
    /// After every tool of the round is terminal; `results` are the round's
    /// result entries in call order.
    pub after_tools: Option<
        Arc<
            dyn Fn(
                    EntryId,
                    Vec<EntryId>,
                    Arc<dyn HookApi>,
                    Context,
                ) -> BoxFuture<Result<(), String>>
                + Send
                + Sync,
        >,
    >,
}

impl std::fmt::Debug for GenerationHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationHooks")
            .field("before_request", &self.before_request.is_some())
            .field("after_response", &self.after_response.is_some())
            .field("on_yield", &self.on_yield.is_some())
            .field("after_tools", &self.after_tools.is_some())
            .finish()
    }
}

/// Decision of `ToolHooks::before_tool`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BeforeToolDecision {
    pub arguments: Option<JsonObject>,
    pub block: Option<String>,
}

/// Tool hook handlers (upstream `ToolHooks`).
#[derive(Clone, Default)]
pub struct ToolHooks {
    /// Before intent; the first `block` wins, otherwise `arguments` replace
    /// the call's arguments. An error blocks.
    pub before_tool: Option<
        Arc<
            dyn Fn(
                    pi_ai::ToolCall,
                    Arc<dyn HookApi>,
                    Context,
                ) -> BoxFuture<Result<Option<BeforeToolDecision>, String>>
                + Send
                + Sync,
        >,
    >,
    /// After execution, before the result entry; replaces the result.
    pub after_tool: Option<
        Arc<
            dyn Fn(
                    pi_ai::ToolCall,
                    ToolExecutionResult,
                    Arc<dyn HookApi>,
                    Context,
                ) -> BoxFuture<Result<Option<ToolExecutionResult>, String>>
                + Send
                + Sync,
        >,
    >,
}

impl std::fmt::Debug for ToolHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolHooks")
            .field("before_tool", &self.before_tool.is_some())
            .field("after_tool", &self.after_tool.is_some())
            .finish()
    }
}

/// Payload of `CompactionHooks::before_compact`.
#[derive(Debug, Clone)]
pub struct BeforeCompactPayload {
    pub reason: CompactionReason,
    /// The active entries the summary replaces, the head marker first.
    pub entries: Vec<EntryRecord>,
    /// Their model context, the summarizer's source.
    pub messages: Vec<Message>,
    pub first_kept: EntryId,
    pub instructions: Option<String>,
}

/// Decision of `CompactionHooks::before_compact` (upstream `{ decline: true }
/// | { summary }`).
#[derive(Debug, Clone, PartialEq)]
pub enum BeforeCompactDecision {
    Decline,
    Summary(String),
}

/// Compaction hook handlers (upstream `CompactionHooks`).
#[derive(Clone, Default)]
pub struct CompactionHooks {
    pub before_compact: Option<
        Arc<
            dyn Fn(
                    BeforeCompactPayload,
                    Arc<dyn HookApi>,
                    Context,
                ) -> BoxFuture<Result<Option<BeforeCompactDecision>, String>>
                + Send
                + Sync,
        >,
    >,
}

impl std::fmt::Debug for CompactionHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactionHooks")
            .field("before_compact", &self.before_compact.is_some())
            .finish()
    }
}

/// Entry query without its conversation (upstream
/// `Omit<EntryQuery, "conversationId">`); `Conversation::entries` supplies
/// the conversation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConversationEntryQuery {
    pub min_entry_id: Option<EntryId>,
    pub max_entry_id: Option<EntryId>,
    /// Scan order; newest first when absent.
    pub order: Option<crate::types::ScanOrder>,
}

impl ConversationEntryQuery {
    /// Bound into a full storage query for one conversation.
    pub fn bounded(&self, conversation_id: ConversationId) -> EntryQuery {
        EntryQuery {
            conversation_id,
            min_entry_id: self.min_entry_id,
            max_entry_id: self.max_entry_id,
            order: self.order,
        }
    }
}

impl From<ConversationStreamOptions> for pi_ai::types::StreamOptions {
    fn from(options: ConversationStreamOptions) -> Self {
        let mut converted = pi_ai::types::StreamOptions::default();
        converted.transport = options.transport;
        converted.timeout_ms = options.timeout_ms;
        converted.max_retries = options.max_retries;
        converted.max_retry_delay_ms = options.max_retry_delay_ms;
        converted.base.headers = options.headers.map(|headers| {
            headers
                .into_iter()
                .map(|(name, value)| (name, Some(value)))
                .collect()
        });
        converted.metadata = options.metadata;
        converted.cache_retention = options.cache_retention;
        converted
    }
}

impl Agent {
    /// pi-ai declarations of [`Agent::tools`] in order (upstream `Tool`
    /// registrations carry the declaration fields in `ToolRegistration::tool`).
    pub fn tool_declarations(&self) -> Vec<pi_ai::Tool> {
        self.tools.iter().map(|tool| tool.tool.clone()).collect()
    }
}

/// Adapter presenting a task invocation's runtime as the hook surface
/// (upstream passes the runtime object itself; the erased Rust traits are
/// distinct, so a wrapper bridges them, D-hooks).
pub struct RuntimeHookApi(pub Arc<dyn crate::tasks::TaskRuntime>);

#[async_trait::async_trait]
impl DocumentReader for RuntimeHookApi {
    async fn snapshot(
        &self,
        token: &DocToken,
        owner: Option<u64>,
        key: Option<&str>,
        context: &Context,
    ) -> Result<Option<Arc<JsonObject>>, String> {
        self.0
            .document_snapshot(token, owner, key, context)
            .await
            .map(|value| value.map(Arc::new))
    }

    async fn snapshot_as_of(
        &self,
        token: &DocToken,
        conversation_id: ConversationId,
        key: Option<&str>,
        at: EntryId,
        context: &Context,
    ) -> Result<Option<JsonObject>, String> {
        self.0
            .document_snapshot_as_of(token, conversation_id, key, at, context)
            .await
    }
}

#[async_trait::async_trait]
impl HookApi for RuntimeHookApi {
    fn task_id(&self) -> TaskId {
        self.0.task_id()
    }

    fn conversation_id(&self) -> ConversationId {
        self.0.conversation_id()
    }

    fn models(&self) -> Arc<dyn pi_ai::Models> {
        self.0.models()
    }

    async fn memo(&self, name: &str, context: &Context) -> Result<Option<Value>, String> {
        self.0.memo(name, context).await
    }

    async fn memo_put(
        &self,
        name: &str,
        candidate: Value,
        context: &Context,
    ) -> Result<Value, String> {
        self.0.memo_put(name, candidate, context).await
    }
}

/// Present a task runtime as the hook surface.
pub fn coerce_hook_api(runtime: Arc<dyn crate::tasks::TaskRuntime>) -> Arc<dyn HookApi> {
    Arc::new(RuntimeHookApi(runtime))
}

impl ToolRegistration {
    /// The pi-ai declaration's tool name (upstream `Tool` spread).
    pub fn name(&self) -> &str {
        &self.tool.name
    }
}

/// Forwarding impl so `Arc<Session>` reads anywhere `DocumentReader` is
/// expected (upstream passes the session object itself).
#[async_trait::async_trait]
impl<T: DocumentReader + ?Sized> DocumentReader for std::sync::Arc<T> {
    async fn snapshot(
        &self,
        token: &DocToken,
        owner: Option<u64>,
        key: Option<&str>,
        context: &Context,
    ) -> Result<Option<Arc<JsonObject>>, String> {
        (**self).snapshot(token, owner, key, context).await
    }

    async fn snapshot_as_of(
        &self,
        token: &DocToken,
        conversation_id: ConversationId,
        key: Option<&str>,
        at: EntryId,
        context: &Context,
    ) -> Result<Option<JsonObject>, String> {
        (**self)
            .snapshot_as_of(token, conversation_id, key, at, context)
            .await
    }
}
