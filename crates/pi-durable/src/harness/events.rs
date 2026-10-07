//! Port of `packages/durable/src/harness/events.ts` (experimental agent events, spec §9.4).

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_ai::types::{AssistantContent, AssistantMessage, Message};
use pi_chord::context::Context;
use pi_chord::delta::{Op, Seg};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{oneshot, Notify};

use crate::harness::inbox::{InboxItem, InboxState};
use crate::harness::live::{CompactionStatus, DeferredPoll, GenerationStatus, LiveState, RetryBackoff, SlotStatus, ToolSlot};
use crate::harness::types::{
	AgentState, CompactionReason, ConversationId, EntryRecord, JsonObject, SubmissionId, SubmissionRecord, TaskId,
	ToolDiagnostic,
};
use crate::harness::usage::UsageState;
use crate::harness::view::{ConversationView, ViewObserver};
use crate::session::observation::{CommittedWatch, ObservedValue, WatchError};
use crate::types::{CommitChange, CommitPublication, TaskQuery, TaskRecord, TaskStateStatus, TableCommitChange, WatchEnd};

/// One change to the in-flight assistant message, relative to that message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", rename_all_fields = "camelCase")]
pub enum MessageChange {
	TextStart { content_index: u32, block: Value },
	ThinkingStart { content_index: u32, block: Value },
	ToolcallStart { content_index: u32, block: Value },
	TextDelta { content_index: u32, delta: String },
	ThinkingDelta { content_index: u32, delta: String },
	ToolcallDelta { content_index: u32, path: Vec<Value>, delta: String },
	Block { content_index: u32, block: Value },
	Message { message: AssistantMessage },
}

/// A currently running tool's output change.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolOutputChange {
	Delta {
		#[serde(rename = "trimStart", skip_serializing_if = "Option::is_none")]
		trim_start: Option<u64>,
		#[serde(skip_serializing_if = "Option::is_none")]
		append: Option<String>,
	},
	Set { set: String },
}

/// The typed snapshot fields carried directly by a `snapshot` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotEvent {
	pub entries: Vec<EntryRecord>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub run: Option<RunSnapshot>,
	/// Current generation attempt: its in-flight partial, retry backoff, or deferred poll.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub generation: Option<GenerationSnapshot>,
	pub tools: Vec<SnapshotToolSlot>,
	/// `pi.live.compactions`: live compactions with their attempt and retry backoff.
	pub compactions: Vec<CompactionStatus>,
	pub inbox: Vec<QueuedItem>,
	/// `pi.agent`; the empty state when absent.
	pub agent: AgentState,
	pub usage: UsageState,
}

/// Tool slot snapshot with JSON details preserved even when explicitly `null`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotToolSlot {
	pub call_id: String,
	pub name: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub task_id: Option<TaskId>,
	pub status: SlotStatus,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub output: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub dropped_bytes: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub dropped_lines: Option<u64>,
	#[serde(default, deserialize_with = "deserialize_present_value", skip_serializing_if = "Option::is_none")]
	pub details: Option<Value>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub diagnostics: Option<Vec<ToolDiagnostic>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub entry: Option<crate::harness::types::EntryId>,
	#[serde(flatten)]
	pub extra: JsonObject,
}

fn deserialize_present_value<'de, D>(deserializer: D) -> Result<Option<Value>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	Value::deserialize(deserializer).map(Some)
}

/// Run state included in an initial or overflow snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunSnapshot {
	pub inputs: Vec<SubmissionId>,
}

/// Generation state included in an initial or overflow snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationSnapshot {
	pub attempt: u64,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub message: Option<AssistantMessage>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub retry: Option<RetryBackoff>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub deferred: Option<DeferredPoll>,
}

/// The mode of a queued inbox item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InboxMode {
	Steer,
	FollowUp,
	Write,
}

/// A queued inbox item as event snapshots carry it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueuedItem {
	pub id: SubmissionId,
	pub mode: InboxMode,
}

/// Experimental agent event, shaped like the coding agent's session events (spec §9.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", rename_all_fields = "camelCase")]
pub enum AgentEvent {
	/// Snapshot fields are flattened beside `type`, as in the upstream union.
	Snapshot { #[serde(flatten)] snapshot: SnapshotEvent },
	RunStart { inputs: Vec<SubmissionId> },
	RunEnd { inputs: Vec<SubmissionId> },
	TurnStart,
	TurnEnd,
	MessageStart { message: Message },
	/// `usage` is the partial's current usage, as in the coding agent's JSON mode.
	MessageUpdate { usage: pi_ai::Usage, changes: Vec<MessageChange> },
	MessageEnd { entry: EntryRecord },
	ToolExecutionStart { tool_call_id: String, tool_name: String, args: JsonObject },
	ToolExecutionUpdate {
		tool_call_id: String,
		tool_name: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		output: Option<ToolOutputChange>,
		/// Outer `None` omits the field; inner `None` sends `null` when details were removed.
		#[serde(skip_serializing_if = "Option::is_none")]
		details: Option<Option<Value>>,
		#[serde(skip_serializing_if = "Option::is_none")]
		diagnostics: Option<Vec<ToolDiagnostic>>,
	},
	/// `entry` is absent when the tool task faulted or was orphaned.
	ToolExecutionEnd {
		tool_call_id: String,
		tool_name: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		entry: Option<EntryRecord>,
	},
	InboxUpdate { items: Vec<QueuedItem> },
	Submission { record: SubmissionRecord },
	AutoRetryStart { attempt: u64, at: u64, error_message: String },
	AutoRetryEnd { attempt: u64 },
	DeferredPoll { poll_at: u64 },
	EntryAppended { entry: EntryRecord },
	AgentChanged { agent: AgentState },
	UsageChanged { usage: UsageState },
	TaskFailed { task_id: TaskId, kind: String, message: String },
	CompactionStart { task_id: TaskId, reason: CompactionReason, blocking: bool },
	/// The task's receipt tells whether it produced a summary; the summary entry has its own events.
	CompactionEnd { task_id: TaskId, reason: CompactionReason },
}

/// Serialized stream of one conversation's event batches, one per commit.
pub struct AgentEventStream {
	/// The typed snapshot captured atomically with observer registration.
	pub snapshot: SnapshotEvent,
	watch: CommittedWatch<EventFrame>,
	legacy_next: Option<LegacyNext>,
}

struct LegacyNext {
	pull: Arc<LegacyPull>,
}

struct LegacyPull {
	state: Mutex<LegacyPullState>,
	wake: Notify,
}

struct LegacyPullState {
	request: Option<oneshot::Sender<Result<Vec<AgentEvent>, String>>>,
	terminal: Option<LegacyTerminal>,
	terminal_delivered: bool,
}

#[derive(Clone)]
enum LegacyTerminal {
	Error(String),
	Closed,
}

impl LegacyPull {
	fn new() -> Self {
		Self {
			state: Mutex::new(LegacyPullState { request: None, terminal: None, terminal_delivered: false }),
			wake: Notify::new(),
		}
	}

	async fn deliver(&self, events: Vec<AgentEvent>) {
		loop {
			let notified = self.wake.notified();
			let (request, terminal) = {
				let mut state = self.state.lock();
				if state.terminal.is_some() {
					(None, true)
				} else {
					(state.request.take(), false)
				}
			};
			if terminal {
				return;
			}
			if let Some(request) = request {
				let _ = request.send(Ok(events));
				return;
			}
			notified.await;
		}
	}

	async fn next(&self) -> Option<Result<Vec<AgentEvent>, String>> {
		let (sender, receiver) = oneshot::channel();
		{
			let mut state = self.state.lock();
			if state.terminal.is_some() {
				if state.terminal_delivered {
					return None;
				}
				state.terminal_delivered = true;
				return match state.terminal.as_ref().expect("terminal was checked") {
					LegacyTerminal::Error(error) => Some(Err(error.clone())),
					LegacyTerminal::Closed => None,
				};
			}
			state.request = Some(sender);
		}
		self.wake.notify_one();
		receiver.await.ok()
	}

	fn terminate(&self, end: WatchEnd) {
		let terminal = match end {
			WatchEnd::SessionClosed => LegacyTerminal::Error(crate::harness::util::closed_error()),
			WatchEnd::ListenerError { error } => LegacyTerminal::Error(error),
			_ => LegacyTerminal::Closed,
		};
		let (request, terminal) = {
			let mut state = self.state.lock();
			if state.terminal.is_some() {
				return;
			}
			let request = state.request.take();
			if request.is_some() {
				state.terminal_delivered = true;
			}
			state.terminal = Some(terminal.clone());
			(request, terminal)
		};
		if let (Some(request), LegacyTerminal::Error(error)) = (request, terminal) {
			let _ = request.send(Err(error));
		}
		self.wake.notify_one();
	}
}

impl AgentEventStream {
	/// Install the stream's sole listener. The initial snapshot remains available through [`Self::snapshot`].
	pub fn start<F, Fut>(&self, listener: F) -> Result<(), WatchError>
	where
		F: Fn(Vec<AgentEvent>, Context) -> Fut + Send + Sync + 'static,
		Fut: Future<Output = ()> + Send + 'static,
	{
		let listener = Arc::new(listener);
		self.watch.start(Arc::new(move |frame, ops, context| -> Pin<Box<dyn Future<Output = ()> + Send>> {
			let listener = listener.clone();
			let events = delivered_events(frame, &ops);
			Box::pin(async move { listener(events, context).await })
		}))
	}

	/// Idempotently stop delivery and return the terminal watch result.
	pub async fn stop(&self) -> WatchEnd {
		self.watch.stop().await
	}

	/// Resolve when stopped, cancelled, or when the owning Session closes.
	pub async fn closed(&self) -> WatchEnd {
		self.watch.closed().await
	}

	/// Legacy Rust convenience: the first batch is the attachment snapshot, followed by publication batches.
	pub async fn next(&mut self) -> Option<Result<Vec<AgentEvent>, String>> {
		if self.legacy_next.is_none() {
			let pull = Arc::new(LegacyPull::new());
			let delivery_pull = pull.clone();
			let started = self.watch.start(Arc::new(
				move |frame, ops, _context| -> Pin<Box<dyn Future<Output = ()> + Send>> {
					let delivery_pull = delivery_pull.clone();
					let events = delivered_events(frame, &ops);
					Box::pin(async move { delivery_pull.deliver(events).await })
				},
			));
			self.legacy_next = Some(LegacyNext { pull: pull.clone() });
			let watch_for_close = match started {
				Ok(()) => Some(self.watch.clone()),
				Err(error) if error.0 == "Watch is stopped" => Some(self.watch.clone()),
				Err(error) => {
					pull.terminate(WatchEnd::ListenerError { error: error.to_string() });
					None
				}
			};
			if let Some(watch) = watch_for_close {
				let close_pull = pull.clone();
				tokio::spawn(async move { close_pull.terminate(watch.closed().await) });
			}
			return Some(Ok(vec![AgentEvent::Snapshot { snapshot: self.snapshot.clone() }]));
		}
		self.legacy_next.as_ref().expect("legacy pull initialized").pull.next().await
	}
}

/// The typed parts of a view read by event translation.
struct Parts {
	live: LiveState,
	inbox: Option<InboxState>,
	agent: Option<AgentState>,
	usage: Option<UsageState>,
}

fn parts(view: &ConversationView) -> Parts {
	fn read<T: serde::de::DeserializeOwned>(view: &ConversationView, name: &str) -> Option<T> {
		view.docs.get(name).map(|document| {
			serde_json::from_value(Value::Object(document.clone()))
				.unwrap_or_else(|error| panic!("mounted {name} document has an invalid shape: {error}"))
		})
	}
	Parts {
		live: read(view, "pi.live").unwrap_or_default(),
		inbox: read(view, "pi.inbox"),
		agent: read(view, "pi.agent"),
		usage: read(view, "pi.usage"),
	}
}

fn snapshot_of(view: &ConversationView) -> SnapshotEvent {
	let Parts { live, inbox, agent, usage } = parts(view);
	SnapshotEvent {
		entries: view.entries.clone(),
		run: live.run.as_ref().map(|run| RunSnapshot { inputs: run.inputs.clone() }),
		generation: live.generation.as_ref().map(generation_snapshot),
		tools: snapshot_tools(view),
		compactions: live.compactions.unwrap_or_default(),
		inbox: queued(inbox.as_ref()),
		agent: agent.unwrap_or_default(),
		usage: usage.unwrap_or_default(),
	}
}

fn snapshot_tools(view: &ConversationView) -> Vec<SnapshotToolSlot> {
	let Some(tools) = view.docs.get("pi.live").and_then(|document| document.get("tools")).and_then(Value::as_array) else {
		return Vec::new();
	};
	serde_json::from_value(Value::Array(tools.clone()))
		.unwrap_or_else(|error| panic!("committed tool slots have an invalid shape: {error}"))
}

fn generation_snapshot(generation: &GenerationStatus) -> GenerationSnapshot {
	GenerationSnapshot {
		attempt: generation.attempt,
		message: generation.message.as_ref().map(|message| {
			serde_json::from_value(message.clone())
				.unwrap_or_else(|error| panic!("committed partial assistant message has an invalid shape: {error}"))
		}),
		retry: generation.retry.clone(),
		deferred: generation.deferred,
	}
}

fn queued(inbox: Option<&InboxState>) -> Vec<QueuedItem> {
	inbox
		.into_iter()
		.flat_map(|inbox| inbox.items.iter())
		.map(|item| {
			let (id, mode) = match item {
				InboxItem::Steer { id, .. } => (*id, InboxMode::Steer),
				InboxItem::FollowUp { id, .. } => (*id, InboxMode::FollowUp),
				InboxItem::Write { id, .. } => (*id, InboxMode::Write),
			};
			QueuedItem { id, mode }
		})
		.collect()
}

/// Experimental: attach to one conversation's events. Snapshot, held task set, and observer registration share the Session line.
pub async fn watch_events(
	harness: &Arc<crate::harness::harness::Harness>,
	conversation_id: ConversationId,
	context: &Context,
) -> Result<AgentEventStream, String> {
	let attached = harness
		.views
		.attach_observer(conversation_id, context, move |initial, storage, context, detach| async move {
			let query = TaskQuery {
				conversation_id: Some(conversation_id),
				kind: Some("pi.generation".to_string()),
				status: Some(TaskStateStatus::Completing),
				..TaskQuery::default()
			};
			let completing = crate::harness::util::scan_all(|cursor| {
				let storage = storage.clone();
				let context = context.clone();
				let query = query.clone();
				async move { storage.scan_tasks(&query, 100, cursor.as_ref(), &context).await }
			})
			.await;
			let held = completing.into_iter().map(|task| task.id).collect();
			let snapshot = snapshot_of(&initial);
			let current_snapshot = Arc::new(Mutex::new(snapshot.clone()));
			let initial_frame = EventFrame {
				events: Vec::new(),
				snapshot: current_snapshot.clone(),
				overflow_snapshot: Arc::new(Mutex::new(None)),
			};
			let watch = CommittedWatch::new(initial_frame, detach);
			Ok((EventObserver { conversation_id, watch, held: Mutex::new(held), snapshot: current_snapshot }, snapshot))
		})
		.await?;
	let watch = attached.observer.watch.clone();
	if let Some(signal) = context.abort_signal() {
		if signal.aborted() {
			watch.cancel();
			return Err(crate::harness::util::abort_reason_text(&signal));
		}
		watch.observe_cancellation(&signal).map_err(|error| error.to_string())?;
	}
	Ok(AgentEventStream { snapshot: attached.value, watch, legacy_next: None })
}

#[derive(Clone)]
struct EventFrame {
	events: Vec<AgentEvent>,
	snapshot: Arc<Mutex<SnapshotEvent>>,
	overflow_snapshot: Arc<Mutex<Option<SnapshotEvent>>>,
}

impl ObservedValue for EventFrame {
	fn to_frame_json(&self) -> Value {
		let snapshot = self.snapshot.lock().clone();
		*self.overflow_snapshot.lock() = Some(snapshot.clone());
		serde_json::to_value([AgentEvent::Snapshot { snapshot }]).expect("agent snapshot serializes")
	}

	fn is_retired(&self) -> bool {
		false
	}
}

fn delivered_events(frame: EventFrame, ops: &[Op]) -> Vec<AgentEvent> {
	if ops.iter().any(|op| matches!(op, Op::Replace(_))) {
		vec![AgentEvent::Snapshot {
			snapshot: frame.overflow_snapshot.lock().take().expect("overflow snapshot was captured"),
		}]
	} else {
		frame.events
	}
}

struct EventObserver {
	conversation_id: ConversationId,
	watch: CommittedWatch<EventFrame>,
	held: Mutex<HashSet<TaskId>>,
	snapshot: Arc<Mutex<SnapshotEvent>>,
}

impl ViewObserver for EventObserver {
	fn publication(
		&self,
		before: &ConversationView,
		after: &ConversationView,
		view_ops: &[Op],
		publication: &CommitPublication,
		context: &Context,
	) {
		let events = translate(self.conversation_id, before, after, view_ops, publication, &mut self.held.lock());
		if !events.is_empty() {
			*self.snapshot.lock() = snapshot_of(after);
			self.watch.advance(
				EventFrame {
					events,
					snapshot: self.snapshot.clone(),
					overflow_snapshot: Arc::new(Mutex::new(None)),
				},
				Vec::new(),
				context.clone(),
			);
		}
	}

	fn close_session(&self) {
		self.watch.close_session();
	}
}


/// Every event one publication causes, in the order of spec §9.4.
fn translate(
	conversation_id: ConversationId,
	before: &ConversationView,
	after: &ConversationView,
	view_ops: &[Op],
	publication: &CommitPublication,
	held: &mut HashSet<TaskId>,
) -> Vec<AgentEvent> {
	let mut entries = Vec::new();
	let mut tasks: Vec<TaskRecord> = Vec::new();
	let mut submissions: Vec<SubmissionRecord> = Vec::new();
	for change in &publication.changes {
		match change {
			CommitChange::Table(TableCommitChange::Entry { value }) if value.conversation_id == conversation_id => {
				entries.push(value.clone());
			}
			CommitChange::Table(TableCommitChange::Task { value }) if value.conversation_id == conversation_id => {
				if let Some(existing) = tasks.iter_mut().find(|task| task.id == value.id) {
					*existing = value.clone();
				} else {
					tasks.push(value.clone());
				}
			}
			CommitChange::Table(TableCommitChange::Submission { value }) if value.conversation_id == conversation_id => {
				submissions.push(value.clone());
			}
			_ => {}
		}
	}
	entries.sort_by_key(|entry| entry.id);
	if view_ops.is_empty() && entries.is_empty() && tasks.is_empty() && submissions.is_empty() {
		return Vec::new();
	}
	// Entry changes use ID order; submission changes are ordered by ID.
	submissions.sort_by_key(|submission| submission.id);
	let was = parts(before);
	let now = parts(after);
	let mut events = Vec::new();

	// Progress: tool starts, the in-flight message, tool updates, retry and deferred state.
	let slots_before = was.live.tools.as_deref().unwrap_or_default();
	let slots = now.live.tools.as_deref().unwrap_or_default();
	for slot in slots {
		let previous = slots_before.iter().find(|previous| previous.call_id == slot.call_id);
		if slot.status != SlotStatus::Running || previous.is_some_and(|previous| previous.status == SlotStatus::Running) {
			continue;
		}
		let checkpoint = slot
			.task_id
			.and_then(|task_id| tasks.iter().find(|task| task.id == task_id))
			.and_then(|task| task.state.checkpoint.as_ref());
		let args = checkpoint
			.and_then(|checkpoint| checkpoint.get("arguments"))
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		events.push(AgentEvent::ToolExecutionStart {
			tool_call_id: slot.call_id.clone(),
			tool_name: slot.name.clone(),
			args,
		});
	}
	let partial_before = partial(&was.live);
	let partial_after = partial(&now.live);
	match (&partial_before, &partial_after) {
		(None, Some(message)) => events.push(AgentEvent::MessageStart { message: Message::Assistant(message.clone()) }),
		(Some(_), Some(message)) if ops_touch_path(view_ops, &partial_path()) => events.push(AgentEvent::MessageUpdate {
			usage: message.usage.clone(),
			changes: message_changes(view_ops, message),
		}),
		_ => {}
	}
	for (index, slot) in slots.iter().enumerate() {
		let Some(previous) = slots_before.iter().find(|previous| previous.call_id == slot.call_id) else {
			continue;
		};
		if slot.status != SlotStatus::Running || previous.status != SlotStatus::Running {
			continue;
		}
		if let Some(update) = tool_update(view_ops, index, slot, previous, before, after) {
			events.push(AgentEvent::ToolExecutionUpdate {
				tool_call_id: slot.call_id.clone(),
				tool_name: slot.name.clone(),
				output: update.output,
				details: update.details,
				diagnostics: update.diagnostics,
			});
		}
	}
	let generation = now.live.generation.as_ref();
	let generation_before = was.live.generation.as_ref();
	if let Some(retry) = generation.and_then(|generation| generation.retry.as_ref()) {
		if generation_before.and_then(|generation| generation.retry.as_ref()).is_none() {
			events.push(AgentEvent::AutoRetryStart {
				attempt: generation.expect("retry belongs to generation").attempt,
				at: retry.at,
				error_message: retry.error.clone(),
			});
		}
	}
	if generation_before.and_then(|generation| generation.retry.as_ref()).is_some()
		&& generation.and_then(|generation| generation.retry.as_ref()).is_none()
	{
		events.push(AgentEvent::AutoRetryEnd {
			attempt: generation_before.expect("retry belongs to previous generation").attempt,
		});
	}
	if let Some(deferred) = generation.and_then(|generation| generation.deferred) {
		if generation_before.and_then(|generation| generation.deferred).map(|previous| previous.poll_at) != Some(deferred.poll_at) {
			events.push(AgentEvent::DeferredPoll { poll_at: deferred.poll_at });
		}
	}

	// Tools that end in this commit: completed slots, unoffered calls, or unfinished slots removed with their run.
	let mut tool_ends: Vec<ToolEnd> = Vec::new();
	for previous in slots_before {
		if previous.status == SlotStatus::Done {
			continue;
		}
		let slot = slots.iter().find(|slot| slot.call_id == previous.call_id);
		if let Some(slot) = slot.filter(|slot| slot.status == SlotStatus::Done) {
			tool_ends.push(ToolEnd {
				call_id: previous.call_id.clone(),
				name: previous.name.clone(),
				entry: slot.entry.and_then(|id| entries.iter().find(|entry| entry.id == id)).cloned(),
			});
		} else if slot.is_none() {
			tool_ends.push(ToolEnd {
				call_id: previous.call_id.clone(),
				name: previous.name.clone(),
				entry: result_of(&entries, &previous.call_id),
			});
		}
	}
	for slot in slots {
		if slot.status == SlotStatus::Done && !slots_before.iter().any(|previous| previous.call_id == slot.call_id) {
			tool_ends.push(ToolEnd {
				call_id: slot.call_id.clone(),
				name: slot.name.clone(),
				entry: slot.entry.and_then(|id| entries.iter().find(|entry| entry.id == id)).cloned(),
			});
		}
	}

	// Entries in append order; a tool's end directly precedes its result message.
	let mut assistant_appended = false;
	for entry in &entries {
		for end in tool_ends.iter().filter(|end| end.entry.as_ref().is_some_and(|ended| ended.id == entry.id)) {
			events.push(tool_end_event(end));
		}
		let message = entry.model.as_ref().and_then(|messages| messages.first());
		let Some(message) = message else {
			events.push(AgentEvent::EntryAppended { entry: entry.clone() });
			continue;
		};
		let streamed = matches!(message, Message::Assistant(_)) && partial_before.is_some() && !assistant_appended;
		if matches!(message, Message::Assistant(_)) {
			assistant_appended = true;
		}
		if !streamed {
			events.push(AgentEvent::MessageStart { message: message.clone() });
		}
		events.push(AgentEvent::MessageEnd { entry: entry.clone() });
	}
	for end in tool_ends.iter().filter(|end| end.entry.is_none()) {
		events.push(tool_end_event(end));
	}

	// Compaction ends, task failures, then turn and run ends.
	let compactions_before = was.live.compactions.as_deref().unwrap_or_default();
	let compactions = now.live.compactions.as_deref().unwrap_or_default();
	for previous in compactions_before {
		if !compactions.iter().any(|status| status.task_id == previous.task_id) {
			events.push(AgentEvent::CompactionEnd { task_id: previous.task_id, reason: previous.reason });
		}
	}
	let mut turn_ended = false;
	for task in &tasks {
		let status = task.state.status;
		if task.kind == "pi.generation" && status == TaskStateStatus::Completing && !held.contains(&task.id) {
			held.insert(task.id);
			turn_ended = true;
		}
		if status != TaskStateStatus::Terminal {
			continue;
		}
		if task.kind == "pi.generation" && !held.remove(&task.id) {
			turn_ended = true;
		}
		let outcome = task.state.outcome.as_ref().expect("terminal task has an outcome");
		match outcome.status {
			crate::types::TaskOutcomeStatus::Faulted => events.push(AgentEvent::TaskFailed {
				task_id: task.id,
				kind: task.kind.clone(),
				message: outcome.error.as_ref().expect("faulted outcome has an error").message.clone(),
			}),
			crate::types::TaskOutcomeStatus::Orphaned => events.push(AgentEvent::TaskFailed {
				task_id: task.id,
				kind: task.kind.clone(),
				message: outcome.reason.clone().expect("orphaned outcome has a reason"),
			}),
			_ => {}
		}
	}
	if turn_ended {
		events.push(AgentEvent::TurnEnd);
	}
	let run = now.live.run.as_ref();
	let run_before = was.live.run.as_ref();
	let run_changed = run.and_then(|run| run.inputs.first()) != run_before.and_then(|run| run.inputs.first());
	if run_before.is_some() && run_changed {
		events.push(AgentEvent::RunEnd { inputs: run_before.expect("checked").inputs.clone() });
	}

	// Submissions, document state, then what began.
	for record in submissions {
		events.push(AgentEvent::Submission { record });
	}
	if ops_touch_path(view_ops, &[PathPart::Key("docs"), PathPart::Key("pi.inbox")]) {
		events.push(AgentEvent::InboxUpdate { items: queued(now.inbox.as_ref()) });
	}
	if ops_touch_path(view_ops, &[PathPart::Key("docs"), PathPart::Key("pi.agent")]) {
		events.push(AgentEvent::AgentChanged { agent: now.agent.clone().unwrap_or_default() });
	}
	if ops_touch_path(view_ops, &[PathPart::Key("docs"), PathPart::Key("pi.usage")]) {
		events.push(AgentEvent::UsageChanged { usage: now.usage.clone().unwrap_or_default() });
	}
	for compaction in compactions {
		if !compactions_before.iter().any(|previous| previous.task_id == compaction.task_id) {
			events.push(AgentEvent::CompactionStart {
				task_id: compaction.task_id,
				reason: compaction.reason,
				blocking: compaction.blocking,
			});
		}
	}
	if let Some(run) = run.filter(|_| run_changed) {
		events.push(AgentEvent::RunStart { inputs: run.inputs.clone() });
	}
	if let Some(run) = run {
		if Some(run.task_id) != run_before.map(|previous| previous.task_id)
			&& tasks.iter().any(|task| task.id == run.task_id && task.kind == "pi.generation")
		{
			events.push(AgentEvent::TurnStart);
		}
	}
	events
}

struct ToolEnd {
	call_id: String,
	name: String,
	entry: Option<EntryRecord>,
}

fn tool_end_event(end: &ToolEnd) -> AgentEvent {
	AgentEvent::ToolExecutionEnd { tool_call_id: end.call_id.clone(), tool_name: end.name.clone(), entry: end.entry.clone() }
}

fn result_of(entries: &[EntryRecord], call_id: &str) -> Option<EntryRecord> {
	entries
		.iter()
		.find(|entry| {
			matches!(
				entry.model.as_ref().and_then(|messages| messages.first()),
				Some(Message::ToolResult(result)) if result.tool_call_id == call_id
			)
		})
		.cloned()
}

fn partial(live: &LiveState) -> Option<AssistantMessage> {
	live.generation
		.as_ref()?
		.message
		.as_ref()
		.map(|message| {
			serde_json::from_value(message.clone())
				.unwrap_or_else(|error| panic!("committed partial assistant message has an invalid shape: {error}"))
		})
}

#[derive(Clone, Copy)]
enum PathPart<'a> {
	Key(&'a str),
	Index(usize),
}

fn partial_path() -> [PathPart<'static>; 4] {
	[
		PathPart::Key("docs"),
		PathPart::Key("pi.live"),
		PathPart::Key("generation"),
		PathPart::Key("message"),
	]
}

fn op_path(op: &Op) -> &[Seg] {
	match op {
		Op::Replace(_) => &[],
		Op::Set { path, .. }
		| Op::Delete { path }
		| Op::Append { path, .. }
		| Op::Truncate { path, .. }
		| Op::Splice { path, .. }
		| Op::Permute { path, .. } => path,
	}
}

fn segment_matches(segment: &Seg, part: PathPart<'_>) -> bool {
	match (segment, part) {
		(Seg::Key(actual), PathPart::Key(expected)) => actual == expected,
		(Seg::Index(actual), PathPart::Index(expected)) => *actual == expected,
		_ => false,
	}
}

fn path_starts_with(path: &[Seg], prefix: &[PathPart<'_>]) -> bool {
	prefix.len() <= path.len() && path.iter().zip(prefix).all(|(segment, part)| segment_matches(segment, *part))
}

fn path_is_prefix_of(path: &[Seg], target: &[PathPart<'_>]) -> bool {
	path.len() <= target.len() && path.iter().zip(target).all(|(segment, part)| segment_matches(segment, *part))
}

fn paths_touch(path: &[Seg], target: &[PathPart<'_>]) -> bool {
	path_starts_with(path, target) || path_is_prefix_of(path, target)
}

fn ops_touch_path(ops: &[Op], target: &[PathPart<'_>]) -> bool {
	ops.iter().any(|op| paths_touch(op_path(op), target))
}

/// Translate view operations on the in-flight message into message changes.
fn message_changes(view_ops: &[Op], message: &AssistantMessage) -> Vec<MessageChange> {
	let prefix = partial_path();
	let mut changes = Vec::new();
	let mut whole = HashSet::new();
	for op in view_ops {
		let path = op_path(op);
		if !path_starts_with(path, &prefix) {
			if path_is_prefix_of(path, &prefix) {
				return vec![MessageChange::Message { message: message.clone() }];
			}
			continue;
		}
		let rest = &path[prefix.len()..];
		if matches!(rest.first(), Some(Seg::Key(key)) if key == "usage") {
			continue;
		}
		if !matches!(rest.first(), Some(Seg::Key(key)) if key == "content") {
			return vec![MessageChange::Message { message: message.clone() }];
		}
		if rest.len() == 1 {
			if let Op::Splice { start, remove: 0, items, .. } = op {
				for (offset, value) in items.iter().enumerate() {
					let block: AssistantContent = serde_json::from_value(value.clone())
						.unwrap_or_else(|error| panic!("committed assistant content block has an invalid shape: {error}"));
					let content_index = u32::try_from(*start + offset).expect("assistant content index fits u32");
					let value = serde_json::to_value(&block).expect("assistant content block serializes");
					let change = match block {
						AssistantContent::Text(_) => MessageChange::TextStart { content_index, block: value },
						AssistantContent::Thinking(_) => MessageChange::ThinkingStart { content_index, block: value },
						AssistantContent::ToolCall(_) => MessageChange::ToolcallStart { content_index, block: value },
					};
					changes.push(change);
				}
				continue;
			}
			return vec![MessageChange::Message { message: message.clone() }];
		}
		let Some(Seg::Index(content_index)) = rest.get(1) else {
			return vec![MessageChange::Message { message: message.clone() }];
		};
		let content_index = u32::try_from(*content_index).expect("assistant content index fits u32");
		let field = rest.get(2).and_then(|segment| match segment {
			Seg::Key(field) => Some(field.as_str()),
			Seg::Index(_) => None,
		});
		if whole.contains(&content_index) {
			continue;
		}
		match (op, field) {
			(Op::Append { text, .. }, Some("text")) if rest.len() == 3 => {
				changes.push(MessageChange::TextDelta { content_index, delta: text.clone() });
			}
			(Op::Append { text, .. }, Some("thinking")) if rest.len() == 3 => {
				changes.push(MessageChange::ThinkingDelta { content_index, delta: text.clone() });
			}
			(Op::Append { text, .. }, Some("arguments")) => {
				changes.push(MessageChange::ToolcallDelta {
					content_index,
					path: rest[3..].iter().map(segment_value).collect(),
					delta: text.clone(),
				});
			}
			_ => {
				whole.insert(content_index);
				let block = message
					.content
					.get(content_index as usize)
					.expect("changed assistant content index exists");
				changes.push(MessageChange::Block {
					content_index,
					block: serde_json::to_value(block).expect("assistant content block serializes"),
				});
			}
		}
	}
	changes
}

fn segment_value(segment: &Seg) -> Value {
	match segment {
		Seg::Key(key) => Value::String(key.clone()),
		Seg::Index(index) => Value::from(*index as u64),
	}
}

struct ToolUpdate {
	output: Option<ToolOutputChange>,
	details: Option<Option<Value>>,
	diagnostics: Option<Vec<ToolDiagnostic>>,
}

/// Output, details, and diagnostics changes of one running slot.
fn tool_update(
	view_ops: &[Op],
	index: usize,
	slot: &ToolSlot,
	previous: &ToolSlot,
	before: &ConversationView,
	after: &ConversationView,
) -> Option<ToolUpdate> {
	let output_path = [
		PathPart::Key("docs"),
		PathPart::Key("pi.live"),
		PathPart::Key("tools"),
		PathPart::Index(index),
		PathPart::Key("output"),
	];
	let mut trim_start: u64 = 0;
	let mut append = String::new();
	let mut set = false;
	for op in view_ops {
		if !path_starts_with(op_path(op), &output_path) {
			continue;
		}
		match op {
			Op::Truncate { count, .. } => trim_start += *count as u64,
			Op::Append { text, .. } => append.push_str(text),
			_ => set = true,
		}
	}
	let output = if set || (slot.output != previous.output && trim_start == 0 && append.is_empty()) {
		Some(ToolOutputChange::Set { set: slot.output.clone().unwrap_or_default() })
	} else if trim_start > 0 || !append.is_empty() {
		Some(ToolOutputChange::Delta {
			trim_start: (trim_start > 0).then_some(trim_start),
			append: (!append.is_empty()).then_some(append),
		})
	} else {
		None
	};
	let details_path = [
		PathPart::Key("docs"),
		PathPart::Key("pi.live"),
		PathPart::Key("tools"),
		PathPart::Index(index),
		PathPart::Key("details"),
	];
	let diagnostics_path = [
		PathPart::Key("docs"),
		PathPart::Key("pi.live"),
		PathPart::Key("tools"),
		PathPart::Index(index),
		PathPart::Key("diagnostics"),
	];
	let previous_details = raw_tool_details(before, &slot.call_id);
	let details = raw_tool_details(after, &slot.call_id);
	let details_equal = previous_details == details;
	let details_changed = details.is_some()
		&& (!details_equal
			|| (matches!(details, Some(Value::Object(_) | Value::Array(_))) && ops_touch_path(view_ops, &details_path)));
	let diagnostics_changed = slot.diagnostics != previous.diagnostics
		|| (slot.diagnostics.is_some() && ops_touch_path(view_ops, &diagnostics_path));
	let update = ToolUpdate {
		output,
		details: details_changed.then(|| Some(details.cloned().unwrap_or(Value::Null))),
		diagnostics: diagnostics_changed.then(|| slot.diagnostics.clone().unwrap_or_default()),
	};
	(update.output.is_some() || update.details.is_some() || update.diagnostics.is_some()).then_some(update)
}

fn raw_tool_details<'a>(view: &'a ConversationView, call_id: &str) -> Option<&'a Value> {
	let slots = view.docs.get("pi.live")?.get("tools")?.as_array()?;
	slots
		.iter()
		.find(|slot| slot.get("callId").and_then(Value::as_str) == Some(call_id))?
		.get("details")
}
