//! Port of `packages/durable/src/harness/task-graph.ts`.
//!
//! Deviations: D1 — erased `TaskRecord` (`state.checkpoint` is raw JSON;
//! `checkpoint["phase"]` names the phase). D2 — the upstream
//! `AttachedReplicatedState` surface is carried by
//! `crate::session::observation` frames like `view.rs`; the public surface
//! is the [`TaskGraphWatch`] watch handle. D3 — the `tasks` record map is a
//! `BTreeMap<String, TaskGraphNode>` keyed by decimal ID (upstream JS
//! object, ordered numerically by insertion; Rust keeps key order lexical —
//! zero-padded IDs preserve ascending order under the map's iteration).

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_chord::context::Context;
use pi_chord::delta::{apply_immutable, Op, Seg};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::Value;

use crate::harness::types::{ConversationId, JoinPolicy, TaskId};
use crate::harness::util::{abort_reason_text, closed_error, scan_all};
use crate::session::observation::{CommittedWatch, CommittedStateSource};
use crate::session::Session;
use crate::types::{CommitPublication, Storage, TaskOutcomeStatus, TaskRecord, TaskStateStatus};

/// A live task's durable status without its checkpoint and outcome payloads
/// (spec §9.5).
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum TaskGraphState {
	#[serde(rename = "pending")]
	Pending { phase: String },
	#[serde(rename = "running")]
	Running { phase: String },
	#[serde(rename = "waiting")]
	Waiting { phase: String, on: Vec<TaskId>, policy: JoinPolicy },
	/// Outcome held until its ordinary owned work drains.
	#[serde(rename = "completing")]
	Completing { outcome: TaskOutcomeStatus },
}

/// One node of the live task graph.
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskGraphNode {
	pub id: TaskId,
	pub kind: String,
	pub conversation_id: ConversationId,
	/// Owner task; absent for a conversation-owned task.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub owner: Option<TaskId>,
	pub background: bool,
	pub abort_requested: bool,
	pub state: TaskGraphState,
	/// Conversations this task owns, in ID order.
	pub conversations: Vec<ConversationId>,
}

/// Every live task of the Session (spec §9.5).
#[derive(Debug, Clone, Default, PartialEq, Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskGraph {
	/// Every live task, keyed by its decimal ID.
	pub tasks: BTreeMap<String, TaskGraphNode>,
}

/// Serialized exact-frame watch of the task graph.
pub type TaskGraphWatch = crate::session::observation::CommittedWatch<TaskGraph>;

const LIVE_STATUSES: [TaskStateStatus; 4] = [
	TaskStateStatus::Pending,
	TaskStateStatus::Running,
	TaskStateStatus::Waiting,
	TaskStateStatus::Completing,
];
const SCAN_PAGE_SIZE: u64 = 256;
fn node_with_changes(
	value: &TaskGraph,
	changed: &BTreeMap<String, Option<TaskGraphNode>>,
	key: &str,
) -> Option<TaskGraphNode> {
	match changed.get(key) {
		Some(node) => node.clone(),
		None => value.tasks.get(key).cloned(),
	}
}

type Mount = Arc<Mutex<TaskGraphMount>>;

trait TaskGraphObserver: Send + Sync {
	fn advance(&self, value: &TaskGraph, ops: &[Op], context: &Context);
	fn close_session(&self);
}

/// Forwards mount revisions into a task graph watch (upstream: the watch is
/// itself the registered mount observer, `task-graph.ts` `#attach`).
struct WatchObserver {
	watch: TaskGraphWatch,
}

impl TaskGraphObserver for WatchObserver {
	fn advance(&self, value: &TaskGraph, ops: &[Op], context: &Context) {
		self.watch.advance(value.clone(), ops.to_vec(), context.clone());
	}
	fn close_session(&self) {
		self.watch.close_session();
	}
}

struct TaskGraphMount {
	value: TaskGraph,
	observers: Vec<Arc<dyn TaskGraphObserver>>,
}

/**
 * The Harness's task graph mount: built on the Session line by its first
 * observer and dropped with its last. It advances from the Session's commit
 * publications, which are durable.
 */
pub struct TaskGraphView {
	session: Arc<Session>,
	storage: Arc<dyn Storage>,
	mount: Mutex<Option<Mount>>,
	closed: Mutex<bool>,
}

impl TaskGraphView {
	pub fn new(session: Arc<Session>, storage: Arc<dyn Storage>) -> Arc<Self> {
		let view = Arc::new(Self {
			session,
			storage,
			mount: Mutex::new(None),
			closed: Mutex::new(false),
		});
		let subscriber = view.clone();
		view.session.subscribe_commits(Arc::new(move |publication: &CommitPublication, context: &Context| {
			let mount = subscriber.mount.lock().clone();
			if let Some(mount) = mount {
				advance(&mount, publication, context);
			}
		}));
		let closer = view.clone();
		view.session.subscribe_close(Arc::new(move || {
			*closer.closed.lock() = true;
			if let Some(mount) = closer.mount.lock().take() {
				let observers = mount.lock().observers.clone();
				for observer in observers {
					observer.close_session();
				}
			}
		}));
		view
	}

	/// A serialized exact-frame watch of the graph; cancelling `context`
	/// stops it.
	pub async fn watch(self: &Arc<Self>, context: &Context) -> Result<TaskGraphWatch, String> {
		let view = self.clone();
		let observer = self
			.session
			.read_on_line(async move {
				let value = view.build_value(context).await?;
				let watch = CommittedWatch::new(value.clone(), Box::new({
					let view = view.clone();
					move || {
						*view.mount.lock() = None;
					}
				}));
				if *view.closed.lock() {
					return Err(closed_error());
				}
				if let Some(signal) = context.abort_signal() {
					if signal.aborted() {
						return Err(abort_reason_text(&signal));
					}
				}
				let mount = Arc::new(Mutex::new(TaskGraphMount {
					value,
					observers: vec![Arc::new(WatchObserver { watch: watch.clone() }) as Arc<dyn TaskGraphObserver>],
				}));
				*view.mount.lock() = Some(mount);
				Ok::<_, String>(watch)
			})
			.await?;
		if let Some(signal) = context.abort_signal() {
			if signal.aborted() {
				observer.cancel();
				return Err(abort_reason_text(&signal));
			}
			let _ = observer.observe_cancellation(&signal);
		}
		Ok(observer)
	}

	async fn build_value(&self, context: &Context) -> Result<TaskGraph, String> {
		let mut records: Vec<TaskRecord> = Vec::new();
		for status in LIVE_STATUSES {
			let query = crate::types::TaskQuery { conversation_id: None, status: Some(status), ..Default::default() };
			records.extend(scan_all(|cursor| {
				let storage = self.storage.clone();
				let query = query.clone();
				async move { storage.scan_tasks(&query, SCAN_PAGE_SIZE, cursor.as_ref(), context).await }
			})
			.await);
		}
		records.sort_by_key(|record| record.id);
		let mut tasks = BTreeMap::new();
		for record in records {
			let query = crate::types::ConversationQuery { owner_task_id: Some(record.id), ..Default::default() };
			let mut owned = scan_all(|cursor| {
				let storage = self.storage.clone();
				let query = query.clone();
				async move { storage.scan_conversations(&query, SCAN_PAGE_SIZE, cursor.as_ref(), context).await }
			})
			.await;
			owned.sort_by_key(|conversation| conversation.id);
			let conversations = owned.into_iter().map(|conversation| conversation.id).collect();
			tasks.insert(record.id.0.to_string(), node_of(&record, conversations));
		}
		Ok(TaskGraph { tasks })
	}
}

/// Derive the mount's operations from one publication, apply them, and hand
/// the revision to every observer.
fn advance(mount: &Mount, publication: &CommitPublication, context: &Context) {
	let mut ops: Vec<Op> = Vec::new();
	// Nodes this publication set or deleted, over the mount's value.
	let mut changed: BTreeMap<String, Option<TaskGraphNode>> = BTreeMap::new();
	let mut value = mount.lock().value.clone();
	for change in &publication.changes {
		let crate::types::CommitChange::Table(crate::types::TableCommitChange::Task { value: record }) = change
		else {
			continue;
		};
		let key = record.id.0.to_string();
		let previous = node_with_changes(&value, &changed, &key);
		if record.state.status == TaskStateStatus::Terminal {
			if previous.is_none() {
				continue;
			}
			ops.push(Op::Delete { path: vec![Seg::key("tasks"), Seg::key(key.clone())] });
			changed.insert(key, None);
			continue;
		}
		let next = node_of(record, previous.as_ref().map(|node| node.conversations.clone()).unwrap_or_default());
		if previous.as_ref() == Some(&next) {
			continue;
		}
		ops.push(Op::Set {
			path: vec![Seg::key("tasks"), Seg::key(key.clone())],
			value: serde_json::to_value(&next).expect("TaskGraphNode serializes"),
		});
		changed.insert(key, Some(next));
	}
	// After the tasks, so a conversation created with its owner task in one
	// commit finds the owner's node. Change order within a publication is
	// unspecified, so each owner's list is sorted again.
	let mut created: BTreeMap<String, Vec<ConversationId>> = BTreeMap::new();
	for change in &publication.changes {
		let crate::types::CommitChange::Table(crate::types::TableCommitChange::Conversation { value: conversation }) =
			change
		else {
			continue;
		};
		let Some(owner) = &conversation.owner else { continue };
		let key = owner.task_id.0.to_string();
		if node_with_changes(&value, &changed, &key).is_some() {
			created.entry(key).or_default().push(conversation.id);
		}
	}
	for (key, mut ids) in created {
		let mut conversations = node_with_changes(&value, &changed, &key).map(|node| node.conversations).unwrap_or_default();
		conversations.append(&mut ids);
		conversations.sort();
		ops.push(Op::Set {
			path: vec![Seg::key("tasks"), Seg::key(key.clone()), Seg::key("conversations")],
			value: serde_json::to_value(&conversations).expect("conversations serialize"),
		});
		if let Some(existing) = changed.get_mut(&key) {
			if let Some(node) = existing {
				node.conversations = conversations;
			}
		} else if let Some(node) = value.tasks.get_mut(&key) {
			node.conversations = conversations;
		}
	}
	if ops.is_empty() {
		return;
	}
	let applied = apply_immutable(
		&serde_json::to_value(&value).expect("TaskGraph serializes"),
		&ops,
	);
	if let Ok(applied) = applied {
		if let Ok(graph) = serde_json::from_value::<TaskGraph>(applied) {
			value = graph;
		}
	}
	mount.lock().value = value;
	let frame_context = pi_chord::context::without_abort_signal(context.clone());
	let observers = mount.lock().observers.clone();
	for observer in observers {
		observer.advance(&mount.lock().value, &ops, &frame_context);
	}
}

fn node_of(record: &TaskRecord, conversations: Vec<ConversationId>) -> TaskGraphNode {
	TaskGraphNode {
		id: record.id,
		kind: record.kind.clone(),
		conversation_id: record.conversation_id,
		owner: record.owner,
		background: record.background,
		abort_requested: record.abort_requested,
		state: state_of(record),
		conversations,
	}
}

fn state_of(record: &TaskRecord) -> TaskGraphState {
	let state = &record.state;
	match state.status {
		TaskStateStatus::Pending => TaskGraphState::Pending { phase: phase_of(state.checkpoint.as_ref()) },
		TaskStateStatus::Running => TaskGraphState::Running { phase: phase_of(state.checkpoint.as_ref()) },
		TaskStateStatus::Waiting => TaskGraphState::Waiting {
			phase: phase_of(state.checkpoint.as_ref()),
			on: state.on.clone().unwrap_or_default(),
			policy: state.policy.unwrap_or(JoinPolicy::AllSettled),
		},
		// Terminal records never reach here: they leave the graph.
		TaskStateStatus::Completing | TaskStateStatus::Terminal => TaskGraphState::Completing {
			outcome: state
				.outcome
				.as_ref()
				.map(|outcome| outcome.status)
				.unwrap_or(TaskOutcomeStatus::Faulted),
		},
	}
}

fn phase_of(checkpoint: Option<&Value>) -> String {
	checkpoint
		.and_then(|checkpoint| checkpoint.get("phase"))
		.and_then(|phase| phase.as_str())
		.unwrap_or_default()
		.to_string()
}

impl crate::session::observation::ObservedValue for TaskGraph {
	fn to_frame_json(&self) -> serde_json::Value {
		serde_json::to_value(self).expect("TaskGraph serializes")
	}

	fn is_retired(&self) -> bool {
		false
	}
}
