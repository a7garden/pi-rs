
//! Port of `packages/durable/src/harness/view.ts`.
//!
//! Deviations: D1 — the upstream `WeakMap` harness→views link becomes an
//! explicit `Arc<ConversationViews>` field on the Harness (no weak maps on
//! `object` identities); `conversation_views` panics upstream, here the
//! Harness carries the mounts directly. D2 — `AttachedReplicatedState` is
//! chord's `ReplicatedStateSource` frame machinery via
//! `crate::session::observation::StateSource` (pub(crate)); the public
//! surface is the [`ConversationWatch`] watch handle.

use std::future::Future;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use pi_chord::context::Context;
use pi_chord::delta::{apply_immutable, Op, Path, Seg};
use parking_lot::Mutex;
use serde_json::Value;

use crate::harness::agent::agent_doc;
use crate::harness::context::{active_entries, capture_context_bounds};
use crate::harness::inbox::inbox_doc;
use crate::harness::live::live_doc;
use crate::harness::provider::provider_doc;
use crate::harness::types::{ConversationId, WatchHandle};
use crate::harness::usage::usage_doc;
use crate::harness::util::{abort_reason_text, closed_error};
use crate::session::observation::{CommittedWatch, CommittedStateSource};
use crate::session::Session;
use crate::types::{
	CommitPublication, ConversationRecord, DocumentId, EntryRecord, JsonObject, Storage,
};

/// Structural mount of one conversation's active transcript and built-in
/// documents (spec §9.3).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConversationView {
	pub conversation: ConversationRecord,
	/// Raw active entries, as `ContextView.entries`: the head marker, then
	/// the non-head entries from its head.
	pub entries: Vec<EntryRecord>,
	/// Built-in conversation documents keyed by kind; absent documents are
	/// absent.
	pub docs: BTreeMap<String, JsonObject>,
}

/// Serialized exact-frame watch of one conversation's view.
pub type ConversationWatch = crate::session::observation::CommittedWatch<ConversationView>;

/// Receives each next revision of a mount, and the Session's close.
pub trait ViewObserver: Send + Sync {
	fn advance(&self, _value: &ConversationView, _ops: &[Op], _context: &Context) {}
	/// Every publication, after the mount took it; `ops` are the mount's,
	/// possibly none.
	fn publication(
		&self,
		_before: &ConversationView,
		_after: &ConversationView,
		_ops: &[Op],
		_publication: &CommitPublication,
		_context: &Context,
	) {
	}
	fn close_session(&self);
}

/// Forwards mount revisions into a view's [`CommittedStateSource`] (upstream:
/// the view's replicated state subscribes to the mounted revisions).
struct SourceObserver {
	source: crate::session::observation::ViewSource,
}

impl ViewObserver for SourceObserver {
	fn advance(&self, value: &ConversationView, ops: &[Op], context: &Context) {
		self.source.advance(value.clone(), ops.to_vec(), context.clone());
	}

	fn close_session(&self) {
		self.source.close_session();
	}
}

fn mounted_kinds() -> Vec<(&'static str, crate::documents::DocToken)> {
	vec![
		("pi.agent", agent_doc()),
		("pi.live", live_doc()),
		("pi.inbox", inbox_doc()),
		("pi.provider", provider_doc()),
		("pi.usage", usage_doc()),
	]
}

/// One conversation's mount: its current revision, the document incarnations
/// it shows, and its observers.
struct ObserverRegistration {
	id: Arc<()>,
	observer: Arc<dyn ViewObserver>,
}

struct Mount {
	value: ConversationView,
	/// Mounted incarnation and definition version per kind; another
	/// incarnation or version is set whole.
	docs: HashMap<String, (DocumentId, u64)>,
	observers: Vec<ObserverRegistration>,
}

/**
 * The Harness's conversation view mounts: at most one per conversation,
 * built on the Session line by its first observer and dropped with its last.
 * Each mount advances from the Session's commit publications, which are
 * durable.
 */
pub struct ConversationViews {
	session: Arc<Session>,
	storage: Arc<dyn Storage>,
	mounts: Arc<Mutex<HashMap<ConversationId, Arc<Mutex<Mount>>>>>,
	closed: Mutex<bool>,
}

/// Observer and acquisition value created atomically on the Session line.
pub(crate) struct Attached<O, V> {
	pub observer: Arc<O>,
	pub value: V,
}

impl ConversationViews {
	pub fn new(session: Arc<Session>, storage: Arc<dyn Storage>) -> Arc<Self> {
		let views = Arc::new(Self {
			session,
			storage,
			mounts: Arc::new(Mutex::new(HashMap::new())),
			closed: Mutex::new(false),
		});
		let subscriber = views.clone();
		views.session.subscribe_commits(Arc::new(move |publication: &CommitPublication, context: &Context| {
			let mounts = subscriber.mounts.lock().iter().map(|(k,v)| (*k, v.clone())).collect::<Vec<_>>();
			for (id, mount) in mounts {
				advance(&id, &mount, publication, context);
			}
		}));
		let closer = views.clone();
		views.session.subscribe_close(Arc::new(move || {
			*closer.closed.lock() = true;
			let mounts = closer.mounts.lock().drain().collect::<Vec<_>>();
			for (_, mount) in mounts {
				let observers = mount.lock().observers.iter().map(|registration| registration.observer.clone()).collect::<Vec<_>>();
				for observer in observers {
					observer.close_session();
				}
			}
		}));
		views
	}

	/// A disposable read-only Chord state of the view.
	pub async fn state(
		self: &Arc<Self>,
		id: ConversationId,
		context: &Context,
	) -> Result<crate::session::observation::DocumentStateView, String> {
		let views = self.clone();
		self.session
			.read_on_line(async move {
				let attach = views.attach_internal(id, context).await?;
				crate::session::observation::new_view(attach)
			})
			.await
	}

	/// A serialized exact-frame watch of the view; cancelling `context` stops
	/// it.
	pub async fn watch(
		self: &Arc<Self>,
		id: ConversationId,
		context: &Context,
	) -> Result<ConversationWatch, String> {
		let views = self.clone();
		let observer = self
			.session
			.read_on_line(async move {
				let attach = views.attach_watch_internal(id, context).await?;
				Ok::<_, String>(attach)
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

	async fn attach_internal(
		self: &Arc<Self>,
		id: ConversationId,
		context: &Context,
	) -> Result<crate::session::observation::ViewSource, String> {
		let mount = self.mount_for(id, context).await?;
		let value = mount.lock().value.clone();
		let registration_id = Arc::new(());
		let release = Box::new({
			let mounts = Arc::downgrade(&self.mounts);
			let mount = Arc::downgrade(&mount);
			let registration_id = Arc::downgrade(&registration_id);
			move || {
				let (Some(mount), Some(registration_id)) = (mount.upgrade(), registration_id.upgrade()) else {
					return;
				};
				let empty = {
					let mut mount = mount.lock();
					mount.observers.retain(|registered| !Arc::ptr_eq(&registered.id, &registration_id));
					mount.observers.is_empty()
				};
				if empty {
					if let Some(mounts) = mounts.upgrade() {
						let mut mounts = mounts.lock();
						if mounts.get(&id).is_some_and(|registered| Arc::ptr_eq(registered, &mount)) {
							mounts.remove(&id);
						}
					}
				}
			}
		});
		let (source, closer) = crate::session::observation::CommittedStateSource::new(value, release);
		let observer = Arc::new(SourceObserver { source: source.clone() });
		// Close or cancellation may begin while the mount hydrates; register
		// nothing then.
		if *self.closed.lock() {
			source.close_session();
			return Err(closed_error());
		}
		if let Some(signal) = context.abort_signal() {
			if signal.aborted() {
				source.close_session();
				return Err(abort_reason_text(&signal));
			}
		}
		mount.lock().observers.push(ObserverRegistration { id: registration_id, observer });
		let _ = closer;
		Ok(source)
	}

	async fn attach_watch_internal(
		self: &Arc<Self>,
		id: ConversationId,
		context: &Context,
	) -> Result<CommittedWatch<ConversationView>, String> {
		let mount = self.mount_for(id, context).await?;
		let value = mount.lock().value.clone();
		let watch = crate::session::observation::CommittedWatch::new(
			value,
			Box::new({
				let mounts = self.mounts.clone();
				let mount = Arc::clone(&mount);
				move || {
					let mut mounts = mounts.lock();
					if let std::collections::hash_map::Entry::Occupied(entry) = mounts.entry(id) {
						if Arc::ptr_eq(entry.get(), &mount) {
							entry.remove();
						}
					}
				}
			}),
		);
		if *self.closed.lock() {
			return Err(closed_error());
		}
		if let Some(signal) = context.abort_signal() {
			if signal.aborted() {
				return Err(abort_reason_text(&signal));
			}
		}
		self.mounts.lock().insert(id, mount);
		Ok(watch)
	}

	async fn mount_for(
		self: &Arc<Self>,
		id: ConversationId,
		context: &Context,
	) -> Result<Arc<Mutex<Mount>>, String> {
		if let Some(mount) = self.mounts.lock().get(&id) {
			return Ok(mount.clone());
		}
		let mount = Arc::new(Mutex::new(self.build(id, context).await?));
		self.mounts.lock().insert(id, mount.clone());
		Ok(mount)
	}

	async fn build(&self, id: ConversationId, context: &Context) -> Result<Mount, String> {
		let conversation = self
			.storage
			.conversation(id, context)
			.await
			.ok_or_else(|| format!("Conversation {id} does not exist"))?;
		let bounds = capture_context_bounds(self.storage.as_ref(), id, context, None).await?;
		let entries = active_entries(self.storage.as_ref(), id, bounds, context).await?;
		let mut docs = BTreeMap::new();
		let mut incarnations = HashMap::new();
		for (kind, token) in &mounted_kinds() {
			let Some(loaded) = self.session.conversation_document_on_line(token, id, context).map_err(|e| e.to_string())? else {
				continue;
			};
			docs.insert(kind.to_string(), loaded.value);
			incarnations.insert(kind.to_string(), (loaded.record.id, loaded.version));
		}
		Ok(Mount {
			value: ConversationView { conversation, entries, docs },
			docs: incarnations,
			observers: Vec::new(),
		})
	}
}

/// Derive the mount's operations from one publication, apply them, and hand
/// the revision to every observer.
fn advance(id: &ConversationId, mount: &Arc<Mutex<Mount>>, publication: &CommitPublication, context: &Context) {
	let mut mount = mount.lock();
	let mut doc_ops: Vec<Op> = Vec::new();
	let mut entry_ops: Vec<Op> = Vec::new();
	let mut entries = mount.value.entries.clone();
	// Entry writes are published in ID order.
	for change in &publication.changes {
		let crate::types::CommitChange::Table(crate::types::TableCommitChange::Entry { value: entry }) = change
		else {
			continue;
		};
		if entry.conversation_id != *id {
			continue;
		}
		let value = serde_json::to_value(entry).expect("EntryRecord serializes");
		if entry.head.is_none() {
			entry_ops.push(Op::Splice {
				path: vec![Seg::key("entries")],
				start: entries.len(),
				remove: 0,
				items: vec![value],
			});
			entries.push(entry.clone());
			continue;
		}
		// A head marker keeps the non-head entries from its head, which are
		// always a suffix, and goes in front.
		let target = entry.head.expect("checked");
		let kept = entries
			.iter()
			.position(|candidate| candidate.head.is_none() && candidate.id >= target)
			.unwrap_or(entries.len());
		entry_ops.push(Op::Splice {
			path: vec![Seg::key("entries")],
			start: 0,
			remove: kept,
			items: vec![value],
		});
		entries = std::iter::once(entry.clone()).chain(entries.into_iter().skip(kept)).collect();
		continue;
	}
	for change in &publication.changes {
		let crate::types::CommitChange::Document(crate::types::DocumentCommitChange::Document {
			record,
			conversation_id,
			version,
			value,
			ops,
		}) = change
		else {
			continue;
		};
		if *conversation_id != Some(*id) {
			continue;
		}
		let kind = record.kind.clone();
		if !mounted_kinds().iter().any(|(name, _)| *name == kind) || record.key.is_some() {
			continue;
		}
		let path: Path = vec![Seg::key("docs"), Seg::key(kind.clone())];
		let mounted = mount.docs.get(&kind).cloned();
		match value {
			None => {
				if mounted.map(|(id, _)| id) != Some(record.id) {
					continue;
				}
				mount.docs.remove(&kind);
				doc_ops.push(Op::Delete { path });
			}
			Some(value) => {
				match mounted {
					Some((id, mounted_version)) if id == record.id && mounted_version == version.expect("version") => {
						for op in ops {
							doc_ops.push(prefixed(op, &path));
						}
					}
					_ => {
						mount.docs.insert(kind.clone(), (record.id, version.expect("version")));
						doc_ops.push(Op::Set { path, value: Value::Object(value.clone()) });
					}
				}
			}
		}
	}
	let before = mount.value.clone();
	let ops: Vec<Op> = doc_ops.iter().cloned().chain(entry_ops.iter().cloned()).collect();
	let frame_context = pi_chord::context::without_abort_signal(context.clone());
	if !ops.is_empty() {
		if doc_ops.is_empty() {
			mount.value.entries = entries;
		} else {
			let applied = apply_immutable(
				&serde_json::to_value(&mount.value).expect("ConversationView serializes"),
				&doc_ops,
			)
			.map_err(|error| format!("View mount delta failed: {error}"));
			match applied {
				Ok(value) => {
					let mut value: ConversationView = serde_json::from_value(value)
						.map_err(|error| format!("View mount shape failed: {error}"))
						.unwrap_or_else(|_| mount.value.clone());
					value.entries = entries;
					mount.value = value;
				}
				Err(_) => {
					mount.value.entries = entries;
				}
			}
		}
		for registration in mount.observers.iter() {
			registration.observer.advance(&mount.value, &ops, &frame_context);
		}
	}
	for registration in mount.observers.iter() {
		registration.observer.publication(&before, &mount.value, &ops, publication, &frame_context);
	}
}

/// A context that keeps no abort signal (upstream `withoutAbortSignal`).
fn without_abort_signal(context: &Context) -> Context {
	pi_chord::context::without_abort_signal(context.clone())
}

/// `op` moved under `prefix`; a root replacement becomes a set of the prefix.
fn prefixed(op: &Op, prefix: &Path) -> Op {
	fn at(path: &[Seg], prefix: &Path) -> Path {
		prefix.iter().cloned().chain(path.iter().cloned()).collect()
	}
	match op {
		Op::Replace(value) => Op::Set { path: prefix.clone(), value: value.clone() },
		Op::Splice { path, start, remove, items } => Op::Splice {
			path: at(path, prefix),
			start: *start,
			remove: *remove,
			items: items.clone(),
		},
		Op::Permute { path, permutation } => Op::Permute { path: at(path, prefix), permutation: permutation.clone() },
		Op::Set { path, value } => Op::Set { path: at(path, prefix), value: value.clone() },
		Op::Delete { path } => Op::Delete { path: at(path, prefix) },
		Op::Append { path, text } => Op::Append { path: at(path, prefix), text: text.clone() },
		Op::Truncate { path, count } => Op::Truncate { path: at(path, prefix), count: *count },
	}
}

impl ConversationViews {

	/// Atomically initialize and register a long-lived observer on the
	/// Session line. `initialize` receives the revision, storage, acquisition
	/// context, and a detach action suitable for the observer's lifetime.
	pub(crate) async fn attach_observer<O, V, F, Fut>(
		self: &Arc<Self>,
		id: ConversationId,
		context: &Context,
		initialize: F,
	) -> Result<Attached<O, V>, String>
	where
		O: ViewObserver + 'static,
		V: Send + 'static,
		F: FnOnce(
				ConversationView,
				Arc<dyn Storage>,
				Context,
				Box<dyn FnOnce() + Send>,
			) -> Fut
			+ Send
			+ 'static,
		Fut: Future<Output = Result<(O, V), String>> + Send + 'static,
	{
		let views = self.clone();
		let context = context.clone();
		self.session
			.read_on_line(async move {
				if *views.closed.lock() {
					return Err(closed_error());
				}
				if let Some(signal) = context.abort_signal() {
					if signal.aborted() {
						return Err(abort_reason_text(&signal));
					}
				}
				let mount = views.mount_for(id, &context).await?;
				let value = mount.lock().value.clone();
				let registration_id = Arc::new(());
				let detach = Box::new({
					let mount = Arc::downgrade(&mount);
					let mounts = Arc::downgrade(&views.mounts);
					let registration_id = Arc::downgrade(&registration_id);
					move || {
						let (Some(mount), Some(registration_id)) = (mount.upgrade(), registration_id.upgrade()) else {
							return;
						};
						let empty = {
							let mut mount = mount.lock();
							mount.observers.retain(|registered| !Arc::ptr_eq(&registered.id, &registration_id));
							mount.observers.is_empty()
						};
						if empty {
							if let Some(mounts) = mounts.upgrade() {
								let mut mounts = mounts.lock();
								if mounts.get(&id).is_some_and(|registered| Arc::ptr_eq(registered, &mount)) {
									mounts.remove(&id);
								}
							}
						}
					}
				});
				let (observer, value) = match initialize(value, views.storage.clone(), context.clone(), detach).await {
					Ok(result) => result,
					Err(error) => return Err(error),
				};
				if *views.closed.lock() {
					observer.close_session();
					return Err(closed_error());
				}
				if let Some(signal) = context.abort_signal() {
					if signal.aborted() {
						observer.close_session();
						return Err(abort_reason_text(&signal));
					}
				}
				let observer = Arc::new(observer);
				mount.lock().observers.push(ObserverRegistration {
					id: registration_id,
					observer: observer.clone(),
				});
				Ok(Attached { observer, value })
			})
			.await
	}
}

impl crate::session::observation::ObservedValue for ConversationView {
	fn to_frame_json(&self) -> serde_json::Value {
		serde_json::to_value(self).expect("ConversationView serializes")
	}

	fn is_retired(&self) -> bool {
		false
	}
}
