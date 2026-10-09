//! Port of `experimental/session-worker-manager.ts` — the server-side
//! Session/process bookkeeping owned by one replaceable server process
//! ([`SessionWorkerManager`]): worker launches, demand reconciliation,
//! routed operation/service plumbing, and the live `workerPids` snapshot
//! consumed by [`crate::experimental::server`].
//!
//! # Deviations (D161)
//!
//! - Upstream `Promise`s become [`PromiseSlot<T>`] — a first-write-wins
//!   shared cell — so awaited launch/demand/operation completions keep the
//!   upstream "settled once, many awaiters" semantics without `Arc`-cloning
//!   boxed errors; launch and demand errors travel as
//!   `Arc<dyn Error + Send + Sync>` (pi-server's error shape) while worker
//!   operation errors stay structured (`WorkerOpError`) so the routed
//!   `ServerError` code survives to the client.
//! - Node `setTimeout` + `unref` become spawned tokio tasks whose timeout
//!   paths re-check the pending maps before acting (settled entries are
//!   already removed, so the late fire is a no-op).
//! - The per-subscription `deliveryTail` promise chain becomes a
//!   [`tokio::sync::Mutex`] held across each delivery; ordering within one
//!   subscription is preserved.
//! - `workerPids` is a [`WorkerPids`] snapshot handle (`Arc` + mutex map)
//!   shared with the running server instead of an exported `readonly` Map.
//! - `process.kill(pid, "SIGKILL")` becomes `libc::kill(pid, SIGKILL)` (the
//!   child may have been reaped by the watcher task; `ESRCH` is swallowed
//!   like upstream).
//! - Operation abort registration maps to a spawned watcher awaiting
//!   [`pi_chord::AbortSignal::aborted_future`] raced against the operation
//!   slot; `operation_cancel` is sent and the slot rejected on abort.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use base64::Engine;
use futures_util::future::BoxFuture;
use parking_lot::Mutex as PLMutex;
use pi_chord::context::abort_error;
use pi_chord::services::errors::{error, AnyError, ChordError};
use pi_chord::types::{BoxFuture as ChordBoxFuture, ServiceCall};
use pi_chord::{
	create_service_unsubscribe_call, decode_service_control_call, parse_service_provider_update,
	BACKGROUND_CONTEXT, Context, JsonValue, ServiceControlCall, TODO_CONTEXT,
};
use pi_server::types::{
	MaybeFuture, RoutedSessionAttachment, RoutedSessionHandle, ServicePublish,
};
use serde_json::Value;

use crate::experimental::coordinator::{CoordinatorConnection, CoordinatorConnectionEvent, EventGuard};
use crate::experimental::process::{
	spawn_internal_process, InternalProcessSpawnOptions, InternalProcessRole,
};
use crate::experimental::session_catalog::SessionCatalogMetadata;
use crate::experimental::session_worker::{
	SessionWorkerEvent, SessionWorkerMetadata, SessionWorkerOptions, WorkerOperationResponse,
	WorkerOperationScope, SESSION_WORKER_CONTROL_ADDRESS_ENV, SESSION_WORKER_CONTROL_TOKEN_ENV,
	SESSION_WORKER_PEER_ID_ENV, SESSION_WORKER_SESSION_KEY_ENV,
};

const WORKER_STARTUP_TIMEOUT_MS: u64 = 15_000;
const WORKER_SHUTDOWN_TIMEOUT_MS: u64 = 10_000;
const WORKER_DISCOVERY_TIMEOUT_MS: u64 = 5_000;
const WORKER_DEMAND_TIMEOUT_MS: u64 = 5_000;

/// Shared error value crossing manager boundaries (`PromiseSlot` payloads).
pub(crate) type SharedError = Arc<dyn std::error::Error + Send + Sync>;

/// Wrap a chord boxed error into the shared shape.
pub(crate) fn shared_error(value: AnyError) -> SharedError {
	Arc::from(value)
}

/// Build a shared error from a message.
fn message_error(message: impl Into<String>) -> SharedError {
	shared_error(error(message))
}

// ---------------------------------------------------------------------------
// Promise slot (D161).
// ---------------------------------------------------------------------------

/// First-write-wins shared cell standing in for an upstream `Promise<T>`.
#[derive(Clone)]
pub(crate) struct PromiseSlot<T: Clone> {
	state: Arc<PromiseSlotState<T>>,
}

struct PromiseSlotState<T> {
	slot: PLMutex<Option<T>>,
	notify: tokio::sync::Notify,
}

impl<T: Clone> PromiseSlot<T> {
	pub fn new() -> Self {
		PromiseSlot {
			state: Arc::new(PromiseSlotState {
				slot: PLMutex::new(None),
				notify: tokio::sync::Notify::new(),
			}),
		}
	}

	/// Resolve the slot; later resolutions are ignored (`resolveTerminated`
	/// and `stopPromise` memoization both rely on this).
	pub fn resolve(&self, value: T) -> bool {
		let mut slot = self.state.slot.lock();
		if slot.is_some() {
			return false;
		}
		*slot = Some(value);
		drop(slot);
		self.state.notify.notify_waiters();
		true
	}

	pub fn is_resolved(&self) -> bool {
		self.state.slot.lock().is_some()
	}

	/// Await the settled value.
	pub async fn get(&self) -> T {
		loop {
			let notified = self.state.notify.notified();
			if let Some(value) = self.state.slot.lock().clone() {
				return value;
			}
			notified.await;
		}
	}
}

// ---------------------------------------------------------------------------
// Errors and record types.
// ---------------------------------------------------------------------------

/// Upstream `SessionPluginSelectionConflictError`.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct SessionPluginSelectionConflictError {
	pub message: String,
}

impl SessionPluginSelectionConflictError {
	fn any(message: impl Into<String>) -> AnyError {
		Box::new(Self { message: message.into() })
	}
}

/// Structured worker operation failure preserving the routed error code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkerOpError {
	pub code: Option<String>,
	pub message: String,
}

/// Upstream `WorkerRecord`.
pub(crate) struct WorkerRecord {
	pub(crate) peer_id: String,
	pub(crate) metadata: SessionCatalogMetadata,
	pub(crate) pid: i32,
	pub(crate) token: String,
	pub(crate) plugin_manifest_paths: Vec<String>,
	/// Resolves with an error for unexpected termination, `None` otherwise.
	pub(crate) terminated: PromiseSlot<Option<SharedError>>,
	pub(crate) attachment_ids: PLMutex<HashSet<String>>,
	pub(crate) expected_stop: AtomicBool,
	pub(crate) stopping: AtomicBool,
	pub(crate) stop_slot: PLMutex<Option<PromiseSlot<()>>>,
	pub(crate) stop_started: AtomicBool,
}

impl WorkerRecord {
	fn same_plugin_paths(&self, manifest_paths: &[String]) -> bool {
		same_strings(&self.plugin_manifest_paths, manifest_paths)
	}
}

/// Upstream `PendingLaunch`.
struct PendingLaunch {
	session_key: String,
	peer_id: String,
	token: String,
	plugin_manifest_paths: Vec<String>,
	child: Arc<tokio::sync::Mutex<tokio::process::Child>>,
	promise: PromiseSlot<Result<Arc<WorkerRecord>, SharedError>>,
}

/// Upstream `PendingDemand`.
struct PendingDemand {
	attachment_id: String,
	attached: bool,
	request_id: String,
	worker: Arc<WorkerRecord>,
	slot: PromiseSlot<Result<(), SharedError>>,
}

/// Upstream `PendingWorkerOperation`.
struct PendingWorkerOperation {
	worker: Arc<WorkerRecord>,
	scope: WorkerOperationScope,
	slot: PromiseSlot<Result<Option<JsonValue>, WorkerOpError>>,
}

/// Upstream `WorkerServiceSubscription`.
struct WorkerServiceSubscription {
	worker: Arc<WorkerRecord>,
	scope: WorkerOperationScope,
	publish: ServicePublish,
	/// Serializes deliveries (upstream `deliveryTail` chaining, D161).
	delivery_lock: Arc<tokio::sync::Mutex<()>>,
}

/// Live `sessionId -> pid` snapshot shared with the running server
/// (upstream the exported `workerPids` Map, D161).
#[derive(Clone, Default)]
pub struct WorkerPids {
	inner: Arc<PLMutex<HashMap<String, i32>>>,
}

impl WorkerPids {
	pub fn get(&self, session_id: &str) -> Option<i32> {
		self.inner.lock().get(session_id).copied()
	}

	pub fn snapshot(&self) -> HashMap<String, i32> {
		self.inner.lock().clone()
	}

	fn set(&self, session_id: &str, pid: i32) {
		self.inner.lock().insert(session_id.to_string(), pid);
	}

	fn remove_if(&self, session_id: &str, pid: i32) {
		let mut pids = self.inner.lock();
		if pids.get(session_id) == Some(&pid) {
			pids.remove(session_id);
		}
	}

	fn clear(&self) {
		self.inner.lock().clear();
	}
}

/// Optional worker model override (`{ provider?, model }`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerModelOptions {
	pub provider: Option<String>,
	pub model: String,
}

#[derive(Default)]
struct ManagerState {
	workers_by_session: HashMap<String, Arc<WorkerRecord>>,
	workers_by_peer: HashMap<String, Arc<WorkerRecord>>,
	pending: HashMap<String, Arc<PendingLaunch>>,
	pending_demand: HashMap<String, PendingDemand>,
	pending_operations: HashMap<String, PendingWorkerOperation>,
	service_subscriptions: HashMap<String, WorkerServiceSubscription>,
	discovery_peers: Option<HashSet<String>>,
	resolve_discovery: Option<PromiseSlot<()>>,
	detached: bool,
	shutting_down: bool,
}

/// Worker model options serialized into the spawn argv (upstream the
/// `parseServerModelOptions` JSON argument).
pub fn serialize_worker_model_options(model: &WorkerModelOptions) -> String {
	let mut object = serde_json::Map::new();
	if let Some(provider) = &model.provider {
		object.insert("provider".to_string(), Value::String(provider.clone()));
	}
	object.insert("model".to_string(), Value::String(model.model.clone()));
	Value::Object(object).to_string()
}

/// `SessionWorkerManager` — Session and process bookkeeping owned by one
/// replaceable server process.
pub struct SessionWorkerManager {
	worker_pids: WorkerPids,
	coordinator: Arc<CoordinatorConnection>,
	session_dir: String,
	model: Option<WorkerModelOptions>,
	on_worker_count_changed: Option<Arc<dyn Fn(usize) + Send + Sync>>,
	state: PLMutex<ManagerState>,
	listener: PLMutex<Option<EventGuard>>,
}

impl SessionWorkerManager {
	pub fn new(
		coordinator: Arc<CoordinatorConnection>,
		session_dir: String,
		model: Option<WorkerModelOptions>,
		on_worker_count_changed: Option<Arc<dyn Fn(usize) + Send + Sync>>,
	) -> Arc<Self> {
		let manager = Arc::new(Self {
			worker_pids: WorkerPids::default(),
			coordinator,
			session_dir,
			model,
			on_worker_count_changed,
			state: PLMutex::new(ManagerState::default()),
			listener: PLMutex::new(None),
		});
		let listener_manager = Arc::clone(&manager);
		let guard = manager.coordinator.on_event(move |event| {
			let event_manager = Arc::clone(&listener_manager);
			let owned = event.clone();
			tokio::spawn(async move { event_manager.handle_coordinator_event(&owned).await });
		});
		*manager.listener.lock() = Some(guard);
		manager
	}

	pub fn worker_pids(&self) -> &WorkerPids {
		&self.worker_pids
	}

	fn server_connection_id(&self) -> &str {
		&self.coordinator.server_connection_id
	}

	/// Upstream `trackedSessions`.
	pub fn tracked_sessions(&self) -> Vec<SessionCatalogMetadata> {
		self.state
			.lock()
			.workers_by_session
			.values()
			.map(|worker| worker.metadata.clone())
			.collect()
	}

	/// Upstream `assertSessionPluginManifestPaths`.
	pub fn assert_session_plugin_manifest_paths(
		&self,
		metadata: &SessionCatalogMetadata,
		manifest_paths: &[String],
	) -> Result<(), AnyError> {
		let state = self.state.lock();
		if let Some(existing) = state.workers_by_session.get(&metadata.path) {
			if !existing.same_plugin_paths(manifest_paths) {
				return Err(SessionPluginSelectionConflictError::any(format!(
					"Session {} is active with a different plugin selection",
					metadata.id
				)));
			}
		}
		if let Some(pending) = state.pending.get(&metadata.path) {
			if !same_strings(&pending.plugin_manifest_paths, manifest_paths) {
				return Err(SessionPluginSelectionConflictError::any(format!(
					"Session {} is starting with a different plugin selection",
					metadata.id
				)));
			}
		}
		Ok(())
	}

	/// Upstream `discover(peerIds)`.
	pub async fn discover(&self, peer_ids: &std::collections::BTreeSet<String>) {
		let (discovered_slot, undiscovered) = {
			let mut state = self.state.lock();
			if state.detached {
				return;
			}
			let undiscovered: HashSet<String> = peer_ids
				.iter()
				.filter(|peer_id| !state.workers_by_peer.contains_key(*peer_id) && state.pending_peer(peer_id).is_none())
				.cloned()
				.collect();
			if undiscovered.is_empty() {
				return;
			}
			let slot = PromiseSlot::new();
			state.discovery_peers = Some(undiscovered.clone());
			state.resolve_discovery = Some(slot.clone());
			(slot, undiscovered)
		};
		let _ = self.coordinator.broadcast(serde_json::json!({ "type": "discover_workers" })).await;
		let _ = tokio::time::timeout(
			std::time::Duration::from_millis(WORKER_DISCOVERY_TIMEOUT_MS),
			discovered_slot.get(),
		)
		.await;
		let mut state = self.state.lock();
		if let Some(peers) = state.discovery_peers.as_mut() {
			for peer_id in &undiscovered {
				peers.remove(peer_id);
			}
			if peers.is_empty() {
				state.discovery_peers = None;
				if let Some(resolver) = state.resolve_discovery.take() {
					resolver.resolve(());
				}
			}
		}
		state.discovery_peers = None;
		state.resolve_discovery = None;
	}

	/// Upstream `openSession(metadata, context, pluginManifestPaths)`.
	pub async fn open_session(
		self: &Arc<Self>,
		metadata: &SessionCatalogMetadata,
		plugin_manifest_paths: &[String],
	) -> Result<Arc<dyn RoutedSessionHandle>, AnyError> {
		{
			let state = self.state.lock();
			if state.detached || state.shutting_down {
				return Err(error("Experimental server is shutting down"));
			}
		}
		self.assert_session_plugin_manifest_paths(metadata, plugin_manifest_paths)?;
		let existing = self.state.lock().workers_by_session.get(&metadata.path).cloned();
		if let Some(worker) = existing {
			return Ok(self.routed_handle(worker));
		}
		let pending = self.state.lock().pending.get(&metadata.path).map(Arc::clone);
		if let Some(pending) = pending {
			let worker = pending
				.promise
				.get()
				.await
				.map_err(|launch_error| -> AnyError { Box::new(ChordError { message: launch_error.to_string() }) })?;
			return Ok(self.routed_handle(worker));
		}
		let worker = self.launch(metadata, plugin_manifest_paths).await?;
		Ok(self.routed_handle(worker))
	}

	/// Upstream `closeSession(metadata, context)`.
	pub async fn close_session(self: &Arc<Self>, metadata: &SessionCatalogMetadata) -> Result<(), AnyError> {
		let worker = { self.state.lock().workers_by_session.get(&metadata.path).cloned() };
		let worker = match worker {
			Some(worker) => Some(worker),
			None => {
				let pending = self.state.lock().pending.get(&metadata.path).map(Arc::clone);
				match pending {
					Some(pending) => pending.promise.get().await.ok(),
					None => None,
				}
			}
		};
		if let Some(worker) = worker {
			self.stop_worker(&worker).await?;
		}
		Ok(())
	}

	fn routed_handle(self: &Arc<Self>, worker: Arc<WorkerRecord>) -> Arc<dyn RoutedSessionHandle> {
		Arc::new(ManagerSessionHandle {
			manager: Arc::clone(self),
			worker,
		})
	}

	async fn attach_client(
		self: &Arc<Self>,
		worker: Arc<WorkerRecord>,
		_context: Context,
	) -> Result<Arc<dyn RoutedSessionAttachment>, AnyError> {
		if self.is_detached_or_stopping() || worker.stopping.load(Ordering::SeqCst) {
			return Err(error("Experimental Session worker is stopping"));
		}
		if !self.state.lock().peer_owns_worker(&worker.peer_id, &worker) {
			return Err(error("Experimental Session worker is no longer available"));
		}
		let attachment_id = new_uuid();
		worker.attachment_ids.lock().insert(attachment_id.clone());
		if let Err(apply_error) = self.apply_demand(&worker, &attachment_id, true, true).await {
			worker.attachment_ids.lock().remove(&attachment_id);
			return Err(apply_error);
		}
		let scope = self.operation_scope(&worker, &attachment_id)?;
		let attachment = Arc::new(ManagerSessionAttachment {
			manager: Arc::clone(self),
			worker: Arc::clone(&worker),
			scope,
			released: Arc::new(AtomicBool::new(false)),
		});
		Ok(attachment)
	}

	async fn invoke_service(
		self: &Arc<Self>,
		worker: &Arc<WorkerRecord>,
		scope: &WorkerOperationScope,
		call: &ServiceCall,
		publish: ServicePublish,
		context: &Context,
	) -> Result<Option<JsonValue>, AnyError> {
		let control = decode_service_control_call(call);
		let mut added_subscription_key: Option<String> = None;
		if let Some(ServiceControlCall::Subscribe { subscription_id, .. }) = &control {
			let key = scoped_service_subscription_key(scope, subscription_id);
			let mut state = self.state.lock();
			if state.service_subscriptions.contains_key(&key) {
				return Err(error("Service subscription ID is already active"));
			}
			state.service_subscriptions.insert(
				key.clone(),
				WorkerServiceSubscription {
					worker: Arc::clone(worker),
					scope: scope.clone(),
					publish,
					delivery_lock: Arc::new(tokio::sync::Mutex::new(())),
				},
			);
			added_subscription_key = Some(key);
		}
		let response = match self.invoke(worker, scope, call, context).await {
			Ok(response) => response,
			// A failed subscribe compensates like upstream.
			Err(invoke_error) => {
				if let (Some(key), Some(ServiceControlCall::Subscribe { subscription_id, .. })) =
					(&added_subscription_key, &control)
				{
					self.state.lock().service_subscriptions.remove(key);
					let worker = Arc::clone(worker);
					let unsubscribe = create_service_unsubscribe_call(subscription_id);
					let scope = scope.clone();
					let manager = Arc::clone(self);
					tokio::spawn(async move {
						let _ = manager.invoke(&worker, &scope, &unsubscribe, &BACKGROUND_CONTEXT.clone()).await;
					});
				}
				return Err(invoke_error);
			}
		};
		if let Some(ServiceControlCall::Unsubscribe { subscription_id, .. }) = &control {
			let key = scoped_service_subscription_key(scope, subscription_id);
			self.state.lock().service_subscriptions.remove(&key);
		}
		Ok(response)
	}

	async fn apply_demand(
		self: &Arc<Self>,
		worker: &Arc<WorkerRecord>,
		attachment_id: &str,
		attached: bool,
		compensate_on_timeout: bool,
	) -> Result<(), AnyError> {
		if !compensate_on_timeout {
			return self.apply_demand_uncorrected(worker, attachment_id, attached).await;
		}
		// Attached demands reconcile on timeout like upstream's compensation
		// task; boxed so the Send bound does not recurse through the
		// reconcile → demand cycle.
		match self.apply_demand_uncorrected(worker, attachment_id, attached).await {
			Ok(()) => Ok(()),
			Err(timeout_error) => {
				let reconcile: futures_util::future::BoxFuture<'_, Result<(), AnyError>> =
					Box::pin(async { self.reconcile_last_demand(worker, attachment_id).await });
				match reconcile.await {
					Ok(()) => {
						if attached {
							Err(timeout_error)
						} else {
							Ok(())
						}
					}
					Err(reconcile_error) => Err(reconcile_error),
				}
			}
		}
	}

	/// Demand update with the plain timeout rejection (upstream
	/// `compensateOnTimeout === false`).
	async fn apply_demand_uncorrected(
		self: &Arc<Self>,
		worker: &Arc<WorkerRecord>,
		attachment_id: &str,
		attached: bool,
	) -> Result<(), AnyError> {
		if worker.stopping.load(Ordering::SeqCst) || !self.state.lock().peer_owns_worker(&worker.peer_id, worker) {
			return Err(error("Experimental Session worker is stopping"));
		}
		let request_id = new_uuid();
		let slot = PromiseSlot::new();
		{
			let mut state = self.state.lock();
			if state.detached || state.shutting_down {
				return Err(error("Experimental server is shutting down"));
			}
			state.pending_demand.insert(
				request_id.clone(),
				PendingDemand {
					attachment_id: attachment_id.to_string(),
					attached,
					request_id: request_id.clone(),
					worker: Arc::clone(worker),
					slot: slot.clone(),
				},
			);
		}
		let payload = serde_json::json!({
			"type": "session_demand",
			"serverConnectionId": self.server_connection_id(),
			"requestId": request_id,
			"attachmentId": attachment_id,
			"attached": attached,
		});
		if let Err(send_error) = self.coordinator.send(&worker.peer_id, payload).await {
			self.reject_demand(&request_id, message_error(send_error));
		}
		match tokio::time::timeout(
			std::time::Duration::from_millis(WORKER_DEMAND_TIMEOUT_MS),
			slot.get(),
		)
		.await
		{
			Ok(result) => {
				self.state.lock().pending_demand.remove(&request_id);
				result.map_err(|demand_error| -> AnyError {
					Box::new(ChordError { message: demand_error.to_string() })
				})
			}
			Err(_elapsed) => {
				self.state.lock().pending_demand.remove(&request_id);
				Err(error("Session worker demand update timed out"))
			}
		}
	}

	/// Worker operations have no wall-clock timeout: completion, disconnect,
	/// replacement, or shutdown settles them.
	async fn invoke(
		self: &Arc<Self>,
		worker: &Arc<WorkerRecord>,
		scope: &WorkerOperationScope,
		call: &ServiceCall,
		context: &Context,
	) -> Result<Option<JsonValue>, AnyError> {
		// Validates attachment liveness.
		self.operation_scope(worker, &scope.attachment_id)?;
		let signal = context.abort_signal();
		if let Some(signal) = &signal {
			if signal.aborted() {
				return Err(abort_error(signal));
			}
		}
		let request_id = new_uuid();
		let slot = PromiseSlot::new();
		{
			let mut state = self.state.lock();
			if state.detached || state.shutting_down {
				return Err(error("Experimental server is shutting down"));
			}
			state.pending_operations.insert(
				request_id.clone(),
				PendingWorkerOperation {
					worker: Arc::clone(worker),
					scope: scope.clone(),
					slot: slot.clone(),
				},
			);
		}
		// Abort watcher: sends `operation_cancel` and rejects on abort; ends
		// as soon as the operation settles.
		if let Some(signal) = &signal {
			let abort_manager = Arc::clone(self);
			let abort_request_id = request_id.clone();
			let abort_worker = Arc::clone(worker);
			let abort_scope = scope.clone();
			let abort_slot = slot.clone();
			let abort_signal = signal.clone();
			tokio::spawn(async move {
				tokio::select! {
					reason = abort_signal.aborted_future() => {
						let _ = reason;
						let pending = abort_manager.state.lock().pending_operations.get(&abort_request_id).is_some();
						if pending {
							let _ = abort_manager
								.coordinator
								.send(
									&abort_worker.peer_id,
									serde_json::json!({
										"type": "operation_cancel",
										"requestId": abort_request_id,
										"scope": abort_scope,
									}),
								)
								.await;
							abort_manager
								.reject_operation(&abort_request_id, shared_error(abort_error(&abort_signal)));
						}
					}
					settled = abort_slot.get() => {
						// The operation finished (or was rejected) first.
						let _ = settled;
					}
				}
			});
		}
		let payload = serde_json::json!({
			"type": "operation",
			"requestId": request_id,
			"scope": scope,
			"call": call,
		});
		let send_manager = Arc::clone(self);
		let send_worker = Arc::clone(worker);
		let send_request_id = request_id.clone();
		if let Err(send_error) = self.coordinator.send(&worker.peer_id, payload).await {
			send_manager.reject_operation(&send_request_id, message_error(send_error));
		}
		let _ = send_worker;
		slot.get().await.map_err(|op_error| -> AnyError {
			if let Some(code) = &op_error.code {
				if let Some(remote) = pi_chord::RemoteServiceErrorCode::from_str(code) {
					return pi_chord::remote_service_error(remote, op_error.message);
				}
			}
			error(format!("Session worker operation failed: {}", op_error.message))
		})
	}

	fn operation_scope(&self, worker: &Arc<WorkerRecord>, attachment_id: &str) -> Result<WorkerOperationScope, AnyError> {
		let state = self.state.lock();
		if state.detached || state.shutting_down || worker.stopping.load(Ordering::SeqCst) {
			return Err(error("Experimental Session worker is stopping"));
		}
		if !state.peer_owns_worker(&worker.peer_id, worker) || !worker.attachment_ids.lock().contains(attachment_id) {
			return Err(error("Experimental Session worker has no active attachment"));
		}
		Ok(WorkerOperationScope {
			server_connection_id: self.server_connection_id().to_string(),
			attachment_id: attachment_id.to_string(),
		})
	}

	fn is_detached_or_stopping(&self) -> bool {
		let state = self.state.lock();
		state.detached || state.shutting_down
	}

	/// Upstream `#stopWorker` — memoized per worker.
	async fn stop_worker(self: &Arc<Self>, worker: &Arc<WorkerRecord>) -> Result<(), AnyError> {
		{
			let state = self.state.lock();
			if state.detached || !state.peer_owns_worker(&worker.peer_id, worker) {
				return Ok(());
			}
		}
		let slot = {
			let mut stop = worker.stop_slot.lock();
			stop.get_or_insert_with(PromiseSlot::new).clone()
		};
		// Only the first caller drives the internal stop; later callers await
		// the shared slot (upstream `stopPromise ??=`).
		if !worker.stop_started.swap(true, Ordering::SeqCst) {
			let stop_manager = Arc::clone(self);
			let stop_worker = Arc::clone(worker);
			let stop_slot = slot.clone();
			tokio::spawn(async move {
				stop_manager.stop_worker_internal(&stop_worker).await;
				stop_slot.resolve(());
			});
		}
		slot.get().await;
		Ok(())
	}

	async fn stop_worker_internal(self: &Arc<Self>, worker: &Arc<WorkerRecord>) {
		self.reject_worker_operations(worker, message_error("Session worker is stopping"));
		worker.stopping.store(true, Ordering::SeqCst);
		worker.expected_stop.store(true, Ordering::SeqCst);
		let _ = self.coordinator.send(&worker.peer_id, serde_json::json!({ "type": "shutdown" })).await;
		let terminated = tokio::time::timeout(
			std::time::Duration::from_millis(WORKER_SHUTDOWN_TIMEOUT_MS),
			worker.terminated.get(),
		)
		.await;
		match terminated {
			Ok(_) => {}
			Err(_) => {
				kill_pid(worker.pid);
				self.remove_worker(worker, None);
				let _ = worker.terminated.get().await;
			}
		}
	}

	/// Upstream `shutdown()`.
	pub async fn shutdown(self: &Arc<Self>) {
		let pending_workers: Vec<Arc<PendingLaunch>> = {
			let mut state = self.state.lock();
			if state.detached || state.shutting_down {
				return;
			}
			state.shutting_down = true;
			state.pending.values().map(Arc::clone).collect()
		};
		for pending in &pending_workers {
			let _ = self
				.coordinator
				.send(&pending.peer_id, serde_json::json!({ "type": "shutdown" }))
				.await;
		}
		let wait_pending = async {
			for pending in &pending_workers {
				let mut child = pending.child.lock().await;
				let _ = tokio::time::timeout(std::time::Duration::from_millis(WORKER_SHUTDOWN_TIMEOUT_MS), child.wait()).await;
			}
		};
		let workers: Vec<Arc<WorkerRecord>> = self
			.state
			.lock()
			.workers_by_session
			.values()
			.map(Arc::clone)
			.collect();
		let stop_all = async {
			for worker in &workers {
				self.stop_worker(worker).await;
			}
		};
		tokio::join!(wait_pending, stop_all);
		self.detach_state();
	}

	/// Forget workers without stopping them when this server is replaced.
	pub fn detach(&self) {
		{
			let mut state = self.state.lock();
			if state.detached {
				return;
			}
			state.detached = true;
			let replaced_error = message_error("Experimental server was replaced");
			for pending in state.pending.values() {
				pending
					.promise
					.resolve(Err(replaced_error.clone()));
			}
			let operation_error = message_error("Experimental server was replaced during a worker operation");
			for request_id in state.pending_operations.keys().cloned().collect::<Vec<_>>() {
				drop(state);
				self.reject_operation(&request_id, operation_error.clone());
				state = self.state.lock();
			}
		}
		self.detach_state();
	}

	async fn handle_coordinator_event(self: &Arc<Self>, event: &CoordinatorConnectionEvent) {
		if self.state.lock().detached {
			return;
		}
		if let CoordinatorConnectionEvent::PeerDisconnected { peer_id } = event {
			self.mark_discovered(peer_id);
			let worker = self.state.lock().workers_by_peer.get(peer_id).cloned();
			if let Some(worker) = worker {
				let error = if worker.expected_stop.load(Ordering::SeqCst) {
					None
				} else {
					Some(message_error(format!(
						"Session worker {} disconnected unexpectedly",
						worker.metadata.id
					)))
				};
				self.remove_worker(&worker, error);
			}
			let pending = self.state.lock().pending_peer(peer_id);
			if let Some(pending) = pending {
				self.fail_pending(&pending.session_key, message_error("Session worker disconnected during startup"));
			}
			return;
		}
		let CoordinatorConnectionEvent::Message { from, payload } = event else {
			return;
		};
		let message = match serde_json::from_value::<SessionWorkerEvent>(payload.clone()) {
			Ok(message) => message,
			Err(_) => {
				if payload.get("type").and_then(Value::as_str) == Some("operation_response") {
					let worker = self.state.lock().workers_by_peer.get(from).cloned();
					if let Some(worker) = worker {
						self.reject_worker_operations(
							&worker,
							message_error("Session worker returned an invalid operation response"),
						);
					}
				}
				return;
			}
		};
		match message {
			SessionWorkerEvent::WorkerFailed { token, session_key, message, .. } => {
				let pending = self.state.lock().pending.get(&session_key).map(Arc::clone);
				if let Some(pending) = pending {
					if pending.peer_id == *from && pending.token == token {
						self.fail_pending(
							&session_key,
							message_error(format!("Session worker failed: {message}")),
						);
					}
				}
			}
			SessionWorkerEvent::DemandApplied { token, session_key, request_id, attachment_id, attached } => {
				self.settle_demand(from, &token, &session_key, &request_id, Some((&attachment_id, attached)), None);
			}
			SessionWorkerEvent::DemandRejected { token, session_key, request_id, message } => {
				self.settle_demand(from, &token, &session_key, &request_id, None, Some(&message));
			}
			SessionWorkerEvent::OperationResponse { token, session_key, response } => {
				self.handle_operation_response(from, &token, &session_key, &response);
			}
			SessionWorkerEvent::ServiceUpdate { token, session_key, scope, subscription_id, update } => {
				self.handle_service_event(from, &token, &session_key, &scope, &subscription_id, update);
			}
			SessionWorkerEvent::WorkerReady { .. } => {
				self.record_ready_worker(from, &message);
			}
		}
	}

	/// Shared validation + settlement for `demand_applied`/`demand_rejected`.
	fn settle_demand(
		&self,
		from: &str,
		token: &str,
		session_key: &str,
		request_id: &str,
		applied: Option<(&str, bool)>,
		reject_message: Option<&str>,
	) {
		let pending = self.state.lock().pending_demand.get(request_id).map(|pending| {
			(
				pending.worker.peer_id.clone(),
				pending.worker.token.clone(),
				pending.worker.metadata.path.clone(),
				pending.attachment_id.clone(),
				pending.attached,
				pending.slot.clone(),
			)
		});
		let Some((peer_id, worker_token, path, pending_attachment_id, pending_attached, slot)) = pending else {
			return;
		};
		if peer_id != from
			|| worker_token != token
			|| path != session_key
			|| (applied.is_some()
				&& (pending_attachment_id != applied.expect("checked").0
					|| pending_attached != applied.expect("checked").1))
		{
			return;
		}
		self.state.lock().pending_demand.remove(request_id);
		if let Some((attachment_id, attached)) = applied {
			slot.resolve(Ok(()));
			let _ = (attachment_id, attached);
		} else {
			slot.resolve(Err(message_error(format!(
				"Session worker rejected demand: {}",
				reject_message.unwrap_or_default()
			))));
		}
	}

	fn handle_operation_response(
		&self,
		peer_id: &str,
		token: &str,
		session_key: &str,
		response: &WorkerOperationResponse,
	) {
		let request_id = response.request_id().to_string();
		let pending = self.state.lock().pending_operations.get(&request_id).map(|pending| {
			(
				pending.worker.peer_id.clone(),
				pending.worker.token.clone(),
				pending.worker.metadata.path.clone(),
				pending.scope.clone(),
				pending.slot.clone(),
			)
		});
		let Some((worker_peer_id, worker_token, path, pending_scope, slot)) = pending else {
			return;
		};
		if worker_peer_id != peer_id
			|| worker_token != token
			|| path != session_key
			|| response.scope().server_connection_id != pending_scope.server_connection_id
			|| response.scope().attachment_id != pending_scope.attachment_id
		{
			self.reject_operation(
				&request_id,
				message_error("Session worker returned a mismatched operation response"),
			);
			return;
		}
		self.state.lock().pending_operations.remove(&request_id);
		match response {
			WorkerOperationResponse::OperationResult { result, .. } => {
				slot.resolve(Ok(result.clone()));
			}
			WorkerOperationResponse::OperationError { code, message, .. } => {
				slot.resolve(Err(WorkerOpError {
					code: code.clone(),
					message: message.clone(),
				}));
			}
		}
	}

	fn handle_service_event(
		&self,
		peer_id: &str,
		token: &str,
		session_key: &str,
		scope: &WorkerOperationScope,
		subscription_id: &str,
		update: Value,
	) {
		let key = scoped_service_subscription_key(scope, subscription_id);
		let entry = self.state.lock().service_subscriptions.get(&key).map(|entry| {
			(
				entry.worker.peer_id.clone(),
				entry.worker.token.clone(),
				entry.worker.metadata.path.clone(),
				entry.scope.clone(),
				entry.publish.clone(),
				entry.delivery_lock.clone(),
			)
		});
		let Some((worker_peer_id, worker_token, path, entry_scope, publish, delivery_lock)) = entry else {
			return;
		};
		if worker_peer_id != peer_id || worker_token != token || path != session_key || entry_scope != *scope {
			return;
		}
		let parsed = match parse_service_provider_update(&update) {
			Ok(parsed) => parsed,
			Err(_) => return,
		};
		let subscription_id = subscription_id.to_string();
		tokio::spawn(async move {
			// Serialize deliveries (upstream `deliveryTail`).
			let _guard = delivery_lock.lock().await;
			let () = publish(subscription_id, parsed, TODO_CONTEXT.clone()).await.ok().unwrap_or(());
		});
	}

	fn record_ready_worker(&self, peer_id: &str, message: &SessionWorkerEvent) {
		let SessionWorkerEvent::WorkerReady {
			token,
			session_key,
			session_id,
			pid,
			metadata,
			plugin_manifest_paths,
		} = message
		else {
			return;
		};
		if session_key != &metadata.path
			|| session_id != &metadata.id
			|| !std::path::Path::new(&metadata.cwd).is_absolute()
			|| !std::path::Path::new(&metadata.path).is_absolute()
		{
			return;
		}
		self.mark_discovered(peer_id);
		{
			let state = self.state.lock();
			if let Some(pending) = state.pending.get(session_key) {
				if pending.peer_id == peer_id && !same_strings(&plugin_manifest_paths, &pending.plugin_manifest_paths) {
					drop(state);
					self.fail_pending(
						session_key,
						message_error("Session worker started with stale plugin packages"),
					);
					return;
				}
			}
			if let Some(existing) = state.workers_by_session.get(session_key) {
				if existing.peer_id != peer_id {
					drop(state);
					let coordinator = self.coordinator.clone();
					let stale_peer = peer_id.to_string();
					tokio::spawn(async move {
						let _ = coordinator.send(&stale_peer, serde_json::json!({ "type": "shutdown" })).await;
					});
					return;
				}
			}
			if let Some(pending) = state.pending.get(session_key) {
                let child_pid = pending
					.child
					.try_lock()
					.ok()
					.and_then(|child| child.id())
					.map(|id| id as i32);
				if pending.peer_id != peer_id || pending.token != *token || child_pid != Some(*pid) {
					return;
				}
			}
			if state.workers_by_session.contains_key(session_key) {
				return;
			}
		}
		let worker = Arc::new(WorkerRecord {
			peer_id: peer_id.to_string(),
			metadata: SessionCatalogMetadata {
				id: metadata.id.clone(),
				created_at: metadata.created_at,
				cwd: metadata.cwd.clone(),
				path: metadata.path.clone(),
			},
			pid: *pid,
			token: token.clone(),
			plugin_manifest_paths: plugin_manifest_paths.clone(),
			terminated: PromiseSlot::new(),
			attachment_ids: PLMutex::new(HashSet::new()),
			expected_stop: AtomicBool::new(false),
			stopping: AtomicBool::new(false),
			stop_slot: PLMutex::new(None),
			stop_started: AtomicBool::new(false),
		});
		{
			let mut state = self.state.lock();
			state
				.workers_by_session
				.insert(session_key.clone(), Arc::clone(&worker));
			state.workers_by_peer.insert(peer_id.to_string(), Arc::clone(&worker));
		}
		self.worker_pids.set(session_id, *pid);
		let pending = self.state.lock().pending.get(session_key).map(Arc::clone);
		if let Some(pending) = pending {
			if pending.peer_id == peer_id {
				self.state.lock().pending.remove(session_key);
				pending.promise.resolve(Ok(Arc::clone(&worker)));
			}
		}
		self.notify_worker_count_changed();
	}

	fn child_exited(&self, pending: &PendingLaunch, code: Option<i32>) {
		let descriptor = format!("{}", code.map(|value| value.to_string()).unwrap_or_else(|| "unknown".to_string()));
		if self
			.state
			.lock()
			.pending
			.get(&pending.session_key)
			.map(|current| current.peer_id == pending.peer_id)
			.unwrap_or(false)
		{
			self.fail_pending(
				&pending.session_key,
				message_error(format!("Session worker exited before readiness ({descriptor})")),
			);
			return;
		}
		let worker = self.state.lock().workers_by_peer.get(&pending.peer_id).cloned();
		if let Some(worker) = worker {
			let error = if worker.expected_stop.load(Ordering::SeqCst) {
				None
			} else {
				Some(message_error(format!(
					"Session worker {} exited unexpectedly ({descriptor})",
					worker.metadata.id
				)))
			};
			self.remove_worker(&worker, error);
		}
	}

	fn fail_pending(&self, session_key: &str, launch_error: SharedError) {
		let pending = self.state.lock().pending.get(session_key).map(Arc::clone);
		let Some(pending) = pending else {
			return;
		};
		self.state.lock().pending.remove(session_key);
		self.notify_worker_count_changed();
		let child = Arc::clone(&pending.child);
		tokio::spawn(async move {
			if let Ok(mut child) = child.try_lock() {
				let _ = child.start_kill();
			}
		});
		pending.promise.resolve(Err(launch_error));
	}

	fn reject_demand(&self, request_id: &str, demand_error: SharedError) {
		let pending = self.state.lock().pending_demand.remove(request_id);
		if let Some(pending) = pending {
			pending.slot.resolve(Err(demand_error));
		}
	}

	fn reject_operation(&self, request_id: &str, operation_error: SharedError) {
		let pending = self.state.lock().pending_operations.remove(request_id);
		if let Some(pending) = pending {
			pending
				.slot
				.resolve(Err(WorkerOpError {
					code: None,
					message: operation_error.to_string(),
				}));
		}
	}

	fn reject_worker_operations(&self, worker: &Arc<WorkerRecord>, worker_error: SharedError) {
		let request_ids: Vec<String> = self
			.state
			.lock()
			.pending_operations
			.iter()
			.filter(|(_, pending)| Arc::ptr_eq(&pending.worker, worker))
			.map(|(request_id, _)| request_id.clone())
			.collect();
		for request_id in request_ids {
			self.reject_operation(&request_id, worker_error.clone());
		}
	}

	/// Upstream `#reconcileDemandTimeout` — re-apply the demand unattached
	/// (compensating), terminating the worker when cleanup fails.
	async fn reconcile_last_demand(
		self: &Arc<Self>,
		worker: &Arc<WorkerRecord>,
		attachment_id: &str,
	) -> Result<(), AnyError> {
		let timeout_message = "Session worker demand update timed out";
		match self.apply_demand_uncorrected(worker, attachment_id, false).await {
			Ok(()) => Ok(()),
			Err(cleanup_error) => {
				self.stop_worker(worker).await?;
				Err(error(format!(
					"{timeout_message}; worker was terminated during reconciliation ({cleanup_error})"
				)))
			}
		}
	}

	fn remove_worker(&self, worker: &Arc<WorkerRecord>, worker_error: Option<SharedError>) {
		{
			let state = self.state.lock();
			if !state.peer_owns_worker(&worker.peer_id, worker) {
				return;
			}
		}
		self.reject_worker_operations(
			worker,
			message_error("Session worker disconnected during an operation"),
		);
		self.remove_service_subscriptions(|entry| Arc::ptr_eq(&entry.worker, worker));
		let stale_requests: Vec<String> = self
			.state
			.lock()
			.pending_demand
			.iter()
			.filter(|(_, pending)| Arc::ptr_eq(&pending.worker, worker))
			.map(|(request_id, _)| request_id.clone())
			.collect();
		for request_id in stale_requests {
			self.reject_demand(&request_id, message_error("Session worker disconnected during demand update"));
		}
		{
			let mut state = self.state.lock();
			state.workers_by_peer.remove(&worker.peer_id);
			state.workers_by_session.remove(&worker.metadata.path);
		}
		self.worker_pids
			.remove_if(&worker.metadata.id, worker.pid);
		worker.terminated.resolve(worker_error);
		self.notify_worker_count_changed();
	}

	fn remove_service_subscriptions(&self, matches: impl Fn(&WorkerServiceSubscription) -> bool) {
		let keys: Vec<String> = self
			.state
			.lock()
			.service_subscriptions
			.iter()
			.filter(|(_, entry)| matches(entry))
			.map(|(key, _)| key.clone())
			.collect();
		let mut state = self.state.lock();
		for key in keys {
			state.service_subscriptions.remove(&key);
		}
	}

	fn notify_worker_count_changed(&self) {
		let count = {
			let state = self.state.lock();
			state.workers_by_session.len() + state.pending.len()
		};
		if let Some(on_change) = &self.on_worker_count_changed {
			on_change(count);
		}
	}

	async fn launch(
		self: &Arc<Self>,
		metadata: &SessionCatalogMetadata,
		plugin_manifest_paths: &[String],
	) -> Result<Arc<WorkerRecord>, AnyError> {
		let session_key = metadata.path.clone();
		let peer_id = format!("worker-{}", new_uuid());
		let token = new_uuid();
		let options = SessionWorkerOptions {
			session_dir: self.session_dir.clone(),
			metadata: SessionWorkerMetadata {
				id: metadata.id.clone(),
				created_at: metadata.created_at,
				cwd: metadata.cwd.clone(),
				path: metadata.path.clone(),
			},
			provider: self.model.as_ref().and_then(|model| model.provider.clone()),
			model: self.model.as_ref().map(|model| model.model.clone()),
			plugin_manifest_paths: plugin_manifest_paths.to_vec(),
		};
		let serialized = serde_json::to_string(&options).map_err(|serialization_error| -> AnyError {
			error(format!("Session worker options failed to serialize: {serialization_error}"))
		})?;
		let mut env = HashMap::new();
		env.insert(
			SESSION_WORKER_CONTROL_ADDRESS_ENV.to_string(),
			self.coordinator.control_path().to_string(),
		);
		env.insert(SESSION_WORKER_CONTROL_TOKEN_ENV.to_string(), token.clone());
		env.insert(
			SESSION_WORKER_SESSION_KEY_ENV.to_string(),
			base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(session_key.as_bytes()),
		);
		env.insert(SESSION_WORKER_PEER_ID_ENV.to_string(), peer_id.clone());
		let child = spawn_internal_process(
			InternalProcessRole::SessionWorker,
			&[serialized],
			&InternalProcessSpawnOptions {
				entry_override: None,
				env,
			},
		)
		.map_err(|spawn_error| -> AnyError { error(spawn_error.to_string()) })?;
		let promise = PromiseSlot::new();
		let pending = Arc::new(PendingLaunch {
			session_key: session_key.clone(),
			peer_id,
			token,
			plugin_manifest_paths: plugin_manifest_paths.to_vec(),
			child: Arc::new(tokio::sync::Mutex::new(child)),
			promise: promise.clone(),
		});
		self.state.lock().pending.insert(session_key.clone(), Arc::clone(&pending));
		self.notify_worker_count_changed();
		// Startup timeout (a settled pending is already removed, so the late
		// fire is inert).
		let timeout_manager = Arc::clone(self);
		let timeout_key = session_key.clone();
		tokio::spawn(async move {
			tokio::time::sleep(std::time::Duration::from_millis(WORKER_STARTUP_TIMEOUT_MS)).await;
			timeout_manager.fail_pending(
				&timeout_key,
				message_error("Session worker startup timed out"),
			);
		});
		// Exit watcher (upstream `error` + `exit` child listeners; spawn
		// failures surface through the same readiness-failure path).
		let exit_manager = Arc::clone(self);
		let exit_pending = Arc::clone(&pending);
		tokio::spawn(async move {
			let mut child = exit_pending.child.lock().await;
			let code = match child.wait().await {
				Ok(status) => status.code(),
				Err(_) => None,
			};
			drop(child);
			exit_manager.child_exited(&exit_pending, code);
		});
		promise
			.get()
			.await
			.map_err(|launch_error| -> AnyError { Box::new(ChordError { message: launch_error.to_string() }) })
	}

	fn mark_discovered(&self, peer_id: &str) {
		let mut state = self.state.lock();
		let Some(discovery_peers) = state.discovery_peers.as_mut() else {
			return;
		};
		if !discovery_peers.remove(peer_id) || !discovery_peers.is_empty() {
			return;
		}
		state.discovery_peers = None;
		if let Some(resolver) = state.resolve_discovery.take() {
			resolver.resolve(());
		}
	}

	/// Upstream `#detachState` — drop the listener and settle every pending
	/// operation without touching workers.
	fn detach_state(&self) {
		*self.listener.lock() = None;
		let operation_error = message_error("Experimental server detached during a worker operation");
		let request_ids: Vec<String> = self.state.lock().pending_operations.keys().cloned().collect();
		for request_id in request_ids {
			self.reject_operation(&request_id, operation_error.clone());
		}
		let demands: Vec<(String, PromiseSlot<Result<(), SharedError>>)> = self
			.state
			.lock()
			.pending_demand
			.iter()
			.map(|(request_id, pending)| (request_id.clone(), pending.slot.clone()))
			.collect();
		{
			let mut state = self.state.lock();
			for (request_id, slot) in demands {
				state.pending_demand.remove(&request_id);
				slot.resolve(Ok(()));
			}
		}
		let mut state = self.state.lock();
		state.pending.clear();
		state.service_subscriptions.clear();
		state.workers_by_peer.clear();
		state.workers_by_session.clear();
		self.worker_pids.clear();
		state.discovery_peers = None;
		if let Some(resolver) = state.resolve_discovery.take() {
			resolver.resolve(());
		}
	}
}

impl ManagerState {
	fn peer_owns_worker(&self, peer_id: &str, worker: &Arc<WorkerRecord>) -> bool {
		self.workers_by_peer
			.get(peer_id)
			.map(|current| Arc::ptr_eq(current, worker))
			.unwrap_or(false)
	}

	fn pending_peer(&self, peer_id: &str) -> Option<Arc<PendingLaunch>> {
		self.pending
			.values()
			.find(|pending| pending.peer_id == peer_id)
			.map(Arc::clone)
	}
}

// ---------------------------------------------------------------------------
// Routed session handle + attachment.
// ---------------------------------------------------------------------------

/// Upstream the `#routedHandle` object literal.
struct ManagerSessionHandle {
	manager: Arc<SessionWorkerManager>,
	worker: Arc<WorkerRecord>,
}

#[async_trait::async_trait]
impl RoutedSessionHandle for ManagerSessionHandle {
	fn attach_client(&self, context: Context) -> MaybeFuture<Result<Arc<dyn RoutedSessionAttachment>, pi_server::types::AnyError>> {
		let manager = Arc::clone(&self.manager);
		let worker = Arc::clone(&self.worker);
		Box::pin(async move {
			manager
				.attach_client(worker, context)
				.await
				.map(|attachment| attachment as Arc<dyn RoutedSessionAttachment>)
				.map_err(shared_error)
		})
	}

	fn terminated(&self) -> Option<ChordBoxFuture<Option<pi_server::types::AnyError>>> {
		let worker = Arc::clone(&self.worker);
		Some(Box::pin(async move { worker.terminated.get().await }))
	}

	async fn close(&self, _context: Context) -> Result<(), pi_server::types::AnyError> {
		self.manager
			.stop_worker(&self.worker)
			.await
			.map_err(shared_error)
	}
}

/// Upstream the `#attachClient` return object literal.
struct ManagerSessionAttachment {
	manager: Arc<SessionWorkerManager>,
	worker: Arc<WorkerRecord>,
	scope: WorkerOperationScope,
	released: Arc<AtomicBool>,
}

impl RoutedSessionAttachment for ManagerSessionAttachment {
	fn invoke_service(
		&self,
		call: ServiceCall,
		publish: ServicePublish,
		context: Context,
	) -> BoxFuture<'static, Result<Option<JsonValue>, pi_server::types::AnyError>> {
		let manager = Arc::clone(&self.manager);
		let worker = Arc::clone(&self.worker);
		let scope = self.scope.clone();
		Box::pin(async move {
			manager
				.invoke_service(&worker, &scope, &call, publish, &context)
				.await
				.map_err(shared_error)
		})
	}

	fn release(&self, _context: Context) -> MaybeFuture<Result<(), pi_server::types::AnyError>> {
		let manager = Arc::clone(&self.manager);
		let worker = Arc::clone(&self.worker);
		let scope = self.scope.clone();
		let released = Arc::clone(&self.released);
		Box::pin(async move {
			if released.swap(true, Ordering::SeqCst) {
				return Ok(());
			}
			let attachment_id = scope.attachment_id.clone();
			let detached = manager.is_detached_or_stopping()
				|| !worker.attachment_ids.lock().contains(&attachment_id);
			if detached {
				worker.attachment_ids.lock().remove(&attachment_id);
				manager.remove_service_subscriptions(|entry| {
					Arc::ptr_eq(&entry.worker, &worker) && same_scope(&entry.scope, &scope)
				});
				return Ok(());
			}
			let result = manager.apply_demand(&worker, &attachment_id, false, true).await;
			worker.attachment_ids.lock().remove(&attachment_id);
			manager.remove_service_subscriptions(|entry| {
				Arc::ptr_eq(&entry.worker, &worker) && same_scope(&entry.scope, &scope)
			});
			let replaced = manager.coordinator.was_replaced();
			match result {
				Ok(()) => Ok(()),
				Err(release_error) if !replaced => Err(shared_error(release_error)),
				Err(_) => Ok(()),
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

fn same_strings(left: &[String], right: &[String]) -> bool {
	left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a == b)
}

fn same_scope(left: &WorkerOperationScope, right: &WorkerOperationScope) -> bool {
	left.server_connection_id == right.server_connection_id && left.attachment_id == right.attachment_id
}

fn scoped_service_subscription_key(scope: &WorkerOperationScope, subscription_id: &str) -> String {
	format!("{}\0{}\0{}", scope.server_connection_id, scope.attachment_id, subscription_id)
}

fn demand_rejected_message(event: &SessionWorkerEvent) -> Option<String> {
	match event {
		SessionWorkerEvent::DemandRejected { message, .. } => Some(message.clone()),
		_ => None,
	}
}

/// `crypto.randomUUID()` equivalent (same generator as the session catalog).
pub(crate) fn new_uuid() -> String {
	crate::experimental::session_catalog::new_session_id()
}

/// `process.kill(pid, "SIGKILL")`; a missing pid is swallowed (`ESRCH`).
fn kill_pid(pid: i32) {
	unsafe {
		libc::kill(pid, libc::SIGKILL);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn same_strings_matches_upstream() {
		assert!(same_strings(&[], &[]));
		assert!(same_strings(&["a".into()], &["a".into()]));
		assert!(!same_strings(&["a".into()], &["b".into()]));
		assert!(!same_strings(&["a".into()], &["a".into(), "b".into()]));
	}

	#[test]
	fn same_scope_compares_both_fields() {
		let base = WorkerOperationScope { server_connection_id: "s".into(), attachment_id: "a".into() };
		assert!(same_scope(&base, &base));
		assert!(!same_scope(
			&base,
			&WorkerOperationScope { server_connection_id: "s".into(), attachment_id: "b".into() }
		));
		assert!(!same_scope(
			&base,
			&WorkerOperationScope { server_connection_id: "t".into(), attachment_id: "a".into() }
		));
	}

	#[test]
	fn subscription_key_embeds_scope_and_id() {
		let scope = WorkerOperationScope { server_connection_id: "s".into(), attachment_id: "a".into() };
		assert_eq!(scoped_service_subscription_key(&scope, "sub"), "s\0a\0sub");
	}

	#[tokio::test]
	async fn promise_slot_settles_once_and_broadcasts() {
		let slot = PromiseSlot::<u32>::new();
		assert!(!slot.is_resolved());
		let reader = tokio::spawn({
			let slot = slot.clone();
			async move { slot.get().await }
		});
		tokio::task::yield_now().await;
		assert!(slot.resolve(7));
		assert!(!slot.resolve(8));
		assert!(slot.is_resolved());
		assert_eq!(slot.get().await, 7);
		assert_eq!(reader.await.unwrap(), 7);
	}

	#[test]
	fn worker_pids_snapshot_tracks_updates() {
		let pids = WorkerPids::default();
		assert_eq!(pids.get("s1"), None);
		pids.set("s1", 42);
		assert_eq!(pids.get("s1"), Some(42));
		pids.remove_if("s1", 7);
		assert_eq!(pids.get("s1"), Some(42));
		pids.remove_if("s1", 42);
		assert_eq!(pids.get("s1"), None);
	}

	#[test]
	fn conflict_error_message_matches_upstream() {
		let error = SessionPluginSelectionConflictError::any("Session s-1 is active with a different plugin selection");
		assert_eq!(
			error.to_string(),
			"Session s-1 is active with a different plugin selection"
		);
	}

	#[test]
	fn worker_model_options_serialization() {
		assert_eq!(
			serialize_worker_model_options(&WorkerModelOptions { provider: None, model: "m".into() }),
			r#"{"model":"m"}"#
		);
		assert_eq!(
			serialize_worker_model_options(&WorkerModelOptions { provider: Some("p".into()), model: "m".into() }),
			r#"{"provider":"p","model":"m"}"#
		);
	}

	#[tokio::test]
	async fn discovery_times_out_without_coordinator_peers() {
		let _guard = crate::experimental::TEST_GUARD.lock();
		// Without a connected coordinator, `broadcast` fails and the discovery
		// window times out; the call must still return.
		let coordinator = CoordinatorConnection::new(crate::experimental::coordinator::CoordinatorConnectionOptions {
			control_path: "/tmp/pi-manager-test-control.sock".into(),
			endpoint: "/tmp/pi-manager-test-endpoint.sock".into(),
			server_connection_id: None,
		});
		let manager = SessionWorkerManager::new(Arc::new(coordinator), "/tmp".into(), None, None);
		let mut peers = std::collections::BTreeSet::new();
		peers.insert("worker-x".to_string());
		manager.discover(&peers).await;
	}
}
