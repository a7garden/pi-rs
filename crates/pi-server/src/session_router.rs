//! Port of `packages/server/src/session-router.ts`.
//!
//! Upstream client identity (the connection object as a `Map` key) becomes a
//! monotonically increasing [`ClientId`] assigned by the server when a
//! connection is accepted. `Promise`-returning attachment methods
//! (`invokeService`) return [`BoxFuture`]s without awaiting them so the
//! per-client operation queue only serializes the call *start* — matching
//! upstream `runForClient(() => this.startServiceCall(...))` semantics.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use pi_chord::context::BACKGROUND_CONTEXT;
use pi_chord::{Context, JsonValue, ServiceCall};
use tokio::sync::{Notify, OnceCell};

use crate::errors::ServerError;
use crate::types::{
	AnyError, RoutedSessionAttachment, RoutedSessionHandle, ServerHost, ServicePublish,
};
use crate::wire;

/// Upstream `SessionCleanupError` (AggregateError subclass); recognized by
/// `close_internal` when classifying in-flight acquisition failures.
#[derive(Debug)]
pub(crate) struct SessionCleanupError;

impl std::fmt::Display for SessionCleanupError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str("Failed to close routed Session acquired while draining")
	}
}

impl std::error::Error for SessionCleanupError {}

pub(crate) type ClientId = u64;

fn insert_attachment(hosted: &Arc<HostedSession>, attachment: &Arc<ClientAttachment>) {
	let mut attachments = hosted.attachments.lock();
	if !attachments.iter().any(|existing| Arc::ptr_eq(existing, attachment)) {
		attachments.push(Arc::clone(attachment));
	}
}

fn remove_attachment(hosted: &Arc<HostedSession>, attachment: &Arc<ClientAttachment>) {
	hosted.attachments.lock().retain(|existing| !Arc::ptr_eq(existing, attachment));
}

fn contains_attachment(hosted: &Arc<HostedSession>, attachment: &Arc<ClientAttachment>) -> bool {
	hosted.attachments.lock().iter().any(|existing| Arc::ptr_eq(existing, attachment))
}

static NEXT_CLIENT_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_client_id() -> ClientId {
	NEXT_CLIENT_ID.fetch_add(1, Ordering::Relaxed)
}

/// uuidv4 (upstream `crypto.randomUUID()`); used for attachment IDs and
/// temp-name suffixes.
pub(crate) fn uuid_v4() -> String {
	let mut bytes = [0u8; 16];
	getrandom::getrandom(&mut bytes).expect("platform CSPRNG availability");
	bytes[6] = (bytes[6] & 0x0f) | 0x40;
	bytes[8] = (bytes[8] & 0x3f) | 0x80;
	let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
	format!(
		"{}-{}-{}-{}-{}",
		&hex[0..8],
		&hex[8..12],
		&hex[12..16],
		&hex[16..20],
		&hex[20..32]
	)
}

/// Shared slot for one in-flight service call (upstream
/// `attachment.operations: Set<Promise>`); resolved when the caller's guard
/// drops.
#[derive(Clone)]
pub(crate) struct OperationSlot {
	done: Arc<tokio::sync::watch::Receiver<bool>>,
}

impl OperationSlot {
	pub(crate) fn new() -> (OperationSlot, OperationGuard) {
		let (tx, rx) = tokio::sync::watch::channel(false);
		(OperationSlot { done: Arc::new(rx) }, OperationGuard { done: Arc::new(tx) })
	}

	pub(crate) async fn settled(&self) {
		let mut rx = self.done.as_ref().clone();
		while !*rx.borrow_and_update() {
			if rx.changed().await.is_err() {
				return;
			}
		}
	}
}

pub(crate) struct OperationGuard {
	done: Arc<tokio::sync::watch::Sender<bool>>,
}

impl Drop for OperationGuard {
	fn drop(&mut self) {
		let _ = self.done.send(true);
	}
}

/// Per-client FIFO operation queue (upstream `runForClient`).
pub(crate) struct ClientQueue {
	lock: tokio::sync::Mutex<()>,
	inflight: parking_lot::Mutex<usize>,
	idle: Notify,
}

impl ClientQueue {
	fn new() -> ClientQueue {
		ClientQueue { lock: tokio::sync::Mutex::new(()), inflight: parking_lot::Mutex::new(0), idle: Notify::new() }
	}

	pub(crate) async fn run<T, F>(&self, operation: F) -> T
	where
		F: Future<Output = T>,
	{
		*self.inflight.lock() += 1;
		let guard = self.lock.lock().await;
		let result = operation.await;
		drop(guard);
		let mut inflight = self.inflight.lock();
		*inflight -= 1;
		if *inflight == 0 {
			self.idle.notify_waiters();
		}
		result
	}

	/// Resolves once every enqueued operation has settled.
	pub(crate) async fn wait_idle(&self) {
		loop {
			let notified = self.idle.notified();
			if *self.inflight.lock() == 0 {
				return;
			}
			notified.await;
		}
	}
}

pub(crate) struct HostedSession {
	pub(crate) id: String,
	pub(crate) handle: Arc<dyn RoutedSessionHandle>,
	pub(crate) attachments: parking_lot::Mutex<Vec<Arc<ClientAttachment>>>,
}

type AttachOutcome = Result<Arc<dyn RoutedSessionAttachment>, AnyError>;
type ReleaseOutcome = Arc<Result<(), AnyError>>;
type OpenOutcome = Result<Arc<HostedSession>, AnyError>;

pub(crate) struct ClientAttachment {
	pub(crate) id: String,
	pub(crate) client: ClientId,
	pub(crate) session: Arc<HostedSession>,
	pub(crate) operations: parking_lot::Mutex<Vec<OperationSlot>>,
	/// Resolved result of `handle.attachClient` (upstream `acquiring`).
	acquiring: tokio::sync::Mutex<Option<AttachOutcome>>,
	acquiring_notify: Notify,
	/// Upstream `lease`.
	lease: parking_lot::Mutex<Option<Arc<dyn RoutedSessionAttachment>>>,
	/// Upstream `releasing` memoized promise.
	releasing: OnceCell<ReleaseOutcome>,
}

impl ClientAttachment {
	fn new(client: ClientId, session: Arc<HostedSession>) -> Arc<ClientAttachment> {
		Arc::new(ClientAttachment {
			id: uuid_v4(),
			client,
			session,
			operations: parking_lot::Mutex::new(Vec::new()),
			acquiring: tokio::sync::Mutex::new(None),
			acquiring_notify: Notify::new(),
			lease: parking_lot::Mutex::new(None),
			releasing: OnceCell::new(),
		})
	}
}

/// Shared in-flight session open (upstream `openingSessions`).
struct Opening {
	cell: OnceCell<Arc<OpenOutcome>>,
	tx: Arc<tokio::sync::watch::Sender<bool>>,
	rx: Arc<tokio::sync::watch::Receiver<bool>>,
	/// Clients that joined this in-flight open (test determinism hook).
	joiners: std::sync::atomic::AtomicUsize,
}

impl Opening {
	fn new() -> Arc<Opening> {
		let (tx, rx) = tokio::sync::watch::channel(false);
		Arc::new(Opening {
			cell: OnceCell::new(),
			tx: Arc::new(tx),
			rx: Arc::new(rx),
			joiners: std::sync::atomic::AtomicUsize::new(0),
		})
	}

	async fn resolved(&self) {
		let mut rx = self.rx.as_ref().clone();
		while !*rx.borrow_and_update() {
			if rx.changed().await.is_err() {
				return;
			}
		}
	}
}

/// Upstream `publishAttachment`.
pub(crate) type PublishAttachment =
	Arc<dyn Fn(ClientId, Option<JsonValue>, Context) -> crate::types::MaybeFuture<Result<(), AnyError>> + Send + Sync>;

pub(crate) struct SessionRouter {
	pub(crate) host: Arc<dyn ServerHost>,
	pub(crate) server_id: String,
	pub(crate) is_closing: Arc<AtomicBool>,
	publish: parking_lot::Mutex<Option<PublishAttachment>>,
	pub(crate) report_error: Arc<dyn Fn(AnyError) + Send + Sync>,
	state: parking_lot::Mutex<RouterState>,
	close_lock: tokio::sync::Mutex<()>,
	close_result: parking_lot::Mutex<Option<Arc<Result<(), AnyError>>>>,
}

struct RouterState {
	hosted_sessions: HashMap<String, Arc<HostedSession>>,
	opening_sessions: HashMap<String, Arc<Opening>>,
	attachments_by_client: HashMap<ClientId, Arc<ClientAttachment>>,
	disconnected_clients: HashSet<ClientId>,
	client_queues: HashMap<ClientId, Arc<ClientQueue>>,
}

impl SessionRouter {
	pub(crate) fn new(
		host: Arc<dyn ServerHost>,
		server_id: String,
		is_closing: Arc<AtomicBool>,
		report_error: Arc<dyn Fn(AnyError) + Send + Sync>,
	) -> SessionRouter {
		SessionRouter {
			host,
			server_id,
			is_closing,
			publish: parking_lot::Mutex::new(None),
			report_error,
			state: parking_lot::Mutex::new(RouterState {
				hosted_sessions: HashMap::new(),
				opening_sessions: HashMap::new(),
				attachments_by_client: HashMap::new(),
				disconnected_clients: HashSet::new(),
				client_queues: HashMap::new(),
			}),
			close_lock: tokio::sync::Mutex::new(()),
			close_result: parking_lot::Mutex::new(None),
		}
	}

	/// Installs the attachment publisher (deferred until the server's
	/// connection table exists).
	pub(crate) fn set_publish_attachment(&self, publish: PublishAttachment) {
		*self.publish.lock() = Some(publish);
	}

	async fn publish_attachment(
		&self,
		client: ClientId,
		attachment: Option<JsonValue>,
		context: Context,
	) -> Result<(), AnyError> {
		let publish = self.publish.lock().clone();
		match publish {
			Some(publish) => publish(client, attachment, context).await,
			None => Ok(()),
		}
	}

	/// Assigns a client identity and its operation queue; called when the
	/// server accepts a connection.
	/// Test determinism hook: clients parked joining the in-flight open.
	#[cfg(test)]
	pub(crate) fn opening_joiners(&self, session_id: &str) -> usize {
		self.state
			.lock()
			.opening_sessions
			.get(session_id)
			.map(|opening| opening.joiners.load(std::sync::atomic::Ordering::SeqCst))
			.unwrap_or(0)
	}

	pub(crate) fn register_client(&self) -> (ClientId, Arc<ClientQueue>) {
		let id = next_client_id();
		let queue = Arc::new(ClientQueue::new());
		self.state.lock().client_queues.insert(id, queue.clone());
		(id, queue)
	}

	fn draining() -> AnyError {
		Arc::new(ServerError::server_draining())
	}

	fn queue_for(&self, client: ClientId) -> Arc<ClientQueue> {
		self.state
			.lock()
			.client_queues
			.get(&client)
			.cloned()
			.unwrap_or_else(|| {
				// Unregistered clients (e.g. after state clear) still serialize
				// on a throwaway queue.
				Arc::new(ClientQueue::new())
			})
	}

	pub(crate) async fn execute_service_call(
		self: &Arc<Self>,
		call: ServiceCall,
		target: &JsonValue,
		client: ClientId,
		publish: ServicePublish,
		context: Context,
	) -> Result<Option<JsonValue>, AnyError> {
		let queue = self.queue_for(client);
		let target = target.clone();
		let (future, guard) = queue
			.run(async {
				self.start_service_call(client, &target, call, publish, context.clone())
					.await
			})
			.await?;
		// The lease future is driven outside the queue (upstream tracks the
		// promise and returns it immediately); the operation guard keeps the
		// attachment's demand alive until the call settles.
		let result = future.await;
		drop(guard);
		result
	}

	async fn start_service_call(
		self: &Arc<Self>,
		client: ClientId,
		target: &JsonValue,
		call: ServiceCall,
		publish: ServicePublish,
		context: Context,
	) -> Result<(crate::types::BoxFuture<Result<Option<JsonValue>, AnyError>>, OperationGuard), AnyError> {
		let attachment = self.require_attachment(client, target).await?;
		let lease = attachment
			.lease
			.lock()
			.clone()
			.ok_or_else(|| -> AnyError { Arc::new(ServerError::session_not_attached()) })?;
		let (slot, guard) = OperationSlot::new();
		attachment.operations.lock().push(slot);
		let future = lease.invoke_service(call, publish, context);
		Ok((future, guard))
	}

	pub(crate) async fn attach_client(
		self: &Arc<Self>,
		client: ClientId,
		session_id: &str,
		context: Context,
	) -> Result<(), AnyError> {
		if self.is_closing.load(Ordering::SeqCst) {
			return Err(Self::draining());
		}
		let this = Arc::clone(self);
		let session_id = session_id.to_string();
		self.queue_for(client)
			.run(async move { this.attach_client_now(client, &session_id, context).await })
			.await
	}

	async fn attach_client_now(
		self: &Arc<Self>,
		client: ClientId,
		session_id: &str,
		context: Context,
	) -> Result<(), AnyError> {
		if self.is_closing.load(Ordering::SeqCst) || self.is_disconnected(client) {
			return Err(Self::draining());
		}
		let current = self.state.lock().attachments_by_client.get(&client).cloned();
		if current.as_ref().map(|a| a.session.id.as_str()) == Some(session_id) {
			return Ok(());
		}
		let hosted = self.acquire(session_id, context.clone()).await?;
		if self.is_closing.load(Ordering::SeqCst) || self.is_disconnected(client) {
			return Err(Self::draining());
		}
		if let Some(current) = current {
			self.release_attachment(&current, context.clone(), false).await?;
		}
		let attachment = ClientAttachment::new(client, hosted.clone());
		insert_attachment(&hosted, &attachment);
		let outcome: AttachOutcome = hosted.handle.attach_client(context.clone()).await;
		*attachment.acquiring.lock().await = Some(outcome.clone());
		attachment.acquiring_notify.notify_waiters();
		if let Err(error) = outcome {
			remove_attachment(&hosted, &attachment);
			return Err(error);
		}
		let lease = outcome.ok().expect("checked above");
		*attachment.lease.lock() = Some(lease);
		let still_valid = {
			let state = self.state.lock();
			let hosted_current = state.hosted_sessions.get(&hosted.id).map(|h| Arc::ptr_eq(h, &hosted)).unwrap_or(false);
			hosted_current
				&& contains_attachment(&hosted, &attachment)
				&& !state.disconnected_clients.contains(&client)
				&& !self.is_closing.load(Ordering::SeqCst)
		};
		if !still_valid {
			self.release_attachment(&attachment, context.clone(), true).await?;
			return Err(Self::draining());
		}
		self.state.lock().attachments_by_client.insert(client, attachment.clone());
		let target = wire::session_target(&self.server_id, session_id, &attachment.id);
		self.publish_attachment(client, Some(target), context).await
	}

	pub(crate) async fn detach_client(self: &Arc<Self>, client: ClientId, context: Context) -> Result<(), AnyError> {
		let this = Arc::clone(self);
		self.queue_for(client)
			.run(async move {
				let attachment = this.state.lock().attachments_by_client.get(&client).cloned();
				if let Some(attachment) = attachment {
					this.release_attachment(&attachment, context, true).await?;
				}
				Ok(())
			})
			.await
	}

	pub(crate) async fn remove_session(
		self: &Arc<Self>,
		session_id: &str,
		context: Context,
	) -> Result<(), AnyError> {
		if self.is_closing.load(Ordering::SeqCst) {
			return Err(Self::draining());
		}
		let hosted = self.state.lock().hosted_sessions.get(session_id).cloned();
		let Some(hosted) = hosted else { return Ok(()) };
		let mut errors: Vec<AnyError> = Vec::new();
		let attachments: Vec<Arc<ClientAttachment>> = hosted.attachments.lock().clone();
		let releases: Vec<_> = attachments
			.into_iter()
			.map(|attachment| {
				let this = Arc::clone(self);
				let context = context.clone();
				tokio::spawn(async move { this.release_attachment(&attachment, context, true).await })
			})
			.collect();
		for release in releases {
			if let Ok(Err(error)) = release.await {
				errors.push(error);
			}
		}
		if let Err(error) = hosted.handle.close(context.clone()).await {
			errors.push(error);
		}
		{
			let mut state = self.state.lock();
			if state.hosted_sessions.get(session_id).map(|h| Arc::ptr_eq(h, &hosted)).unwrap_or(false) {
				state.hosted_sessions.remove(session_id);
			}
		}
		match errors.len() {
			0 => Ok(()),
			1 => Err(errors.pop().expect("one error")),
			_ => Err(crate::types::aggregate(format!("Failed to close Session {session_id}"), errors)),
		}
	}

	pub(crate) async fn disconnect(self: &Arc<Self>, client: ClientId, context: Context) -> Result<(), AnyError> {
		self.state.lock().disconnected_clients.insert(client);
		let result = async {
			let this = Arc::clone(self);
			self.queue_for(client)
				.run(async move {
					let attachment = this.state.lock().attachments_by_client.get(&client).cloned();
					if let Some(attachment) = attachment {
						this.release_attachment(&attachment, context, false).await?;
					}
					Ok(())
				})
				.await
		}
		.await;
		self.state.lock().disconnected_clients.remove(&client);
		result
	}

	pub(crate) async fn close(self: &Arc<Self>, context: Context) -> Result<(), AnyError> {
		// Upstream `closePromise ??=`: serialize callers on one shutdown run.
		let _guard = self.close_lock.lock().await;
		if let Some(existing) = &*self.close_result.lock() {
			return (**existing).clone();
		}
		let outcome = Arc::new(self.close_internal(context).await);
		*self.close_result.lock() = Some(outcome.clone());
		(*outcome).clone()
	}

	async fn close_internal(self: &Arc<Self>, context: Context) -> Result<(), AnyError> {
		let (queues, openings) = {
			let state = self.state.lock();
			(state.client_queues.values().cloned().collect::<Vec<_>>(), state.opening_sessions.values().cloned().collect::<Vec<_>>())
		};
		for queue in &queues {
			queue.wait_idle().await;
		}
		let mut close_errors: Vec<AnyError> = Vec::new();
		for opening in &openings {
			opening.resolved().await;
			if let Some(outcome) = opening.cell.get() {
				if let Err(error) = &**outcome {
					(self.report_error)(error.clone());
					if error.downcast_ref::<SessionCleanupError>().is_some() {
						close_errors.push(error.clone());
					}
				}
			}
		}
		// Release every live attachment (upstream allSettled over attachments).
		let attachments: Vec<Arc<ClientAttachment>> = {
			let state = self.state.lock();
			state
				.hosted_sessions
				.values()
				.flat_map(|session| session.attachments.lock().iter().cloned().collect::<Vec<_>>())
				.collect()
		};
		let releases: Vec<_> = attachments
			.into_iter()
			.map(|attachment| {
				let this = Arc::clone(self);
				let context = context.clone();
				tokio::spawn(async move { this.release_attachment(&attachment, context, true).await })
			})
			.collect();
		for release in releases {
			if let Ok(Err(error)) = release.await {
				close_errors.push(error);
			}
		}
		let hosted: Vec<Arc<HostedSession>> = {
			self.state.lock().hosted_sessions.values().cloned().collect()
		};
		let closes: Vec<_> = hosted
			.into_iter()
			.map(|session| {
				let context = context.clone();
				tokio::spawn(async move { session.handle.close(context).await.map(|_| session) })
			})
			.collect();
		for close in closes {
			match close.await {
				Ok(Ok(session)) => {
					let mut state = self.state.lock();
					if state.hosted_sessions.get(&session.id).map(|h| Arc::ptr_eq(h, &session)).unwrap_or(false) {
						state.hosted_sessions.remove(&session.id);
					}
				}
				Ok(Err(error)) => {
					(self.report_error)(error.clone());
					close_errors.push(error);
				}
				Err(join_error) => {
					let error: AnyError = Arc::new(pi_chord::services::errors::ChordError {
						message: join_error.to_string(),
					});
					(self.report_error)(error.clone());
					close_errors.push(error);
				}
			}
		}
		{
			let mut state = self.state.lock();
			state.attachments_by_client.clear();
			state.opening_sessions.clear();
		}
		if !close_errors.is_empty() {
			return Err(crate::types::aggregate("Failed to close routed Sessions", close_errors));
		}
		Ok(())
	}

	async fn await_operations(&self, attachment: &ClientAttachment) {
		let slots: Vec<OperationSlot> = attachment.operations.lock().clone();
		for slot in slots {
			slot.settled().await;
		}
	}

	async fn wait_for_acquiring(&self, attachment: &ClientAttachment) -> Result<Arc<dyn RoutedSessionAttachment>, AnyError> {
		loop {
			let notified = attachment.acquiring_notify.notified();
			{
				let acquiring = attachment.acquiring.lock().await;
				if let Some(outcome) = &*acquiring {
					return outcome.clone();
				}
			}
			notified.await;
		}
	}

	pub(crate) async fn release_attachment(
		self: &Arc<Self>,
		attachment: &Arc<ClientAttachment>,
		context: Context,
		publish: bool,
	) -> Result<(), AnyError> {
		let this = Arc::clone(self);
		let inner_attachment = Arc::clone(attachment);
		let outcome = attachment
			.releasing
			.get_or_init(move || async move {
				Arc::new(this.release_attachment_inner(&inner_attachment, context, publish).await)
			})
			.await;
		(**outcome).clone()
	}

	async fn release_attachment_inner(
		self: &Arc<Self>,
		attachment: &Arc<ClientAttachment>,
		context: Context,
		publish: bool,
	) -> Result<(), AnyError> {
		// Wait for in-flight service calls (upstream allSettled(operations)).
		self.await_operations(attachment).await;
		let mut errors: Vec<AnyError> = Vec::new();
		let lease = { attachment.lease.lock().clone() };
		let release_result = async {
			let lease = match lease {
				Some(lease) => lease,
				None => self.wait_for_acquiring(attachment).await?,
			};
			lease.release(context.clone()).await
		}
		.await;
		if let Err(error) = release_result {
			errors.push(error);
		}
		let released = match errors.len() {
			0 => Ok(()),
			1 => Err(errors.pop().expect("one error")),
			_ => Err(crate::types::aggregate("Failed to release Session attachment", errors)),
		};
		// finally: clearAttachment (a clearing failure replaces the outcome).
		if let Err(error) = self.clear_attachment(attachment, context, publish).await {
			return Err(error);
		}
		released
	}

	async fn clear_attachment(
		self: &Arc<Self>,
		attachment: &Arc<ClientAttachment>,
		context: Context,
		publish: bool,
	) -> Result<(), AnyError> {
		remove_attachment(&attachment.session, attachment);
		let was_current = {
			let mut state = self.state.lock();
			state.attachments_by_client.get(&attachment.client).map(|current| Arc::ptr_eq(current, attachment)).unwrap_or(false)
				&& state.attachments_by_client.remove(&attachment.client).is_some()
		};
		if was_current && publish {
			return self.publish_attachment(attachment.client, None, context).await;
		}
		Ok(())
	}

	fn is_disconnected(&self, client: ClientId) -> bool {
		self.state.lock().disconnected_clients.contains(&client)
	}

	async fn require_attachment(self: &Arc<Self>, client: ClientId, target: &JsonValue) -> Result<Arc<ClientAttachment>, AnyError> {
		if self.is_closing.load(Ordering::SeqCst) || self.is_disconnected(client) {
			return Err(Self::draining());
		}
		let Some((session_id, attachment_id)) = wire::target_session(target) else {
			return Err(Arc::new(ServerError::session_not_attached()));
		};
		let state = self.state.lock();
		let attachment = state
			.attachments_by_client
			.get(&client)
			.ok_or_else(|| -> AnyError { Arc::new(ServerError::session_not_attached()) })?;
		if attachment.session.id != session_id || attachment.id != attachment_id {
			return Err(Arc::new(ServerError::session_not_attached()));
		}
		Ok(attachment.clone())
	}

	async fn acquire(self: &Arc<Self>, session_id: &str, context: Context) -> Result<Arc<HostedSession>, AnyError> {
		{
			let state = self.state.lock();
			if let Some(existing) = state.hosted_sessions.get(session_id) {
				return Ok(existing.clone());
			}
		}
		let opening = {
			let mut state = self.state.lock();
			if let Some(existing) = state.hosted_sessions.get(session_id) {
				return Ok(existing.clone());
			}
			match state.opening_sessions.get(session_id).cloned() {
				Some(existing) => {
					existing
						.joiners
						.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
					existing
				}
				None => {
					let created = Opening::new();
					state.opening_sessions.insert(session_id.to_string(), Arc::clone(&created));
					created
				}
			}
		};
		let this = Arc::clone(self);
		let session_id = session_id.to_string();
		let done = Arc::clone(&opening.tx);
		let opening_for_init = Arc::clone(&opening);
		let outcome = opening
			.cell
			.get_or_init(move || async move {
				let result = Arc::new(this.open(&session_id, context).await);
				// Remove the in-flight entry once settled (upstream `finally`).
				{
					let mut state = this.state.lock();
					if state
						.opening_sessions
						.get(&session_id)
						.map(|o| Arc::ptr_eq(o, &opening_for_init))
						.unwrap_or(false)
					{
						state.opening_sessions.remove(&session_id);
					}
				}
				let _ = done.send(true);
				result
			})
			.await;
		(**outcome).clone()
	}

	async fn open(self: &Arc<Self>, session_id: &str, context: Context) -> Result<Arc<HostedSession>, AnyError> {
		let metadata = self.host.resolve_session(session_id, context.clone()).await?;
		let handle = self.host.open_session(metadata.clone(), context.clone()).await?;
		if self.is_closing.load(Ordering::SeqCst) {
			if let Err(error) = handle.close(context).await {
				(self.report_error)(error.clone());
				return Err(Arc::new(SessionCleanupError));
			}
			return Err(Self::draining());
		}
		let hosted = Arc::new(HostedSession {
			id: metadata.id().to_string(),
			handle,
			attachments: parking_lot::Mutex::new(Vec::new()),
		});
		self.state.lock().hosted_sessions.insert(hosted.id.clone(), hosted.clone());
		if let Some(terminated) = hosted.handle.terminated() {
			let this = Arc::clone(self);
			let hosted_for_task = Arc::clone(&hosted);
			tokio::spawn(async move {
				let error = terminated.await;
				this.invalidate(&hosted_for_task, error).await;
			});
		}
		Ok(hosted)
	}

	async fn invalidate(self: &Arc<Self>, hosted: &Arc<HostedSession>, error: Option<AnyError>) {
		{
			let mut state = self.state.lock();
			if state.hosted_sessions.get(&hosted.id).map(|h| Arc::ptr_eq(h, hosted)).unwrap_or(false) {
				state.hosted_sessions.remove(&hosted.id);
			} else {
				return;
			}
		}
		let attachments: Vec<Arc<ClientAttachment>> = hosted.attachments.lock().clone();
		for attachment in attachments {
			let this = Arc::clone(self);
			tokio::spawn(async move {
				if let Err(release_error) = this.release_attachment(&attachment, BACKGROUND_CONTEXT.clone(), true).await {
					(this.report_error)(release_error);
				}
			});
		}
		if let Some(error) = error {
			(self.report_error)(error);
		}
	}
}
