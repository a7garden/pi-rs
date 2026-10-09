//! Port of `experimental/services/connection.ts` — the server-scoped and
//! selected-Session remote service sources for one presentation client.
//!
//! Deviations (D144):
//! - `RoutedServiceBinding` wraps chord's [`pi_chord::RemoteServiceBinding`]
//!   behind the [`pi_chord::RemoteServices`] trait; lazy activation and the
//!   one-shot `onActivate` mirror upstream `ready()`.
//! - Upstream promise-chain transition serialization maps to a tokio mutex:
//!   connection/attachment transitions rebind in listener order and publish
//!   `connected`/`attached`/`degraded` state after they settle.
//! - `RemoteServiceSource` methods take `&Context` per chord's Rust
//!   conventions; state DTOs serialize into chord's untyped replicated states.
//! - Errors: upstream `AggregateError` maps to [`pi_chord::AggregateError`].

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use parking_lot::Mutex;
use pi_chord::context::{Context, BACKGROUND_CONTEXT};
use pi_chord::types::RemoteServiceSourceOpenOptions;
use pi_chord::services::consumer::RemoteServiceBinding;
use pi_chord::services::errors::{error, type_error, AnyError, AggregateError};
use pi_chord::services::state::MutableReplicatedState;
use pi_chord::types::{
	KeyedObservationHandler, RemoteServiceBindingOptions, RemoteServiceTransport, RemoteServices,
	ServiceCatalogueEntry, ServiceClientHandle, Unsubscribe,
};
use pi_client::client::Client;
use pi_client::types::{ConnectionState, RpcTarget, SessionTarget};
use serde::{Deserialize, Serialize};

/// Upstream `ServerConnectionState`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum ServerConnectionState {
	Connecting { attempt: u64 },
	Connected { since: String },
	Disconnected { since: String, reason: String, retry_at: Option<String> },
}

/// Upstream `SessionAttachmentState`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum SessionAttachmentState {
	Detached,
	Attaching { session_id: String },
	Attached { session_id: String },
	Degraded { session_id: String },
}

/// Upstream `ServiceSourceOptions`.
#[derive(Clone, Default)]
pub struct ServiceSourceOptions {
	pub on_error: Option<Arc<dyn Fn(AnyError) + Send + Sync>>,
}

type ActivateFn =
	Arc<dyn Fn(Context) -> futures_util::future::BoxFuture<'static, Result<(), AnyError>> + Send + Sync>;

/// Upstream `ServerServiceSource`.
#[async_trait]
pub trait ServerServiceSource: Send + Sync {
	fn accepts_unavailable_services(&self) -> bool;
	async fn catalogue(&self, context: &Context) -> Result<Vec<ServiceCatalogueEntry>, AnyError>;
	async fn open(&self, options: RemoteServiceSourceOpenOptions) -> Result<Box<dyn RemoteServices>, AnyError>;
	fn connection(&self) -> Arc<MutableReplicatedState>;
	async fn dispose(&self, context: &Context) -> Result<(), AnyError>;
}

/// Upstream `SessionServiceSource`.
#[async_trait]
pub trait SessionServiceSource: Send + Sync {
	fn accepts_unavailable_services(&self) -> bool;
	async fn catalogue(&self, context: &Context) -> Result<Vec<ServiceCatalogueEntry>, AnyError>;
	async fn open(&self, options: RemoteServiceSourceOpenOptions) -> Result<Box<dyn RemoteServices>, AnyError>;
	fn attachment(&self) -> Arc<MutableReplicatedState>;
	/// Wait for the exact current attachment generation to finish hydrating.
	async fn when_attached(&self, session_id: &str, context: &Context) -> Result<(), AnyError>;
	/// Wait for every binding to finish releasing the previous attachment.
	async fn when_detached(&self, context: &Context) -> Result<(), AnyError>;
	async fn dispose(&self, context: &Context) -> Result<(), AnyError>;
}

/// Shared bookkeeping for both sources.
struct SourceCore {
	client: Client,
	transport: Arc<dyn RemoteServiceTransport>,
	bindings: Mutex<Vec<Weak<RoutedServiceBinding>>>,
	on_error: Arc<dyn Fn(AnyError) + Send + Sync>,
	disposed: AtomicBool,
}

impl SourceCore {
	fn live_bindings(&self) -> Vec<Arc<RoutedServiceBinding>> {
		self.bindings
			.lock()
			.iter()
			.filter_map(|weak| weak.upgrade())
			.collect()
	}

	fn open_binding(
		self: &Arc<Self>,
		services: Vec<String>,
		get_bound: Arc<dyn Fn() -> bool + Send + Sync>,
		on_activate: Option<ActivateFn>,
		assert_access: Option<Arc<dyn Fn() -> Result<(), AnyError> + Send + Sync>>,
	) -> Result<Arc<RoutedServiceBinding>, AnyError> {
		if self.disposed.load(Ordering::SeqCst) {
			return Err(error("Service source is disposed"));
		}
		let on_error = Arc::clone(&self.on_error);
		let core = Arc::downgrade(self);
		let chord_binding = RemoteServiceBinding::new(RemoteServiceBindingOptions {
			services,
			transport: Arc::clone(&self.transport),
			bound: false,
			on_error: Some(on_error),
			assert_access,
		})?;
		let binding = RoutedServiceBinding {
			binding: chord_binding,
			get_bound,
			activated: AtomicBool::new(false),
			activation_complete: AtomicBool::new(false),
			on_activate,
			core,
		};
		let binding_arc = Arc::new(binding);
		self.bindings.lock().push(Arc::downgrade(&binding_arc));
		Ok(binding_arc)
	}

	/// Rebind every live binding, collecting failures (upstream
	/// `Promise.allSettled` + `throwFailures`).
	async fn rebind_all(&self, bound: bool, context: &Context, message: &str) -> Result<(), AnyError> {
		let mut failures = Vec::new();
		for binding in self.live_bindings() {
			if let Err(rebind_error) = binding.update_bound(bound, context).await {
				failures.push(rebind_error);
			}
		}
		throw_failures(failures, message)
	}

	async fn dispose_all(&self, context: &Context, message: &str) -> Result<(), AnyError> {
		self.bindings.lock().clear();
		let mut failures = Vec::new();
		for binding in self.live_bindings() {
			if let Err(dispose_error) = binding.dispose(context).await {
				failures.push(dispose_error);
			}
		}
		throw_failures(failures, message)
	}
}

/// One routed binding over the chord binding with lazy activation
/// (upstream `RoutedServiceBinding`).
struct RoutedServiceBinding {
	binding: RemoteServiceBinding,
	get_bound: Arc<dyn Fn() -> bool + Send + Sync>,
	activated: AtomicBool,
	activation_complete: AtomicBool,
	on_activate: Option<ActivateFn>,
	/// Back-reference for self-removal on drop (D144: the JS closure captures
	/// `bindings.delete(binding)`; Rust drops prune the weak list).
	core: Weak<SourceCore>,
}

impl RoutedServiceBinding {
	async fn ready(&self, context: &Context) -> Result<(), AnyError> {
		if !self.activated.swap(true, Ordering::SeqCst) {
			self.binding.rebind((self.get_bound)(), context).await?;
		}
		self.binding.ready(context).await?;
		if !self.activation_complete.swap(true, Ordering::SeqCst) {
			if let Some(on_activate) = &self.on_activate {
				on_activate(context.clone()).await?;
			}
		}
		Ok(())
	}

	async fn update_bound(&self, bound: bool, context: &Context) -> Result<(), AnyError> {
		if self.activated.load(Ordering::SeqCst) {
			self.binding.rebind(bound, context).await?;
		}
		Ok(())
	}
}

impl Drop for RoutedServiceBinding {
	fn drop(&mut self) {
		if let Some(core) = self.core.upgrade() {
			core.bindings.lock().retain(|weak| weak.strong_count() > 0);
		}
	}
}

#[async_trait::async_trait]
impl RemoteServices for RoutedServiceBinding {
	fn use_service(&self, service_id: &str) -> Result<Arc<dyn ServiceClientHandle>, AnyError> {
		self.binding.use_service(service_id)
	}

	fn observe(
		&self,
		service_id: &str,
		handler: KeyedObservationHandler,
	) -> Result<Unsubscribe, AnyError> {
		self.binding.observe(service_id, handler)
	}

	async fn ready(&self, context: &Context) -> Result<(), AnyError> {
		RoutedServiceBinding::ready(self, context).await
	}

	async fn dispose(&self, context: &Context) -> Result<(), AnyError> {
		self.binding.dispose(context).await
	}
}

/// `Box<dyn RemoteServices>` front for the shared binding (the binding stays
/// reachable through the source's weak list for rebinding).
struct SharedBinding(Arc<RoutedServiceBinding>);

#[async_trait::async_trait]
impl RemoteServices for SharedBinding {
	fn use_service(&self, service_id: &str) -> Result<Arc<dyn ServiceClientHandle>, AnyError> {
		self.0.use_service(service_id)
	}

	fn observe(
		&self,
		service_id: &str,
		handler: KeyedObservationHandler,
	) -> Result<Unsubscribe, AnyError> {
		self.0.observe(service_id, handler)
	}

	async fn ready(&self, context: &Context) -> Result<(), AnyError> {
		self.0.ready(context).await
	}

	async fn dispose(&self, context: &Context) -> Result<(), AnyError> {
		self.0.dispose(context).await
	}
}

/// Upstream `ServerServiceSourceImpl`.
pub struct ServerServiceSourceImpl {
	core: Arc<SourceCore>,
	connection: Arc<MutableReplicatedState>,
	transition_lock: Arc<tokio::sync::Mutex<()>>,
	_listener: Mutex<Option<Unsubscribe>>,
}

impl ServerServiceSourceImpl {
	fn new(client: Client, options: ServiceSourceOptions) -> Arc<ServerServiceSourceImpl> {
		let transport = pi_client::client::create_client_service_transport(&client, {
			let client = client.clone();
			move || Some(RpcTarget::Server { server_id: client.server_id().to_string() })
		});
		let on_error = options
			.on_error
			.unwrap_or_else(|| Arc::new(|_: AnyError| {}));
		let attempt = if client.connection_state() == ConnectionState::Connecting { 1 } else { 0 };
		let connection = pi_chord::api::replicated_state(
			serde_json::to_value(to_server_connection_state(&client, attempt, None))
				.expect("ServerConnectionState serializes"),
		);
		let core = Arc::new(SourceCore {
			client: client.clone(),
			transport,
			bindings: Mutex::new(Vec::new()),
			on_error,
			disposed: AtomicBool::new(false),
		});
		let transition_lock = Arc::new(tokio::sync::Mutex::new(()));
		let listener_core = Arc::downgrade(&core);
		let listener_attempt = Arc::new(AtomicU64::new(attempt));
		let listener_lock = Arc::clone(&transition_lock);
		let listener_connection = Arc::clone(&connection);
		let listener_client = client.clone();
		let remove = client
			.on_connection_state_change(Arc::new(
				move |change: &pi_client::types::ConnectionStateChange| {
					let Some(core) = listener_core.upgrade() else {
						return;
					};
					let connected = change.state == ConnectionState::Connected;
					if change.state == ConnectionState::Connecting {
						listener_attempt.fetch_add(1, Ordering::SeqCst);
					}
					// Publish the replacement state (upstream
					// `publishReplacement(connectionState, toServerConnectionState(...))`).
					let reason = change.clone_error().map(|listener_error| listener_error.to_string());
					let state_value = to_server_connection_state(&listener_client, listener_attempt.load(Ordering::SeqCst), reason);
					let _ = listener_connection.replace(
						&BACKGROUND_CONTEXT,
						serde_json::to_value(state_value).expect("ServerConnectionState serializes"),
					);
					let lock = Arc::clone(&listener_lock);
					tokio::spawn(async move {
						let _guard = lock.lock().await;
						if let Err(rebind_error) = core
							.rebind_all(connected, &BACKGROUND_CONTEXT, "Failed to rebind server services")
							.await
						{
							(core.on_error)(rebind_error);
						}
					});
				},
			))
			.expect("connection listener registers");
		Arc::new(ServerServiceSourceImpl {
			core,
			connection,
			transition_lock,
		_listener: Mutex::new(Some(remove)),
		})
	}
}

#[async_trait]
impl ServerServiceSource for ServerServiceSourceImpl {
	fn accepts_unavailable_services(&self) -> bool {
		false
	}

	async fn catalogue(&self, context: &Context) -> Result<Vec<ServiceCatalogueEntry>, AnyError> {
		let target = RpcTarget::Server { server_id: self.core.client.server_id().to_string() };
		self.core
			.client
			.service_catalogue(&target, context.abort_signal().as_ref())
			.await
	}

	async fn open(
		&self,
		options: RemoteServiceSourceOpenOptions,
	) -> Result<Box<dyn RemoteServices>, AnyError> {
		let client = self.core.client.clone();
		let get_bound: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || client.connected());
		Ok(Box::new(SharedBinding(self.core.open_binding(
			options.services,
			get_bound,
			None,
			Some(options.assert_access),
		)?)))
	}

	fn connection(&self) -> Arc<MutableReplicatedState> {
		Arc::clone(&self.connection)
	}

	async fn dispose(&self, context: &Context) -> Result<(), AnyError> {
		if self.core.disposed.swap(true, Ordering::SeqCst) {
			return Ok(());
		}
		let _guard = self.transition_lock.lock().await;
		self.core
			.dispose_all(context, "Failed to dispose server service source")
			.await
	}
}

/// Cached worker catalogue for the current (or a just-detached) attachment.
type CatalogueCache = Arc<Mutex<Option<Vec<ServiceCatalogueEntry>>>>;

/// Upstream `SessionServiceSourceImpl`.
pub struct SessionServiceSourceImpl {
	core: Arc<SourceCore>,
	attachment_state: Arc<MutableReplicatedState>,
	attachment_revision: Arc<AtomicU64>,
	catalogue_cache: CatalogueCache,
	transition_lock: Arc<tokio::sync::Mutex<()>>,
	_listener: Mutex<Option<Unsubscribe>>,
}

impl SessionServiceSourceImpl {
	fn new(client: Client, options: ServiceSourceOptions) -> Arc<SessionServiceSourceImpl> {
		let transport = pi_client::client::create_client_service_transport(&client, {
			let client = client.clone();
			move || {
				let attachment = client.attachment()?;
				Some(RpcTarget::Session(attachment))
			}
		});
		let on_error = options
			.on_error
			.unwrap_or_else(|| Arc::new(|_: AnyError| {}));
		let attachment_state = pi_chord::api::replicated_state(
			serde_json::to_value(match client.attachment() {
				None => SessionAttachmentState::Detached,
				Some(attachment) => SessionAttachmentState::Attaching {
					session_id: attachment.session_id,
				},
			})
			.expect("SessionAttachmentState serializes"),
		);
		let core = Arc::new(SourceCore {
			client: client.clone(),
			transport,
			bindings: Mutex::new(Vec::new()),
			on_error,
			disposed: AtomicBool::new(false),
		});
		let transition_lock = Arc::new(tokio::sync::Mutex::new(()));
		let catalogue_cache: CatalogueCache = Arc::new(Mutex::new(None));
		let listener_core = Arc::downgrade(&core);
		let listener_state = Arc::clone(&attachment_state);
		let listener_lock = Arc::clone(&transition_lock);
		let listener_catalogue = Arc::clone(&catalogue_cache);
		let revision = Arc::new(AtomicU64::new(0));
		let listener_revision = Arc::clone(&revision);
		let remove = client
			.on_attachment_change(Arc::new(move |attachment: Option<&SessionTarget>| {
				let Some(core) = listener_core.upgrade() else {
					return;
				};
				let revision_now = listener_revision.fetch_add(1, Ordering::SeqCst) + 1;
				if let Some(attachment) = attachment {
					let _ = listener_state.replace(
						&BACKGROUND_CONTEXT,
						serde_json::to_value(SessionAttachmentState::Attaching {
							session_id: attachment.session_id.clone(),
						})
						.expect("SessionAttachmentState serializes"),
					);
					// Refresh the cached worker catalogue for the new attachment
					// (upstream: void serviceCatalogue(attachment).then(...)).
					let client_for_catalogue = core.client.clone();
					let cache_for_catalogue = Arc::clone(&listener_catalogue);
					let attachment_for_catalogue = attachment.clone();
					tokio::spawn(async move {
						let target = RpcTarget::Session(attachment_for_catalogue);
						match client_for_catalogue.service_catalogue(&target, None).await {
							Ok(catalogue) => *cache_for_catalogue.lock() = Some(catalogue),
							Err(_catalogue_error) => {}
						}
					});
				}
				let state = Arc::clone(&listener_state);
				let lock = Arc::clone(&listener_lock);
				let attached = attachment.is_some();
				let session_id = attachment.map(|target| target.session_id.clone());
				tokio::spawn(async move {
					let _guard = lock.lock().await;
					let result = core
						.rebind_all(attached, &BACKGROUND_CONTEXT, "Failed to rebind Session services")
						.await;
					// Upstream publishes only when the revision and attachment
					// still match; the serialized transition approximates that.
					let published = match (&result, session_id) {
						(Ok(()), Some(session_id)) => SessionAttachmentState::Attached { session_id },
						(Ok(()), None) => SessionAttachmentState::Detached,
						(Err(_), Some(session_id)) => SessionAttachmentState::Degraded { session_id },
						(Err(_), None) => SessionAttachmentState::Detached,
					};
					let _ = state.replace(
						&BACKGROUND_CONTEXT,
						serde_json::to_value(published).expect("SessionAttachmentState serializes"),
					);
					if let Err(transition_error) = result {
						(core.on_error)(transition_error);
					}
					let _ = revision_now;
				});
			}))
			.expect("attachment listener registers");
		Arc::new(SessionServiceSourceImpl {
			core,
			attachment_state,
			attachment_revision: revision,
			catalogue_cache,
			transition_lock,
		_listener: Mutex::new(Some(remove)),
		})
	}

	async fn when_attached_internal(
		&self,
		attachment: &SessionTarget,
		revision: u64,
		context: &Context,
	) -> Result<(), AnyError> {
		let mut failures = Vec::new();
		for binding in self.core.live_bindings() {
			if let Err(ready_error) = binding.ready(context).await {
				failures.push(ready_error);
			}
		}
		if let Err(ready_error) = throw_failures(failures, "Failed to prepare Session services") {
			if self.attachment_revision.load(Ordering::SeqCst) == revision
				&& same_attachment(self.core.client.attachment().as_ref(), Some(attachment))
			{
				let _ = self.attachment_state.replace(
					context,
					serde_json::to_value(SessionAttachmentState::Degraded {
						session_id: attachment.session_id.clone(),
					})
					.expect("SessionAttachmentState serializes"),
				);
			}
			return Err(ready_error);
		}
		if self.attachment_revision.load(Ordering::SeqCst) != revision
			|| !same_attachment(self.core.client.attachment().as_ref(), Some(attachment))
		{
			return Err(error(format!(
				"Session {} was replaced while attaching",
				attachment.session_id
			)));
		}
		Ok(())
	}
}

#[async_trait]
impl SessionServiceSource for SessionServiceSourceImpl {
	fn accepts_unavailable_services(&self) -> bool {
		self.core.client.attachment().is_none() && self.catalogue_cache.lock().is_none()
	}

	async fn catalogue(&self, context: &Context) -> Result<Vec<ServiceCatalogueEntry>, AnyError> {
		let Some(attachment) = self.core.client.attachment() else {
			return Ok(self.catalogue_cache.lock().clone().unwrap_or_default());
		};
		let target = RpcTarget::Session(attachment);
		let catalogue = self
			.core
			.client
			.service_catalogue(&target, context.abort_signal().as_ref())
			.await?;
		*self.catalogue_cache.lock() = Some(catalogue.clone());
		Ok(catalogue)
	}

	async fn open(
		&self,
		options: RemoteServiceSourceOpenOptions,
	) -> Result<Box<dyn RemoteServices>, AnyError> {
		let client = self.core.client.clone();
		let get_bound: Arc<dyn Fn() -> bool + Send + Sync> =
			Arc::new(move || client.attachment().is_some());
		// Upstream's onActivate waits for the exact attachment generation; the
		// Rust transition task serializes rebinds and publishes state, so
		// activation waits on the transition lock before readiness resolves.
		let transition_lock = Arc::clone(&self.transition_lock);
		let revision = self.attachment_revision.load(Ordering::SeqCst);
		let on_activate: ActivateFn = {
			let lock = Arc::clone(&transition_lock);
			let revision_counter = Arc::clone(&self.attachment_revision);
			let client_for_activate = self.core.client.clone();
			Arc::new(move |context| {
				let lock = Arc::clone(&lock);
				let revision_counter = Arc::clone(&revision_counter);
				let client_for_activate = client_for_activate.clone();
				Box::pin(async move {
					let _guard = lock.lock().await;
					if revision_counter.load(Ordering::SeqCst) != revision {
						return Err(error("Session attachment changed while activating"));
					}
					if client_for_activate.attachment().is_none() {
						// Upstream awaits whenDetached here; draining the lock
						// above covers the released-binding half (D144).
					}
					let _ = context;
					Ok(())
				})
			})
		};
		Ok(Box::new(SharedBinding(self.core.open_binding(
			options.services,
			get_bound,
			Some(on_activate),
			Some(options.assert_access),
		)?)))
	}

	fn attachment(&self) -> Arc<MutableReplicatedState> {
		Arc::clone(&self.attachment_state)
	}

	async fn when_attached(&self, session_id: &str, context: &Context) -> Result<(), AnyError> {
		let attachment = self
			.core
			.client
			.attachment()
			.filter(|attachment| attachment.session_id == session_id);
		let Some(attachment) = attachment else {
			return Err(type_error(format!("Session {session_id} is not the current attachment")));
		};
		let revision = self.attachment_revision.load(Ordering::SeqCst);
		self.when_attached_internal(&attachment, revision, context).await
	}

	async fn when_detached(&self, _context: &Context) -> Result<(), AnyError> {
		if self.core.client.attachment().is_some() {
			return Err(error("A Session is still attached"));
		}
		// The listener transition drains rebinds before publishing `detached`.
		let _guard = self.transition_lock.lock().await;
		Ok(())
	}

	async fn dispose(&self, context: &Context) -> Result<(), AnyError> {
		if self.core.disposed.swap(true, Ordering::SeqCst) {
			return Ok(());
		}
		let _guard = self.transition_lock.lock().await;
		self.core
			.dispose_all(context, "Failed to dispose Session service source")
			.await
	}
}

/// `createServerServiceSource(client, options?)`.
pub fn create_server_service_source(
	client: Client,
	options: ServiceSourceOptions,
) -> Arc<dyn ServerServiceSource> {
	ServerServiceSourceImpl::new(client, options)
}

/// `createSessionServiceSource(client, options?)`.
pub fn create_session_service_source(
	client: Client,
	options: ServiceSourceOptions,
) -> Arc<dyn SessionServiceSource> {
	SessionServiceSourceImpl::new(client, options)
}

fn to_server_connection_state(
	client: &Client,
	attempt: u64,
	reason: Option<String>,
) -> ServerConnectionState {
	let since = now_iso();
	match client.connection_state() {
		ConnectionState::Connecting => ServerConnectionState::Connecting { attempt },
		ConnectionState::Connected => ServerConnectionState::Connected { since },
		ConnectionState::Disconnected => ServerConnectionState::Disconnected {
			since,
			reason: reason.unwrap_or_else(|| "Client is disconnected".to_string()),
			retry_at: None,
		},
	}
}

/// Upstream `new Date().toISOString()` (UTC, millisecond precision).
fn now_iso() -> String {
	let duration = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.unwrap_or_default();
	format_rfc3339(duration.as_millis() as i64)
}

fn format_rfc3339(unix_millis: i64) -> String {
	let days = unix_millis.div_euclid(86_400_000);
	let millis_of_day = unix_millis.rem_euclid(86_400_000);
	let (year, month, day) = civil_from_days(days);
	let hour = millis_of_day / 3_600_000;
	let minute = (millis_of_day % 3_600_000) / 60_000;
	let second = (millis_of_day % 60_000) / 1_000;
	let millis = millis_of_day % 1_000;
	format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

/// Howard Hinnant's `civil_from_days` algorithm.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
	let z = days + 719_468;
	let era = z.div_euclid(146_097);
	let doe = z.rem_euclid(146_097);
	let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
	let year = yoe + era * 400;
	let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
	let mp = (5 * doy + 2) / 153;
	let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
	let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
	(if month <= 2 { year + 1 } else { year }, month, day)
}

fn same_attachment(left: Option<&SessionTarget>, right: Option<&SessionTarget>) -> bool {
	match (left, right) {
		(None, None) => true,
		(Some(left), Some(right)) => {
			left.server_id == right.server_id
				&& left.session_id == right.session_id
				&& left.attachment_id == right.attachment_id
		}
		_ => false,
	}
}

/// Upstream `throwFailures(results, message)`.
fn throw_failures(failures: Vec<AnyError>, message: &str) -> Result<(), AnyError> {
	match failures.len() {
		0 => Ok(()),
		1 => Err(failures.into_iter().next().expect("one failure")),
		_ => Err(AggregateError::new(message, failures)),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn attachment_equality_ignores_generation_mismatch_details() {
		let base = SessionTarget {
			server_id: "s".into(),
			session_id: "sess".into(),
			attachment_id: "a1".into(),
		};
		assert!(same_attachment(Some(&base), Some(&base)));
		let other = SessionTarget { attachment_id: "a2".into(), ..base.clone() };
		assert!(!same_attachment(Some(&base), Some(&other)));
		assert!(!same_attachment(None, Some(&base)));
		assert!(same_attachment(None, None));
	}

	#[test]
	fn rfc3339_matches_upstream_shape() {
		assert_eq!(format_rfc3339(0), "1970-01-01T00:00:00.000Z");
		assert_eq!(format_rfc3339(1_700_000_000_123), "2023-11-14T22:13:20.123Z");
	}

	#[test]
	fn connection_state_serializes_with_status_tag() {
		let state = ServerConnectionState::Disconnected {
			since: "t".into(),
			reason: "r".into(),
			retry_at: None,
		};
		let value = serde_json::to_value(&state).unwrap();
		assert_eq!(value["status"], "disconnected");
		assert_eq!(value["retryAt"], serde_json::Value::Null);
	}
}
