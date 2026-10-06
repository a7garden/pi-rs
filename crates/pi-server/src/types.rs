//! Shared types for `pi-server` (upstream: `packages/server/src/types.ts` plus
//! crate-wide error/future aliases).
//!
//! Upstream generics collapse to trait objects: `TMetadata extends
//! SessionMetadata` becomes [`Arc<dyn SessionMetadata>`] (identity of the
//! concrete metadata value is preserved by the `Arc`, which the conformance
//! test "attach passes concrete repository metadata" relies on).

use std::error::Error;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;

use crate::listener::ServerListener;

/// Owned error value crossing server boundaries (upstream `Error`/`unknown`).
/// `Arc` rather than `Box` so lifecycle results (upstream `close()`'s
/// rejection identity) can be observed by multiple awaiters.
pub type AnyError = Arc<dyn Error + Send + Sync>;

/// Upstream `Promise<T>` where only the settled outcome is shared.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Wrap any error value into [`AnyError`].
pub fn any_error(error: impl Error + Send + Sync + 'static) -> AnyError {
	Arc::new(error)
}

/// Convert a chord-owned error (`Box<dyn Error + Send + Sync>`) into [`AnyError`].
pub fn into_any_error(error: pi_chord::AnyError) -> AnyError {
	Arc::from(error)
}

/// Wrap an I/O error into [`AnyError`].
pub fn io_error(error: std::io::Error) -> AnyError {
	Arc::new(error)
}

/// Wrap a concrete error value into [`AnyError`].
pub fn any<E: Error + Send + Sync + 'static>(error: E) -> AnyError {
	Arc::new(error)
}

/// Build a chord `AggregateError` from server errors. Individual errors are
/// flattened to their Display text (chord aggregates own chord-owned errors;
/// the aggregate message is the observable surface in tests).
pub fn aggregate(message: impl Into<String>, errors: Vec<AnyError>) -> AnyError {
	let flattened: Vec<pi_chord::AnyError> = errors
		.into_iter()
		.map(|error| -> pi_chord::AnyError { Box::new(ChordError { message: error.to_string() }) })
		.collect();
	into_any_error(pi_chord::AggregateError::new(message, flattened))
}

use pi_chord::ChordError;

/// `MaybePromise<T>` for methods that return a promise the callee drives.
/// Sync returns collapse into an immediately-polled future by convention.
pub type MaybeFuture<T> = BoxFuture<T>;

/// Upstream `ServerOptions`.
#[derive(Clone)]
pub struct ServerOptions {
	pub listeners: Vec<Arc<dyn ServerListener>>,
	/// Stable logical server identity supplied by the installation or profile.
	pub server_id: String,
	pub max_frame_length: Option<usize>,
	pub handshake_timeout_ms: Option<u64>,
	pub on_connection_count_changed: Option<ConnectionCountCallback>,
	pub on_error: Option<ErrorCallback>,
}

/// Upstream `onConnectionCountChanged`.
pub type ConnectionCountCallback = Arc<dyn Fn(usize) + Send + Sync>;
/// Upstream `onError`.
pub type ErrorCallback = Arc<dyn Fn(AnyError) + Send + Sync>;

impl std::fmt::Debug for ServerOptions {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ServerOptions")
			.field("server_id", &self.server_id)
			.field("listeners", &self.listeners.len())
			.field("max_frame_length", &self.max_frame_length)
			.field("handshake_timeout_ms", &self.handshake_timeout_ms)
			.finish()
	}
}

/// Minimal durable Session metadata the server needs for routing.
/// Applications may extend it (upstream: interface extension; Rust: trait impl).
pub trait SessionMetadata: Send + Sync {
	fn id(&self) -> &str;
}

/// Concrete default metadata (upstream `SessionMetadata` base shape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicSessionMetadata {
	pub id: String,
}

impl SessionMetadata for BasicSessionMetadata {
	fn id(&self) -> &str {
		&self.id
	}
}

/// One presentation connection's live capability for a hosted Session
/// (upstream `RoutedSessionAttachment`).
///
/// `invoke_service` returns the call's future without awaiting it (upstream
/// returns a promise immediately); the caller decides when to drive it.
pub trait RoutedSessionAttachment: Send + Sync {
	fn invoke_service(
		&self,
		call: pi_chord::ServiceCall,
		publish: ServicePublish,
		context: pi_chord::Context,
	) -> BoxFuture<Result<Option<pi_chord::JsonValue>, AnyError>>;
	fn release(&self, context: pi_chord::Context) -> MaybeFuture<Result<(), AnyError>>;
}

/// Upstream `publish: (subscriptionId, update, context) => MaybePromise<void>`.
pub type ServicePublish = Arc<dyn Fn(String, pi_chord::ServiceProviderUpdate, pi_chord::Context) -> MaybeFuture<Result<(), AnyError>> + Send + Sync>;

/// Presentation-scoped routing capabilities available to server service
/// implementations (upstream `RoutedServerPresentation`).
#[async_trait]
pub trait RoutedServerPresentation: Send + Sync {
	async fn attach_session(&self, session_id: &str, context: pi_chord::Context) -> Result<(), AnyError>;
	async fn detach_session(&self, context: pi_chord::Context) -> Result<(), AnyError>;
	/// Release routed attachments and handles before the application deletes
	/// durable metadata.
	async fn prepare_session_removal(&self, session_id: &str, context: pi_chord::Context) -> Result<(), AnyError>;
}

/// One connection's server-scoped service endpoint
/// (upstream `RoutedServerServiceAttachment`).
pub trait RoutedServerServiceAttachment: Send + Sync {
	fn invoke_service(
		&self,
		call: pi_chord::ServiceCall,
		publish: ServicePublish,
		context: pi_chord::Context,
	) -> BoxFuture<Result<Option<pi_chord::JsonValue>, AnyError>>;
	fn release(&self, context: pi_chord::Context) -> MaybeFuture<Result<(), AnyError>>;
}

/// Upstream `RoutedServerServiceHost`.
#[async_trait]
pub trait RoutedServerServiceHost: Send + Sync {
	async fn attach_client(
		&self,
		presentation: Arc<dyn RoutedServerPresentation>,
		context: pi_chord::Context,
	) -> Result<Arc<dyn RoutedServerServiceAttachment>, AnyError>;
}

/// A process-safe handle that acquires presentation-scoped Session capabilities
/// (upstream `RoutedSessionHandle`).
#[async_trait]
pub trait RoutedSessionHandle: Send + Sync {
	/// Upstream `attachClient(): MaybePromise<RoutedSessionAttachment>`.
	fn attach_client(
		&self,
		context: pi_chord::Context,
	) -> MaybeFuture<Result<Arc<dyn RoutedSessionAttachment>, AnyError>>;
	/// Upstream `terminated?: Promise<Error | undefined>` — resolves with an
	/// error for unexpected termination, or `None` after an expected close.
	/// `None` at the trait level means the handle has no termination signal.
	fn terminated(&self) -> Option<BoxFuture<Option<AnyError>>>;
	async fn close(&self, context: pi_chord::Context) -> Result<(), AnyError>;
}

/// Application capabilities used by server-wide management and Session routing
/// (upstream `ServerHost`).
#[async_trait]
pub trait ServerHost: Send + Sync {
	fn server_services(&self) -> Arc<dyn RoutedServerServiceHost>;
	/// Resolve one durable Session ID or throw a bounded routing error.
	async fn resolve_session(&self, session_id: &str, context: pi_chord::Context)
		-> Result<Arc<dyn SessionMetadata>, AnyError>;
	async fn open_session(
		&self,
		metadata: Arc<dyn SessionMetadata>,
		context: pi_chord::Context,
	) -> Result<Arc<dyn RoutedSessionHandle>, AnyError>;
}
