//! Port of `packages/client/src/types.ts`.
//!
//! Upstream imports `RpcTarget`/`SessionTarget` as types from `pi-protocol`;
//! the Rust protocol port models them as validated strict-JSON shapes without
//! typed structs, so the client owns the typed definitions here and converts
//! through [`RpcTarget::to_json`] / [`SessionTarget::to_json`].

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use pi_chord::json::JsonValue;
use pi_chord::services::errors::AnyError;
use pi_chord::types::ServiceSubscriptionSnapshot;
use serde::{Deserialize, Serialize};

/// Disposer returned by listener registration APIs (upstream `Unsubscribe`).
pub type Unsubscribe = Box<dyn FnOnce() + Send>;

/// Reports subscriber failures (upstream `ListenerErrorHandler`). Errors are
/// owned chord errors.
pub type ListenerErrorHandler = Arc<dyn Fn(AnyError) + Send + Sync>;

/// Attachment change callback (upstream `AttachmentChangeListener`).
pub type AttachmentChangeListener = Arc<dyn Fn(Option<&SessionTarget>) + Send + Sync>;

/// Upstream `SessionTarget`: `{ serverId, sessionId, attachmentId }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionTarget {
	#[serde(rename = "serverId")]
	pub server_id: String,
	#[serde(rename = "sessionId")]
	pub session_id: String,
	#[serde(rename = "attachmentId")]
	pub attachment_id: String,
}

impl SessionTarget {
	pub fn to_json(&self) -> JsonValue {
		serde_json::to_value(self).expect("SessionTarget serializes to strict JSON")
	}

	/// Reads a session target from a validated protocol message value.
	pub fn from_json(value: &JsonValue) -> Option<SessionTarget> {
		let object = value.as_object()?;
		Some(SessionTarget {
			server_id: object.get("serverId")?.as_str()?.to_string(),
			session_id: object.get("sessionId")?.as_str()?.to_string(),
			attachment_id: object.get("attachmentId")?.as_str()?.to_string(),
		})
	}
}

/// Upstream `RpcTarget`: a server target (`{ serverId }`) or a session target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RpcTarget {
	Server { server_id: String },
	Session(SessionTarget),
}

impl RpcTarget {
	pub fn to_json(&self) -> JsonValue {
		match self {
			RpcTarget::Server { server_id } => serde_json::json!({ "serverId": server_id }),
			RpcTarget::Session(session) => session.to_json(),
		}
	}

	/// Reads a routed target from a validated protocol message value.
	pub fn from_json(value: &JsonValue) -> Option<RpcTarget> {
		let object = value.as_object()?;
		let server_id = object.get("serverId")?.as_str()?.to_string();
		if object.len() == 1 {
			return Some(RpcTarget::Server { server_id });
		}
		if object.len() == 3 {
			return Some(RpcTarget::Session(SessionTarget {
				server_id,
				session_id: object.get("sessionId")?.as_str()?.to_string(),
				attachment_id: object.get("attachmentId")?.as_str()?.to_string(),
			}));
		}
		None
	}
}

/// Upstream `ConnectionState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConnectionState {
	Disconnected,
	Connecting,
	Connected,
}

impl ConnectionState {
	pub fn as_str(&self) -> &'static str {
		match self {
			ConnectionState::Disconnected => "disconnected",
			ConnectionState::Connecting => "connecting",
			ConnectionState::Connected => "connected",
		}
	}
}

/// Upstream `ConnectionStateChange`.
pub struct ConnectionStateChange {
	pub state: ConnectionState,
	/// Upstream optional `error`. Cloning goes through
	/// [`crate::client::clone_any_error`] (`AnyError` is not `Clone`).
	pub error: Option<AnyError>,
}

impl ConnectionStateChange {
	pub fn clone_error(&self) -> Option<AnyError> {
		self.error.as_ref().map(crate::client::clone_any_error)
	}
}

pub type ConnectionChangeListener = Arc<dyn Fn(&ConnectionStateChange) + Send + Sync>;

/// Handle returned by [`crate::Client::subscribe_service`] (upstream
/// `ServiceSubscription`). The snapshot is installed before [`start`](Self::start)
/// begins ordered update delivery; [`dispose`](Self::dispose) is idempotent.
pub struct ServiceSubscription {
	/// Upstream `ServiceSubscription.id`.
	pub id: String,
	/// Upstream `ServiceSubscription.target`.
	pub target: RpcTarget,
	/// Upstream `ServiceSubscription.snapshot`.
	pub snapshot: ServiceSubscriptionSnapshot,
	start_impl: Box<dyn Fn() + Send + Sync>,
	dispose_impl: Box<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>,
}

impl ServiceSubscription {
	pub(crate) fn new(
		id: String,
		target: RpcTarget,
		snapshot: ServiceSubscriptionSnapshot,
		start_impl: Box<dyn Fn() + Send + Sync>,
		dispose_impl: Box<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>,
	) -> ServiceSubscription {
		ServiceSubscription { id, target, snapshot, start_impl, dispose_impl }
	}

	/// Begin ordered update delivery after the caller has installed the snapshot.
	pub fn start(&self) {
		(self.start_impl)();
	}

	/// Stops delivery and unsubscribes on the wire when still possible.
	pub async fn dispose(&self) {
		(self.dispose_impl)().await;
	}
}

/// Upstream `ClientOptions`.
pub struct ClientOptions {
	pub transport_factory: crate::transport::ByteTransportFactory,
	/// Logical server identity expected at the physical endpoint.
	pub server_id: String,
	pub max_frame_length: Option<usize>,
	/// Reports subscriber failures without allowing them to corrupt client state.
	pub on_listener_error: Option<ListenerErrorHandler>,
}
