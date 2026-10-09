//! Port of `experimental/services/sessions.ts` — server-scoped session
//! directory/management DTOs and their remote service tokens.
//!
//! Deviations (D144):
//! - Upstream `ServerId` (pi-protocol brand) is a plain `String` here; the
//!   Rust transport carries it as a string.
//! - `ReplicatedState<SessionDirectoryState>` becomes chord's untyped
//!   [`pi_chord::MutableReplicatedState`] over the serialized
//!   [`SessionDirectoryState`] JSON (typed at the seam).

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use pi_chord::api::define_service;

/// Upstream `SessionAddress`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionAddress {
	pub server_id: String,
	pub session_id: String,
}

/// Upstream `SessionSummary`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSummary {
	pub server_id: String,
	pub session_id: String,
	pub created_at: u64,
}

impl SessionSummary {
	pub fn address(&self) -> SessionAddress {
		SessionAddress {
			server_id: self.server_id.clone(),
			session_id: self.session_id.clone(),
		}
	}
}

/// Upstream `SessionCreateOptions`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCreateOptions {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub id: Option<String>,
}

/// Upstream `SessionDirectoryState`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionDirectoryState {
	pub revision: u64,
	pub sessions: Vec<SessionSummary>,
}

/// `defineService<SessionDirectory>("pi.session-directory")`; remote-only
/// (D144 marker token).
pub struct SessionDirectoryToken;

pub static SESSION_DIRECTORY: LazyLock<pi_chord::Service<SessionDirectoryToken>> =
	LazyLock::new(|| define_service("pi.session-directory", false).expect("service id is valid"));

/// `defineService<SessionManagement>("pi.session-management")`; remote-only
/// (D144 marker token).
pub struct SessionManagementToken;

pub static SESSION_MANAGEMENT: LazyLock<pi_chord::Service<SessionManagementToken>> =
	LazyLock::new(|| define_service("pi.session-management", false).expect("service id is valid"));

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn directory_state_serializes_camel_case() {
		let state = SessionDirectoryState {
			revision: 3,
			sessions: vec![SessionSummary {
				server_id: "s".into(),
				session_id: "sess".into(),
				created_at: 42,
			}],
		};
		let value = serde_json::to_value(&state).unwrap();
		assert_eq!(value["revision"], 3);
		assert_eq!(value["sessions"][0]["serverId"], "s");
		assert_eq!(value["sessions"][0]["sessionId"], "sess");
		assert_eq!(value["sessions"][0]["createdAt"], 42);
	}

	#[test]
	fn create_options_omit_absent_id() {
		let value = serde_json::to_value(SessionCreateOptions::default()).unwrap();
		assert_eq!(value, serde_json::json!({}));
	}
}
