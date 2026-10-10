//! Port of `experimental/services/plugins.ts` — server-built presentation
//! plugin service and the session worker's plugin reload service.
//!
//! Deviations (D144):
//! - `PresentationPlugins` is remote-only; its method calls travel as JSON
//!   members (`prepareSession`, `reload`) over the chord provider.
//! - `SessionPlugins` is process-local and typed by the concrete
//!   [`SessionPluginsService`] handle.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};

use serde::{Deserialize, Serialize};

use pi_chord::api::define_service;
use pi_chord::context::Context;
use pi_chord::json::JsonValue;

/// Upstream `PresentationPlugins.prepareSession` request object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrepareSessionRequest {
    pub session_id: String,
    /// `null` upstream when unselected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_paths: Option<Vec<String>>,
}

/// `defineService<PresentationPlugins>("pi.presentation-plugins")`; remote-only
/// (D144 marker token).
pub struct PresentationPluginsToken;

pub static PRESENTATION_PLUGINS: LazyLock<pi_chord::Service<PresentationPluginsToken>> =
    LazyLock::new(|| {
        define_service("pi.presentation-plugins", false).expect("service id is valid")
    });

/// `defineService<SessionPlugins>("pi.session-plugins")`; local service typed
/// by the concrete reload handle (D144).
#[derive(Clone)]
pub struct SessionPluginsService {
    reload: Arc<
        dyn Fn(Context) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>> + Send + Sync,
    >,
}

impl SessionPluginsService {
    pub fn new(
        reload: impl Fn(Context) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
            + Send
            + Sync
            + 'static,
    ) -> SessionPluginsService {
        SessionPluginsService {
            reload: Arc::new(reload),
        }
    }

    /// Cheap shared clone (the reload closure is behind an `Arc`).
    pub fn clone_handle(&self) -> SessionPluginsService {
        self.clone()
    }

    /// Upstream `SessionPlugins.reload(context)`.
    pub async fn reload(&self, context: Context) -> Result<(), String> {
        (self.reload)(context).await
    }
}

pub static SESSION_PLUGINS: LazyLock<pi_chord::Service<SessionPluginsService>> =
    LazyLock::new(|| define_service("pi.session-plugins", true).expect("service id is valid"));

/// JSON member names of the remote `PresentationPlugins` service (upstream
/// object keys).
pub mod members {
    pub const PREPARE_SESSION: &str = "prepareSession";
    pub const RELOAD: &str = "reload";
}

/// Serialize `prepareSession` arguments the way the TS caller passes them (one
/// positional object argument).
pub fn prepare_session_args(request: &PrepareSessionRequest) -> JsonValue {
    serde_json::to_value(request).expect("PrepareSessionRequest serializes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_session_request_serializes_camel_case() {
        let request = PrepareSessionRequest {
            session_id: "s".into(),
            package_paths: Some(vec!["/pkg".into()]),
        };
        let value = prepare_session_args(&request);
        assert_eq!(value["sessionId"], "s");
        assert_eq!(value["packagePaths"][0], "/pkg");
        let none = PrepareSessionRequest {
            session_id: "s".into(),
            package_paths: None,
        };
        assert_eq!(
            prepare_session_args(&none),
            serde_json::json!({"sessionId": "s"})
        );
    }

    #[tokio::test]
    async fn session_plugins_reload_delegates() {
        let service =
            SessionPluginsService::new(|_context| Box::pin(async { Err("not ready".to_string()) }));
        assert_eq!(
            service
                .reload(pi_chord::context::BACKGROUND_CONTEXT.clone())
                .await,
            Err("not ready".to_string())
        );
    }
}
