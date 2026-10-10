//! Port of `experimental/radius-auth.ts` — resolves explicit or stored Radius
//! credentials anew for every relay connection attempt.
//!
//! Deviations (D151):
//! - `AuthInput` (upstream `cli/experimental/command-options.ts`, not yet
//!   ported) is defined here as [`AuthInput`]; it migrates to the CLI
//!   experimental slice when that lands.
//! - `signal?: AbortSignal` becomes [`pi_chord::context::AbortSignal`];
//!   `throwIfAborted` is `aborted()` + an error return.
//! - The memoized `ModelRuntime` creation uses a tokio `OnceCell` standing
//!   in for the lazily-assigned promise field.

use std::sync::Arc;

use pi_chord::context::{AbortController, AbortSignal};
use pi_chord::services::errors::{error, AnyError};

use crate::cli::auth_command::get_auth_credential;
use crate::core::model_runtime::{
    CreateModelRuntimeOptions, ModelRuntime, ModelRuntimeAuthOverrides,
};
pub use crate::core::radius::ENV_RADIUS_GATEWAY;
use crate::core::radius::{get_radius_gateway_url, RADIUS_PROVIDER_ID};
use crate::utils::paths::resolve_path_cwd;

/// Upstream `AuthInput` (token or token file) from
/// `cli/experimental/command-options.ts`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthInput {
    /// `--auth-token`.
    Token { token: String },
    /// `--auth-token-file`.
    File { path: String },
}

/// Upstream `RadiusRelayAuth`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadiusRelayAuth {
    pub gateway: String,
    pub token: String,
}

/// Resolve options (upstream `resolve(options)`).
pub struct RadiusRelayResolveOptions {
    pub required: bool,
    pub signal: Option<AbortSignal>,
}

/// Upstream `RadiusRelayAuthResolver`.
pub struct RadiusRelayAuthResolver {
    input: Option<AuthInput>,
    gateway: String,
    model_runtime: tokio::sync::OnceCell<Arc<ModelRuntime>>,
}

impl RadiusRelayAuthResolver {
    /// `new RadiusRelayAuthResolver(input?, gateway?)`.
    pub fn new(input: Option<AuthInput>) -> Self {
        Self::with_gateway(input, get_radius_gateway_url())
    }

    pub fn with_gateway(input: Option<AuthInput>, gateway: String) -> Self {
        Self {
            input,
            gateway: pi_ai::providers::radius_config::normalize_radius_gateway_url(&gateway),
            model_runtime: tokio::sync::OnceCell::new(),
        }
    }

    /// Upstream `gateway` getter.
    pub fn gateway(&self) -> &str {
        &self.gateway
    }

    /// Resolve explicit or stored Radius credentials anew for every relay
    /// connection attempt.
    pub async fn resolve(
        &self,
        options: &RadiusRelayResolveOptions,
    ) -> Result<Option<RadiusRelayAuth>, AnyError> {
        check_signal(options.signal.as_ref())?;
        if std::env::var("PI_OFFLINE").is_ok() {
            if options.required {
                return Err(error(
                    "Radius relay connections are unavailable in offline mode",
                ));
            }
            return Ok(None);
        }

        if let Some(explicit) = self.explicit_token(options.signal.as_ref()).await? {
            return Ok(Some(RadiusRelayAuth {
                gateway: self.gateway.clone(),
                token: explicit,
            }));
        }

        let runtime = self
            .model_runtime
            .get_or_try_init(|| async {
                ModelRuntime::create(CreateModelRuntimeOptions {
                    refresh_on_create: false,
                    allow_model_network: false,
                    ..Default::default()
                })
                .await
                .map_err(|thrown| error(thrown_message(&thrown)))
            })
            .await?;
        check_signal(options.signal.as_ref())?;
        let auth = runtime
            .get_auth_for_provider(
                RADIUS_PROVIDER_ID,
                Some(ModelRuntimeAuthOverrides {
                    min_oauth_validity_ms: Some(5 * 60_000),
                    signal: bridge_signal_to_ai(options.signal.as_ref()),
                    ..Default::default()
                }),
            )
            .await
            .map_err(|thrown| error(thrown_message(&thrown)))?;
        let token = get_auth_credential(auth.as_ref());
        if let Some(token) = token.filter(|token| !token.is_empty()) {
            return Ok(Some(RadiusRelayAuth {
                gateway: self.gateway.clone(),
                token,
            }));
        }
        if options.required {
            return Err(error(
                "Radius authentication is required; start Pi and run /login radius, then retry",
            ));
        }
        Ok(None)
    }

    async fn explicit_token(
        &self,
        signal: Option<&AbortSignal>,
    ) -> Result<Option<String>, AnyError> {
        let Some(input) = &self.input else {
            return Ok(None);
        };
        check_signal(signal)?;
        let value = match input {
            AuthInput::Token { token } => token.clone(),
            AuthInput::File { path } => tokio::fs::read_to_string(resolve_path_cwd(path))
                .await
                .map_err(|io_error| {
                    error(format!(
                        "Could not read the Radius auth token file: {io_error}"
                    ))
                })?,
        };
        let token = value.trim().to_string();
        if token.is_empty() {
            return Err(error("Radius authentication token must not be empty"));
        }
        Ok(Some(token))
    }
}

/// `throwIfAborted` — an abort surfaces as an error carrying the reason.
fn check_signal(signal: Option<&AbortSignal>) -> Result<(), AnyError> {
    let Some(signal) = signal else {
        return Ok(());
    };
    if signal.aborted() {
        return Err(match signal.reason() {
            Some(reason) => error(reason.to_string()),
            None => error("This operation was aborted"),
        });
    }
    Ok(())
}

fn thrown_message(thrown: &pi_ai::utils::thrown::ThrownValue) -> String {
    match thrown {
        pi_ai::utils::thrown::ThrownValue::Text(message) => message.clone(),
        pi_ai::utils::thrown::ThrownValue::Error(error) => error.to_string(),
    }
}

/// Bridge a chord abort signal into the pi-ai signal type the model runtime
/// consumes (same forward-on-abort shape as `bridge_crate_signal_to_ai`).
fn bridge_signal_to_ai(signal: Option<&AbortSignal>) -> Option<pi_ai::utils::abort::AbortSignal> {
    let signal = signal?;
    let bridged = pi_ai::utils::abort::AbortSignal::new();
    let target = bridged.clone();
    let forward = signal.clone();
    tokio::spawn(async move {
        forward.aborted_future().await;
        pi_ai::utils::abort::AbortSignal::abort(&target);
    });
    Some(bridged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offline() -> bool {
        std::env::var("PI_OFFLINE").is_ok()
    }

    #[tokio::test]
    async fn explicit_token_short_circuits() {
        if offline() {
            return;
        }
        let resolver = RadiusRelayAuthResolver::new(Some(AuthInput::Token {
            token: "  secret  ".into(),
        }));
        let auth = resolver
            .resolve(&RadiusRelayResolveOptions {
                required: false,
                signal: None,
            })
            .await
            .unwrap();
        assert_eq!(auth.map(|auth| auth.token), Some("secret".to_string()));
    }

    #[tokio::test]
    async fn empty_explicit_token_rejected() {
        if offline() {
            return;
        }
        let resolver = RadiusRelayAuthResolver::new(Some(AuthInput::Token {
            token: "   ".into(),
        }));
        let error = resolver
            .resolve(&RadiusRelayResolveOptions {
                required: false,
                signal: None,
            })
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Radius authentication token must not be empty"
        );
    }

    #[tokio::test]
    async fn offline_mode_gates_resolution() {
        let _guard = crate::experimental::TEST_GUARD.lock();
        let original = std::env::var("PI_OFFLINE").ok();
        std::env::set_var("PI_OFFLINE", "1");
        let resolver = RadiusRelayAuthResolver::new(None);
        let optional = resolver
            .resolve(&RadiusRelayResolveOptions {
                required: false,
                signal: None,
            })
            .await;
        assert_eq!(optional.unwrap(), None);
        let required = resolver
            .resolve(&RadiusRelayResolveOptions {
                required: true,
                signal: None,
            })
            .await;
        assert_eq!(
            required.unwrap_err().to_string(),
            "Radius relay connections are unavailable in offline mode"
        );
        match original {
            Some(value) => std::env::set_var("PI_OFFLINE", value),
            None => std::env::remove_var("PI_OFFLINE"),
        }
    }

    #[tokio::test]
    async fn aborted_signal_rejects() {
        let controller = AbortController::new();
        controller.abort("stopped");
        let resolver = RadiusRelayAuthResolver::new(None);
        let error = resolver
            .resolve(&RadiusRelayResolveOptions {
                required: false,
                signal: Some(controller.signal()),
            })
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "stopped");
    }

    #[test]
    fn gateway_normalizes() {
        let resolver = RadiusRelayAuthResolver::with_gateway(None, "gateway.example".into());
        assert_eq!(resolver.gateway(), "https://gateway.example");
    }
}
