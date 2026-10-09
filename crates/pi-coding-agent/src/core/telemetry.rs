//! Port of `packages/coding-agent/src/core/telemetry.ts` — install-telemetry
//! opt-in resolution.
//!
//! Upstream reads the env var at call time (`process.env.PI_TELEMETRY`);
//! Rust passes it explicitly (no global environment in library code — D71a:
//! callers thread `telemetry_env` or pass `None` to read
//! [`std::env::var`] at the call site equivalent).
//!
//! No upstream direct suite; covered by [`provider_attribution`] tests and
//! the sdk consumers.

use crate::core::settings_manager::{Settings, SettingsManager, SettingsManagerCreateOptions};

fn is_truthy_env_flag(value: Option<&str>) -> bool {
	match value {
		None => false,
		Some(value) => matches!(
			value.to_ascii_lowercase().as_str(),
			"1" | "true" | "yes"
		),
	}
}

/// Whether install telemetry attribution headers are enabled
/// (`isInstallTelemetryEnabled`): `PI_TELEMETRY` overrides the settings
/// value when set.
pub fn is_install_telemetry_enabled(settings_manager: &SettingsManager) -> bool {
	is_install_telemetry_enabled_with_env(
		settings_manager,
		std::env::var("PI_TELEMETRY").ok().as_deref(),
	)
}

/// Env-injecting variant of [`is_install_telemetry_enabled`].
pub fn is_install_telemetry_enabled_with_env(
	settings_manager: &SettingsManager,
	telemetry_env: Option<&str>,
) -> bool {
	match telemetry_env {
		Some(_) => is_truthy_env_flag(telemetry_env),
		None => settings_manager.get_enable_install_telemetry(),
	}
}

#[cfg(test)]
mod telemetry_tests {
	use super::*;

	#[test]
	fn env_flag_overrides_settings() {
		let mut settings =
			SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default());
		settings.set_enable_install_telemetry(true);
		assert!(!is_install_telemetry_enabled_with_env(&settings, Some("0")));
		assert!(is_install_telemetry_enabled_with_env(&settings, Some("1")));
		assert!(is_install_telemetry_enabled_with_env(&settings, Some("TRUE")));
		assert!(is_install_telemetry_enabled_with_env(&settings, Some("yes")));
		assert!(!is_install_telemetry_enabled_with_env(&settings, Some("no")));
	}

	#[test]
	fn falls_back_to_settings_without_env() {
		let mut settings =
			SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default());
		// Settings default is enabled (`unwrap_or(true)`); pin it explicitly.
		settings.set_enable_install_telemetry(false);
		assert!(!is_install_telemetry_enabled_with_env(&settings, None));
		settings.set_enable_install_telemetry(true);
		assert!(is_install_telemetry_enabled_with_env(&settings, None));
	}
}
