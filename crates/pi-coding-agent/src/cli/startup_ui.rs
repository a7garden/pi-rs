//! `cli/startup-ui.ts` — logic-only port.
//!
//! The TUI surfaces (startup selector/input, first-time-setup dialog,
//! session picker host, startup theme loading) are excluded with
//! interactive mode. What remains portable is the first-time-setup gate.

use crate::config::{get_settings_path, APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR, PACKAGE_NAME};

const OFFICIAL_PACKAGE_NAME: &str = "@earendil-works/pi-coding-agent";
const OFFICIAL_APP_NAME: &str = "pi";
const OFFICIAL_CONFIG_DIR_NAME: &str = ".pi";

struct DistributionMetadata<'a> {
    package_name: &'a str,
    app_name: &'a str,
    config_dir_name: &'a str,
}

fn is_official_distribution(metadata: &DistributionMetadata<'_>) -> bool {
    metadata.package_name == OFFICIAL_PACKAGE_NAME
        && metadata.app_name == OFFICIAL_APP_NAME
        && metadata.config_dir_name == OFFICIAL_CONFIG_DIR_NAME
}

/// Whether experimental features are enabled (`PI_EXPERIMENTAL=1`).
///
/// The authoritative module (`core/experimental.ts`) has no dedicated port;
/// this mirrors its env read (`areExperimentalFeaturesEnabled`).
fn are_experimental_features_enabled() -> bool {
    matches!(
        std::env::var("PI_EXPERIMENTAL").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

/// First-time setup runs when all of these hold (`shouldRunFirstTimeSetup`):
/// - this is the official Pi distribution (not a fork/rebrand)
/// - experimental features are enabled (PI_EXPERIMENTAL=1)
/// - the default agent directory is used (no custom agent dir override)
/// - setup was not completed before (settings.json does not exist)
pub fn should_run_first_time_setup() -> bool {
    should_run_first_time_setup_at(&get_settings_path())
}

pub fn should_run_first_time_setup_at(settings_path: &str) -> bool {
    if !is_official_distribution(&DistributionMetadata {
        package_name: PACKAGE_NAME,
        app_name: APP_NAME,
        config_dir_name: CONFIG_DIR_NAME,
    }) {
        return false;
    }
    if !are_experimental_features_enabled() {
        return false;
    }
    if std::env::var(ENV_AGENT_DIR).is_ok() {
        return false;
    }
    !std::path::Path::new(settings_path).exists()
}
