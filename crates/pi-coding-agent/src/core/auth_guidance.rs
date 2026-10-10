//! Port of `packages/coding-agent/src/core/auth-guidance.ts` — login-help
//! strings shown when no models/API keys are available.
//!
//! There is no upstream `auth-guidance.test.ts`: every consumer
//! (`core/sdk.ts`, `core/agent-session.ts`, `main.ts`, `cli/list-models.ts`)
//! is a deferred module, so the suite travels with those ports (deviation
//! D69). The functions are exported now because they are dependency-free.

use crate::config::get_docs_path;
use crate::utils::paths::join;

const UNKNOWN_PROVIDER: &str = "unknown";

/// `getProviderLoginHelp`.
pub fn get_provider_login_help() -> String {
    [
        "Use /login to log into a provider via OAuth or API key. See:".to_string(),
        join(&get_docs_path(), "providers.md"),
        join(&get_docs_path(), "models.md"),
    ]
    .join("\n")
}

/// `formatNoModelsAvailableMessage`.
pub fn format_no_models_available_message() -> String {
    format!("No models available. {}", get_provider_login_help())
}

/// `formatNoModelSelectedMessage`.
pub fn format_no_model_selected_message() -> String {
    format!(
        "No model selected.\n\n{}\n\nThen use /model to select a model.",
        get_provider_login_help()
    )
}

/// `formatNoApiKeyFoundMessage`.
pub fn format_no_api_key_found_message(provider: &str) -> String {
    let provider_display = if provider == UNKNOWN_PROVIDER {
        "the selected model"
    } else {
        provider
    };
    format!(
        "No API key found for {provider_display}.\n\n{}",
        get_provider_login_help()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // No upstream suite (see module docs); guard the observable strings.
    #[test]
    fn messages_mention_login_help_and_substitute_unknown_provider() {
        let help = get_provider_login_help();
        assert!(help.starts_with("Use /login to log into a provider via OAuth or API key. See:\n"));
        assert!(help.contains("providers.md"));
        assert!(help.contains("models.md"));

        assert_eq!(
            format_no_models_available_message(),
            format!("No models available. {help}")
        );
        assert_eq!(
            format_no_model_selected_message(),
            format!("No model selected.\n\n{help}\n\nThen use /model to select a model.")
        );
        assert_eq!(
            format_no_api_key_found_message("unknown"),
            format!("No API key found for the selected model.\n\n{help}")
        );
        assert_eq!(
            format_no_api_key_found_message("anthropic"),
            format!("No API key found for anthropic.\n\n{help}")
        );
    }
}
