//! Port of `packages/ai/src/providers/google.ts`.
//!
//! Deviations:
//! - D15: upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid.
//! - D16: upstream imports the API via the `.lazy.ts` twin; the Rust port
//!   links statically (crate deviation D10), so the factory calls
//!   [`crate::api::google_generative_ai::google_generative_ai_api`]
//!   directly.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `googleProvider()`.
pub fn google_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: "google".to_string(),
        name: Some("Google".to_string()),
        base_url: Some("https://generativelanguage.googleapis.com/v1beta".to_string()),
        headers: None,
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("Gemini API key", &["GEMINI_API_KEY"])),
            oauth: None,
        },
        models: super::google_models::google_any_models(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::Single(
            crate::api::google_generative_ai::google_generative_ai_api(),
        )),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("static google provider configuration is valid")
}

// ---------------------------------------------------------------------------
// Tests (ports of the providers.test.ts google cases)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::get_supported_thinking_levels;
    use crate::types::Model;

    fn find_model(id: &str) -> Model {
        super::super::google_models::google_models()
            .into_iter()
            .find(|model| model.id() == id)
            .unwrap_or_else(|| panic!("catalog model google/{id}"))
    }

    #[test]
    fn records_known_direct_provider_image_request_limits_for_google() {
        let model = find_model("gemini-2.5-flash");
        let limits = model.base.input_limits.as_ref().expect("inputLimits");
        assert_eq!(limits.max_request_bytes, Some(20 * 1024 * 1024));
        let images = limits.images.as_ref().expect("images limits");
        assert_eq!(images.max_per_request, Some(3600));
    }

    #[test]
    fn uses_models_dev_effort_levels_for_google_thinking_models() {
        // Regression test for https://github.com/earendil-works/pi/issues/9455
        assert!(
            get_supported_thinking_levels(&find_model("gemini-3.6-flash"))
                .contains(&crate::types::ModelThinkingLevel::Minimal)
        );
        assert_eq!(
            get_supported_thinking_levels(&find_model("gemini-3.8-flash")),
            vec![
                crate::types::ModelThinkingLevel::Low,
                crate::types::ModelThinkingLevel::Medium,
                crate::types::ModelThinkingLevel::High,
            ]
        );
        assert_eq!(
            get_supported_thinking_levels(&find_model("gemini-3.1-pro-preview")),
            vec![
                crate::types::ModelThinkingLevel::Low,
                crate::types::ModelThinkingLevel::Medium,
                crate::types::ModelThinkingLevel::High,
            ]
        );
        assert_eq!(
            get_supported_thinking_levels(&find_model("gemma-4-31b-it")),
            vec![
                crate::types::ModelThinkingLevel::Minimal,
                crate::types::ModelThinkingLevel::High,
            ]
        );
    }

    #[tokio::test]
    async fn google_provider_registers_models_and_api() {
        let provider = google_provider();
        assert_eq!(provider.id(), "google");
        assert_eq!(provider.name(), "Google");
        let models = provider.get_models();
        assert!(!models.is_empty());
        assert!(models.iter().all(|model| model.provider() == "google"));
        assert!(models
            .iter()
            .all(|model| model.api() == "google-generative-ai"));
    }
}
