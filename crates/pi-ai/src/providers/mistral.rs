//! Port of `packages/ai/src/providers/mistral.ts`.
//!
//! Deviations:
//! - D15: the `MISTRAL_API_KEY` name constant (upstream `env-api-keys.ts`,
//!   out of scope) is defined locally as a `pub const` with the upstream name
//!   and value.
//! - D16: upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `MISTRAL_API_KEY_ENV`.
pub const MISTRAL_API_KEY_ENV: &str = "MISTRAL_API_KEY";

/// Upstream `mistralProvider()`.
pub fn mistral_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: "mistral".to_string(),
        name: Some("Mistral".to_string()),
        base_url: Some("https://api.mistral.ai".to_string()),
        headers: None,
        auth: ProviderAuth {
            api_key: Some(crate::auth::helpers::env_api_key_auth(
                "Mistral API key",
                &["MISTRAL_API_KEY"],
            )),
            oauth: None,
        },
        models: super::mistral_models::mistral_any_models(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::Single(
            crate::api::mistral_conversations::mistral_conversations_api(),
        )),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("static mistral provider configuration is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_exposes_catalog_models_and_api_key_auth() {
        let provider = mistral_provider();
        assert_eq!(provider.id(), "mistral");
        assert_eq!(provider.name(), "Mistral");
        assert_eq!(provider.base_url(), Some("https://api.mistral.ai"));
        let models = provider.get_models();
        assert_eq!(
            models.len(),
            super::super::mistral_models::mistral_models().len()
        );
        assert!(models
            .iter()
            .any(|model| model.id() == "mistral-large-latest"));
    }
}
