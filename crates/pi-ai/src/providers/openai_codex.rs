//! Port of `packages/ai/src/providers/openai-codex.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::helpers::{lazy_oauth, LazyOAuthInput};
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `openaiCodexProvider()`.
pub fn openai_codex_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: "openai-codex".to_string(),
        name: Some("OpenAI Codex (legacy)".to_string()),
        base_url: Some("https://chatgpt.com/backend-api".to_string()),
        headers: None,
        auth: ProviderAuth {
            api_key: None,
            oauth: Some(Arc::new(lazy_oauth(LazyOAuthInput {
                name: "OpenAI (ChatGPT Plus/Pro)".to_string(),
                is_subscription: true,
                login_label: None,
                load: crate::auth::oauth::load::load_openai_codex_oauth,
            }))),
        },
        models: super::openai_codex_models::openai_codex_any_models(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::Single(
            crate::api::openai_codex_responses::openai_codex_responses_api(),
        )),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("static openai-codex provider configuration is valid")
}
