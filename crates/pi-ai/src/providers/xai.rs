//! Port of `packages/ai/src/providers/xai.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::helpers::{env_api_key_auth, lazy_oauth, LazyOAuthInput};
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `xaiProvider()`.
pub fn xai_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: "xai".to_string(),
        name: Some("xAI".to_string()),
        base_url: Some("https://api.x.ai/v1".to_string()),
        headers: None,
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("xAI API key", &["XAI_API_KEY"])),
            oauth: Some(Arc::new(lazy_oauth(LazyOAuthInput {
                name: "xAI (Grok/X subscription)".to_string(),
                is_subscription: true,
                login_label: Some("Sign in with SuperGrok or X Premium".to_string()),
                load: crate::auth::oauth::load::load_xai_oauth,
            }))),
        },
        models: super::xai_models::xai_any_models(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::Single(
            crate::api::openai_responses::openai_responses_api(),
        )),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("static xai provider configuration is valid")
}
