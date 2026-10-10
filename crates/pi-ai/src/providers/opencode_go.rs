//! Port of `packages/ai/src/providers/opencode-go.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{
    create_provider, ApiDispatch, CreateProviderOptions, Provider, ProviderStreams,
};
use crate::providers::opencode_headers::with_opencode_session_header;

/// Upstream `opencodeGoProvider()`.
pub fn opencode_go_provider() -> Arc<dyn Provider> {
    let mut api: BTreeMap<String, Arc<dyn ProviderStreams>> = BTreeMap::new();
    api.insert(
        "anthropic-messages".to_string(),
        with_opencode_session_header(crate::api::anthropic_messages::anthropic_messages_api()),
    );
    api.insert(
        "openai-completions".to_string(),
        with_opencode_session_header(crate::api::openai_completions::openai_completions_api()),
    );
    api.insert(
        "openai-responses".to_string(),
        with_opencode_session_header(crate::api::openai_responses::openai_responses_api()),
    );
    create_provider(CreateProviderOptions {
        id: "opencode-go".to_string(),
        name: Some("OpenCode Go".to_string()),
        base_url: None,
        headers: None,
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("OpenCode API key", &["OPENCODE_API_KEY"])),
            oauth: None,
        },
        models: super::opencode_go_models::opencode_go_any_models(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::ByApi(api)),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("static opencode-go provider configuration is valid")
}
