//! Port of `packages/ai/src/providers/xiaomi-token-plan-sgp.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `xiaomi_token_plan_sgpProvider()`.
pub fn xiaomi_token_plan_sgp_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: "xiaomi-token-plan-sgp".to_string(),
        name: Some("Xiaomi Token Plan SGP".to_string()),
        base_url: Some("https://token-plan-sgp.xiaomimimo.com/v1".to_string()),
        headers: None,
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "Xiaomi Token Plan SGP API key",
                &["XIAOMI_TOKEN_PLAN_SGP_API_KEY"],
            )),
            oauth: None,
        },
        models: super::xiaomi_token_plan_sgp_models::xiaomi_token_plan_sgp_any_models(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::Single(
            crate::api::openai_completions::openai_completions_api(),
        )),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("static xiaomi-token-plan-sgp provider configuration is valid")
}
