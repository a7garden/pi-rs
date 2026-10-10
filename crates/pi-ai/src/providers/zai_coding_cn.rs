//! Port of `packages/ai/src/providers/zai-coding-cn.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `zai_coding_cnProvider()`.
pub fn zai_coding_cn_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: "zai-coding-cn".to_string(),
        name: Some("Z.AI Coding CN".to_string()),
        base_url: Some("https://open.bigmodel.cn/api/coding/paas/v4".to_string()),
        headers: None,
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "Z.AI Coding CN API key",
                &["ZAI_CODING_CN_API_KEY"],
            )),
            oauth: None,
        },
        models: super::zai_coding_cn_models::zai_coding_cn_any_models(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::Single(
            crate::api::openai_completions::openai_completions_api(),
        )),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("static zai-coding-cn provider configuration is valid")
}
