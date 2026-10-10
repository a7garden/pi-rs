//! Port of `packages/ai/src/providers/qwen-token-plan-cn.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `qwen_token_plan_cnProvider()`.
pub fn qwen_token_plan_cn_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: "qwen-token-plan-cn".to_string(),
        name: Some("Qwen Token Plan CN".to_string()),
        base_url: Some(
            "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1".to_string(),
        ),
        headers: None,
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "Qwen Token Plan CN API key",
                &["QWEN_TOKEN_PLAN_CN_API_KEY"],
            )),
            oauth: None,
        },
        models: super::qwen_token_plan_cn_models::qwen_token_plan_cn_any_models(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::Single(
            crate::api::openai_completions::openai_completions_api(),
        )),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("static qwen-token-plan-cn provider configuration is valid")
}
