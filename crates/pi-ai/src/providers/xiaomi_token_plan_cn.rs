//! Port of `packages/ai/src/providers/xiaomi-token-plan-cn.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `xiaomi_token_plan_cnProvider()`.
pub fn xiaomi_token_plan_cn_provider() -> Arc<dyn Provider> {
	create_provider(CreateProviderOptions {
		id: "xiaomi-token-plan-cn".to_string(),
		name: Some("Xiaomi Token Plan CN".to_string()),
		base_url: Some("https://token-plan-cn.xiaomimimo.com/v1".to_string()),
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("Xiaomi Token Plan CN API key", &["XIAOMI_TOKEN_PLAN_CN_API_KEY"])),
			oauth: None,
		},
		models: super::xiaomi_token_plan_cn_models::xiaomi_token_plan_cn_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(crate::api::openai_completions::openai_completions_api())),
		images: Default::default(),
		classifiers: Default::default(),
	})
	.expect("static xiaomi-token-plan-cn provider configuration is valid")
}
