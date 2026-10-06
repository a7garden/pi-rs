//! Port of `packages/ai/src/providers/minimax-cn.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `minimax_cnProvider()`.
pub fn minimax_cn_provider() -> Arc<dyn Provider> {
	create_provider(CreateProviderOptions {
		id: "minimax-cn".to_string(),
		name: Some("MiniMax CN".to_string()),
		base_url: Some("https://api.minimaxi.com/anthropic".to_string()),
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("MiniMax CN API key", &["MINIMAX_CN_API_KEY"])),
			oauth: None,
		},
		models: super::minimax_cn_models::minimax_cn_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(crate::api::anthropic_messages::anthropic_messages_api())),
		images: Default::default(),
		classifiers: Default::default(),
	})
	.expect("static minimax-cn provider configuration is valid")
}
