//! Port of `packages/ai/src/providers/ant-ling.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `ant_lingProvider()`.
pub fn ant_ling_provider() -> Arc<dyn Provider> {
	create_provider(CreateProviderOptions {
		id: "ant-ling".to_string(),
		name: Some("Ant Ling".to_string()),
		base_url: Some("https://api.ant-ling.com/v1".to_string()),
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("Ant Ling API key", &["ANT_LING_API_KEY"])),
			oauth: None,
		},
		models: super::ant_ling_models::ant_ling_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(crate::api::openai_completions::openai_completions_api())),
		images: Default::default(),
		classifiers: Default::default(),
	})
	.expect("static ant-ling provider configuration is valid")
}
