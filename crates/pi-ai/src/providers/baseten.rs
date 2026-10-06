//! Port of `packages/ai/src/providers/baseten.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `basetenProvider()`.
pub fn baseten_provider() -> Arc<dyn Provider> {
	create_provider(CreateProviderOptions {
		id: "baseten".to_string(),
		name: Some("Baseten".to_string()),
		base_url: Some("https://inference.baseten.co/v1".to_string()),
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("Baseten API key", &["BASETEN_API_KEY"])),
			oauth: None,
		},
		models: super::baseten_models::baseten_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(crate::api::openai_completions::openai_completions_api())),
		images: Default::default(),
		classifiers: Default::default(),
	})
	.expect("static baseten provider configuration is valid")
}
