//! Port of `packages/ai/src/providers/kimi-coding.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::helpers::{env_api_key_auth, lazy_oauth, LazyOAuthInput};
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `kimiCodingProvider()`.
pub fn kimi_coding_provider() -> Arc<dyn Provider> {
	create_provider(CreateProviderOptions {
		id: "kimi-coding".to_string(),
		name: Some("Kimi For Coding".to_string()),
		base_url: Some("https://api.kimi.com/coding".to_string()),
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("Kimi API key", &["KIMI_API_KEY"])),
			oauth: Some(Arc::new(lazy_oauth(LazyOAuthInput {
				name: "Kimi Code (subscription)".to_string(),
				is_subscription: true,
				login_label: Some("Sign in with Kimi Code".to_string()),
				load: crate::auth::oauth::load::load_kimi_coding_oauth,
			}))),
		},
		models: super::kimi_coding_models::kimi_coding_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(crate::api::anthropic_messages::anthropic_messages_api())),
		images: Default::default(),
		classifiers: Default::default(),
	})
	.expect("static kimi-coding provider configuration is valid")
}
