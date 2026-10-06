//! Port of `packages/ai/src/providers/openai.ts`.
//!
//! Deviations:
//! - D15: upstream `oauth: lazyOAuth(...)` is omitted — `auth/oauth/load.ts`
//!   (the OpenAI ChatGPT OAuth flow) is out of scope for this wave; the
//!   provider advertises api-key auth only.
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::types::ProviderAuth;
use crate::auth::helpers::env_api_key_auth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `openaiProvider()`. The OAuth auth method is omitted (D15).
pub fn openai_provider() -> Arc<dyn Provider> {
	create_provider(CreateProviderOptions {
		id: "openai".to_string(),
		name: Some("OpenAI".to_string()),
		base_url: Some("https://api.openai.com/v1".to_string()),
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("OpenAI API key", &["OPENAI_API_KEY"])),
			oauth: Some(Arc::new(crate::auth::helpers::lazy_oauth(crate::auth::helpers::LazyOAuthInput {
				name: "OpenAI (ChatGPT subscription)".to_string(),
				is_subscription: true,
				login_label: Some("Sign in with ChatGPT".to_string()),
				load: crate::auth::oauth::load::load_openai_chatgpt_oauth,
			}))),
		},
		models: super::openai_models::openai_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(crate::api::openai_responses::openai_responses_api())),
		images: Default::default(),
		classifiers: Default::default(),
	})
	.expect("static openai provider configuration is valid")
}
