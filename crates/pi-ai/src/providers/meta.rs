//! Port of `packages/ai/src/providers/meta.ts`.
//!
//! Deviations:
//! - D15: upstream `oauth: lazyOAuth(...)` is omitted — `auth/oauth/load.ts`
//!   (the Meta Muse OAuth flow) is out of scope for this wave; the provider
//!   advertises api-key auth only.
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::types::ProviderAuth;
use crate::auth::helpers::env_api_key_auth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `metaProvider()`. The OAuth auth method is omitted (D15).
pub fn meta_provider() -> Arc<dyn Provider> {
	create_provider(CreateProviderOptions {
		id: "meta".to_string(),
		name: Some("Meta".to_string()),
		base_url: Some("https://api.meta.ai/v1".to_string()),
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("Meta Model API key", &["META_API_KEY"])),
			oauth: Some(Arc::new(crate::auth::helpers::lazy_oauth(crate::auth::helpers::LazyOAuthInput {
				name: "Meta (Muse subscription)".to_string(),
				is_subscription: true,
				login_label: Some("Sign in with Meta".to_string()),
				load: crate::auth::oauth::load::load_meta_oauth,
			}))),
		},
		models: super::meta_models::meta_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(crate::api::openai_responses::openai_responses_api())),
		images: Default::default(),
		classifiers: Default::default(),
	})
	.expect("static meta provider configuration is valid")
}
