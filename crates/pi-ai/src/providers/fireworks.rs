//! Port of `packages/ai/src/providers/fireworks.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};

/// Upstream `fireworksProvider()`.
pub fn fireworks_provider() -> Arc<dyn Provider> {
	let mut api = BTreeMap::new();
	api.insert(
		"anthropic-messages".to_string(),
		crate::api::anthropic_messages::anthropic_messages_api(),
	);
	api.insert(
		"openai-completions".to_string(),
		crate::api::openai_completions::openai_completions_api(),
	);
	create_provider(CreateProviderOptions {
		id: "fireworks".to_string(),
		name: Some("Fireworks".to_string()),
		base_url: Some("https://api.fireworks.ai/inference".to_string()),
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("Fireworks API key", &["FIREWORKS_API_KEY"])),
			oauth: None,
		},
		models: super::fireworks_models::fireworks_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::ByApi(api)),
		images: Default::default(),
		classifiers: Default::default(),
	})
	.expect("static fireworks provider configuration is valid")
}

#[cfg(test)]
mod tests {
	// Data-level port of upstream test/fireworks-models.test.ts (the streaming
	// payload cases depend on the shared compat/env-api-keys surface and are
	// covered by the openai-completions adapter's own tests).
	#[test]
	fn fireworks_catalog_keeps_provider_and_apis() {
		let models = super::super::fireworks_models::fireworks_models();
		assert!(!models.is_empty());
		let apis: std::collections::BTreeSet<&str> = models.iter().map(|m| m.api()).collect();
		assert!(
			apis.iter().all(|api| *api == "anthropic-messages" || *api == "openai-completions"),
			"unexpected fireworks api values: {apis:?}"
		);
	}
}
