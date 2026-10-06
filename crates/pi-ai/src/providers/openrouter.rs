//! Port of `packages/ai/src/providers/openrouter.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::auth::helpers::{env_api_key_auth, lazy_oauth, LazyOAuthInput};
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider, ProviderClassifier, ProviderImages, ProviderStreams};

/// Upstream `openrouterProvider()`.
pub fn openrouter_provider() -> Arc<dyn Provider> {
	let mut api: BTreeMap<String, Arc<dyn ProviderStreams>> = BTreeMap::new();
	api.insert(
		"anthropic-messages".to_string(),
		crate::api::anthropic_messages::anthropic_messages_api(),
	);
	api.insert(
		"openai-completions".to_string(),
		crate::api::openai_completions::openai_completions_api(),
	);
	let mut images: BTreeMap<String, Arc<dyn ProviderImages>> = BTreeMap::new();
	images.insert(
		"openrouter-images".to_string(),
		crate::api::openrouter_images::openrouter_images_api(),
	);
	let mut classifiers: BTreeMap<String, Arc<dyn ProviderClassifier>> = BTreeMap::new();
	// OpenRouter serves TypeSafe's System One protocol at /api/v1/systemone.
	classifiers.insert(
		"typesafe-system-one".to_string(),
		crate::api::typesafe_system_one::typesafe_system_one_api(),
	);
	create_provider(CreateProviderOptions {
		id: "openrouter".to_string(),
		name: Some("OpenRouter".to_string()),
		base_url: Some("https://openrouter.ai/api/v1".to_string()),
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("OpenRouter API key", &["OPENROUTER_API_KEY"])),
			// Upstream has no `isSubscription` (defaults to false).
			oauth: Some(Arc::new(lazy_oauth(LazyOAuthInput {
				name: "OpenRouter OAuth".to_string(),
				is_subscription: false,
				login_label: Some("Sign in with OpenRouter".to_string()),
				load: crate::auth::oauth::load::load_openrouter_oauth,
			}))),
		},
		models: super::openrouter_models::openrouter_all_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::ByApi(api)),
		images,
		classifiers,
	})
	.expect("static openrouter provider configuration is valid")
}

#[cfg(test)]
mod tests {
	// Data-level port of upstream test/openrouter-cache-control-models.test.ts:
	// the shard marks cache-control-capable models, asserted on the embedded
	// data (streaming propagation is covered by the anthropic-messages /
	// openai-completions adapter tests).
	#[test]
	fn openrouter_catalog_carries_cache_control_compat() {
		let models = super::super::openrouter_models::openrouter_models();
		assert!(!models.is_empty());
		let apis: std::collections::BTreeSet<&str> = models.iter().map(|m| m.api()).collect();
		assert!(
			apis.iter().all(|api| *api == "anthropic-messages" || *api == "openai-completions"),
			"unexpected openrouter api values: {apis:?}"
		);
		// Upstream test/openrouter-cache-control-models.test.ts: the
		// `~anthropic/*-latest` aliases keep completions cache control.
		for id in [
			"~anthropic/claude-fable-latest",
			"~anthropic/claude-haiku-latest",
			"~anthropic/claude-opus-latest",
			"~anthropic/claude-sonnet-latest",
		] {
			let model = models
				.iter()
				.find(|model| model.id() == id)
				.unwrap_or_else(|| panic!("missing openrouter alias {id}"));
			assert_eq!(model.api(), "openai-completions");
			let compat = model.compat.as_ref().expect("alias models carry compat");
			assert!(
				matches!(compat.cache_control_format, Some(crate::types::CacheControlFormat::Anthropic)),
				"alias {id} loses cache control"
			);
		}
	}
}
