//! Port of `packages/ai/src/providers/vercel-ai-gateway.ts`.
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

/// Upstream `vercelAIGatewayProvider()`.
pub fn vercel_ai_gateway_provider() -> Arc<dyn Provider> {
	let mut models = super::vercel_ai_gateway_models::vercel_ai_gateway_any_models();
	models.extend(super::vercel_ai_gateway_models::vercel_ai_gateway_classifier_any_models());
	let mut classifiers: BTreeMap<String, Arc<dyn crate::models::ProviderClassifier>> = BTreeMap::new();
	// AI Gateway serves TypeSafe's System One protocol at /typesafe/v1/systemone.
	classifiers.insert(
		"typesafe-system-one".to_string(),
		crate::api::typesafe_system_one::typesafe_system_one_api(),
	);
	create_provider(CreateProviderOptions {
		id: "vercel-ai-gateway".to_string(),
		name: Some("Vercel AI Gateway".to_string()),
		base_url: Some("https://ai-gateway.vercel.sh".to_string()),
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("Vercel AI Gateway API key", &["AI_GATEWAY_API_KEY"])),
			oauth: None,
		},
		models,
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(
			crate::api::anthropic_messages::anthropic_messages_api(),
		)),
		images: Default::default(),
		classifiers,
	})
	.expect("static vercel-ai-gateway provider configuration is valid")
}

#[cfg(test)]
mod tests {
	// Data-level check mirroring upstream test/classifier-models.test.ts cases
	// for the vercel-ai-gateway shard (data-level only).
	#[test]
	fn vercel_catalog_keeps_classifier_models() {
		let models = super::super::vercel_ai_gateway_models::vercel_ai_gateway_models();
		assert!(!models.is_empty());
		for model in &models {
			assert_eq!(model.provider(), "vercel-ai-gateway");
			assert_eq!(model.api(), "anthropic-messages");
		}
		let classifiers = super::super::vercel_ai_gateway_models::vercel_ai_gateway_classifier_models();
		for model in &classifiers {
			assert_eq!(model.provider(), "vercel-ai-gateway");
			assert_eq!(model.api(), "typesafe-system-one");
		}
	}
}
