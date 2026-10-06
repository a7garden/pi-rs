//! Port of `packages/ai/src/providers/azure.ts` — the Azure provider factory.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid.
//! - The `azure-openai-responses` api entry is the static constructor of
//!   `azure-openai-responses.lazy.ts` (`azureOpenAIResponsesApi`, crate D10);
//!   the `openai-completions` entry is the `azureStreams` wrapper hosted in
//!   [`crate::api::azure_openai_completions`] (upstream has no
//!   `api/azure-openai-completions.ts`).

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider, ProviderStreams};
use crate::types::{api_ids, provider_ids};

/// Upstream `azureProvider()`.
pub fn azure_provider() -> Arc<dyn Provider> {
	let mut api: BTreeMap<String, Arc<dyn ProviderStreams>> = BTreeMap::new();
	api.insert(
		api_ids::AZURE_OPENAI_RESPONSES.to_string(),
		crate::api::azure_openai_responses::azure_openai_responses_api(),
	);
	api.insert(
		api_ids::OPENAI_COMPLETIONS.to_string(),
		crate::api::azure_openai_completions::azure_openai_completions_api(),
	);
	create_provider(CreateProviderOptions {
		id: provider_ids::AZURE.to_string(),
		name: Some("Azure".to_string()),
		base_url: None,
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("Azure OpenAI API key", &["AZURE_OPENAI_API_KEY"])),
			oauth: None,
		},
		models: super::azure_models::azure_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::ByApi(api)),
		images: BTreeMap::new(),
		classifiers: BTreeMap::new(),
	})
	.expect("static azure provider configuration is valid")
}
