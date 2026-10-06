//! Port of `packages/ai/src/providers/cloudflare-workers-ai.ts`.
//!
//! Deviations (per-file ledger):
//! - **D15** — upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid (the
//!   anthropic.rs D17 pattern).
//! - **D16** — deviation D10 (crate docs): the upstream `.lazy.ts` imports
//!   resolve statically, so the provider calls the API constructors directly.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};
use crate::providers::cloudflare_auth::cloudflare_workers_ai_auth;
use crate::providers::cloudflare_stream::{cloudflare_classifier, cloudflare_streams};

/// Upstream `cloudflareWorkersAIProvider()`.
pub fn cloudflare_workers_ai_provider() -> Arc<dyn Provider> {
	let mut classifiers = BTreeMap::new();
	classifiers.insert(
		"cloudflare-workers-ai-system-one".to_string(),
		cloudflare_classifier(crate::api::cloudflare_workers_ai_system_one::cloudflare_workers_ai_system_one_api()),
	);
	create_provider(CreateProviderOptions {
		id: "cloudflare-workers-ai".to_string(),
		name: Some("Cloudflare Workers AI".to_string()),
		base_url: None,
		headers: None,
		auth: ProviderAuth { api_key: Some(cloudflare_workers_ai_auth()), oauth: None },
		models: super::cloudflare_workers_ai_models::cloudflare_workers_ai_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(cloudflare_streams(
			crate::api::openai_completions::openai_completions_api(),
		))),
		images: Default::default(),
		classifiers,
	})
	.expect("static cloudflare-workers-ai provider configuration is valid")
}
