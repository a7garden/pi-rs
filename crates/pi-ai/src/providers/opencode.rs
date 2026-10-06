//! Port of `packages/ai/src/providers/opencode.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider, ProviderClassifier, ProviderStreams};
use crate::providers::opencode_headers::with_opencode_session_header;

type OpenCodeApiMap = BTreeMap<String, Arc<dyn ProviderStreams>>;

/// Upstream `opencodeProvider()`.
pub fn opencode_provider() -> Arc<dyn Provider> {
	let mut api: OpenCodeApiMap = BTreeMap::new();
	api.insert(
		"anthropic-messages".to_string(),
		with_opencode_session_header(crate::api::anthropic_messages::anthropic_messages_api()),
	);
	api.insert(
		"google-generative-ai".to_string(),
		with_opencode_session_header(crate::api::google_generative_ai::google_generative_ai_api()),
	);
	api.insert(
		"openai-completions".to_string(),
		with_opencode_session_header(crate::api::openai_completions::openai_completions_api()),
	);
	api.insert(
		"openai-responses".to_string(),
		with_opencode_session_header(crate::api::openai_responses::openai_responses_api()),
	);
	let mut classifiers: BTreeMap<String, Arc<dyn ProviderClassifier>> = BTreeMap::new();
	// OpenCode Zen serves TypeSafe's System One protocol at /zen/v1/systemone.
	classifiers.insert(
		"typesafe-system-one".to_string(),
		crate::api::typesafe_system_one::typesafe_system_one_api(),
	);
	create_provider(CreateProviderOptions {
		id: "opencode".to_string(),
		name: Some("OpenCode Zen".to_string()),
		base_url: None,
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("OpenCode API key", &["OPENCODE_API_KEY"])),
			oauth: None,
		},
		models: {
			let mut models = super::opencode_models::opencode_any_models();
			models.extend(super::opencode_models::opencode_classifier_any_models());
			models
		},
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::ByApi(api)),
		images: Default::default(),
		classifiers,
	})
	.expect("static opencode provider configuration is valid")
}
