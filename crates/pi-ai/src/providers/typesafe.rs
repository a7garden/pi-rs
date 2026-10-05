//! Port of `packages/ai/src/providers/typesafe.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid.

use std::sync::Arc;

use crate::auth::types::ProviderAuth;
use crate::auth::helpers::env_api_key_auth;
use crate::models::{create_provider, CreateProviderOptions, Provider};

/// Upstream `typesafeProvider()`: a classifier-only provider (no chat `api`).
pub fn typesafe_provider() -> Arc<dyn Provider> {
	let mut classifiers = std::collections::BTreeMap::new();
	classifiers
		.insert("typesafe-system-one".to_string(), crate::api::typesafe_system_one::typesafe_system_one_api());
	create_provider(CreateProviderOptions {
		id: "typesafe".to_string(),
		name: Some("TypeSafe".to_string()),
		base_url: None,
		headers: None,
		auth: ProviderAuth { api_key: Some(env_api_key_auth("TypeSafe API key", &["TYPESAFE_API_KEY"])), oauth: None },
		models: super::typesafe_models::typesafe_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: None,
		images: Default::default(),
		classifiers,
	})
	.expect("static typesafe provider configuration is valid")
}
