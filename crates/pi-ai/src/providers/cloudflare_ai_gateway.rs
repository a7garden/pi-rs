//! Port of `packages/ai/src/providers/cloudflare-ai-gateway.ts`.
//!
//! Deviations (per-file ledger):
//! - **D15** — upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid (the
//!   anthropic.rs D17 pattern).
//! - **D16** — upstream `api` is a partial record keyed by API id; Rust
//!   expresses the three pinned entries as [`crate::models::ApiDispatch::ByApi`]
//!   (keyed dispatch on `model.api`).

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::auth::types::ProviderAuth;
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};
use crate::providers::cloudflare_auth::cloudflare_ai_gateway_auth;
use crate::providers::cloudflare_stream::cloudflare_streams;

/// Upstream `type CloudflareAIGatewayApi` — the APIs the provider pins,
/// including `openai-completions` even when the generated catalog happens to
/// contain none.
pub const CLOUDFLARE_AI_GATEWAY_APIS: &[&str] = &[
    "anthropic-messages",
    "openai-completions",
    "openai-responses",
];

/// Upstream `cloudflareAIGatewayProvider()`.
pub fn cloudflare_ai_gateway_provider() -> Arc<dyn Provider> {
    // The api map is pinned to all three APIs: models.dev's gateway catalog drops and
    // restores `workers-ai/*` (openai-completions) entries over time, and inference from
    // `models` alone would otherwise reject the openai-completions entry whenever the
    // generated catalog happens to contain none.
    let mut api = BTreeMap::new();
    api.insert(
        "anthropic-messages".to_string(),
        cloudflare_streams(crate::api::anthropic_messages::anthropic_messages_api()),
    );
    api.insert(
        "openai-completions".to_string(),
        cloudflare_streams(crate::api::openai_completions::openai_completions_api()),
    );
    api.insert(
        "openai-responses".to_string(),
        cloudflare_streams(crate::api::openai_responses::openai_responses_api()),
    );
    create_provider(CreateProviderOptions {
        id: "cloudflare-ai-gateway".to_string(),
        name: Some("Cloudflare AI Gateway".to_string()),
        base_url: None,
        headers: None,
        auth: ProviderAuth {
            api_key: Some(cloudflare_ai_gateway_auth()),
            oauth: None,
        },
        models: super::cloudflare_ai_gateway_models::cloudflare_ai_gateway_any_models(),
        fetch_models: None,
        filter_models: None,
        filter_all_models: None,
        api: Some(ApiDispatch::ByApi(api)),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("static cloudflare-ai-gateway provider configuration is valid")
}
