//! Port of `packages/ai/src/providers/radius.ts` — the Radius gateway
//! provider with a persisted, dynamically refreshed catalog.
//!
//! Upstream builds the provider object literal with private `dynamicModels`
//! state; the Rust port routes the refresh flow through
//! [`crate::models::create_provider`]'s `fetch_models` hook, which owns the
//! same stored-restore + overlay merge semantics (crate `create_provider`).
//!
//! # Deviations (per-file ledger)
//!
//! - **D15 — OAuth is wired through `lazy_oauth`** (`auth/oauth/radius.rs`
//!   via `load_radius_oauth`), capturing the resolved `name`/`gateway` the
//!   way upstream's `lazyOAuth({ name, load: () => loadRadiusOAuth({...}) })`
//!   does; api-key auth is kept alongside.
//! - **D16 — the legacy pre-ModelsStore catalog import happens inside the
//!   `fetch_models` hook**, so it only runs when refresh reaches the network
//!   phase. Upstream publishes the legacy import even when network access is
//!   disabled; under `create_provider` the network phase is skipped entirely
//!   then. The observable overlay for the common online path is unchanged.
//! - **D17 — `RadiusProviderOptions.fetch`** injects the `FetchFn` used by
//!   `loadRadiusGatewayConfig` (upstream uses `globalThis.fetch`); tests
//!   inject fakes through it. `None` uses the crate default adapter.
//! - **D18 — factory returns `Arc<dyn Provider>`** via
//!   [`crate::models::create_provider`] (`expect`ed static configuration),
//!   following anthropic.rs D17.

use std::future::Future;
use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::{Credential, ProviderAuth};
use crate::http::FetchFn;
use crate::models::{
	create_provider, ApiDispatch, CreateProviderOptions, ModelsPublication, ModelsStoreWrite, Provider,
	RefreshModelsContext,
};
use crate::providers::radius_config::{
	get_radius_models, get_radius_models_from_config, load_radius_gateway_config, normalize_radius_gateway_url,
	DEFAULT_RADIUS_GATEWAY,
};
use crate::providers::radius_models::radius_models;
use crate::types::AnyModel;
use crate::utils::diagnostics::now_ms;
use crate::utils::thrown::ThrownValue;

/// Upstream `RadiusProviderOptions`.
#[derive(Default)]
pub struct RadiusProviderOptions {
	pub id: Option<String>,
	pub name: Option<String>,
	pub gateway: Option<String>,
	/// Deviation D17: fetch seam for the gateway config request.
	pub fetch: Option<FetchFn>,
}

fn is_default_gateway(gateway: &str) -> bool {
	gateway == normalize_radius_gateway_url(DEFAULT_RADIUS_GATEWAY)
}

/// Upstream `radiusProvider()`. The OAuth auth method is omitted (D15).
pub fn radius_provider(options: RadiusProviderOptions) -> Arc<dyn Provider> {
	let id = options.id.unwrap_or_else(|| "radius".to_string());
	let name = options.name.unwrap_or_else(|| "Radius".to_string());
	let gateway = normalize_radius_gateway_url(options.gateway.as_deref().unwrap_or(DEFAULT_RADIUS_GATEWAY));
	let baseline_models: Vec<AnyModel> = if is_default_gateway(&gateway) {
		radius_models()
			.into_iter()
			.map(|mut model| {
				model.base.provider = id.clone();
				AnyModel::Chat(model)
			})
			.collect()
	} else {
		Vec::new()
	};
	let fetch_models = build_fetch_models(id.clone(), gateway.clone(), options.fetch);
	create_provider(CreateProviderOptions {
		id: id.clone(),
		name: Some(name.clone()),
		base_url: None,
		headers: None,
		auth: ProviderAuth {
			api_key: Some(env_api_key_auth("Radius API key", &["RADIUS_API_KEY"])),
			oauth: Some(Arc::new(crate::auth::helpers::lazy_oauth(crate::auth::helpers::LazyOAuthInput {
				name: name.clone(),
				is_subscription: false,
				login_label: None,
				load: move || {
					crate::auth::oauth::load::load_radius_oauth(crate::auth::oauth::radius::RadiusOAuthOptions {
						name: name.clone(),
						gateway: gateway.clone(),
					})
				},
			}))),
		},
		models: baseline_models,
		fetch_models: Some(fetch_models),
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(crate::api::pi_messages::pi_messages_api())),
		images: Default::default(),
		classifiers: Default::default(),
	})
	.expect("static radius provider configuration is valid")
}

/// The `refreshModels` body threaded through `create_provider`'s
/// `fetch_models` hook: legacy catalog import, then a gateway config fetch.
fn build_fetch_models(
	provider_id: String,
	gateway: String,
	fetch: Option<FetchFn>,
) -> Box<dyn Fn(&RefreshModelsContext) -> std::pin::Pin<Box<dyn Future<Output = Result<Vec<AnyModel>, ThrownValue>> + Send>> + Send + Sync>
{
	Box::new(move |context: &RefreshModelsContext| {
		let provider_id = provider_id.clone();
		let gateway = gateway.clone();
		let fetch = fetch.clone().unwrap_or_default();
		let credential = context.credential.clone();
		let stored_present = context.stored.is_some();
		let allow_network = context.allow_network;
		let signal = context.signal.clone();
		let publish = context.publish.clone();
		Box::pin(async move {
			// Import catalogs cached by the pre-ModelsStore Radius
			// implementation (deviation D16: only on the network path).
			if !stored_present {
				if let Some(Credential::OAuth(oauth)) = &credential {
					let legacy = get_radius_models(&provider_id, Some(oauth));
					if !legacy.is_empty() {
						let models: Vec<AnyModel> = legacy.into_iter().map(AnyModel::Chat).collect();
						let published = publish
							.publish(ModelsPublication {
								persist: Some(ModelsStoreWrite::Write(crate::models_store::ModelsStoreEntry {
									models: models.clone(),
									last_modified: None,
									checked_at: Some(now_ms()),
									etag: None,
								})),
								update: None,
							})
							.await;
						if !published {
							return Ok(Vec::new());
						}
					}
				}
			}

			if !allow_network || signal.aborted() {
				return Ok(Vec::new());
			}
			let api_key = match &credential {
				Some(Credential::OAuth(oauth)) => Some(oauth.credentials.access.clone()),
				Some(Credential::ApiKey(key)) => key.key.clone(),
				None => None,
			};
			let config = load_radius_gateway_config(&gateway, api_key.as_deref(), Some(signal), fetch)
				.await
				.map_err(ThrownValue::message)?;
			Ok(get_radius_models_from_config(&provider_id, &config)
				.into_iter()
				.map(AnyModel::Chat)
				.collect())
		})
	})
}
// ---------------------------------------------------------------------------
// Tests (port of test/radius-provider.test.ts)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use crate::auth::types::CredentialStore;
	use crate::auth::credential_store::InMemoryCredentialStore;
	use crate::http::{HttpBody, HttpRequest, HttpResponse};
	use crate::models::{create_models, CreateModelsOptions, Models, ModelsRefreshOptions, MutableModels};
	use crate::models_store::{InMemoryModelsStore, ModelsStore};
	use serde_json::json;
	use parking_lot::Mutex;

	fn radius_config_json() -> serde_json::Value {
		json!({
			"baseUrl": "https://radius.example/v1",
			"models": [
				{
					"id": "balanced",
					"name": "Fresh Balanced",
					"reasoning": true,
					"input": ["text"],
					"cost": { "input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 0 },
					"contextWindow": 424242,
					"maxTokens": 32000
				},
				{
					"id": "organization-only",
					"name": "Organization Only",
					"reasoning": false,
					"input": ["text"],
					"cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
					"contextWindow": 128000,
					"maxTokens": 16000
				}
			]
		})
	}

	fn config_fetch(body: serde_json::Value, requests: Arc<Mutex<Vec<String>>>) -> FetchFn {
		let body = serde_json::to_vec(&body).expect("config serializes");
		FetchFn::new(move |request: HttpRequest| {
			requests.lock().push(request.url.clone());
			let body = body.clone();
			Box::pin(async move {
				Ok(HttpResponse {
					status: 200,
					headers: vec![("content-type".to_string(), "application/json".to_string())],
					body: HttpBody::Bytes(body),
				})
			})
		})
	}

	#[test]
	fn ships_a_static_public_catalog_for_the_default_gateway() {
		let provider = radius_provider(RadiusProviderOptions::default());
		let models = provider.get_models();
		assert!(!models.is_empty());
		let balanced = models.iter().find(|model| model.id() == "balanced").expect("balanced entry");
		assert_eq!(balanced.provider(), "radius");
		assert_eq!(balanced.api(), "pi-messages");
	}

	#[test]
	fn does_not_apply_the_public_radius_catalog_to_custom_gateways() {
		let provider = radius_provider(RadiusProviderOptions {
			id: Some("radius-dev".to_string()),
			gateway: Some("http://localhost:8788".to_string()),
			..Default::default()
		});
		assert!(provider.get_models().is_empty());
	}

	#[tokio::test]
	async fn overlays_refreshed_models_on_the_static_public_catalog() {
		let requests = Arc::new(Mutex::new(Vec::<String>::new()));
		let fetch = config_fetch(radius_config_json(), requests.clone());
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		credentials
			.modify(
				"radius",
				Box::new(move |_current| {
					Box::pin(async move {
						Ok(Some(Credential::ApiKey(crate::auth::types::ApiKeyCredential {
							key: Some("radius-key".to_string()),
							env: None,
						})))
					})
				}),
				None,
			)
			.await
			.expect("credential write");
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials),
			..Default::default()
		}));
		models.set_provider(radius_provider(RadiusProviderOptions { fetch: Some(fetch), ..Default::default() }));

		let result = models.refresh(Some(ModelsRefreshOptions { providers: Some(vec!["radius".to_string()]), ..Default::default() })).await;

		assert!(result.errors.is_empty(), "{:?}", result.errors.keys());
		let balanced = models.get_model("radius", "balanced").expect("balanced");
		assert_eq!(balanced.base.name, "Fresh Balanced");
		assert_eq!(balanced.base.base_url, "https://radius.example/v1");
		assert_eq!(balanced.context_window, 424242);
		assert!(models.get_model("radius", "organization-only").is_some());
		assert!(models.get_models(Some("radius")).len() > radius_config_json()["models"].as_array().expect("array").len());
	}

	#[tokio::test]
	async fn overlays_a_cached_effective_catalog_without_network_access() {
		let store: Arc<dyn ModelsStore> = Arc::new(InMemoryModelsStore::new());
		// Same config as `radiusConfig()` upstream.
		let config: crate::providers::radius_config::RadiusGatewayConfig =
			serde_json::from_value(radius_config_json()).expect("config parses");
		let cached = get_radius_models_from_config("radius", &config);
		store
			.write(
				"radius",
				crate::models_store::ModelsStoreEntry {
					models: cached.into_iter().map(AnyModel::Chat).collect(),
					last_modified: None,
					checked_at: Some(now_ms()),
					etag: None,
				},
				None,
			)
			.await
			.expect("cached write");
		let models = create_models(Some(CreateModelsOptions { models_store: Some(store), ..Default::default() }));
		models.set_provider(radius_provider(RadiusProviderOptions::default()));

		models
			.refresh(Some(ModelsRefreshOptions {
				providers: Some(vec!["radius".to_string()]),
				allow_network: Some(false),
				..Default::default()
			}))
			.await;

		assert_eq!(
			models.get_model("radius", "balanced").map(|model| model.base.name.clone()),
			Some("Fresh Balanced".to_string())
		);
		assert!(models.get_model("radius", "organization-only").is_some());
	}
}
