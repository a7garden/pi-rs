//! Port of `packages/ai/src/providers/cloudflare-ai-gateway.models.ts`.
//!
//! Upstream imports `data/cloudflare-ai-gateway.json` and re-exports
//! flattened catalogs via `model-catalog.ts`; here the JSON is embedded with
//! `include_str!` and flattened with
//! [`crate::model_catalog::flatten_*_model_catalog`] (D15: JSON embedded
//! instead of an import attribute; the provider argument is unused upstream
//! and kept for signature parity).
//!
//! D16: upstream's catalogs are keyed records; Rust exposes the values as
//! ordered `Vec`s (insertion order matches upstream `Object.values`).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/cloudflare-ai-gateway.json`.
pub const CLOUDFLARE_AI_GATEWAY_DATA_JSON: &str = include_str!("data/cloudflare-ai-gateway.json");

fn groups() -> JsonValue {
	serde_json::from_str(CLOUDFLARE_AI_GATEWAY_DATA_JSON).expect("embedded cloudflare-ai-gateway.json parses")
}

/// `CLOUDFLARE_AI_GATEWAY_MODELS` (`Object.values` order).
pub fn cloudflare_ai_gateway_models() -> Vec<Model> {
	crate::model_catalog::flatten_chat_model_catalog("cloudflare-ai-gateway", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Chat(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// `CLOUDFLARE_AI_GATEWAY_IMAGE_MODELS` (`Object.values` order).
pub fn cloudflare_ai_gateway_image_models() -> Vec<AnyModel> {
	crate::model_catalog::flatten_image_model_catalog("cloudflare-ai-gateway", &groups()).into_models()
}

/// `CLOUDFLARE_AI_GATEWAY_CLASSIFIER_MODELS` (`Object.values` order).
pub fn cloudflare_ai_gateway_classifier_models() -> Vec<ClassifierModel> {
	crate::model_catalog::flatten_classifier_model_catalog("cloudflare-ai-gateway", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Classifier(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// All chat entries as `AnyModel` (what `create_provider` consumes).
pub fn cloudflare_ai_gateway_any_models() -> Vec<AnyModel> {
	cloudflare_ai_gateway_models().into_iter().map(AnyModel::Chat).collect()
}
