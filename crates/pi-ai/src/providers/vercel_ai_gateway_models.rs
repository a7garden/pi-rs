//! Port of `packages/ai/src/providers/vercel-ai-gateway.models.ts`.
//!
//! Deviations:
//! - D22: JSON embedded with `include_str!` instead of an import attribute.
//! - D23: catalogs are exposed as ordered `Vec`s (`Object.values` order).
//! - D24: provider id argument kept for signature parity (unused upstream).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/vercel-ai-gateway.json`.
pub const VERCEL_AI_GATEWAY_DATA_JSON: &str = include_str!("data/vercel-ai-gateway.json");

fn groups() -> JsonValue {
	serde_json::from_str(VERCEL_AI_GATEWAY_DATA_JSON).expect("embedded vercel-ai-gateway.json parses")
}

/// `VERCEL_AI_GATEWAY_MODELS` (`Object.values` order).
pub fn vercel_ai_gateway_models() -> Vec<Model> {
	crate::model_catalog::flatten_chat_model_catalog("vercel-ai-gateway", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Chat(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// `VERCEL_AI_GATEWAY_IMAGE_MODELS` (`Object.values` order).
pub fn vercel_ai_gateway_image_models() -> Vec<AnyModel> {
	crate::model_catalog::flatten_image_model_catalog("vercel-ai-gateway", &groups()).into_models()
}

/// `VERCEL_AI_GATEWAY_CLASSIFIER_MODELS` (`Object.values` order).
pub fn vercel_ai_gateway_classifier_models() -> Vec<ClassifierModel> {
	crate::model_catalog::flatten_classifier_model_catalog("vercel-ai-gateway", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Classifier(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// All chat entries as `AnyModel` (what `create_provider` consumes).
pub fn vercel_ai_gateway_any_models() -> Vec<AnyModel> {
	vercel_ai_gateway_models().into_iter().map(AnyModel::Chat).collect()
}

/// All classifier entries as `AnyModel`.
pub fn vercel_ai_gateway_classifier_any_models() -> Vec<AnyModel> {
	vercel_ai_gateway_classifier_models().into_iter().map(AnyModel::Classifier).collect()
}

/// Chat + image + classifier combined view.
pub fn vercel_ai_gateway_all_any_models() -> Vec<AnyModel> {
	let mut all = vercel_ai_gateway_any_models();
	all.extend(vercel_ai_gateway_image_models());
	all.extend(vercel_ai_gateway_classifier_any_models());
	all
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn catalog_flattens_with_stable_ids() {
		let models = vercel_ai_gateway_models();
		assert!(!models.is_empty(), "chat catalog must not be empty");
		for model in &models {
			assert_eq!(model.provider(), "vercel-ai-gateway");
			assert_eq!(model.api(), "anthropic-messages");
		}
	}
}
