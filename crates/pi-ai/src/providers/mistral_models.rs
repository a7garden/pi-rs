//! Port of `packages/ai/src/providers/mistral.models.ts`.
//!
//! Upstream imports `data/mistral.json` and re-exports flattened catalogs
//! via `model-catalog.ts`; here the JSON is embedded with `include_str!` and
//! flattened with [`crate::model_catalog::flatten_*_model_catalog`] (crate
//! deviation D22: JSON embedded instead of an import attribute; the provider
//! argument is unused upstream and kept for signature parity). Catalogs are
//! ordered `Vec`s in `Object.values` order (D23).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/mistral.json`.
pub const MISTRAL_DATA_JSON: &str = include_str!("data/mistral.json");

fn groups() -> JsonValue {
	serde_json::from_str(MISTRAL_DATA_JSON).expect("embedded mistral.json parses")
}

/// `MISTRAL_MODELS` (`Object.values` order).
pub fn mistral_models() -> Vec<Model> {
	crate::model_catalog::flatten_chat_model_catalog("mistral", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Chat(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// `MISTRAL_IMAGE_MODELS` (`Object.values` order).
pub fn mistral_image_models() -> Vec<AnyModel> {
	crate::model_catalog::flatten_image_model_catalog("mistral", &groups()).into_models()
}

/// `MISTRAL_CLASSIFIER_MODELS` (`Object.values` order).
pub fn mistral_classifier_models() -> Vec<ClassifierModel> {
	crate::model_catalog::flatten_classifier_model_catalog("mistral", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Classifier(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// All chat entries as `AnyModel` (what `create_provider` consumes).
pub fn mistral_any_models() -> Vec<AnyModel> {
	mistral_models().into_iter().map(AnyModel::Chat).collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	// Port of the mistral portions of test/model-catalog and the generated
	// data checks: the embedded mistral.json flattens into chat models only.
	#[test]
	fn catalog_flattens_to_chat_models_in_json_order() {
		let models = mistral_models();
		assert!(!models.is_empty());
		assert!(mistral_image_models().is_empty());
		assert!(mistral_classifier_models().is_empty());
		assert_eq!(models.len(), mistral_any_models().len());
		assert_eq!(models[0].id(), "codestral-latest");
		assert!(models.iter().all(|model| model.provider() == "mistral"));
		assert!(models.iter().all(|model| model.api() == "mistral-conversations"));
	}

	#[test]
	fn catalog_lookup_matches_upstream_get_model_entries() {
		let models = mistral_models();
		let large = models.iter().find(|model| model.id() == "mistral-large-latest").expect("entry");
		assert_eq!(large.base.base_url, "https://api.mistral.ai");
		assert!(!large.reasoning);
		let glm = models.iter().find(|model| model.id() == "zai-glm-5-3").expect("entry");
		assert!(glm.reasoning);
		let low = glm.thinking_level_map.as_ref().and_then(|map| map.get(&crate::types::ModelThinkingLevel::Low)).cloned();
		assert_eq!(low, Some(Some("low".to_string())));
	}
}
