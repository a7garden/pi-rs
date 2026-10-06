//! Port of `packages/ai/src/providers/openai-codex.models.ts`.
//!
//! Deviations:
//! - D22: JSON embedded with `include_str!` instead of an import attribute.
//! - D23: catalogs are exposed as ordered `Vec`s (`Object.values` order).
//! - D24: provider id argument kept for signature parity (unused upstream).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/openai-codex.json`.
pub const OPENAICODEX_DATA_JSON: &str = include_str!("data/openai-codex.json");

fn groups() -> JsonValue {
	serde_json::from_str(OPENAICODEX_DATA_JSON).expect("embedded openai-codex.json parses")
}

/// `OPENAICODEX_MODELS` (`Object.values` order).
pub fn openai_codex_models() -> Vec<Model> {
	crate::model_catalog::flatten_chat_model_catalog("openai-codex", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Chat(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// `OPENAICODEX_IMAGE_MODELS` (`Object.values` order).
pub fn openai_codex_image_models() -> Vec<AnyModel> {
	crate::model_catalog::flatten_image_model_catalog("openai-codex", &groups()).into_models()
}

/// `OPENAICODEX_CLASSIFIER_MODELS` (`Object.values` order).
pub fn openai_codex_classifier_models() -> Vec<ClassifierModel> {
	crate::model_catalog::flatten_classifier_model_catalog("openai-codex", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Classifier(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// All chat entries as `AnyModel` (what `create_provider` consumes).
pub fn openai_codex_any_models() -> Vec<AnyModel> {
	openai_codex_models().into_iter().map(AnyModel::Chat).collect()
}

/// All classifier entries as `AnyModel`.
pub fn openai_codex_classifier_any_models() -> Vec<AnyModel> {
	openai_codex_classifier_models().into_iter().map(AnyModel::Classifier).collect()
}

/// Chat + image + classifier combined view (upstream array spreads).
pub fn openai_codex_all_any_models() -> Vec<AnyModel> {
	let mut all = openai_codex_any_models();
	all.extend(openai_codex_image_models());
	all.extend(openai_codex_classifier_any_models());
	all
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn catalog_flattens_with_stable_ids() {
		let models = openai_codex_models();
		assert!(!models.is_empty(), "chat catalog must not be empty");
		for model in &models {
			assert_eq!(model.provider(), "openai-codex");
			assert!(
				model.api() == "openai-codex-responses",
				"unexpected api for model {}",
				model.id()
			);
		}
		let ids: Vec<&str> = models.iter().map(|m| m.id()).collect();
		let mut sorted = ids.clone();
		sorted.sort();
		sorted.dedup();
		assert_eq!(ids.len(), sorted.len(), "duplicate model ids in catalog");
	}
}
