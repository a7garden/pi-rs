//! Port of `packages/ai/src/providers/openrouter.models.ts`.
//!
//! Deviations:
//! - D22: JSON embedded with `include_str!` instead of an import attribute.
//! - D23: catalogs are exposed as ordered `Vec`s (`Object.values` order).
//! - D24: provider id argument kept for signature parity (unused upstream).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/openrouter.json`.
pub const OPENROUTER_DATA_JSON: &str = include_str!("data/openrouter.json");

fn groups() -> JsonValue {
	serde_json::from_str(OPENROUTER_DATA_JSON).expect("embedded openrouter.json parses")
}

/// `OPENROUTER_MODELS` (`Object.values` order).
pub fn openrouter_models() -> Vec<Model> {
	crate::model_catalog::flatten_chat_model_catalog("openrouter", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Chat(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// `OPENROUTER_IMAGE_MODELS` (`Object.values` order).
pub fn openrouter_image_models() -> Vec<AnyModel> {
	crate::model_catalog::flatten_image_model_catalog("openrouter", &groups()).into_models()
}

/// `OPENROUTER_CLASSIFIER_MODELS` (`Object.values` order).
pub fn openrouter_classifier_models() -> Vec<ClassifierModel> {
	crate::model_catalog::flatten_classifier_model_catalog("openrouter", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Classifier(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// All chat entries as `AnyModel` (what `create_provider` consumes).
pub fn openrouter_any_models() -> Vec<AnyModel> {
	openrouter_models().into_iter().map(AnyModel::Chat).collect()
}

/// All classifier entries as `AnyModel`.
pub fn openrouter_classifier_any_models() -> Vec<AnyModel> {
	openrouter_classifier_models().into_iter().map(AnyModel::Classifier).collect()
}

/// Chat + image + classifier combined view (upstream array spreads).
pub fn openrouter_all_any_models() -> Vec<AnyModel> {
	let mut all = openrouter_any_models();
	all.extend(openrouter_image_models());
	all.extend(openrouter_classifier_any_models());
	all
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn catalog_flattens_with_stable_ids() {
		let models = openrouter_models();
		assert!(!models.is_empty(), "chat catalog must not be empty");
		for model in &models {
			assert_eq!(model.provider(), "openrouter");
			assert!(
				matches!(model.api(), "anthropic-messages" | "openai-completions"),
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
