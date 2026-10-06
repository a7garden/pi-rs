//! Port of `packages/ai/src/providers/github-copilot.models.ts`.
//!
//! Deviations:
//! - D22: JSON embedded with `include_str!` instead of an import attribute.
//! - D23: catalogs are exposed as ordered `Vec`s (`Object.values` order).
//! - D24: provider id argument kept for signature parity (unused upstream).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/github-copilot.json`.
pub const GITHUBCOPILOT_DATA_JSON: &str = include_str!("data/github-copilot.json");

fn groups() -> JsonValue {
	serde_json::from_str(GITHUBCOPILOT_DATA_JSON).expect("embedded github-copilot.json parses")
}

/// `GITHUBCOPILOT_MODELS` (`Object.values` order).
pub fn github_copilot_models() -> Vec<Model> {
	crate::model_catalog::flatten_chat_model_catalog("github-copilot", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Chat(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// `GITHUBCOPILOT_IMAGE_MODELS` (`Object.values` order).
pub fn github_copilot_image_models() -> Vec<AnyModel> {
	crate::model_catalog::flatten_image_model_catalog("github-copilot", &groups()).into_models()
}

/// `GITHUBCOPILOT_CLASSIFIER_MODELS` (`Object.values` order).
pub fn github_copilot_classifier_models() -> Vec<ClassifierModel> {
	crate::model_catalog::flatten_classifier_model_catalog("github-copilot", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Classifier(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// All chat entries as `AnyModel` (what `create_provider` consumes).
pub fn github_copilot_any_models() -> Vec<AnyModel> {
	github_copilot_models().into_iter().map(AnyModel::Chat).collect()
}

/// All classifier entries as `AnyModel`.
pub fn github_copilot_classifier_any_models() -> Vec<AnyModel> {
	github_copilot_classifier_models().into_iter().map(AnyModel::Classifier).collect()
}

/// Chat + image + classifier combined view (upstream array spreads).
pub fn github_copilot_all_any_models() -> Vec<AnyModel> {
	let mut all = github_copilot_any_models();
	all.extend(github_copilot_image_models());
	all.extend(github_copilot_classifier_any_models());
	all
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn catalog_flattens_with_stable_ids() {
		let models = github_copilot_models();
		assert!(!models.is_empty(), "chat catalog must not be empty");
		for model in &models {
			assert_eq!(model.provider(), "github-copilot");
			assert!(
				matches!(model.api(), "anthropic-messages" | "openai-completions" | "openai-responses"),
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
