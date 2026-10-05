//! Port of `packages/ai/src/providers/openai.models.ts`.
//!
//! Upstream imports `data/openai.json` and re-exports flattened catalogs via
//! `model-catalog.ts`; here the JSON is embedded with `include_str!` and
//! flattened with [`crate::model_catalog::flatten_*_model_catalog`] (D22:
//! JSON embedded instead of an import attribute; the provider argument is
//! unused upstream and kept for signature parity).
//!
//! D23: upstream's catalogs are keyed records; Rust exposes the values as
//! ordered `Vec`s (insertion order matches upstream `Object.values`).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/openai.json`.
pub const OPENAI_DATA_JSON: &str = include_str!("data/openai.json");

fn groups() -> JsonValue {
	serde_json::from_str(OPENAI_DATA_JSON).expect("embedded openai.json parses")
}

/// `OPENAI_MODELS` (`Object.values` order).
pub fn openai_models() -> Vec<Model> {
	crate::model_catalog::flatten_chat_model_catalog("openai", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Chat(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// `OPENAI_IMAGE_MODELS` (`Object.values` order).
pub fn openai_image_models() -> Vec<AnyModel> {
	crate::model_catalog::flatten_image_model_catalog("openai", &groups()).into_models()
}

/// `OPENAI_CLASSIFIER_MODELS` (`Object.values` order).
pub fn openai_classifier_models() -> Vec<ClassifierModel> {
	crate::model_catalog::flatten_classifier_model_catalog("openai", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Classifier(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// All chat entries as `AnyModel` (what `create_provider` consumes).
pub fn openai_any_models() -> Vec<AnyModel> {
	openai_models().into_iter().map(AnyModel::Chat).collect()
}
