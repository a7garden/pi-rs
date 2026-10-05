//! Port of `packages/ai/src/providers/typesafe.models.ts`.
//!
//! Upstream imports `data/typesafe.json` and re-exports flattened catalogs via
//! `model-catalog.ts`; here the JSON is embedded with `include_str!` and
//! flattened with [`crate::model_catalog::flatten_*_model_catalog`] (D22:
//! JSON embedded instead of an import attribute; the provider argument is
//! unused upstream and kept for signature parity).
//!
//! D23: upstream's catalogs are keyed records; Rust exposes the values as
//! ordered `Vec`s (insertion order matches upstream `Object.values`).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/typesafe.json`.
pub const TYPESAFE_DATA_JSON: &str = include_str!("data/typesafe.json");

fn groups() -> JsonValue {
	serde_json::from_str(TYPESAFE_DATA_JSON).expect("embedded typesafe.json parses")
}

/// `TYPESAFE_MODELS` (`Object.values` order).
pub fn typesafe_models() -> Vec<Model> {
	crate::model_catalog::flatten_chat_model_catalog("typesafe", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Chat(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// `TYPESAFE_IMAGE_MODELS` (`Object.values` order).
pub fn typesafe_image_models() -> Vec<AnyModel> {
	crate::model_catalog::flatten_image_model_catalog("typesafe", &groups()).into_models()
}

/// `TYPESAFE_CLASSIFIER_MODELS` (`Object.values` order).
pub fn typesafe_classifier_models() -> Vec<ClassifierModel> {
	crate::model_catalog::flatten_classifier_model_catalog("typesafe", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Classifier(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// All chat entries as `AnyModel` (what `create_provider` consumes).
pub fn typesafe_any_models() -> Vec<AnyModel> {
	typesafe_models().into_iter().map(AnyModel::Chat).collect()
}
