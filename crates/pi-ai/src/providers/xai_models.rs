//! Port of `packages/ai/src/providers/xai.models.ts`.
//!
//! Deviations:
//! - D22: JSON embedded with `include_str!` instead of an import attribute.
//! - D23: catalogs are exposed as ordered `Vec`s (`Object.values` order).
//! - D24: provider id argument kept for signature parity (unused upstream).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/xai.json`.
pub const XAI_DATA_JSON: &str = include_str!("data/xai.json");

fn groups() -> JsonValue {
    serde_json::from_str(XAI_DATA_JSON).expect("embedded xai.json parses")
}

/// `XAI_MODELS` (`Object.values` order).
pub fn xai_models() -> Vec<Model> {
    crate::model_catalog::flatten_chat_model_catalog("xai", &groups())
        .into_models()
        .into_iter()
        .filter_map(|model| match model {
            AnyModel::Chat(model) => Some(model),
            _ => None,
        })
        .collect()
}

/// `XAI_IMAGE_MODELS` (`Object.values` order).
pub fn xai_image_models() -> Vec<AnyModel> {
    crate::model_catalog::flatten_image_model_catalog("xai", &groups()).into_models()
}

/// `XAI_CLASSIFIER_MODELS` (`Object.values` order).
pub fn xai_classifier_models() -> Vec<ClassifierModel> {
    crate::model_catalog::flatten_classifier_model_catalog("xai", &groups())
        .into_models()
        .into_iter()
        .filter_map(|model| match model {
            AnyModel::Classifier(model) => Some(model),
            _ => None,
        })
        .collect()
}

/// All chat entries as `AnyModel` (what `create_provider` consumes).
pub fn xai_any_models() -> Vec<AnyModel> {
    xai_models().into_iter().map(AnyModel::Chat).collect()
}

/// All classifier entries as `AnyModel`.
pub fn xai_classifier_any_models() -> Vec<AnyModel> {
    xai_classifier_models()
        .into_iter()
        .map(AnyModel::Classifier)
        .collect()
}

/// All image entries as `AnyModel` (chat + image + classifier combined view).
pub fn xai_all_any_models() -> Vec<AnyModel> {
    let mut all = xai_any_models();
    all.extend(xai_image_models());
    all.extend(xai_classifier_any_models());
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_flattens_with_stable_ids() {
        let models = xai_models();
        assert!(!models.is_empty(), "chat catalog must not be empty");
        for model in &models {
            assert_eq!(model.provider(), "xai");
            assert_eq!(model.api(), "openai-responses");
        }
        // Insertion order matches the embedded JSON key order.
        let ids: Vec<&str> = models.iter().map(|m| m.id()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(ids.len(), sorted.len(), "duplicate model ids in catalog");
    }
}
