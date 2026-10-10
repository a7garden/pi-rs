//! Port of `packages/ai/src/providers/cloudflare-workers-ai.models.ts`.
//!
//! Upstream imports `data/cloudflare-workers-ai.json` and re-exports
//! flattened catalogs via `model-catalog.ts`; here the JSON is embedded with
//! `include_str!` and flattened with
//! [`crate::model_catalog::flatten_*_model_catalog`] (D15: JSON embedded
//! instead of an import attribute; the provider argument is unused upstream
//! and kept for signature parity).
//!
//! D16: upstream's catalogs are keyed records; Rust exposes the values as
//! ordered `Vec`s (insertion order matches upstream `Object.values`).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/cloudflare-workers-ai.json`.
pub const CLOUDFLARE_WORKERS_AI_DATA_JSON: &str = include_str!("data/cloudflare-workers-ai.json");

fn groups() -> JsonValue {
    serde_json::from_str(CLOUDFLARE_WORKERS_AI_DATA_JSON)
        .expect("embedded cloudflare-workers-ai.json parses")
}

/// `CLOUDFLARE_WORKERS_AI_MODELS` (`Object.values` order).
pub fn cloudflare_workers_ai_models() -> Vec<Model> {
    crate::model_catalog::flatten_chat_model_catalog("cloudflare-workers-ai", &groups())
        .into_models()
        .into_iter()
        .filter_map(|model| match model {
            AnyModel::Chat(model) => Some(model),
            _ => None,
        })
        .collect()
}

/// `CLOUDFLARE_WORKERS_AI_IMAGE_MODELS` (`Object.values` order).
pub fn cloudflare_workers_ai_image_models() -> Vec<AnyModel> {
    crate::model_catalog::flatten_image_model_catalog("cloudflare-workers-ai", &groups())
        .into_models()
}

/// `CLOUDFLARE_WORKERS_AI_CLASSIFIER_MODELS` (`Object.values` order).
pub fn cloudflare_workers_ai_classifier_models() -> Vec<ClassifierModel> {
    crate::model_catalog::flatten_classifier_model_catalog("cloudflare-workers-ai", &groups())
        .into_models()
        .into_iter()
        .filter_map(|model| match model {
            AnyModel::Classifier(model) => Some(model),
            _ => None,
        })
        .collect()
}

/// Chat plus classifier entries as `AnyModel`, in the order the provider
/// factory lists them (`[...MODELS, ...CLASSIFIER_MODELS]` upstream).
pub fn cloudflare_workers_ai_any_models() -> Vec<AnyModel> {
    cloudflare_workers_ai_models()
        .into_iter()
        .map(AnyModel::Chat)
        .chain(
            cloudflare_workers_ai_classifier_models()
                .into_iter()
                .map(AnyModel::Classifier),
        )
        .collect()
}
