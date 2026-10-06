//! Port of `packages/ai/src/providers/amazon-bedrock.models.ts`.
//!
//! Upstream imports `data/amazon-bedrock.json` and re-exports flattened
//! catalogs via `model-catalog.ts`; here the JSON is embedded with
//! `include_str!` and flattened with
//! [`crate::model_catalog::flatten_*_model_catalog`].
//!
//! Deviations (per-file ledger):
//! - D15: upstream's import attribute becomes an embedded string (D22 pattern
//!   of the reference modules); the provider argument is unused upstream and
//!   kept for signature parity.
//! - D16: upstream's catalogs are keyed records; Rust exposes the values as
//!   ordered `Vec`s (insertion order matches upstream `Object.values`).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/amazon-bedrock.json`.
pub const AMAZON_BEDROCK_DATA_JSON: &str = include_str!("data/amazon-bedrock.json");

fn groups() -> JsonValue {
    serde_json::from_str(AMAZON_BEDROCK_DATA_JSON).expect("embedded amazon-bedrock.json parses")
}

/// `AMAZON_BEDROCK_MODELS` (`Object.values` order).
pub fn amazon_bedrock_models() -> Vec<Model> {
    crate::model_catalog::flatten_chat_model_catalog("amazon-bedrock", &groups())
        .into_models()
        .into_iter()
        .filter_map(|model| match model {
            AnyModel::Chat(model) => Some(model),
            _ => None,
        })
        .collect()
}

/// `AMAZON_BEDROCK_IMAGE_MODELS` (`Object.values` order).
pub fn amazon_bedrock_image_models() -> Vec<AnyModel> {
    crate::model_catalog::flatten_image_model_catalog("amazon-bedrock", &groups()).into_models()
}

/// `AMAZON_BEDROCK_CLASSIFIER_MODELS` (`Object.values` order).
pub fn amazon_bedrock_classifier_models() -> Vec<ClassifierModel> {
    crate::model_catalog::flatten_classifier_model_catalog("amazon-bedrock", &groups())
        .into_models()
        .into_iter()
        .filter_map(|model| match model {
            AnyModel::Classifier(model) => Some(model),
            _ => None,
        })
        .collect()
}

/// All chat entries as `AnyModel` (what `create_provider` consumes).
pub fn amazon_bedrock_any_models() -> Vec<AnyModel> {
    amazon_bedrock_models()
        .into_iter()
        .map(AnyModel::Chat)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port of `test/bedrock-models.test.ts` (the `hasBedrockCredentials()`
    /// per-model live sweep is credentials-gated upstream and stays deferred —
    /// the offline assertions travel).
    #[test]
    fn gets_all_available_bedrock_models() {
        let models = amazon_bedrock_models();
        assert!(models.len() > 0);
    }

    /// "exposes Claude Opus 5 through an inference profile only".
    #[test]
    fn exposes_claude_opus_5_through_an_inference_profile_only() {
        let models = amazon_bedrock_models();
        assert!(models
            .iter()
            .any(|model| model.base.id == "global.anthropic.claude-opus-5"));
        assert!(!models
            .iter()
            .any(|model| model.base.id == "anthropic.claude-opus-5"));
    }

    /// Port of the amazon-bedrock case of `test/providers.test.ts`
    /// "records known direct-provider image request limits".
    #[test]
    fn records_known_image_request_limits_for_bedrock_haiku() {
        let model = amazon_bedrock_models()
            .into_iter()
            .find(|model| model.base.id == "anthropic.claude-haiku-4-5-20251001-v1:0")
            .expect("haiku 4.5 model present");
        let images = model
            .base
            .input_limits
            .as_ref()
            .and_then(|limits| limits.images.as_ref())
            .expect("image limits present");
        assert_eq!(images.max_per_message, Some(20));
    }

    /// Port of the bedrock cases of `test/supports-xhigh.test.ts` (the
    /// `getSupportedThinkingLevels` machinery lives in `models.rs`; the catalog
    /// data driving it is this module's).
    #[test]
    fn supports_xhigh_cases() {
        let supported = |id: &str| {
            let model = amazon_bedrock_models()
                .into_iter()
                .find(|model| model.base.id == id)
                .unwrap_or_else(|| panic!("catalog model not found: {id}"));
            crate::models::get_supported_thinking_levels(&model)
                .into_iter()
                .map(|level| serde_json::to_value(level).expect("level serializes"))
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        };

        // "includes xhigh and max for Bedrock Claude Opus 5"
        let opus_5 = supported("global.anthropic.claude-opus-5");
        assert!(opus_5.iter().any(|level| level == "xhigh"));
        assert!(opus_5.iter().any(|level| level == "max"));

        // "includes xhigh and max but not off for Bedrock Claude Fable 5"
        let fable_5 = supported("global.anthropic.claude-fable-5");
        assert!(fable_5.iter().any(|level| level == "xhigh"));
        assert!(fable_5.iter().any(|level| level == "max"));
        assert!(!fable_5.iter().any(|level| level == "off"));
    }
}
