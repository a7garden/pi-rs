//! Port of `packages/ai/src/providers/radius.models.ts`.
//!
//! Upstream imports `data/radius.json` and re-exports flattened catalogs via
//! `model-catalog.ts`; here the JSON is embedded with `include_str!` and
//! flattened with [`crate::model_catalog::flatten_*_model_catalog`]
//! (anthropic_models D22/D23 pattern: JSON embedded instead of an import
//! attribute; catalogs exposed as ordered `Vec`s).

use crate::types::{AnyModel, ClassifierModel, JsonValue, Model};

/// Raw embedded contents of upstream `providers/data/radius.json`.
pub const RADIUS_DATA_JSON: &str = include_str!("data/radius.json");

fn groups() -> JsonValue {
	serde_json::from_str(RADIUS_DATA_JSON).expect("embedded radius.json parses")
}

/// `RADIUS_MODELS` (`Object.values` order).
pub fn radius_models() -> Vec<Model> {
	crate::model_catalog::flatten_chat_model_catalog("radius", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Chat(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// `RADIUS_IMAGE_MODELS` (`Object.values` order).
pub fn radius_image_models() -> Vec<AnyModel> {
	crate::model_catalog::flatten_image_model_catalog("radius", &groups()).into_models()
}

/// `RADIUS_CLASSIFIER_MODELS` (`Object.values` order).
pub fn radius_classifier_models() -> Vec<ClassifierModel> {
	crate::model_catalog::flatten_classifier_model_catalog("radius", &groups())
		.into_models()
		.into_iter()
		.filter_map(|model| match model {
			AnyModel::Classifier(model) => Some(model),
			_ => None,
		})
		.collect()
}

/// All chat entries as `AnyModel` (what `create_provider` consumes).
pub fn radius_any_models() -> Vec<AnyModel> {
	radius_models().into_iter().map(AnyModel::Chat).collect()
}

#[cfg(test)]
mod tests {
	// Flag-level port of test/xiaomi-token-plan-ams-anthropic-empty-signature-smoke.test.ts.
	//
	// The upstream case is a live-network smoke test (skipped without a
	// XIAOMI_TOKEN_PLAN_AMS_API_KEY) whose model is hand-built rather than
	// catalog-derived. Only its data/flag-level assertions travel here; the
	// live empty-signature round-trip itself needs the xiaomi provider
	// modules (a sibling's slice) and network access, so it is deferred.
	use crate::models_generated;
	use crate::types::AnyModel;

	#[test]
	fn xiaomi_token_plan_ams_catalog_carries_the_mimo_models_with_their_compat_flags() {
		let entry = models_generated::models().get("xiaomi-token-plan-ams").expect("provider shard");
		let AnyModel::Chat(model) = entry.get("mimo-v2.5-pro").expect("catalog entry") else {
			panic!("mimo-v2.5-pro is a chat model");
		};
		assert_eq!(model.base.api, "openai-completions");
		assert_eq!(model.base.provider, "xiaomi-token-plan-ams");
		assert_eq!(model.base.base_url, "https://token-plan-ams.xiaomimimo.com/v1");
		let compat = model.compat.as_ref().expect("compat flags");
		assert_eq!(compat.supports_strict_mode, Some(true));
		assert_eq!(compat.requires_reasoning_content_on_assistant_messages, Some(true));
		assert_eq!(compat.thinking_format, Some(crate::types::ThinkingFormat::Deepseek));
		assert!(model.reasoning);
	}

	#[test]
	fn the_embedded_catalog_advertises_allow_empty_signature_for_anthropic_models() {
		// The compat flag the smoke test exercises (`allowEmptySignature`)
		// is a real, landed flag across the embedded anthropic-messages
		// catalog; the anthropic API honors it (anthropic_messages
		// `get_anthropic_compat`).
		let mut flagged = 0;
		for (provider, entries) in models_generated::models().iter() {
			for (id, model) in entries.iter() {
				let AnyModel::Chat(model) = model else { continue };
				let Some(compat) = &model.compat else { continue };
				if compat.allow_empty_signature == Some(true) {
					assert_eq!(model.base.api, "anthropic-messages", "{provider}/{id}");
					flagged += 1;
				}
			}
		}
		assert!(flagged > 0, "expected allowEmptySignature entries in the embedded catalog");
	}
}
