//! Port of `packages/ai/src/image-models.ts` — compat reads of the generated
//! catalog restricted to image models.
//!
//! New code uses `Models::get_model_of_type("image", ...)` or
//! `providers::all::get_builtin_image_model()`; these reads are deprecated
//! upstream and kept for the compat surface.
//!
//! Deviation D36: upstream's type-level `BuiltinImageProvider` /
//! `BuiltinImageModel` markers are TypeScript compile-time filters with no
//! runtime behavior; the Rust functions are dynamically typed over the same
//! embedded catalog (`models_generated::image_models`).

use crate::model_catalog::OrderedMap;
use crate::types::{AnyModel, ImageModel};

/// Providers with at least one image model in the generated catalog, in
/// upstream key order.
pub fn get_image_providers() -> Vec<String> {
    crate::models_generated::image_models()
        .iter()
        .filter(|(_, catalog)| !catalog.is_empty())
        .map(|(provider, _)| provider.to_string())
        .collect()
}

/// Upstream `getImageModel` — `None` when the provider or model id is absent.
pub fn get_image_model(provider: &str, model_id: &str) -> Option<ImageModel> {
    match crate::models_generated::image_models()
        .get(provider)?
        .get(model_id)?
    {
        AnyModel::Image(model) => Some(model.clone()),
        _ => None,
    }
}

/// Upstream `getImageModels` — every image model of one provider, in catalog
/// order.
pub fn get_image_models(provider: &str) -> Vec<ImageModel> {
    let Some(catalog) = crate::models_generated::image_models().get(provider) else {
        return Vec::new();
    };
    catalog
        .iter()
        .filter_map(|(_, model)| match model {
            AnyModel::Image(model) => Some(model.clone()),
            _ => None,
        })
        .collect()
}

/// Provider catalog accessor used by tests (keeps the OrderedMap import
/// meaningful for callers embedding their own shards).
pub type ProviderImageCatalog = OrderedMap<AnyModel>;

#[cfg(test)]
mod tests {
    use super::*;

    // Port of test/images-models.test.ts "keeps existing built-in and compat
    // model reads chat-only" — the compat image reads only return image models.
    #[test]
    fn reads_only_image_models_from_the_generated_catalog() {
        assert!(!get_image_providers().is_empty());
        assert!(get_image_providers().contains(&"openrouter".to_string()));

        let openrouter = get_image_models("openrouter");
        assert!(!openrouter.is_empty());
        for model in &openrouter {
            assert_eq!(model.model_type(), crate::types::ModelType::Image);
            assert_eq!(model.api(), "openrouter-images");
        }

        let specific = get_image_model("openrouter", "google/gemini-2.5-flash-image");
        assert!(specific.is_some());
        assert_eq!(
            specific.expect("model").model_type(),
            crate::types::ModelType::Image
        );

        assert_eq!(get_image_model("openrouter", "does-not-exist"), None);
        assert_eq!(
            get_image_models("no-such-provider"),
            Vec::<ImageModel>::new()
        );
    }
}
