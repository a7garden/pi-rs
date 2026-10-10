//! Port of `packages/ai/src/images.ts` — global image generation dispatched
//! on `model.api` through the images api registry.
//!
//! Upstream also has a top-level `import "./providers/images/register-builtins.ts"`
//! side effect; that registration is final-integration-owned (it depends on
//! the openrouter images API module landing in the same wave — see the
//! hand-off notes in `api/openrouter_images.rs`).

use crate::images_api_registry::get_images_api_provider;
use crate::providers::images::register_builtins::ensure_built_in_images_api_providers;
use crate::types::{AssistantImages, ImageModel, ImagesContext, ImagesOptions, ImagesStopReason};

/// Upstream `resolveImagesApiProvider` + `generateImages`: auth must be passed
/// explicitly via `options.api_key`; prefer `Models::generate_images`, which
/// resolves provider auth.
pub async fn generate_images(
    model: &ImageModel,
    context: &ImagesContext,
    options: Option<ImagesOptions>,
) -> AssistantImages {
    ensure_built_in_images_api_providers();
    let options = options.unwrap_or_default();
    let Some(provider) = get_images_api_provider(model.api()) else {
        let mut output = error_output(
            model,
            &options,
            &format!("No API provider registered for api: {}", model.api()),
        );
        output.stop_reason = ImagesStopReason::Error;
        return output;
    };
    provider.generate_images(model, context, &options).await
}

pub(crate) fn error_output(
    model: &ImageModel,
    options: &ImagesOptions,
    message: &str,
) -> AssistantImages {
    AssistantImages {
        api: model.api().to_string(),
        provider: model.provider().to_string(),
        model: model.id().to_string(),
        output: Vec::new(),
        response_id: None,
        usage: None,
        stop_reason: if options
            .signal
            .as_ref()
            .is_some_and(|signal| signal.aborted())
        {
            ImagesStopReason::Aborted
        } else {
            ImagesStopReason::Error
        },
        error_message: Some(message.to_string()),
        timestamp: crate::utils::diagnostics::now_ms(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Port of test/images.test.ts dispatch error path (the happy paths hit live
    // endpoints behind OPENROUTER_API_KEY and are skipped, like upstream's
    // describe.skipIf).
    #[tokio::test]
    async fn errors_when_no_provider_is_registered_for_the_api() {
        let model: ImageModel = serde_json::from_value(serde_json::json!({
            "type": "image",
            "id": "test-image-model",
            "name": "Test Image Model",
            "api": "unregistered-images",
            "provider": "test-provider",
            "baseUrl": "https://example.invalid/v1",
            "input": ["text"],
            "output": ["image"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        }))
        .expect("image model");
        let context = ImagesContext {
            input: vec![crate::types::UserContent::Text(
                crate::types::TextContent::new("a red circle"),
            )],
        };
        let result = generate_images(&model, &context, None).await;
        assert_eq!(result.stop_reason, ImagesStopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("No API provider registered for api: unregistered-images")
        );
        assert!(result.timestamp > 0);
    }
}
