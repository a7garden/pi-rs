//! Port of `packages/ai/src/images-api-registry.ts` — the global registry of
//! image-generation API implementations.
//!
//! Upstream stores `{ provider: { api, generateImages }, sourceId }` in a
//! module-level `Map`; the Rust port stores the same dispatch behind the
//! crate's [`ProviderImages`] trait object in a process-global `RwLock` map
//! (models_store/faux precedent, deviation D35).

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc};
use {parking_lot::RwLock};

use crate::models::ProviderImages;
use crate::types::{AssistantImages, ImageModel, ImagesContext, ImagesOptions};
use crate::utils::thrown::ThrownValue;

/// The uniform image-generation function of an API module
/// (upstream `ImagesApiFunction`).
pub type ImagesApiFunction = Arc<
	dyn Fn(ImageModel, ImagesContext, ImagesOptions) -> futures_util::future::BoxFuture<'static, AssistantImages>
		+ Send
		+ Sync,
>;

/// Upstream `ImagesApiProvider<TApi, TOptions>` — the registration input.
pub struct ImagesApiProvider {
	pub api: String,
	pub generate_images: ImagesApiFunction,
}

struct RegisteredImagesApiProvider {
	provider: Arc<dyn ProviderImages>,
	/// Upstream keeps this for `unregisterApiProviders(sourceId)`-style
	/// bookkeeping; the images registry never unregisters today.
	#[allow(dead_code)]
	source_id: Option<String>,
}

fn registry() -> &'static RwLock<BTreeMap<String, RegisteredImagesApiProvider>> {
	static REGISTRY: std::sync::LazyLock<RwLock<BTreeMap<String, RegisteredImagesApiProvider>>> =
		std::sync::LazyLock::new(|| RwLock::new(BTreeMap::new()));
	&REGISTRY
}

/// Adapter from the upstream function shape to the crate's
/// [`ProviderImages`] trait (the registry is consumed through the trait).
struct FunctionProviderImages {
	api: String,
	generate_images: ImagesApiFunction,
}

impl ProviderImages for FunctionProviderImages {
	fn generate_images<'a>(
		&'a self,
		model: &'a ImageModel,
		context: &'a ImagesContext,
		options: &'a ImagesOptions,
	) -> std::pin::Pin<Box<dyn Future<Output = AssistantImages> + Send + 'a>> {
		Box::pin((self.generate_images)(model.clone(), context.clone(), options.clone()))
	}
}

/// Upstream `wrapGenerateImages`: guards the model's `api` before dispatch
/// (upstream throws synchronously; the Rust function-form panics for the same
/// programming-error contract, and [`check_api_mismatch`] exposes the check
/// to result-returning dispatch sites).
fn wrap_generate_images(api: String, generate_images: ImagesApiFunction) -> ImagesApiFunction {
	Arc::new(move |model, context, options| {
		let api = api.clone();
		let generate_images = generate_images.clone();
		Box::pin(async move {
			check_api_mismatch(&model, &api).expect("mismatched api");
			let fut = generate_images(model, context, options);
			fut.await
		})
	})
}

/// Upstream `registerImagesApiProvider` — replaces any existing entry for the
/// same api id.
pub fn register_images_api_provider(provider: ImagesApiProvider, source_id: Option<String>) {
	let api = provider.api.clone();
	registry().write().insert(
		api.clone(),
		RegisteredImagesApiProvider {
			provider: Arc::new(FunctionProviderImages {
				api: api.clone(),
				generate_images: wrap_generate_images(api, provider.generate_images),
			}),
			source_id,
		},
	);
}

/// Upstream `getImagesApiProvider`.
pub fn get_images_api_provider(api: &str) -> Option<Arc<dyn ProviderImages>> {
	registry().read().get(api).map(|entry| entry.provider.clone())
}

/// Upstream's synchronous `model.api !== api` guard, exposed to
/// result-returning dispatch sites.
pub fn check_api_mismatch(model: &ImageModel, api: &str) -> Result<(), ThrownValue> {
	if model.api() != api {
		return Err(ThrownValue::message(format!("Mismatched api: {} expected {}", model.api(), api)));
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	fn passthrough(api: &'static str) -> ImagesApiFunction {
		Arc::new(move |model, _context, _options| {
			Box::pin(async move { AssistantImages {
				api: api.to_string(),
				provider: model.provider().to_string(),
				model: model.id().to_string(),
				output: Vec::new(),
				response_id: None,
				usage: None,
				stop_reason: crate::types::ImagesStopReason::Stop,
				error_message: None,
				timestamp: crate::utils::diagnostics::now_ms(),
			} })
		})
	}

	// Port of the registry contract exercised by test/images.test.ts dispatch:
	// registration replaces per api id and lookup returns the implementation.
	#[test]
	fn registers_and_returns_api_providers() {
		register_images_api_provider(
			ImagesApiProvider { api: "test-images".to_string(), generate_images: passthrough("test-images") },
			None,
		);
		assert!(get_images_api_provider("test-images").is_some());
		// A second registration replaces the first without error.
		register_images_api_provider(
			ImagesApiProvider { api: "test-images".to_string(), generate_images: passthrough("test-images") },
			Some("source".to_string()),
		);
		assert!(get_images_api_provider("other-images").is_none());
	}
}
