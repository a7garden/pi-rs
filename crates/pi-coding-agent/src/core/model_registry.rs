//! Port of `packages/coding-agent/src/core/model-registry.ts`.
//!
//! Synchronous compatibility facade exposed to extensions. Coding-agent
//! internals use [`ModelRuntime`](crate::core::model_runtime::ModelRuntime)
//! directly. Replaces the D29 stand-in in `extensions/types.rs` with a
//! re-export (crate docs D64).
//!
//! Deviations:
//! - **D64 — inherent methods replace overloads.** Upstream
//!   `getAuth`/`registerProvider` overload on the first argument; the port
//!   carries distinct method names (`get_provider_auth` for the provider
//!   overload, `register_provider`/`register_native_provider`).

use std::sync::Arc;

use pi_ai::auth::types::{AuthResult, Credential};
use pi_ai::models::{
	Models, ModelsApiStreamOptions, ModelsClassifierOptions, ModelsDeferredFetchOptions, ModelsImagesOptions,
	ModelsRefreshOptions, ModelsRefreshResult, ModelsSimpleStreamOptions, Provider,
};
use pi_ai::types::{
	AnyModel, AssistantImages, AssistantMessage, ClassifierContext, ClassifierModel, ClassifierResult, Context,
	ImageModel, ImagesContext, Model, ModelType, ProviderEnv, ProviderHeaders,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::thrown::ThrownValue;

use crate::core::model_runtime::ModelRuntime;
use crate::core::provider_composer::{AuthStatus, ProviderConfigInput};

/// Successful/failed request-auth resolution (`ResolvedRequestAuth`).
#[derive(Debug, Clone)]
pub enum ResolvedRequestAuth {
	Ok {
		api_key: Option<String>,
		headers: Option<ProviderHeaders>,
		base_url: Option<String>,
		env: Option<ProviderEnv>,
	},
	Error {
		error: String,
	},
}

/// Upstream `ModelRegistry`: a facade delegating to [`ModelRuntime`].
pub struct ModelRegistry {
	runtime: Arc<ModelRuntime>,
}

impl ModelRegistry {
	pub fn new(runtime: Arc<ModelRuntime>) -> ModelRegistry {
		ModelRegistry { runtime }
	}

	pub fn runtime(&self) -> &Arc<ModelRuntime> {
		&self.runtime
	}

	/// Reload models.json asynchronously. Await before making synchronous
	/// registry reads.
	pub async fn refresh(&self, options: Option<ModelsRefreshOptions>) -> ModelsRefreshResult {
		self.runtime.refresh(options).await
	}

	pub fn get_error(&self) -> Option<String> {
		self.runtime.get_error()
	}

	pub fn get_all(&self) -> Vec<Model> {
		self.runtime.get_models(None)
	}

	pub fn get_available(&self) -> Vec<Model> {
		self.runtime.get_available_snapshot()
	}

	pub fn find(&self, provider: &str, model_id: &str) -> Option<Model> {
		self.runtime.get_model(provider, model_id)
	}

	/// Find a model of a non-chat type, e.g. `find_of_type(ModelType::Classifier, "typesafe", "jev-latest")`.
	pub fn find_of_type(&self, type_: ModelType, provider: &str, model_id: &str) -> Option<AnyModel> {
		self.runtime.get_model_of_type(type_, provider, model_id)
	}

	pub fn has_configured_auth(&self, model: &Model) -> bool {
		self.runtime.has_configured_auth(model.provider())
	}

	/// `getApiKeyAndHeaders` — never rejects; failures become
	/// [`ResolvedRequestAuth::Error`].
	pub async fn get_api_key_and_headers(&self, model: &AnyModel) -> ResolvedRequestAuth {
		match self.runtime.get_auth_for_model(model, None).await {
			Ok(Some(resolution)) => ResolvedRequestAuth::Ok {
				api_key: resolution.auth.api_key.clone(),
				headers: resolution.auth.headers.clone(),
				base_url: resolution.auth.base_url.clone(),
				env: resolution.env.clone(),
			},
			Ok(None) => {
				let compatibility = self.runtime.get_compatibility_request_config(model);
				if compatibility.auth_header {
					ResolvedRequestAuth::Error {
						error: format!("No API key found for \"{}\"", model.provider()),
					}
				} else {
					ResolvedRequestAuth::Ok {
						api_key: None,
						headers: compatibility.headers,
						base_url: None,
						env: None,
					}
				}
			}
			Err(error) => {
				let message = match &error {
					ThrownValue::Text(text) => text.clone(),
					ThrownValue::Error(cause) => cause.to_string(),
				};
				ResolvedRequestAuth::Error {
					error: if message == "authHeader requires a resolved API key" {
						format!("No API key found for \"{}\"", model.provider())
					} else {
						message
					},
				}
			}
		}
	}

	pub fn get_provider_auth_status(&self, provider: &str) -> AuthStatus {
		self.runtime.get_provider_auth_status(provider)
	}

	pub fn get_provider(&self, provider: &str) -> Option<Arc<dyn Provider>> {
		self.runtime.get_provider(provider)
	}

	/// Stream through the configured provider with request-time authentication.
	pub fn stream(&self, model: &Model, context: &Context, options: Option<ModelsApiStreamOptions>) -> AssistantMessageEventStream {
		self.runtime.stream(model, context, options)
	}

	/// Stream with provider-neutral options and request-time authentication.
	pub fn stream_simple(&self, model: &Model, context: &Context, options: Option<ModelsSimpleStreamOptions>) -> AssistantMessageEventStream {
		self.runtime.stream_simple(model, context, options)
	}

	pub async fn complete(&self, model: &Model, context: &Context, options: Option<ModelsApiStreamOptions>) -> AssistantMessage {
		self.runtime.complete(model, context, options).await
	}

	/// Every known model of a type (chat, image, classifier), optionally for
	/// one provider.
	pub fn get_models_of_type(&self, type_: ModelType, provider: Option<&str>) -> Vec<AnyModel> {
		self.runtime.get_models_of_type(type_, provider)
	}

	/// Models of a type whose provider has working credentials.
	pub async fn get_available_of_type(
		&self,
		type_: ModelType,
		provider: Option<&str>,
		options: Option<pi_ai::auth::types::AuthOperationOptions>,
	) -> Result<Vec<AnyModel>, ThrownValue> {
		self.models_get_available_of_type(type_, provider, options).await
	}

	pub fn get_model_of_type(&self, type_: ModelType, provider: &str, model_id: &str) -> Option<AnyModel> {
		self.runtime.get_model_of_type(type_, provider, model_id)
	}

	/// Classify structured state with request-time authentication. Never rejects.
	pub async fn classify(&self, model: &ClassifierModel, context: &ClassifierContext, options: Option<ModelsClassifierOptions>) -> ClassifierResult {
		self.runtime.classify(model, context, options).await
	}

	/// Generate images with request-time authentication. Never rejects.
	pub async fn generate_images(&self, model: &ImageModel, context: &ImagesContext, options: Option<ModelsImagesOptions>) -> AssistantImages {
		self.runtime.generate_images(model, context, options).await
	}

	pub fn get_provider_display_name(&self, provider: &str) -> String {
		self.runtime
			.get_provider(provider)
			.map(|provider| provider.name().to_string())
			.unwrap_or_else(|| provider.to_string())
	}

	pub async fn get_provider_auth(&self, provider: &str) -> Result<Option<AuthResult>, ThrownValue> {
		self.runtime.get_auth_for_provider(provider, None).await
	}

	pub async fn get_api_key_for_provider(&self, provider: &str) -> Option<String> {
		self.runtime
			.get_auth_for_provider(provider, None)
			.await
			.ok()
			.flatten()
			.and_then(|resolution| resolution.auth.api_key.clone())
	}

	pub fn is_using_oauth(&self, model: &Model) -> bool {
		self.runtime.is_using_oauth(model.provider())
	}

	/// Register a configured provider by name (`registerProvider(name, config)`).
	pub fn register_provider(&self, provider_name: &str, config: ProviderConfigInput) -> Result<(), ThrownValue> {
		self.runtime.register_provider(provider_name, config)
	}

	/// Register a native provider object (`registerProvider(provider)`).
	pub fn register_native_provider(&self, provider: Arc<dyn Provider>) -> Result<(), ThrownValue> {
		self.runtime.register_native_provider(provider)
	}

	pub fn unregister_provider(&self, provider_name: &str) {
		self.runtime.unregister_provider(provider_name)
	}

	pub fn register_virtual_model(&self, definition: crate::core::virtual_models::VirtualModelDefinition) -> Result<(), ThrownValue> {
		self.runtime.register_virtual_model(definition)
	}

	pub fn unregister_virtual_model(&self, provider_name: &str, id: &str) {
		self.runtime.unregister_virtual_model(provider_name, id)
	}

	pub fn get_registered_provider_config(&self, provider_name: &str) -> Option<ProviderConfigInput> {
		self.runtime.get_registered_provider_config(provider_name)
	}

	pub fn get_registered_native_provider(&self, provider_name: &str) -> Option<Arc<dyn Provider>> {
		self.runtime.get_registered_native_provider(provider_name)
	}

	pub fn get_registered_provider_ids(&self) -> Vec<String> {
		self.runtime.get_registered_provider_ids()
	}

	/// `fetchDeferred` convenience.
	pub async fn fetch_deferred(
		&self,
		model: &Model,
		handle: &pi_ai::types::DeferredHandle,
		options: Option<ModelsDeferredFetchOptions>,
	) -> AssistantMessage {
		self.runtime.fetch_deferred(model, handle, options).await
	}

	/// Credential listing passthrough (`listCredentials`).
	pub async fn list_credentials(
		&self,
		options: Option<pi_ai::auth::types::AuthOperationOptions>,
	) -> Result<Vec<pi_ai::auth::types::CredentialInfo>, ThrownValue> {
		self.runtime.list_credentials(options).await
	}

	/// Login passthrough.
	pub async fn login(
		&self,
		provider_id: &str,
		type_: pi_ai::auth::types::AuthType,
		interaction: pi_ai::auth::types::AuthInteraction,
		options: Option<pi_ai::auth::types::LoginOptions>,
	) -> Result<Credential, ThrownValue> {
		self.runtime.login(provider_id, type_, interaction, options).await
	}

	/// Logout passthrough.
	pub async fn logout(&self, provider_id: &str, options: Option<pi_ai::auth::types::AuthOperationOptions>) -> Result<(), ThrownValue> {
		self.runtime.logout(provider_id, options).await
	}
}

impl ModelRegistry {
	async fn models_get_available_of_type(
		&self,
		type_: ModelType,
		provider: Option<&str>,
		options: Option<pi_ai::auth::types::AuthOperationOptions>,
	) -> Result<Vec<AnyModel>, ThrownValue> {
		match type_ {
			ModelType::Chat => {
				Ok(self.runtime.get_available(provider, options).await?.into_iter().map(AnyModel::Chat).collect())
			}
			ModelType::Image | ModelType::Classifier => {
				Ok(self
					.runtime
					.get_models_of_type(type_, provider)
					.into_iter()
					.filter(|model| match provider {
						Some(provider) => model.provider() == provider,
						None => true,
					})
					.collect())
			}
		}
	}
}
