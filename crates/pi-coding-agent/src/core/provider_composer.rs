//! Port of `packages/coding-agent/src/core/provider-composer.ts` — composing
//! built-in, `models.json`, and extension provider layers into one
//! [`pi_ai::models::Provider`] (`composeModelProvider`), plus the
//! request-config helpers (`validateExtensionProvider`,
//! `resolveConfiguredModelHeaders`, `resolveCompatibilityRequestConfig`,
//! `configuredRequestAuthStatus`).
//!
//! Upstream test coverage for this module arrives with the `model-runtime.ts`
//! port (`test/model-runtime-{cloudflare,modify-models}-compat.test.ts` and
//! the `model-registry.test.ts` suites construct a `ModelRuntime`).
//!
//! # Deviations
//!
//! - **D49 — `ProviderConfigInput` aliases the extension `ProviderConfig`.**
//!   Upstream declares structurally identical interfaces in
//!   `provider-composer.ts` (`ProviderConfigInput`/`ExtensionOAuthConfig`)
//!   and `extensions/types.ts` (`ProviderConfig`, inline oauth object); the
//!   port keeps one type ([`ProviderConfigInput`]) and aliases it here. The
//!   only interface delta — composer chat model configs carry
//!   `samplingParamsByThinkingLevel` — is unreachable through the extension
//!   interface upstream too (that interface omits the field), so no data is
//!   lost at the extension→composer boundary.
//! - **D50 — throws become `Result`; composed state is interior-mutable;
//!   capability presence is approximated.** Synchronous upstream throws
//!   (`applyModelsJson`/`modelFromJson`/`extensionModelFromDefinition`
//!   validation, `withConfiguredAuth`'s authHeader check, the no-auth error)
//!   return `Err` with the exact upstream strings, and
//!   [`compose_model_provider`] still validates eagerly so registration
//!   reports structural errors immediately. The refresh-path captured state
//!   (`extensionOAuthCredential`, `refreshedExtensionModels`) lives in a
//!   `parking_lot::Mutex`; the `context.publish` update validates the
//!   refreshed list before assigning and drops the publication on validation
//!   failure (upstream throws inside the update callback, rejecting the
//!   refresh). Unlike JS objects, Rust traits cannot introspect whether a
//!   base provider overrides `refreshModels`/`generateImages`/`classify`, so
//!   the composed provider exposes those operations whenever a base provider
//!   exists at all; unsupported operations then surface through the same
//!   default/absent paths (`Models` treats a `None` refresh as static).
//! - **Option casting at the extension stream boundary.** When an
//!   extension's `streamSimple` claims a model's api, upstream casts
//!   `StreamOptions` to `SimpleStreamOptions` for both dispatch paths; the
//!   port wraps full-path options into [`SimpleStreamOptions`] with defaults
//!   for the provider-neutral fields.
//! - **OAuth credential `env`.** `resolveHeadersOrThrow` for OAuth `toAuth`
//!   reads the credential's `env` map; non-string entries are skipped
//!   (upstream would pass arbitrary JSON through and fail at header use).
//! - Plain (uncolored) error text; `chalk`/TUI coloring is excluded with the
//!   interactive mode.

use std::collections::BTreeMap;
use std::sync::{Arc, Weak};
use std::future::Future;
use std::pin::Pin;

use futures_util::future::BoxFuture;
use parking_lot::Mutex;
use pi_ai::auth::types::{
	ApiLoginFn, ApiKeyAuth, ApiKeyAuthInput, ApiKeyCheckFn, ApiKeyCredential, ApiKeyResolveFn, AuthCheck, AuthEvent,
	AuthPrompt, AuthPromptKind, AuthResult, AuthSelectOption, AuthType, Credential, LoginOptions, ModelAuth,
	OAuthAuth, OAuthCredential, ProviderAuth, ProviderAuthInteraction, OAuthLoginFn, OAuthRefreshFn, OAuthToAuthFn,
};
use pi_ai::compat::extension_oauth_types::OAuthLoginCallbacks;
use pi_ai::compat::{get_api_provider, ApiProviderInternal};
use pi_ai::models::{ModelsPublication, Provider, ProviderClassifier, ProviderImages, RefreshModelsContext};
use pi_ai::types::{
	AnyModel, ClassifierModel, ClassifierOptions, DeferredCancelOptions, DeferredFetchOptions, DeferredHandle,
	ImageModel, ImagesContext, ImagesOptions, ImageTypeTag, Model, ModelCompat, ModelCost, ModelImageInputLimits,
	ModelImageResizeOptions, ModelInput, ModelInputLimits, ModelType, SimpleStreamOptions, StreamOptions,
	TranscriptContext,
};
use pi_ai::utils::abort::AbortSignal;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::model_operations::{classifier_error_result, image_error_result};
use pi_ai::utils::thrown::ThrownValue;

use crate::core::extensions::types::{ProviderConfig, ProviderModelConfig, ProviderOAuthConfig};
use crate::core::model_config::{
	ModelConfig, ModelsJsonCostOverride, ModelsJsonModel, ModelsJsonModelOverride, ModelsJsonProvider,
};
use crate::core::resolve_config_value::{
	clear_config_value_cache, get_config_value_env_var_names, is_command_config_value, is_config_value_configured,
	resolve_config_value_or_throw, resolve_headers_or_throw,
};

/// `ExtensionOAuthConfig`: structurally identical to the ported extension
/// type (crate docs D49).
pub type ExtensionOAuthConfig = ProviderOAuthConfig;

/// `ProviderConfigInput`: alias of the extension `ProviderConfig` (D49).
pub type ProviderConfigInput = ProviderConfig;

/// Auth source label for [`AuthStatus`] (`AuthStatus.source`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthStatusSource {
	Stored,
	Runtime,
	Environment,
	Fallback,
	ModelsJsonKey,
	ModelsJsonCommand,
}

impl AuthStatusSource {
	pub fn as_str(&self) -> &'static str {
		match self {
			AuthStatusSource::Stored => "stored",
			AuthStatusSource::Runtime => "runtime",
			AuthStatusSource::Environment => "environment",
			AuthStatusSource::Fallback => "fallback",
			AuthStatusSource::ModelsJsonKey => "models_json_key",
			AuthStatusSource::ModelsJsonCommand => "models_json_command",
		}
	}
}

/// Whether a provider's API key is configured, and from where (`AuthStatus`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthStatus {
	pub configured: bool,
	pub source: Option<AuthStatusSource>,
	pub label: Option<String>,
}

/// `clearApiKeyCache` (re-exported upstream).
pub fn clear_api_key_cache() {
	clear_config_value_cache();
}

/// HRTB closure constructor (pi-ai `ApiKeyAuth.login`).
fn api_login_fn(
	f: impl Fn(ProviderAuthInteraction) -> BoxFuture<'static, Result<ApiKeyCredential, ThrownValue>> + Send + Sync + 'static,
) -> ApiLoginFn {
	Box::new(f)
}

/// HRTB closure constructor (pi-ai `ApiKeyAuth.check`).
fn api_check_fn(
	f: impl for<'a> Fn(ApiKeyAuthInput<'a>) -> BoxFuture<'a, Result<Option<AuthCheck>, ThrownValue>>
	+ Send
	+ Sync
	+ 'static,
) -> ApiKeyCheckFn {
	Box::new(f)
}

/// HRTB closure constructor (pi-ai `ApiKeyAuth.resolve`).
fn api_resolve_fn(
	f: impl for<'a> Fn(ApiKeyAuthInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ThrownValue>>
	+ Send
	+ Sync
	+ 'static,
) -> ApiKeyResolveFn {
	Box::new(f)
}

fn get_all_provider_models(base: Option<&Arc<dyn Provider>>) -> Vec<AnyModel> {
	match base {
		// `provider.getAllModels?.() ?? provider.getModels()` — pi-ai's
		// `get_all_models` default already delegates to `get_models`.
		Some(provider) => provider.get_all_models(),
		None => Vec::new(),
	}
}

const NESTED_COMPAT_KEYS: [&str; 4] =
	["openRouterRouting", "vercelGatewayRouting", "chatTemplateKwargs", "chatTemplateArgs"];

/// Shallow-merge compat objects with one-level deep merges for the nested
/// routing/kwarg keys (`mergeCompat`).
fn merge_compat(base: Option<&ModelCompat>, over: Option<&ModelCompat>) -> Option<ModelCompat> {
	let over = over?;
	let Some(base) = base else { return Some(over.clone()) };
	let mut merged: serde_json::Map<String, serde_json::Value> = serde_json::to_value(base)
		.ok()
		.and_then(|value| value.as_object().cloned())
		.expect("ModelCompat serializes to an object");
	let over_object: serde_json::Map<String, serde_json::Value> = serde_json::to_value(over)
		.ok()
		.and_then(|value| value.as_object().cloned())
		.expect("ModelCompat serializes to an object");
	for (key, override_value) in over_object {
		let base_value = merged.get(&key).cloned();
		if NESTED_COMPAT_KEYS.contains(&key.as_str())
			&& (base_value.as_ref().is_some_and(serde_json::Value::is_object) || override_value.is_object())
		{
			let mut merged_nested = base_value.unwrap_or_else(|| serde_json::Value::Object(Default::default()));
			if let (Some(base_object), Some(over_nested)) = (merged_nested.as_object_mut(), override_value.as_object())
			{
				for (nested_key, nested_value) in over_nested {
					base_object.insert(nested_key.clone(), nested_value.clone());
				}
				merged.insert(key, merged_nested);
			} else {
				merged.insert(key, override_value);
			}
		} else {
			merged.insert(key, override_value);
		}
	}
	serde_json::from_value(serde_json::Value::Object(merged)).ok()
}

/// Merge per-model input limits (`mergeInputLimits`): override values win per
/// field, image entries merge one level deep.
fn merge_input_limits(base: Option<ModelInputLimits>, over: Option<ModelInputLimits>) -> Option<ModelInputLimits> {
	let over = over?;
	let base = base.unwrap_or_default();
	Some(ModelInputLimits {
		max_request_bytes: over.max_request_bytes.or(base.max_request_bytes),
		images: match over.images {
			None => base.images,
			Some(over_images) => {
				let base_images = base.images.unwrap_or_default();
				Some(ModelImageInputLimits {
					max_per_message: over_images.max_per_message.or(base_images.max_per_message),
					max_per_request: over_images.max_per_request.or(base_images.max_per_request),
					resize: match (over_images.resize, base_images.resize) {
						(None, base_resize) => base_resize,
						(Some(over_resize), base_resize) => {
							let base_resize = base_resize.unwrap_or_default();
							Some(ModelImageResizeOptions {
								max_width: over_resize.max_width.or(base_resize.max_width),
								max_height: over_resize.max_height.or(base_resize.max_height),
								max_bytes: over_resize.max_bytes.or(base_resize.max_bytes),
								jpeg_quality: over_resize.jpeg_quality.or(base_resize.jpeg_quality),
							})
						}
					},
				})
			}
		},
	})
}

/// Merge per-thinking-level sampling params (`mergeSamplingParamsByThinkingLevel`).
fn merge_sampling_params_by_thinking_level(
	base: Option<pi_ai::types::SamplingParamsByThinkingLevel>,
	over: Option<pi_ai::types::SamplingParamsByThinkingLevel>,
) -> Option<pi_ai::types::SamplingParamsByThinkingLevel> {
	let over = over?;
	let mut merged = base.clone().unwrap_or_default();
	for level in pi_ai::types::ModelThinkingLevel::LADDER {
		if let Some(params) = over.get(&level) {
			let entry = merged.entry(level).or_default();
			for (key, value) in params {
				entry.insert(key.clone(), value.clone());
			}
		}
	}
	Some(merged)
}

/// Apply a `models.json` per-model override to a chat model
/// (`applyModelOverride`).
fn apply_model_override(mut model: Model, over: &ModelsJsonModelOverride) -> Model {
	if let Some(name) = &over.name {
		model.base.name = name.clone();
	}
	if let Some(reasoning) = over.reasoning {
		model.reasoning = reasoning;
	}
	if let Some(map) = &over.thinking_level_map {
		let mut merged = model.thinking_level_map.clone().unwrap_or_default();
		for (level, value) in map {
			merged.insert(*level, value.clone());
		}
		model.thinking_level_map = Some(merged);
	}
	if let Some(input) = &over.input {
		model.base.input = input.clone();
	}
	model.base.input_limits = merge_input_limits(model.base.input_limits.take(), over.input_limits.clone());
	if let Some(over_cost) = &over.cost {
		let ModelsJsonCostOverride { input, output, cache_read, cache_write, tiers } = over_cost;
		let mut rates = model.base.cost.rates.clone();
		if let Some(input) = input {
			rates.input = *input;
		}
		if let Some(output) = output {
			rates.output = *output;
		}
		if let Some(cache_read) = cache_read {
			rates.cache_read = *cache_read;
		}
		if let Some(cache_write) = cache_write {
			rates.cache_write = *cache_write;
		}
		let tiers = tiers.clone().or_else(|| model.base.cost.tiers.take());
		model.base.cost = ModelCost { rates, tiers };
	}
	if let Some(over_cache) = &over.prompt_cache {
		let mut merged = model.prompt_cache.clone().unwrap_or_default();
		if over_cache.short.is_some() {
			merged.short = over_cache.short;
		}
		if over_cache.long.is_some() {
			merged.long = over_cache.long;
		}
		model.prompt_cache = Some(merged);
	}
	if let Some(context_window) = over.context_window {
		model.context_window = context_window;
	}
	if let Some(max_tokens) = over.max_tokens {
		model.max_tokens = max_tokens;
	}
	if let Some(over_params) = &over.sampling_params {
		let mut merged = model.sampling_params.clone().unwrap_or_default();
		for (key, value) in over_params {
			merged.insert(key.clone(), value.clone());
		}
		model.sampling_params = Some(merged);
	}
	model.sampling_params_by_thinking_level = merge_sampling_params_by_thinking_level(
		model.sampling_params_by_thinking_level.take(),
		over.sampling_params_by_thinking_level.clone(),
	);
	model.compat = merge_compat(model.compat.as_ref(), over.compat.as_ref());
	model
}

/// Build a chat model from a `models.json` definition (`modelFromJson`).
fn model_from_json(
	provider_id: &str,
	definition: &ModelsJsonModel,
	provider_config: &ModelsJsonProvider,
	defaults: Option<&Model>,
) -> Result<Model, String> {
	let api = definition
		.api
		.clone()
		.or_else(|| provider_config.api.clone())
		.or_else(|| defaults.map(|defaults| defaults.api().to_string()));
	let Some(api) = api else {
		return Err(format!(
			"Provider {provider_id}, model {}: no \"api\" specified. Set at provider or model level.",
			definition.id
		));
	};
	let base_url = definition
		.base_url
		.clone()
		.or_else(|| provider_config.base_url.clone())
		.or_else(|| defaults.map(|defaults| defaults.base.base_url.clone()));
	let Some(base_url) = base_url else {
		return Err(format!("Provider {provider_id}: \"baseUrl\" is required when defining custom models."));
	};
	if definition.context_window.is_some_and(|context_window| context_window == 0) {
		return Err(format!("Provider {provider_id}, model {}: invalid contextWindow", definition.id));
	}
	if definition.max_tokens.is_some_and(|max_tokens| max_tokens == 0) {
		return Err(format!("Provider {provider_id}, model {}: invalid maxTokens", definition.id));
	}
	Ok(Model {
		base: pi_ai::types::BaseModelFields {
			id: definition.id.clone(),
			name: definition.name.clone().unwrap_or_else(|| definition.id.clone()),
			api,
			provider: provider_id.to_string(),
			base_url,
			input: definition.input.clone().unwrap_or_else(|| vec![ModelInput::Text]),
			input_limits: definition.input_limits.clone(),
			cost: definition.cost.clone().unwrap_or_default(),
			headers: None,
		},
		model_type: None,
		reasoning: definition.reasoning.unwrap_or(false),
		thinking_level_map: definition.thinking_level_map.clone(),
		prompt_cache: definition.prompt_cache.clone(),
		context_window: definition.context_window.unwrap_or(128000),
		max_tokens: definition.max_tokens.unwrap_or(16384),
		sampling_params: definition.sampling_params.clone(),
		sampling_params_by_thinking_level: definition.sampling_params_by_thinking_level.clone(),
		compat: merge_compat(provider_config.compat.as_ref(), definition.compat.as_ref()),
	})
}

/// Default catalog model reused when building `models.json` definitions
/// (`findModelDefaults`).
fn find_model_defaults(models: &[AnyModel], model_id: &str, api: Option<&str>) -> Option<Model> {
	let chat_models: Vec<&Model> = models
		.iter()
		.filter_map(|model| match model {
			AnyModel::Chat(model) => Some(model),
			_ => None,
		})
		.collect();
	let by_id = chat_models.iter().find(|model| model.id() == model_id).copied();
	let by_api = api.and_then(|api| chat_models.iter().find(|model| model.api() == api).copied());
	let by_openai = chat_models.iter().find(|model| model.api() == "openai-completions").copied();
	let first = chat_models.first().copied();
	by_id.or(by_api).or(by_openai).or(first).cloned()
}

/// Shared base fields of any catalog entry.
fn any_model_base(model: &AnyModel) -> &pi_ai::types::BaseModelFields {
	match model {
		AnyModel::Chat(model) => &model.base,
		AnyModel::Image(model) => &model.base,
		AnyModel::Classifier(model) => &model.base,
	}
}

/// Mutable shared base fields of any catalog entry.
fn any_model_base_mut(model: &mut AnyModel) -> &mut pi_ai::types::BaseModelFields {
	match model {
		AnyModel::Chat(model) => &mut model.base,
		AnyModel::Image(model) => &mut model.base,
		AnyModel::Classifier(model) => &mut model.base,
	}
}

fn definition_type(definition: &ProviderModelConfig) -> ModelType {
	match definition {
		ProviderModelConfig::Chat(_) => ModelType::Chat,
		ProviderModelConfig::Image(_) => ModelType::Image,
		ProviderModelConfig::Classifier(_) => ModelType::Classifier,
	}
}

fn definition_base(definition: &ProviderModelConfig) -> &crate::core::extensions::types::ProviderModelConfigBase {
	match definition {
		ProviderModelConfig::Chat(config) => &config.base,
		ProviderModelConfig::Image(config) => &config.base,
		ProviderModelConfig::Classifier(config) => &config.base,
	}
}

fn definition_api(definition: &ProviderModelConfig) -> Option<&String> {
	match definition {
		ProviderModelConfig::Chat(config) => config.api.as_ref(),
		ProviderModelConfig::Image(config) => config.api.as_ref(),
		ProviderModelConfig::Classifier(config) => config.api.as_ref(),
	}
}

fn definition_headers(definition: &ProviderModelConfig) -> Option<&BTreeMap<String, String>> {
	definition_base(definition).headers.as_ref()
}

/// Default catalog entry reused for extension model definitions
/// (`findExtensionModelDefaults`).
fn find_extension_model_defaults(models: &[AnyModel], definition: &ProviderModelConfig) -> Option<AnyModel> {
	let type_ = definition_type(definition);
	let candidates: Vec<&AnyModel> = models.iter().filter(|model| model.model_type() == type_).collect();
	let by_id = candidates.iter().find(|model| model.id() == definition_base(definition).id).copied();
	let by_api = definition_api(definition)
		.and_then(|api| candidates.iter().find(|model| model.api() == api.as_str()).copied());
	let by_openai_completions = (type_ == ModelType::Chat)
		.then(|| candidates.iter().find(|model| model.api() == "openai-completions").copied())
		.flatten();
	let first = candidates.first().copied();
	by_id.or(by_api).or(by_openai_completions).or(first).cloned()
}

/// Build an [`AnyModel`] from an extension model definition
/// (`extensionModelFromDefinition`).
fn extension_model_from_definition(
	provider_id: &str,
	models: &[AnyModel],
	config: Option<&ProviderConfigInput>,
	definition: &ProviderModelConfig,
) -> Result<AnyModel, String> {
	let type_ = definition_type(definition);
	let base = definition_base(definition);
	let defaults = find_extension_model_defaults(models, definition);
	let provider_api = match (type_, config) {
		(ModelType::Chat, Some(config)) => config.api.clone(),
		_ => None,
	};
	let api = definition_api(definition)
		.cloned()
		.or(provider_api)
		.or_else(|| defaults.as_ref().map(|defaults| defaults.api().to_string()));
	let Some(api) = api else {
		return Err(format!(
			"Provider {provider_id}, model {}: no \"api\" specified. Set it at model level{}.",
			base.id,
			if type_ == ModelType::Chat { " or provider level" } else { "" }
		));
	};
	let base_url = base
		.base_url
		.clone()
		.or_else(|| config.and_then(|config| config.base_url.clone()))
		.or_else(|| defaults.as_ref().map(|defaults| any_model_base(defaults).base_url.clone()));
	let Some(base_url) = base_url else {
		return Err(format!("Provider {provider_id}: \"baseUrl\" is required when defining custom models."));
	};
	let fields = pi_ai::types::BaseModelFields {
		id: base.id.clone(),
		name: base.name.clone(),
		api,
		provider: provider_id.to_string(),
		base_url,
		input: base.input.clone(),
		input_limits: base.input_limits.clone(),
		cost: base.cost.clone().unwrap_or_default(),
		headers: None,
	};
	match definition {
		ProviderModelConfig::Chat(chat) => Ok(AnyModel::Chat(Model {
			base: fields,
			model_type: None,
			reasoning: chat.reasoning,
			thinking_level_map: chat.thinking_level_map.clone(),
			prompt_cache: chat.prompt_cache.clone(),
			context_window: chat.context_window,
			max_tokens: chat.max_tokens,
			sampling_params: chat.sampling_params.clone(),
			// The extension interface carries no per-level sampling params
			// (crate docs D49).
			sampling_params_by_thinking_level: None,
			compat: chat.compat.clone(),
		})),
		ProviderModelConfig::Image(image) => Ok(AnyModel::Image(ImageModel {
			base: fields,
			model_type: ImageTypeTag::Image,
			output: image.output.clone(),
		})),
		ProviderModelConfig::Classifier(classifier) => Ok(AnyModel::Classifier(ClassifierModel {
			base: fields,
			model_type: pi_ai::types::ClassifierTypeTag::Classifier,
			context_window: classifier.context_window,
		})),
	}
}

/// Compose built-in models with the `models.json` layer (`applyModelsJson`).
fn apply_models_json(
	provider_id: &str,
	base_models: &[AnyModel],
	config: Option<&ModelsJsonProvider>,
) -> Result<Vec<AnyModel>, String> {
	let Some(config) = config else {
		return Ok(base_models.to_vec());
	};
	if config.oauth.is_some() && config.base_url.is_none() {
		return Err(format!("Provider {provider_id}: \"baseUrl\" is required when \"oauth\" is set."));
	}
	let has_overrides = config.model_overrides.as_ref().is_some_and(|overrides| !overrides.is_empty());
	let has_models = config.models.as_ref().is_some_and(|models| !models.is_empty());
	if !has_models
		&& config.base_url.is_none()
		&& config.headers.is_none()
		&& config.compat.is_none()
		&& !has_overrides
		&& config.api_key.is_none()
		&& config.oauth.is_none()
		&& config.auth_header.is_none()
	{
		return Err(format!(
			"Provider {provider_id}: must specify \"baseUrl\", \"headers\", \"compat\", \"modelOverrides\", or \"models\"."
		));
	}

	let radius_oauth = config.oauth.as_deref() == Some("radius");
	let mut models: Vec<AnyModel> = base_models
		.iter()
		.map(|model| {
			let base_url = if radius_oauth {
				any_model_base(model).base_url.clone()
			} else {
				config.base_url.clone().unwrap_or_else(|| any_model_base(model).base_url.clone())
			};
			let mut model = model.clone();
			match &mut model {
				AnyModel::Chat(chat) => {
					chat.base.base_url = base_url;
					chat.compat = merge_compat(chat.compat.as_ref(), config.compat.as_ref());
				}
				other => {
					any_model_base_mut(other).base_url = base_url;
				}
			}
			model
		})
		.collect();
	for definition in config.models.iter().flatten() {
		let existing_index = models
			.iter()
			.position(|model| matches!(model, AnyModel::Chat(chat) if chat.id() == definition.id));
		let defaults =
			find_model_defaults(&models, &definition.id, definition.api.as_deref().or(config.api.as_deref()));
		let model = AnyModel::Chat(model_from_json(provider_id, definition, config, defaults.as_ref())?);
		match existing_index {
			Some(existing_index) => models[existing_index] = model,
			None => models.push(model),
		}
	}
	Ok(models)
}

/// Compose extension models over the base layer (`applyExtension`).
fn apply_extension(
	provider_id: &str,
	models: &[AnyModel],
	config: Option<&ProviderConfigInput>,
) -> Result<Vec<AnyModel>, String> {
	let Some(config) = config else {
		return Ok(models.to_vec());
	};
	let Some(definitions) = &config.models else {
		return Ok(match &config.base_url {
			Some(base_url) => models
				.iter()
				.map(|model| {
					let mut model = model.clone();
					any_model_base_mut(&mut model).base_url = base_url.clone();
					model
				})
				.collect(),
			None => models.to_vec(),
		});
	};
	definitions
		.iter()
		.map(|definition| extension_model_from_definition(provider_id, models, Some(config), definition))
		.collect()
}

/// Bridge a pi-ai abort signal onto the crate's abort signal (extension
/// oauth callbacks carry the crate signal; crate docs D50).
fn bridge_abort_signal(signal: pi_ai::utils::abort::AbortSignal) -> crate::utils::abort::AbortSignal {
	let controller = crate::utils::abort::AbortController::new();
	let local = controller.signal().clone();
	tokio::spawn(async move {
		signal.aborted_future().await;
		controller.abort();
	});
	local
}

/// Map extension login callbacks onto the pi-ai interaction surface
/// (`adaptOAuth.login`).
fn adapt_oauth_login(
	config: ProviderOAuthConfig,
	interaction: ProviderAuthInteraction,
) -> BoxFuture<'static, Result<OAuthCredential, ThrownValue>> {
	// Notify and prompt are shared by several callbacks; Arc keeps each
	// closure callable (the Boxed hooks are not Clone).
	let notify: Arc<dyn Fn(AuthEvent) + Send + Sync> = Arc::from(interaction.notify);
	let prompt: Arc<dyn Fn(AuthPrompt) -> BoxFuture<'static, Result<String, ThrownValue>> + Send + Sync> =
		Arc::from(interaction.prompt);
	Box::pin(async move {
		let callbacks = OAuthLoginCallbacks {
			signal: Some(interaction.signal.clone()),
			on_auth: {
				let notify = Arc::clone(&notify);
				Box::new(move |info| notify(AuthEvent::AuthUrl { url: info.url, instructions: info.instructions }))
			},
			on_device_code: {
				let notify = Arc::clone(&notify);
				Box::new(move |info| {
					notify(AuthEvent::DeviceCode {
						user_code: info.user_code,
						verification_uri: info.verification_uri,
						interval_seconds: info.interval_seconds,
						expires_in_seconds: info.expires_in_seconds,
					})
				})
			},
			on_prompt: {
				let prompt = prompt.clone();
				Box::new(move |prompt_info| {
					let prompt = prompt.clone();
					Box::pin(async move {
						prompt(AuthPrompt {
							signal: None,
							kind: AuthPromptKind::Text {
								message: prompt_info.message,
								placeholder: prompt_info.placeholder,
							},
						})
						.await
						.unwrap_or_default()
					}) as BoxFuture<'static, String>
				})
			},
			on_progress: {
				let notify = Arc::clone(&notify);
				Some(Box::new(move |message: &str| {
					notify(AuthEvent::Progress { message: message.to_string() })
				}))
			},
			on_manual_code_input: {
				let prompt = prompt.clone();
				Some(Box::new(move || {
					let prompt = prompt.clone();
					Box::pin(async move {
						prompt(AuthPrompt {
							signal: None,
							kind: AuthPromptKind::ManualCode {
								message: "Paste the authorization code".to_string(),
								placeholder: None,
							},
						})
						.await
						.unwrap_or_default()
					}) as BoxFuture<'static, String>
				}))
			},
			on_select: {
				let prompt = prompt.clone();
				Box::new(move |select| {
					let prompt = prompt.clone();
					Box::pin(async move {
						prompt(AuthPrompt {
							signal: None,
							kind: AuthPromptKind::Select {
								message: select.message,
								options: select
									.options
									.into_iter()
									.map(|option| AuthSelectOption {
										id: option.id,
										label: option.label,
										description: None,
									})
									.collect(),
							},
						})
						.await
						.ok()
					}) as BoxFuture<'static, Option<String>>
				})
			},
		};
		let credential = (config.login)(callbacks).await.map_err(ThrownValue::message)?;
		Ok(OAuthCredential { credentials: credential })
	})
}

/// Merge resolved string headers into request auth, applying `authHeader`
/// (`withConfiguredAuth`).
fn with_configured_auth(
	auth: ModelAuth,
	headers: Option<&BTreeMap<String, String>>,
	auth_header: bool,
) -> Result<ModelAuth, String> {
	let mut merged_headers: Option<pi_ai::types::ProviderHeaders> = if auth.headers.is_some() || headers.is_some() {
		let mut merged = auth.headers.clone().unwrap_or_default();
		if let Some(headers) = headers {
			for (key, value) in headers {
				merged.insert(key.clone(), Some(value.clone()));
			}
		}
		Some(merged)
	} else {
		None
	};
	if auth_header {
		let Some(api_key) = &auth.api_key else {
			return Err("authHeader requires a resolved API key".to_string());
		};
		merged_headers
			.get_or_insert_with(Default::default)
			.insert("Authorization".to_string(), Some(format!("Bearer {api_key}")));
	}
	Ok(ModelAuth { headers: merged_headers, ..auth })
}

/// Extension key over `models.json` key (`configuredApiKey`).
fn configured_api_key(
	config: Option<&ModelsJsonProvider>,
	extension: Option<&ProviderConfigInput>,
) -> Option<String> {
	extension
		.and_then(|extension| extension.api_key.clone())
		.or_else(|| config.and_then(|config| config.api_key.clone()))
}

/// Provider headers with extension values winning (`configuredHeaders`).
fn configured_headers(
	config: Option<&ModelsJsonProvider>,
	extension: Option<&ProviderConfigInput>,
) -> Option<BTreeMap<String, String>> {
	let config_headers = config.and_then(|config| config.headers.clone());
	let extension_headers = extension.and_then(|extension| extension.headers.clone());
	if config_headers.is_none() && extension_headers.is_none() {
		return None;
	}
	let mut merged = config_headers.unwrap_or_default();
	for (key, value) in extension_headers.unwrap_or_default() {
		merged.insert(key, value);
	}
	Some(merged)
}

/// Environment lookup over referenced env-var names (`configContextEnv`);
/// `explicit` entries win.
async fn config_context_env(
	values: &[String],
	ctx: &Arc<dyn pi_ai::auth::types::AuthContext>,
	explicit: Option<BTreeMap<String, String>>,
) -> Option<BTreeMap<String, String>> {
	let mut env = explicit.unwrap_or_default();
	let mut names: Vec<String> = Vec::new();
	for value in values {
		for name in get_config_value_env_var_names(value) {
			if !names.contains(&name) {
				names.push(name);
			}
		}
	}
	for name in names {
		if env.contains_key(&name) {
			continue;
		}
		if let Some(value) = ctx.env(&name).await {
			env.insert(name, value);
		}
	}
	(!env.is_empty()).then_some(env)
}

/// Compose the API-key auth method (`composeApiKeyAuth`).
fn compose_api_key_auth(
	provider_id: &str,
	base: Option<&Arc<dyn Provider>>,
	config: Option<&ModelsJsonProvider>,
	extension: Option<&ProviderConfigInput>,
) -> Option<ApiKeyAuth> {
	let raw_key = configured_api_key(config, extension);
	let oauth_present = extension.is_some_and(|extension| extension.oauth.is_some())
		|| base.is_some_and(|base| base.auth().oauth.is_some());
	// OAuth-only providers get no fabricated API-key login method.
	if base.is_none_or(|base| base.auth().api_key.is_none()) && raw_key.is_none() && oauth_present {
		return None;
	}
	let raw_headers = configured_headers(config, extension);
	let auth_header = extension
		.and_then(|extension| extension.auth_header)
		.or(config.and_then(|config| config.auth_header))
		.unwrap_or(false);
	let base = base.cloned();
	let provider_id = provider_id.to_string();

	let name = base
		.as_ref()
		.and_then(|base| base.auth().api_key.as_ref())
		.map(|inherited| inherited.name.clone())
		.unwrap_or_else(|| "API key".to_string());

	// login: inherited?.login ?? default secret prompt
	let login: ApiLoginFn = {
		let base = base.clone();
		Box::new(move |interaction: ProviderAuthInteraction| {
			let inherited_login = base
				.as_ref()
				.and_then(|base| base.auth().api_key.as_ref())
				.and_then(|api_key| api_key.login.as_ref());
			if let Some(inherited_login) = inherited_login {
				return inherited_login(interaction);
			}
			let prompt = interaction.prompt;
			Box::pin(async move {
				let key = prompt(AuthPrompt {
					signal: None,
					kind: AuthPromptKind::Secret { message: "Enter API key".to_string(), placeholder: None },
				})
				.await?;
				Ok(ApiKeyCredential { key: Some(key), env: None })
			})
		})
	};

	let check: ApiKeyCheckFn = {
		let base = base.clone();
		let raw_key = raw_key.clone();
		Box::new(move |input: ApiKeyAuthInput<'_>| {
			let base = base.clone();
			let raw_key = raw_key.clone();
			Box::pin(async move {
				let inherited = base.as_ref().and_then(|base| base.auth().api_key.as_ref());
				if let Some(credential) = input.credential {
					if let Some(inherited_check) = inherited.and_then(|inherited| inherited.check.as_ref()) {
						return inherited_check(input).await;
					}
					if credential.key.is_some() {
						return Ok(Some(AuthCheck {
							source: Some("stored credential".to_string()),
							check_type: AuthType::ApiKey,
						}));
					}
					let resolved = match inherited.map(|inherited| inherited.resolve.as_ref()) {
						Some(resolve) => resolve(input).await?,
						None => None,
					};
					return Ok(resolved.map(|resolved| AuthCheck { source: resolved.source, check_type: AuthType::ApiKey }));
				}
				if let Some(raw_key) = raw_key {
					if is_command_config_value(&raw_key) {
						return Ok(Some(AuthCheck {
							source: Some("configured API key".to_string()),
							check_type: AuthType::ApiKey,
						}));
					}
					for name in get_config_value_env_var_names(&raw_key) {
						if input.ctx.env(&name).await.is_none() {
							return Ok(None);
						}
					}
					return Ok(Some(AuthCheck {
						source: Some("configured API key".to_string()),
						check_type: AuthType::ApiKey,
					}));
				}
				if let Some(inherited_check) = inherited.and_then(|inherited| inherited.check.as_ref()) {
					return inherited_check(input).await;
				}
				let resolved = match inherited.map(|inherited| inherited.resolve.as_ref()) {
					Some(resolve) => resolve(input).await?,
					None => None,
				};
				Ok(resolved.map(|resolved| AuthCheck { source: resolved.source, check_type: AuthType::ApiKey }))
			})
		})
	};

	let resolve: ApiKeyResolveFn = {
		let base = base.clone();
		let raw_key = raw_key.clone();
		let raw_headers = raw_headers.clone();
		let provider_id = provider_id.clone();
		Box::new(move |input: ApiKeyAuthInput<'_>| {
			let base = base.clone();
			let raw_key = raw_key.clone();
			let raw_headers = raw_headers.clone();
			let provider_id = provider_id.clone();
			Box::pin(async move {
				let inherited = base.as_ref().and_then(|base| base.auth().api_key.as_ref());
				let ctx = Arc::clone(&input.ctx);
				let credential = input.credential;
				let mut result: Option<AuthResult> = None;
				if credential.is_some() {
					result = match inherited.map(|inherited| inherited.resolve.as_ref()) {
						Some(resolve) => resolve(input).await?,
						None => match credential.and_then(|credential| credential.key.clone()) {
							Some(key) => Some(AuthResult {
								auth: ModelAuth { api_key: Some(key), ..ModelAuth::default() },
								env: credential.and_then(|credential| credential.env.clone()),
								source: Some("stored credential".to_string()),
							}),
							None => None,
						},
					};
				} else if let Some(raw_key) = raw_key {
					let env = config_context_env(&[raw_key.clone()], &ctx, None).await;
					let key = resolve_config_value_or_throw(
						&raw_key,
						&format!("API key for provider \"{provider_id}\""),
						env.as_ref(),
					)
					.map_err(ThrownValue::message)?;
					result = match inherited.map(|inherited| inherited.resolve.as_ref()) {
						Some(resolve) => {
							let credential = ApiKeyCredential { key: Some(key), env: None };
							let forwarded = ApiKeyAuthInput {
								ctx: Arc::clone(&ctx),
								credential: Some(&credential),
								signal: input.signal.clone(),
							};
							resolve(forwarded).await?
						}
						None => Some(AuthResult {
							auth: ModelAuth { api_key: Some(key), ..ModelAuth::default() },
							env: None,
							source: Some("configured API key".to_string()),
						}),
					};
				} else if let Some(resolve) = inherited.map(|inherited| inherited.resolve.as_ref()) {
					result = resolve(input).await?;
				}
				let Some(mut result) = result else {
					return Ok(None);
				};
				let mut explicit_env = credential.and_then(|credential| credential.env.clone()).unwrap_or_default();
				if let Some(env) = result.env.take() {
					for (key, value) in env {
						explicit_env.insert(key, value);
					}
				}
				let header_values: Vec<String> = raw_headers
					.as_ref()
					.map(|headers| headers.values().cloned().collect())
					.unwrap_or_default();
				let header_env = config_context_env(&header_values, &ctx, Some(explicit_env)).await;
				let headers = resolve_headers_or_throw(
					raw_headers.as_ref(),
					&format!("provider \"{provider_id}\""),
					header_env.as_ref(),
				)
				.map_err(ThrownValue::message)?;
				result.auth = with_configured_auth(result.auth, headers.as_ref(), auth_header)?;
				Ok(Some(result))
			})
		})
	};

	Some(ApiKeyAuth {
		name,
		login: Some(api_login_fn(login)),
		check: Some(api_check_fn(check)),
		resolve: api_resolve_fn(resolve),
	})
}

/// Shared OAuth `toAuth` implementations wrapped by
/// [`compose_oauth_auth`]: extension `getApiKey` or the base provider's
/// derived auth.
type SharedOAuthToAuth = Arc<dyn for<'a> Fn(&'a OAuthCredential) -> BoxFuture<'a, Result<ModelAuth, ThrownValue>> + Send + Sync>;

/// Compose the OAuth auth method (`composeOAuthAuth`).
fn compose_oauth_auth(
	provider_id: &str,
	base: Option<&Arc<dyn Provider>>,
	config: Option<&ModelsJsonProvider>,
	extension: Option<&ProviderConfigInput>,
) -> Option<Arc<OAuthAuth>> {
	let raw_headers = configured_headers(config, extension);
	let auth_header = extension
		.and_then(|extension| extension.auth_header)
		.or(config.and_then(|config| config.auth_header))
		.unwrap_or(false);
	let provider_id = provider_id.to_string();

	let (name, is_subscription, login_label, login, refresh, inner_to_auth): (
		String,
		bool,
		Option<String>,
		pi_ai::auth::types::OAuthLoginFn,
		pi_ai::auth::types::OAuthRefreshFn,
		SharedOAuthToAuth,
	) = match extension.and_then(|extension| extension.oauth.clone()) {
		Some(extension_oauth) => {
			let login: OAuthLoginFn = {
				let extension_oauth = extension_oauth.clone();
				Box::new(move |interaction: ProviderAuthInteraction, _options: Option<LoginOptions>| {
					let extension_oauth = extension_oauth.clone();
					Box::pin(async move { adapt_oauth_login(extension_oauth, interaction).await })
				})
			};
			let refresh: OAuthRefreshFn = {
				let refresh_token = extension_oauth.refresh_token.clone();
				Box::new(move |credential: OAuthCredential, signal: AbortSignal| {
					let refresh_token = refresh_token.clone();
					Box::pin(async move {
						let local_signal = bridge_abort_signal(signal);
						let refreshed =
							refresh_token(credential.credentials, local_signal).await.map_err(ThrownValue::message)?;
						Ok(OAuthCredential { credentials: refreshed })
					})
				})
			};
			let to_auth: SharedOAuthToAuth = {
				let get_api_key = extension_oauth.get_api_key.clone();
				Arc::new(move |credential: &OAuthCredential| {
					let get_api_key = get_api_key.clone();
					Box::pin(async move {
						Ok(ModelAuth { api_key: Some(get_api_key(&credential.credentials)), ..ModelAuth::default() })
					})
				})
			};
			(
				extension_oauth.name.clone(),
				extension_oauth.is_subscription.unwrap_or(false),
				None,
				login,
				refresh,
				to_auth,
			)
		}
		None => {
			let base_oauth = base.and_then(|base| base.auth().oauth.clone())?;
			let for_login = Arc::clone(&base_oauth);
			let for_refresh = Arc::clone(&base_oauth);
			let for_to_auth = Arc::clone(&base_oauth);
			let login: pi_ai::auth::types::OAuthLoginFn = Box::new(
				move |interaction: ProviderAuthInteraction, options: Option<LoginOptions>| {
					(for_login.login)(interaction, options)
				},
			);
			let refresh: pi_ai::auth::types::OAuthRefreshFn = Box::new(
				move |credential: OAuthCredential, signal: AbortSignal| (for_refresh.refresh)(credential, signal),
			);
			let to_auth: SharedOAuthToAuth = Arc::new(move |credential: &OAuthCredential| {
				(for_to_auth.to_auth)(credential)
			});
			(
				base_oauth.name.clone(),
				base_oauth.is_subscription,
				base_oauth.login_label.clone(),
				login,
				refresh,
				to_auth,
			)
		}
	};

	// Wrap toAuth to apply configured headers/authHeader.
	let to_auth: OAuthToAuthFn = {
		let raw_headers = raw_headers.clone();
		Box::new(move |credential: &OAuthCredential| {
			let inner_to_auth = Arc::clone(&inner_to_auth);
			let raw_headers = raw_headers.clone();
			let provider_id = provider_id.clone();
			Box::pin(async move {
				let auth = inner_to_auth(credential).await?;
				let env: Option<BTreeMap<String, String>> = Some(
					credential
						.credentials
						.extra
						.iter()
						.filter_map(|(key, value)| value.as_str().map(|value| (key.clone(), value.to_string())))
						.collect(),
				);
				let headers = resolve_headers_or_throw(
					raw_headers.as_ref(),
					&format!("provider \"{provider_id}\""),
					env.as_ref(),
				)
				.map_err(ThrownValue::message)?;
				with_configured_auth(auth, headers.as_ref(), auth_header).map_err(ThrownValue::message)
			})
		})
	};

	Some(Arc::new(OAuthAuth { name, is_subscription, login_label, login, refresh, to_auth }))
}

/// Raw per-model headers from `models.json` definitions/overrides and the
/// extension definition (`rawModelHeaders`).
fn raw_model_headers(
	model: &AnyModel,
	config: Option<&ModelsJsonProvider>,
	extension: Option<&ProviderConfigInput>,
) -> Option<BTreeMap<String, String>> {
	// models.json definitions and overrides are chat-only. Extension definitions
	// are matched by operation and id so colliding models cannot share headers.
	let mut headers = BTreeMap::new();
	if let AnyModel::Chat(chat) = model {
		let override_headers = config
			.and_then(|config| config.model_overrides.as_ref())
			.and_then(|overrides| overrides.get(chat.id()))
			.and_then(|over| over.headers.as_ref());
		if let Some(override_headers) = override_headers {
			for (key, value) in override_headers {
				headers.insert(key.clone(), value.clone());
			}
		}
		let definition_headers = config
			.and_then(|config| config.models.as_ref())
			.and_then(|models| models.iter().find(|entry| entry.id == chat.id()))
			.and_then(|entry| entry.headers.as_ref());
		if let Some(definition_headers) = definition_headers {
			for (key, value) in definition_headers {
				headers.insert(key.clone(), value.clone());
			}
		}
	}
	let extension_model_headers = extension
		.and_then(|extension| extension.models.as_ref())
		.and_then(|models| {
			models.iter().find(|entry| {
				definition_type(entry) == model.model_type() && definition_base(entry).id == model.id()
			})
		})
		.and_then(definition_headers);
	if let Some(extension_model_headers) = extension_model_headers {
		for (key, value) in extension_model_headers {
			headers.insert(key.clone(), value.clone());
		}
	}
	(!headers.is_empty()).then_some(headers)
}

/// Validate an extension provider registration before it lands
/// (`validateExtensionProvider`).
pub fn validate_extension_provider(
	provider_id: &str,
	base: Option<&Arc<dyn Provider>>,
	models_config: Option<&ModelsJsonProvider>,
	extension: &ProviderConfigInput,
) -> Result<(), String> {
	if extension.stream_simple.is_some() && extension.api.is_none() {
		return Err(format!("Provider {provider_id}: \"api\" is required when registering streamSimple."));
	}
	let base_models = apply_models_json(provider_id, &get_all_provider_models(base), models_config)?;
	apply_extension(provider_id, &base_models, Some(extension))?;
	Ok(())
}

/// Mutable refresh-path state captured by the composed provider's
/// `refreshModels` (crate docs D50).
#[derive(Default)]
struct ComposedRefreshState {
	extension_oauth_credential: Option<OAuthCredential>,
	refreshed_extension_models: Option<Vec<ProviderModelConfig>>,
}

/// The composed provider object (the `provider` literal in
/// `composeModelProvider`).
struct ComposedProvider {
	provider_id: String,
	display_name: String,
	base_url: Option<String>,
	headers: Option<pi_ai::types::ProviderHeaders>,
	auth: ProviderAuth,
	base: Option<Arc<dyn Provider>>,
	config: Option<ModelsJsonProvider>,
	extension: Option<ProviderConfigInput>,
	has_refresh: bool,
	images: BTreeMap<String, Arc<dyn ProviderImages>>,
	classifiers: BTreeMap<String, Arc<dyn ProviderClassifier>>,
	refresh_state: Mutex<ComposedRefreshState>,
	self_weak: Weak<ComposedProvider>,
}

impl ComposedProvider {
	/// The base model list under `models.json` and extension layers, with
	/// `models.json` overrides applied last (`getAllModels`).
	fn get_all_models(&self) -> Result<Vec<AnyModel>, String> {
		let current_extension: Option<ProviderConfigInput> = {
			let state = self.refresh_state.lock();
			match (&self.extension, &state.refreshed_extension_models) {
				(Some(extension), Some(refreshed)) => {
					let mut current = extension.clone();
					current.models = Some(refreshed.clone());
					Some(current)
				}
				(Some(extension), None) => Some(extension.clone()),
				_ => None,
			}
		};
		let base_models =
			apply_models_json(&self.provider_id, &get_all_provider_models(self.base.as_ref()), self.config.as_ref())?;
		let mut models = apply_extension(&self.provider_id, &base_models, current_extension.as_ref())?;
		let oauth_hook = self
			.extension
			.as_ref()
			.and_then(|extension| extension.oauth.as_ref())
			.and_then(|oauth| oauth.modify_models.clone());
		if let Some(modify_models) = oauth_hook {
			let credential = self.refresh_state.lock().extension_oauth_credential.clone();
			if let Some(credential) = credential {
				// The extension hook is chat-only; other model types pass through untouched.
				let chat_models: Vec<Model> = models
					.iter()
					.filter_map(|model| match model {
						AnyModel::Chat(chat) => Some(chat.clone()),
						_ => None,
					})
					.collect();
				let projected = modify_models(chat_models, &credential.credentials);
				let mut combined: Vec<AnyModel> = projected.into_iter().map(AnyModel::Chat).collect();
				combined.extend(models.into_iter().filter(|model| model.model_type() != ModelType::Chat));
				models = combined;
			}
		}
		for model in models.iter_mut() {
			let AnyModel::Chat(chat) = model else { continue };
			if let Some(over) = self
				.config
				.as_ref()
				.and_then(|config| config.model_overrides.as_ref())
				.and_then(|overrides| overrides.get(chat.id()))
			{
				*model = AnyModel::Chat(apply_model_override(chat.clone(), over));
			}
		}
		Ok(models)
	}

	/// `refreshModels` when base/extension refresh hooks or the OAuth
	/// `modifyModels` projection exist (crate docs D50).
	fn run_refresh(self: Arc<Self>, context: RefreshModelsContext) -> BoxFuture<'static, Result<(), ThrownValue>> {
		Box::pin(async move {
			// Base refresh first.
			if let Some(base) = &self.base {
				if let Some(base_refresh) = base.refresh_models(clone_refresh_context(&context)) {
					base_refresh.await?;
				}
			}
			let mut refreshed: Option<Vec<ProviderModelConfig>> = None;
			if let Some(extension_refresh) =
				self.extension.as_ref().and_then(|extension| extension.refresh_models.clone())
			{
				refreshed = Some(extension_refresh(clone_refresh_context(&context)).await.map_err(ThrownValue::message)?);
			}
			if context.signal.aborted() {
				return Ok(());
			}
			let oauth_credential = match &context.credential {
				Some(Credential::OAuth(credential)) => Some(credential.clone()),
				_ => None,
			};
			let publish = context.publish.clone();
			let this = Arc::clone(&self);
			let update = Box::new(move || {
				let mut state = this.refresh_state.lock();
				if let Some(refreshed) = &refreshed {
					// Validate before publishing the new synchronous list.
					let mut candidate = this.extension.clone();
					if let Some(candidate) = candidate.as_mut() {
						candidate.models = Some(refreshed.clone());
					}
					let validated = apply_models_json(
						&this.provider_id,
						&get_all_provider_models(this.base.as_ref()),
						this.config.as_ref(),
					)
					.and_then(|models| apply_extension(&this.provider_id, &models, candidate.as_ref()));
					if validated.is_err() {
						return;
					}
					state.refreshed_extension_models = Some(refreshed.clone());
				}
				state.extension_oauth_credential = oauth_credential.clone();
			});
			publish.publish(ModelsPublication { persist: None, update: Some(update) }).await;
			Ok(())
		})
	}

	/// Shared dispatch for `stream`/`streamSimple` (`streamWith`).
	fn stream_dispatch(
		&self,
		model: &Model,
		context: &TranscriptContext,
		simple: bool,
		options_stream: Option<StreamOptions>,
		options_simple: Option<SimpleStreamOptions>,
	) -> Result<AssistantMessageEventStream, ThrownValue> {
		if let Some(extension) = &self.extension {
			if let (Some(stream_simple), Some(extension_api)) = (&extension.stream_simple, &extension.api) {
				if model.api() == extension_api.as_str() {
					let options = if simple {
						options_simple
					} else {
						// Upstream casts StreamOptions to SimpleStreamOptions.
						options_stream.map(|options| SimpleStreamOptions { base: options, ..SimpleStreamOptions::default() })
					};
					return Ok(stream_simple(model.clone(), context.clone(), options));
				}
			}
		}
		let supports_base_api = self
			.base
			.as_ref()
			.map(|base| base.get_models().iter().any(|entry| entry.api() == model.api()))
			.unwrap_or(false);
		if let Some(base) = &self.base {
			if supports_base_api {
				return Ok(if simple {
					let options = options_simple.expect("simple dispatch carries simple options");
					base.stream_simple(model, context, &options)
				} else {
					let options = options_stream.expect("full dispatch carries stream options");
					base.stream(model, context, &options)
				});
			}
		}
		let Some(api) = get_api_provider(model.api()) else {
			return Err(ThrownValue::message(format!("No API provider registered for api: {}", model.api())));
		};
		Ok(dispatch_api(&api, model, context, simple, options_stream, options_simple))
	}
}

fn dispatch_api(
	api: &ApiProviderInternal,
	model: &Model,
	context: &TranscriptContext,
	simple: bool,
	options_stream: Option<StreamOptions>,
	options_simple: Option<SimpleStreamOptions>,
) -> AssistantMessageEventStream {
	if simple {
		let options = options_simple.expect("simple dispatch carries simple options");
		(api.stream_simple)(model, context, &options)
	} else {
		let options = options_stream.expect("full dispatch carries stream options");
		(api.stream)(model, context, &options)
	}
}

/// Clone a [`RefreshModelsContext`] for a second consumer (upstream passes
/// the same object to base and extension refreshes).
fn clone_refresh_context(context: &RefreshModelsContext) -> RefreshModelsContext {
	RefreshModelsContext {
		credential: context.credential.clone(),
		stored: context.stored.clone(),
		publish: context.publish.clone(),
		allow_network: context.allow_network,
		force: context.force,
		signal: context.signal.clone(),
	}
}

impl Provider for ComposedProvider {
	fn id(&self) -> &str {
		&self.provider_id
	}

	fn name(&self) -> &str {
		&self.display_name
	}

	fn base_url(&self) -> Option<&str> {
		self.base_url.as_deref()
	}

	fn headers(&self) -> Option<&pi_ai::types::ProviderHeaders> {
		self.headers.as_ref()
	}

	fn auth(&self) -> &ProviderAuth {
		&self.auth
	}

	fn get_models(&self) -> Vec<Model> {
		self.get_all_models()
			.unwrap_or_default()
			.into_iter()
			.filter_map(|model| match model {
				AnyModel::Chat(model) => Some(model),
				_ => None,
			})
			.collect()
	}

	fn get_all_models(&self) -> Vec<AnyModel> {
		self.get_all_models().unwrap_or_default()
	}

	fn refresh_models(
		&self,
		context: RefreshModelsContext,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		if !self.has_refresh {
			return None;
		}
		let this = self.self_weak.upgrade()?;
		// The returned future is 'static (it owns an Arc handle); the trait's
		// elided '_ borrow is satisfied trivially.
		Some(Box::pin(async move { this.run_refresh(context).await }))
	}

	fn filter_models(&self, models: &[Model], credential: Option<&Credential>) -> Option<Vec<Model>> {
		self.base.as_ref()?.filter_models(models, credential)
	}

	fn filter_all_models(&self, models: &[AnyModel], credential: Option<&Credential>) -> Option<Vec<AnyModel>> {
		self.base.as_ref()?.filter_all_models(models, credential)
	}

	fn stream(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
		let this = self.self_weak.clone();
		let setup_model = model.clone();
		let model = model.clone();
		let context = context.clone();
		let options = options.clone();
		pi_ai::api::lazy::lazy_stream(&setup_model, async move {
			let this = this.upgrade().ok_or_else(|| ThrownValue::message("provider dropped"))?;
			this.stream_dispatch(&model, &context, false, Some(options), None)
		})
	}

	fn stream_simple(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: &SimpleStreamOptions,
	) -> AssistantMessageEventStream {
		let this = self.self_weak.clone();
		let setup_model = model.clone();
		let model = model.clone();
		let context = context.clone();
		let options = options.clone();
		pi_ai::api::lazy::lazy_stream(&setup_model, async move {
			let this = this.upgrade().ok_or_else(|| ThrownValue::message("provider dropped"))?;
			this.stream_dispatch(&model, &context, true, None, Some(options))
		})
	}

	fn fetch_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: &DeferredFetchOptions,
	) -> Option<AssistantMessageEventStream> {
		self.base.as_ref()?.fetch_deferred(model, handle, options)
	}

	fn cancel_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: &DeferredCancelOptions,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		self.base.as_ref()?.cancel_deferred(model, handle, options)
	}

	/// Present when a base provider exists (crate docs D50: trait objects
	/// cannot report capability overrides).
	fn generate_images<'a>(
		&'a self,
		model: &'a ImageModel,
		context: &'a ImagesContext,
		options: &'a ImagesOptions,
	) -> Option<Pin<Box<dyn Future<Output = pi_ai::types::AssistantImages> + Send + 'a>>> {
		let implementation = self.images.get(model.api()).cloned();
		let base = self.base.clone();
		let provider_id = self.provider_id.clone();
		if implementation.is_none() && base.is_none() {
			return None;
		}
		Some(Box::pin(async move {
			if let Some(implementation) = implementation {
				return implementation.generate_images(model, context, options).await;
			}
			if let Some(base) = base {
				if let Some(future) = base.generate_images(model, context, options) {
					return future.await;
				}
			}
			image_error_result(
				model,
				&ThrownValue::message(format!(
					"Provider {provider_id} has no image implementation for \"{}\"",
					model.api()
				)),
				false,
			)
		}))
	}

	/// Present when a base provider exists (crate docs D50).
	fn classify<'a>(
		&'a self,
		model: &'a ClassifierModel,
		context: &'a pi_ai::types::ClassifierContext,
		options: &'a ClassifierOptions,
	) -> Option<Pin<Box<dyn Future<Output = pi_ai::types::ClassifierResult> + Send + 'a>>> {
		let implementation = self.classifiers.get(model.api()).cloned();
		let base = self.base.clone();
		let provider_id = self.provider_id.clone();
		if implementation.is_none() && base.is_none() {
			return None;
		}
		Some(Box::pin(async move {
			if let Some(implementation) = implementation {
				return implementation.classify(model, context, options).await;
			}
			if let Some(base) = base {
				if let Some(future) = base.classify(model, context, options) {
					return future.await;
				}
			}
			classifier_error_result(
				model,
				&ThrownValue::message(format!(
					"Provider {provider_id} has no classifier implementation for \"{}\"",
					model.api()
				)),
				false,
			)
		}))
	}
}

/// Compose built-in, `models.json`, and extension layers without reading
/// credentials (`composeModelProvider`).
pub fn compose_model_provider(
	provider_id: &str,
	base: Option<Arc<dyn Provider>>,
	model_config: &ModelConfig,
	extension: Option<ProviderConfigInput>,
) -> Result<Arc<dyn Provider>, String> {
	let config = model_config.get_provider(provider_id).cloned();

	let display_name = extension
		.as_ref()
		.and_then(|extension| extension.name.clone())
		.or_else(|| config.as_ref().and_then(|config| config.name.clone()))
		.or_else(|| base.as_ref().map(|base| base.name().to_string()))
		.or_else(|| {
			extension
				.as_ref()
				.and_then(|extension| extension.oauth.as_ref())
				.map(|oauth| oauth.name.clone())
		})
		.unwrap_or_else(|| provider_id.to_string());
	let base_url = extension
		.as_ref()
		.and_then(|extension| extension.base_url.clone())
		.or_else(|| config.as_ref().and_then(|config| config.base_url.clone()))
		.or_else(|| base.as_ref().and_then(|base| base.base_url().map(str::to_string)));
	let headers = base.as_ref().and_then(|base| base.headers().cloned());

	let api_key = compose_api_key_auth(provider_id, base.as_ref(), config.as_ref(), extension.as_ref());
	let oauth = compose_oauth_auth(provider_id, base.as_ref(), config.as_ref(), extension.as_ref());
	if api_key.is_none() && oauth.is_none() {
		return Err(format!("Provider {provider_id}: no authentication method configured."));
	}
	let auth = ProviderAuth { api_key, oauth };

	let has_refresh = base.is_some()
		|| extension
			.as_ref()
			.is_some_and(|extension| extension.refresh_models.is_some() || extension.oauth.as_ref().is_some_and(|oauth| oauth.modify_models.is_some()));
	let images = extension
		.as_ref()
		.and_then(|extension| extension.images.clone())
		.unwrap_or_default();
	let classifiers = extension
		.as_ref()
		.and_then(|extension| extension.classifiers.clone())
		.unwrap_or_default();

	let composed = Arc::new_cyclic(|self_weak: &Weak<ComposedProvider>| ComposedProvider {
		provider_id: provider_id.to_string(),
		display_name,
		base_url,
		headers,
		auth,
		base,
		config,
		extension,
		has_refresh,
		images,
		classifiers,
		refresh_state: Mutex::new(ComposedRefreshState::default()),
		self_weak: self_weak.clone(),
	});

	// Validate eagerly so registration/reload reports structural errors immediately.
	composed.get_all_models()?;

	Ok(composed)
}

/// Resolved headers for one model request
/// (`resolveConfiguredModelHeaders`).
pub fn resolve_configured_model_headers(
	model: &AnyModel,
	config: Option<&ModelsJsonProvider>,
	extension: Option<&ProviderConfigInput>,
	env: Option<&BTreeMap<String, String>>,
) -> Result<Option<BTreeMap<String, String>>, String> {
	resolve_headers_or_throw(
		raw_model_headers(model, config, extension).as_ref(),
		&format!("model \"{}/{}\"", model.provider(), model.id()),
		env,
	)
}

/// Per-request compatibility config (`CompatibilityRequestConfig`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompatibilityRequestConfig {
	pub headers: Option<pi_ai::types::ProviderHeaders>,
	pub auth_header: bool,
}

/// Headers and `authHeader` flag for one request
/// (`resolveCompatibilityRequestConfig`).
pub fn resolve_compatibility_request_config(
	model: &AnyModel,
	config: Option<&ModelsJsonProvider>,
	extension: Option<&ProviderConfigInput>,
) -> Result<CompatibilityRequestConfig, String> {
	// `{ ...configuredHeaders(...), ...rawModelHeaders(...) }`
	let mut merged = configured_headers(config, extension).unwrap_or_default();
	for (key, value) in raw_model_headers(model, config, extension).unwrap_or_default() {
		merged.insert(key, value);
	}
	let configured = resolve_headers_or_throw(
		Some(&merged),
		&format!("model \"{}/{}\"", model.provider(), model.id()),
		None,
	)?;
	let mut configured_headers_map: pi_ai::types::ProviderHeaders = configured
		.map(|headers| headers.into_iter().map(|(key, value)| (key, Some(value))).collect())
		.unwrap_or_default();
	let model_headers_set = any_model_base(model).headers.is_some();
	if let Some(model_headers) = &any_model_base(model).headers {
		for (key, value) in model_headers {
			configured_headers_map.insert(key.clone(), Some(value.clone()));
		}
	}
	let headers = (model_headers_set || !configured_headers_map.is_empty()).then_some(configured_headers_map);
	let auth_header = extension
		.and_then(|extension| extension.auth_header)
		.or(config.and_then(|config| config.auth_header))
		.unwrap_or(false);
	Ok(CompatibilityRequestConfig { headers, auth_header })
}

/// Whether a models.json/extension API key is configured, and from where
/// (`configuredRequestAuthStatus`).
pub fn configured_request_auth_status(
	config: Option<&ModelsJsonProvider>,
	extension: Option<&ProviderConfigInput>,
) -> Option<AuthStatus> {
	let value = configured_api_key(config, extension)?;
	if is_command_config_value(&value) {
		return Some(AuthStatus {
			configured: true,
			source: Some(AuthStatusSource::ModelsJsonCommand),
			label: None,
		});
	}
	let names = get_config_value_env_var_names(&value);
	if !names.is_empty() {
		return Some(if is_config_value_configured(&value, None) {
			AuthStatus {
				configured: true,
				source: Some(AuthStatusSource::Environment),
				label: Some(names.join(", ")),
			}
		} else {
			AuthStatus { configured: false, source: None, label: None }
		});
	}
	Some(AuthStatus {
		configured: true,
		source: Some(if extension.is_some_and(|extension| extension.api_key.is_some()) {
			AuthStatusSource::Fallback
		} else {
			AuthStatusSource::ModelsJsonKey
		}),
		label: None,
	})
}
