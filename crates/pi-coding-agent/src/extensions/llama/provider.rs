//! Port of `packages/coding-agent/src/extensions/llama/provider.ts` — the
//! native `llama.cpp` pi-ai provider: api-key auth against a llama.cpp
//! server, dynamic model catalog from `/models`, chat models on the
//! `openai-completions` API, and classifiers on `llama-cpp-classify`.
//!
//! # Deviations (D97/D98 range, shared with [`super`])
//!
//! - **D98d** — upstream's `Provider<"openai-completions">` object literal
//!   becomes [`LlamaProvider`] implementing `pi_ai::Provider`:
//!   `stream`/`streamSimple` delegate to the `pi-ai` `openai-completions`
//!   module exports (the upstream `pi-ai/compat` import), `classify` to the
//!   `llama-cpp-classify` API (upstream `llamaCppClassifyApi`).
//! - **D98e** — `setCatalog`'s upstream `options.routerAutoload` boolean
//!   becomes a plain `bool` parameter.
//! - **D98f** — the model catalog lives in a shared `parking_lot::RwLock`;
//!   the `publish({ update })` closures clone the handle like upstream's
//!   captured variables.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use pi_ai::auth::types::{
	ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthCheck, AuthContext, AuthPrompt, AuthPromptKind, AuthResult,
	AuthType, ModelAuth, ProviderAuth, ProviderAuthInteraction,
};
use pi_ai::models::{ModelsPublication, ModelsStoreWrite, Provider, RefreshModelsContext};
use pi_ai::types::{
	AnyModel, BaseModelFields, ClassifierModel, ClassifierTypeTag, Model, ModelCompat, ModelCost, ModelCostRates,
	ModelInput, ThinkingFormat, ThinkingLevelMap,
};
use pi_ai::utils::abort::AbortSignal;
use pi_ai::utils::thrown::ThrownValue;
use pi_agent::BoxFuture;
use futures_util::TryStreamExt;

use super::client::{
	inference_url_of, normalize_llama_server_url, LlamaClient, LlamaModelInfo, LlamaModelStatus, LlamaServerProps,
};

/// Upstream `LLAMA_PROVIDER_ID`.
pub const LLAMA_PROVIDER_ID: &str = "llama.cpp";
/// Upstream `DEFAULT_LLAMA_SERVER_URL`.
pub const DEFAULT_LLAMA_SERVER_URL: &str = "http://127.0.0.1:8080";

/// Classifier API id (upstream `"llama-cpp-classify"`).
const LLAMA_CLASSIFY_API: &str = "llama-cpp-classify";

/// `LLAMA_BASE_URL` key inside `credential.env`.
const LLAMA_BASE_URL_KEY: &str = "LLAMA_BASE_URL";

/// Render a [`ThrownValue`] as the error string the JS paths would carry.
pub(crate) fn thrown_text(error: &ThrownValue) -> String {
	match error {
		ThrownValue::Text(text) => text.clone(),
		ThrownValue::Error(error) => error.to_string(),
	}
}

/// Upstream `credentialServerUrl`.
fn credential_server_url(credential: Option<&ApiKeyCredential>) -> Option<String> {
	let value = credential
		.and_then(|credential| credential.env.as_ref())
		.and_then(|env| env.get(LLAMA_BASE_URL_KEY))?;
	let value = value.trim();
	if value.is_empty() {
		return None;
	}
	normalize_llama_server_url(value).ok()
}

/// Upstream `resolveServerUrl`.
async fn resolve_server_url(context: &dyn AuthContext, credential: Option<&ApiKeyCredential>) -> Option<String> {
	let configured = match credential_server_url(credential) {
		Some(url) => Some(url),
		None => context.env(LLAMA_BASE_URL_KEY).await.map(|value| value.trim().to_string()),
	};
	match configured {
		Some(configured) if !configured.is_empty() => normalize_llama_server_url(&configured).ok(),
		_ => None,
	}
}

/// Upstream `modelIsSelectable`.
fn model_is_selectable(model: &LlamaModelInfo, router_autoload: bool) -> bool {
	match model.status.value {
		LlamaModelStatus::Loaded => true,
		// llama.cpp reports idle-slept models as "sleeping"; requests wake them automatically.
		LlamaModelStatus::Sleeping => true,
		// Unloaded presets are routable only when llama.cpp router autoload can load them on first use.
		LlamaModelStatus::Unloaded => {
			router_autoload && model.status.failed != Some(true) && model.source.as_deref() == Some("preset")
		}
		_ => false,
	}
}

/// Upstream `routerAutoloadEnabled`.
async fn router_autoload_enabled(client: &LlamaClient, catalog: &[LlamaModelInfo], signal: &AbortSignal) -> bool {
	let has_unloaded_preset = catalog
		.iter()
		.any(|model| model.status.value == LlamaModelStatus::Unloaded && model.source.as_deref() == Some("preset"));
	if !has_unloaded_preset {
		return false;
	}
	match client.props(None, Some(signal)).await {
		Ok(props) => props.models_autoload == Some(true),
		Err(_) => false,
	}
}

/// Upstream `configuredContextWindow`: `--ctx-size`/`-c`/`-ctx` in the
/// server launch args.
fn configured_context_window(model: &LlamaModelInfo) -> Option<u64> {
	let args = model.status.args.as_deref()?;
	if args.len() < 2 {
		return None;
	}
	for index in 0..args.len() - 1 {
		if args[index] != "--ctx-size" && args[index] != "-c" && args[index] != "-ctx" {
			continue;
		}
		let Ok(value) = args[index + 1].parse::<f64>() else {
			continue;
		};
		// `Number.isSafeInteger(contextWindow) && contextWindow > 0`.
		if value.fract() == 0.0 && value > 0.0 && value <= 9_007_199_254_740_991.0 {
			return Some(value as u64);
		}
	}
	None
}

/// Upstream `contextWindowOf`.
fn context_window_of(model: &LlamaModelInfo, cached_context_window: Option<u64>) -> u64 {
	let meta = model.meta.as_ref();
	if let Some(runtime) = meta.and_then(|meta| meta.n_ctx).filter(|value| *value > 0) {
		return runtime;
	}
	if let Some(configured) = configured_context_window(model).filter(|value| *value > 0) {
		return configured;
	}
	if let Some(cached) = cached_context_window.filter(|value| *value > 0) {
		return cached;
	}
	let training = meta.and_then(|meta| meta.n_ctx_train).filter(|value| *value > 0);
	training.unwrap_or(128_000)
}

/// Zero cost, like upstream.
fn zero_cost() -> ModelCost {
	ModelCost {
		rates: ModelCostRates { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0 },
		tiers: None,
	}
}

/// Upstream `thinkingLevelMap` for reasoning-capable chat templates.
fn reasoning_thinking_level_map() -> ThinkingLevelMap {
	use pi_ai::types::ModelThinkingLevel;
	[
		(ModelThinkingLevel::Off, Some("off".to_string())),
		(ModelThinkingLevel::Minimal, None),
		(ModelThinkingLevel::Low, None),
		(ModelThinkingLevel::Medium, Some("medium".to_string())),
		(ModelThinkingLevel::High, None),
		(ModelThinkingLevel::Xhigh, None),
	]
	.into_iter()
	.collect()
}

/// The same llama.cpp model used as a classifier: answers are read from
/// next-token label probabilities (upstream `toPiClassifierModel`).
fn to_pi_classifier_model(model: &LlamaModelInfo, server_url: &str, cached_context_window: Option<u64>) -> ClassifierModel {
	ClassifierModel {
		base: BaseModelFields {
			id: model.id.clone(),
			name: model.id.clone(),
			api: LLAMA_CLASSIFY_API.to_string(),
			provider: LLAMA_PROVIDER_ID.to_string(),
			// The classifier talks to the management server, not `/v1`.
			base_url: server_url.to_string(),
			input: vec![ModelInput::Text],
			input_limits: None,
			cost: zero_cost(),
			headers: None,
		},
		model_type: ClassifierTypeTag::TAG,
		context_window: context_window_of(model, cached_context_window),
	}
}

/// Upstream `toPiModel`.
fn to_pi_model(
	model: &LlamaModelInfo,
	server_url: &str,
	props: Option<&LlamaServerProps>,
	cached_context_window: Option<u64>,
) -> Model {
	let context_window = context_window_of(model, cached_context_window);
	let reasoning = props
		.and_then(|props| props.chat_template.as_deref())
		.is_some_and(|template| template.contains("enable_thinking"));
	Model {
		base: BaseModelFields {
			id: model.id.clone(),
			name: model.id.clone(),
			api: "openai-completions".to_string(),
			provider: LLAMA_PROVIDER_ID.to_string(),
			base_url: inference_url_of(server_url),
			input: if model
				.architecture
				.as_ref()
				.and_then(|architecture| architecture.input_modalities.as_ref())
				.is_some_and(|modalities| modalities.iter().any(|modality| modality == "image"))
			{
				vec![ModelInput::Text, ModelInput::Image]
			} else {
				vec![ModelInput::Text]
			},
			input_limits: None,
			cost: zero_cost(),
			headers: None,
		},
		model_type: None,
		reasoning,
		thinking_level_map: reasoning.then(reasoning_thinking_level_map),
		prompt_cache: None,
		context_window,
		max_tokens: context_window,
		sampling_params: None,
		sampling_params_by_thinking_level: None,
		compat: Some(ModelCompat {
			supports_store: Some(false),
			supports_developer_role: Some(false),
			supports_reasoning_effort: Some(false),
			supports_usage_in_streaming: Some(true),
			supports_strict_mode: Some(false),
			max_tokens_field: Some(pi_ai::types::MaxTokensField::MaxTokens),
			thinking_format: reasoning.then_some(ThinkingFormat::QwenChatTemplate),
			..Default::default()
		}),
	}
}

/// The in-memory catalog behind [`LlamaProvider`].
#[derive(Default)]
struct LlamaCatalog {
	models: Vec<Model>,
	classifiers: Vec<ClassifierModel>,
}

/// Native llama.cpp provider (upstream `Provider<"openai-completions">`).
pub struct LlamaProvider {
	catalog: Arc<parking_lot::RwLock<LlamaCatalog>>,
	base_url: String,
	auth: ProviderAuth,
	classifier: Arc<dyn pi_ai::ProviderClassifier>,
}

impl std::fmt::Debug for LlamaProvider {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.debug_struct("LlamaProvider").field("id", &LLAMA_PROVIDER_ID).finish_non_exhaustive()
	}
}

impl Provider for LlamaProvider {
	fn id(&self) -> &str {
		LLAMA_PROVIDER_ID
	}

	fn name(&self) -> &str {
		"llama.cpp"
	}

	fn base_url(&self) -> Option<&str> {
		Some(&self.base_url)
	}

	fn auth(&self) -> &ProviderAuth {
		&self.auth
	}

	fn get_models(&self) -> Vec<Model> {
		self.catalog.read().models.clone()
	}

	fn get_all_models(&self) -> Vec<AnyModel> {
		let catalog = self.catalog.read();
		catalog
			.models
			.iter()
			.map(|model| AnyModel::Chat(model.clone()))
			.chain(catalog.classifiers.iter().map(|model| AnyModel::Classifier(model.clone())))
			.collect()
	}

	fn refresh_models(
		&self,
		context: RefreshModelsContext,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		let catalog = Arc::clone(&self.catalog);
		Some(Box::pin(async move { refresh_catalog(&catalog, context).await }))
	}

	fn stream(
		&self,
		model: &Model,
		context: &pi_ai::types::TranscriptContext,
		options: &pi_ai::types::StreamOptions,
	) -> pi_ai::AssistantMessageEventStream {
		// Upstream imports `stream` from `pi-ai/compat` — D98d.
		pi_ai::api::openai_completions::stream(model, context, options)
	}

	fn stream_simple(
		&self,
		model: &Model,
		context: &pi_ai::types::TranscriptContext,
		options: &pi_ai::types::SimpleStreamOptions,
	) -> pi_ai::AssistantMessageEventStream {
		pi_ai::api::openai_completions::stream_simple(model, context, options)
	}

	fn classify<'a>(
		&'a self,
		model: &'a ClassifierModel,
		context: &'a pi_ai::types::ClassifierContext,
		options: &'a pi_ai::types::ClassifierOptions,
	) -> Option<Pin<Box<dyn Future<Output = pi_ai::types::ClassifierResult> + Send + 'a>>> {
		Some(self.classifier.classify(model, context, options))
	}
}

/// Upstream `refreshModels` body.
async fn refresh_catalog(
	catalog: &Arc<parking_lot::RwLock<LlamaCatalog>>,
	context: RefreshModelsContext,
) -> Result<(), ThrownValue> {
	let mut cached_context_windows: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
	if let Some(stored) = context.stored.as_ref() {
		let restored: Vec<Model> = stored
			.models
			.iter()
			.filter_map(|model| match model {
				AnyModel::Chat(model)
					if model.provider() == LLAMA_PROVIDER_ID && model.api() == "openai-completions" =>
				{
					Some(model.clone())
				}
				_ => None,
			})
			.collect();
		let restored_classifiers: Vec<ClassifierModel> = stored
			.models
			.iter()
			.filter_map(|model| match model {
				AnyModel::Classifier(model)
					if model.provider() == LLAMA_PROVIDER_ID && model.api() == LLAMA_CLASSIFY_API =>
				{
					Some(model.clone())
				}
				_ => None,
			})
			.collect();
		for model in &restored {
			cached_context_windows.insert(model.id().to_string(), model.context_window);
		}
		for model in &restored_classifiers {
			cached_context_windows.insert(model.id().to_string(), model.context_window);
		}
		let models = restored.clone();
		let classifiers = restored_classifiers.clone();
		let catalog_for_update = Arc::clone(catalog);
		let accepted = context
			.publish
			.publish(ModelsPublication {
				persist: None,
				update: Some(Box::new(move || {
					let mut guard = catalog_for_update.write();
					guard.models = models;
					guard.classifiers = classifiers;
				})),
			})
			.await;
		if !accepted {
			return Ok(());
		}
	}

	let Some(pi_ai::auth::types::Credential::ApiKey(credential)) = context.credential.as_ref() else {
		return Ok(());
	};
	if !context.allow_network || context.signal.aborted() {
		return Ok(());
	}
	let Some(server_url) = credential_server_url(Some(credential)) else {
		return Ok(());
	};
	let client = LlamaClient::new(&server_url, credential.key.as_deref()).map_err(ThrownValue::message)?;
	let list = client.list(false, Some(&context.signal)).await.map_err(ThrownValue::message)?;
	if context.signal.aborted() {
		return Ok(());
	}
	let router_autoload = router_autoload_enabled(&client, &list, &context.signal).await;
	if context.signal.aborted() {
		return Ok(());
	}
	let selectable: Vec<LlamaModelInfo> = list
		.iter()
		.filter(|model| model_is_selectable(model, router_autoload))
		.cloned()
		.collect();
	// Upstream Promise.all — the per-model `/props` queries run concurrently.
	let prop_queries: futures_util::stream::FuturesOrdered<_> = selectable
		.into_iter()
		.map(|model| {
			let client = client.clone();
			let server_url = server_url.clone();
			let signal = context.signal.clone();
			let cached = cached_context_windows.get(&model.id).copied();
			// Only loaded models expose their template without side effects. Unloaded autoload presets
			// would need to be loaded, while querying sleeping models may wake them. Those models remain
			// unclassified until they are loaded or woken and a later catalog refresh discovers them.
			async move {
				let props = if model.status.value != LlamaModelStatus::Loaded {
					None
				} else {
					Some(client.props(Some(&model.id), Some(&signal)).await.map_err(ThrownValue::message)?)
				};
				Ok::<_, ThrownValue>((
					to_pi_model(&model, &server_url, props.as_ref(), cached),
					to_pi_classifier_model(&model, &server_url, cached),
				))
			}
		})
		.collect();
	let pairs = prop_queries.try_collect::<Vec<_>>().await?;
	let refreshed: Vec<Model> = pairs.iter().map(|(model, _)| model.clone()).collect();
	let refreshed_classifiers: Vec<ClassifierModel> = pairs.into_iter().map(|(_, classifier)| classifier).collect();
	if context.signal.aborted() {
		return Ok(());
	}
	let models = refreshed.clone();
	let classifiers = refreshed_classifiers.clone();
	let entry_models: Vec<AnyModel> = refreshed
		.into_iter()
		.map(AnyModel::Chat)
		.chain(refreshed_classifiers.into_iter().map(AnyModel::Classifier))
		.collect();
	let catalog_for_update = Arc::clone(catalog);
	context
		.publish
		.publish(ModelsPublication {
			persist: Some(ModelsStoreWrite::Write(pi_ai::ModelsStoreEntry {
				models: entry_models,
				checked_at: Some(
					std::time::SystemTime::now()
						.duration_since(std::time::UNIX_EPOCH)
						.map(|duration| duration.as_millis() as i64)
						.unwrap_or(0),
				),
				..Default::default()
			})),
			update: Some(Box::new(move || {
				let mut guard = catalog_for_update.write();
				guard.models = models;
				guard.classifiers = classifiers;
			})),
		})
		.await;
	Ok(())
}

/// Controller returned by [`create_llama_provider`]
/// (`LlamaProviderController`).
pub struct LlamaProviderController {
	/// The registered provider (upstream `provider`).
	pub provider: Arc<LlamaProvider>,
}

impl LlamaProviderController {
	/// Upstream `setCatalog(models, serverUrl, options)` (D98e).
	pub fn set_catalog(&self, catalog: &[LlamaModelInfo], server_url: &str, router_autoload: bool) {
		let selectable: Vec<&LlamaModelInfo> =
			catalog.iter().filter(|model| model_is_selectable(model, router_autoload)).collect();
		let models: Vec<Model> = selectable.iter().map(|model| to_pi_model(model, server_url, None, None)).collect();
		let classifiers: Vec<ClassifierModel> =
			selectable.iter().map(|model| to_pi_classifier_model(model, server_url, None)).collect();
		let mut guard = self.provider.catalog.write();
		guard.models = models;
		guard.classifiers = classifiers;
	}

	/// Convenience passthrough to `provider.getModels()`.
	pub fn get_models(&self) -> Vec<Model> {
		self.provider.get_models()
	}

	/// Convenience passthrough to `provider.getAllModels()`.
	pub fn get_all_models(&self) -> Vec<AnyModel> {
		self.provider.get_all_models()
	}
}

/// Build the llama.cpp provider (`createLlamaProvider`).
pub fn create_llama_provider() -> LlamaProviderController {
	let provider = Arc::new(LlamaProvider {
		catalog: Arc::new(parking_lot::RwLock::new(LlamaCatalog::default())),
		base_url: inference_url_of(DEFAULT_LLAMA_SERVER_URL),
		auth: ProviderAuth {
			api_key: Some(ApiKeyAuth {
				name: "llama.cpp server".to_string(),
				login: Some(Box::new(
					|interaction: ProviderAuthInteraction| -> BoxFuture<'static, Result<ApiKeyCredential, ThrownValue>> {
						Box::pin(async move {
						let entered = (interaction.prompt)(AuthPrompt {
							signal: Some(interaction.signal.clone()),
							kind: AuthPromptKind::Text {
								message: "llama.cpp server URL".to_string(),
								placeholder: Some(
									std::env::var(LLAMA_BASE_URL_KEY)
										.ok()
										.filter(|value| !value.is_empty())
										.unwrap_or_else(|| DEFAULT_LLAMA_SERVER_URL.to_string()),
								),
							},
						})
						.await?;
						let entered = entered.trim();
						let configured = if !entered.is_empty() {
							entered.to_string()
						} else {
							std::env::var(LLAMA_BASE_URL_KEY)
								.ok()
								.filter(|value| !value.trim().is_empty())
								.unwrap_or_else(|| DEFAULT_LLAMA_SERVER_URL.to_string())
						};
						let server_url = normalize_llama_server_url(&configured).map_err(ThrownValue::message)?;
						let api_key = (interaction.prompt)(AuthPrompt {
							signal: Some(interaction.signal.clone()),
							kind: AuthPromptKind::Secret { message: "API key (optional)".to_string(), placeholder: None },
						})
						.await?
						.trim()
						.to_string();
						// Validate connectivity before storing the credential.
						LlamaClient::new(&server_url, Some(api_key.as_str()).filter(|key| !key.is_empty()))
							.map_err(ThrownValue::message)?
							.list(false, Some(&interaction.signal))
							.await
							.map_err(ThrownValue::message)?;
						Ok(ApiKeyCredential {
							key: if api_key.is_empty() { None } else { Some(api_key) },
							env: Some([(LLAMA_BASE_URL_KEY.to_string(), server_url)].into_iter().collect()),
						})
					})
				})),
				check: Some(Box::new(
					|input: ApiKeyAuthInput<'_>| -> BoxFuture<'_, Result<Option<AuthCheck>, ThrownValue>> {
						Box::pin(async move {
							let server_url = resolve_server_url(input.ctx.as_ref(), input.credential).await;
							Ok(server_url.map(|_| AuthCheck {
								source: Some(auth_source(input.credential).to_string()),
								check_type: AuthType::ApiKey,
							}))
						})
					},
				)),
				resolve: Box::new(
					|input: ApiKeyAuthInput<'_>| -> BoxFuture<'_, Result<Option<AuthResult>, ThrownValue>> {
						Box::pin(async move {
							let Some(server_url) = resolve_server_url(input.ctx.as_ref(), input.credential).await else {
								return Ok(None);
							};
							let ambient_key = input.ctx.env("LLAMA_API_KEY").await;
							let api_key = input
								.credential
								.and_then(|credential| credential.key.clone())
								.or(ambient_key)
								.unwrap_or_else(|| "local".to_string());
							Ok(Some(AuthResult {
								auth: ModelAuth {
									api_key: Some(api_key),
									headers: None,
									base_url: Some(inference_url_of(&server_url)),
								},
								env: {
									let mut env =
										input.credential.and_then(|credential| credential.env.clone()).unwrap_or_default();
									env.insert(LLAMA_BASE_URL_KEY.to_string(), server_url);
									Some(env)
								},
								source: Some(auth_source(input.credential).to_string()),
							}))
						})
					},
				),
			}),
			oauth: None,
		},
		classifier: pi_ai::api::llama_cpp_classify::llama_cpp_classify_api(),
	});
	LlamaProviderController { provider }
}

/// Upstream check/resolve `source` label.
fn auth_source(credential: Option<&ApiKeyCredential>) -> &'static str {
	if credential.is_some() {
		"stored credential"
	} else {
		LLAMA_BASE_URL_KEY
	}
}

// ============================================================================
// Tests — port of the provider cases from `test/llama-extension.test.ts`.
// ============================================================================

#[cfg(test)]
mod provider_tests {
	use super::*;
	use crate::extensions::llama::test_support::{listen, Handler, TestRequest, TestResponse};
	use pi_ai::ModelsStoreEntry;
	use pi_ai::models::PublishHandle;
	use serde_json::{json, Value};
	use std::collections::VecDeque;
	use std::sync::Arc;

	fn handler(function: impl Fn(TestRequest) -> BoxFuture<'static, TestResponse> + Send + Sync + 'static) -> Handler {
		Arc::new(function)
	}

	fn merge_json(target: &mut Value, source: &Value) {
		match (target, source) {
			(Value::Object(target), Value::Object(source)) => {
				for (key, entry) in source {
					merge_json(target.entry(key.clone()).or_insert(Value::Null), entry);
				}
			}
			(target, source) => *target = source.clone(),
		}
	}

	fn model_info(id: &str, status: LlamaModelStatus, extra: Value) -> LlamaModelInfo {
		let mut value = json!({ "id": id, "status": { "value": status } });
		// The extra object overrides fields (`status.args`), it does not
		// replace the whole `status` object.
		merge_json(&mut value, &extra);
		serde_json::from_value(value).expect("model info fixture")
	}

	type PersistStore = Arc<parking_lot::Mutex<Option<ModelsStoreEntry>>>;

	fn publish_handle(store: PersistStore) -> PublishHandle {
		PublishHandle::new(move |publication: ModelsPublication| {
			let mut guard = store.lock();
			if let Some(update) = publication.update {
				update();
			}
			match publication.persist {
				Some(ModelsStoreWrite::Write(entry)) => *guard = Some(entry),
				Some(ModelsStoreWrite::Delete) => *guard = None,
				None => {}
			}
			Box::pin(async { true })
		})
	}

	fn refresh_context(
		credential: Option<pi_ai::auth::types::Credential>,
		stored: Option<ModelsStoreEntry>,
		publish: PublishHandle,
		allow_network: bool,
	) -> RefreshModelsContext {
		RefreshModelsContext {
			credential,
			stored,
			publish,
			allow_network,
			force: None,
			signal: AbortSignal::new(),
		}
	}

	fn llama_credential(key: &str, base_url: &str) -> pi_ai::auth::types::Credential {
		pi_ai::auth::types::Credential::ApiKey(ApiKeyCredential {
			key: Some(key.to_string()),
			env: Some([(LLAMA_BASE_URL_KEY.to_string(), base_url.to_string())].into_iter().collect()),
		})
	}

	async fn refresh(
		controller: &LlamaProviderController,
		context: RefreshModelsContext,
	) -> Result<(), ThrownValue> {
		match controller.provider.refresh_models(context) {
			Some(future) => future.await,
			None => panic!("llama provider refresh_models must be present"),
		}
	}

	#[test]
	fn exposes_loaded_and_sleeping_models_with_router_metadata() {
		let controller = create_llama_provider();
		controller.set_catalog(
			&[
				model_info(
					"loaded",
					LlamaModelStatus::Loaded,
					json!({
						"status": { "args": ["llama-server", "--n-gpu-layers", "999"] },
						"architecture": { "input_modalities": ["text", "image"] },
						"meta": { "n_ctx": 65536, "n_ctx_train": 131072 },
					}),
				),
				model_info("sleeping", LlamaModelStatus::Sleeping, json!({})),
				model_info("unloaded", LlamaModelStatus::Unloaded, json!({})),
				model_info("loading", LlamaModelStatus::Loading, json!({})),
			],
			"http://localhost:8080",
			false,
		);

		let models = controller.get_models();
		assert_eq!(models.len(), 2);
		let loaded = &models[0];
		assert_eq!(loaded.base.id, "loaded");
		assert_eq!(loaded.base.base_url, "http://localhost:8080/v1");
		assert_eq!(loaded.context_window, 65536);
		assert_eq!(loaded.max_tokens, 65536);
		assert_eq!(loaded.base.input, vec![ModelInput::Text, ModelInput::Image]);
		let sleeping = &models[1];
		assert_eq!(sleeping.base.id, "sleeping");
		assert_eq!(sleeping.base.base_url, "http://localhost:8080/v1");
	}

	// Regression test for #9528.
	#[tokio::test]
	async fn discovers_chat_template_thinking_support_for_loaded_models() {
		let props_requests = Arc::new(parking_lot::Mutex::new(0usize));
		let request_counter = Arc::clone(&props_requests);
		let server = listen(handler(move |request| {
			let request_counter = Arc::clone(&request_counter);
			Box::pin(async move {
				match request.target.as_str() {
					"/models" => {
						TestResponse::Json(200, json!({ "data": [{ "id": "qwen", "status": { "value": "loaded" }, "meta": { "n_ctx": 32768 } }] }))
					}
					"/props?model=qwen&autoload=false" => {
						*request_counter.lock() += 1;
						TestResponse::Json(200, json!({ "chat_template": "{% if enable_thinking %}think{% endif %}" }))
					}
					_ => TestResponse::Empty(404),
				}
			})
		}))
		.await;

		let controller = create_llama_provider();
		refresh(
			&controller,
			refresh_context(
				Some(llama_credential("local", &server.url)),
				None,
				publish_handle(Arc::new(parking_lot::Mutex::new(None))),
				true,
			),
		)
		.await
		.expect("refresh succeeds");

		assert_eq!(*props_requests.lock(), 1);
		let models = controller.get_models();
		assert_eq!(models.len(), 1);
		let model = &models[0];
		assert_eq!(model.base.id, "qwen");
		assert!(model.reasoning);
		let thinking_level_map = model.thinking_level_map.clone().expect("thinking level map");
		assert_eq!(
			thinking_level_map
				.iter()
				.map(|(level, mapped)| (format!("{level:?}"), mapped.clone()))
				.collect::<Vec<_>>(),
			vec![
				("Off".to_string(), Some("off".to_string())),
				("Minimal".to_string(), None),
				("Low".to_string(), None),
				("Medium".to_string(), Some("medium".to_string())),
				("High".to_string(), None),
				("Xhigh".to_string(), None),
			]
		);
		assert_eq!(
			model.compat.as_ref().and_then(|compat| compat.thinking_format),
			Some(ThinkingFormat::QwenChatTemplate)
		);
	}

	#[tokio::test]
	async fn persists_and_restores_selectable_models_for_cache_only_startup_refreshes() {
		let store: PersistStore = Arc::new(parking_lot::Mutex::new(None));
		let server = listen(handler(move |request| {
			Box::pin(async move {
				match request.target.as_str() {
					"/models" => TestResponse::Json(
						200,
						json!({
							"data": [
								{ "id": "loaded", "status": { "value": "loaded" }, "meta": { "n_ctx": 32768 } },
								{ "id": "sleeping", "status": { "value": "sleeping" }, "meta": { "n_ctx": 32768 } },
								{ "id": "unloaded", "status": { "value": "unloaded" } },
							],
						}),
					),
					"/props?model=loaded&autoload=false" => TestResponse::Json(200, json!({})),
					_ => TestResponse::Empty(404),
				}
			})
		}))
		.await;

		let first = create_llama_provider();
		refresh(&first, refresh_context(Some(llama_credential("local", &server.url)), None, publish_handle(Arc::clone(&store)), true))
			.await
			.expect("first refresh succeeds");
		assert_eq!(first.get_models().iter().map(|model| model.base.id.clone()).collect::<Vec<_>>(), ["loaded", "sleeping"]);
		let entry = store.lock().clone().expect("persisted entry");
		assert_eq!(
			entry
				.models
				.iter()
				.map(|model| (model.id().to_string(), model.api().to_string()))
				.collect::<Vec<_>>(),
			[
				("loaded".to_string(), "openai-completions".to_string()),
				("sleeping".to_string(), "openai-completions".to_string()),
				("loaded".to_string(), "llama-cpp-classify".to_string()),
				("sleeping".to_string(), "llama-cpp-classify".to_string()),
			]
		);

		let second = create_llama_provider();
		// The `MutexGuard` must not be a temporary here: it would live across
		// the `.await` and self-deadlock the publish's `store.lock()`.
		let stored = store.lock().clone();
		refresh(
			&second,
			refresh_context(Some(llama_credential("local", &server.url)), stored, publish_handle(Arc::clone(&store)), false),
		)
		.await
		.expect("second refresh succeeds");
		let models = second.get_models();
		assert_eq!(models.iter().map(|model| model.base.id.as_str()).collect::<Vec<_>>(), ["loaded", "sleeping"]);
		for model in &models {
			assert_eq!(model.base.base_url, format!("{}/v1", server.url));
			assert_eq!(model.context_window, 32768);
		}
		let classifiers: Vec<pi_ai::types::ClassifierModel> = second
			.get_all_models()
			.into_iter()
			.filter_map(|model| match model {
				AnyModel::Classifier(classifier) => Some(classifier),
				_ => None,
			})
			.collect();
		assert_eq!(classifiers.len(), 2);
		assert_eq!(
			classifiers.iter().map(|classifier| classifier.base.id.as_str()).collect::<Vec<_>>(),
			["loaded", "sleeping"]
		);
		for classifier in &classifiers {
			assert_eq!(classifier.base.api, "llama-cpp-classify");
			assert_eq!(classifier.base.base_url, server.url);
			assert_eq!(classifier.context_window, 32768);
		}
	}

	#[tokio::test]
	async fn preserves_cached_llama_cpp_context_for_unloaded_autoload_presets() {
		let state = Arc::new(parking_lot::Mutex::new((true, None::<Vec<String>>)));
		let store: PersistStore = Arc::new(parking_lot::Mutex::new(None));
		let state_for_handler = Arc::clone(&state);
		let server = listen(handler(move |request| {
			let state = Arc::clone(&state_for_handler);
			Box::pin(async move {
				match request.target.as_str() {
					"/models" => {
						let (loaded, unloaded_args) = state.lock().clone();
						let entry = if loaded {
							json!({
								"id": "qwen",
								"status": { "value": "loaded" },
								"source": "preset",
								"meta": { "n_ctx": 65536, "n_ctx_train": 128000 },
							})
						} else {
							let mut status = json!({ "value": "unloaded" });
							if let Some(args) = unloaded_args {
								status["args"] = json!(args);
							}
							json!({
								"id": "qwen",
								"status": status,
								"source": "preset",
								"meta": { "n_ctx_train": 128000 },
							})
						};
						TestResponse::Json(200, json!({ "data": [entry] }))
					}
					"/props?model=qwen&autoload=false" => TestResponse::Json(200, json!({})),
					"/props" => TestResponse::Json(200, json!({ "role": "router", "models_autoload": true })),
					_ => TestResponse::Empty(404),
				}
			})
		}))
		.await;
		let credential = llama_credential("local", &server.url);

		let first = create_llama_provider();
		refresh(&first, refresh_context(Some(credential.clone()), None, publish_handle(Arc::clone(&store)), true))
			.await
			.expect("first refresh succeeds");
		let stored_windows = || {
			store
				.lock()
				.as_ref()
				.map(|entry| {
					entry
						.models
						.iter()
						.map(|model| match model {
							AnyModel::Chat(model) => model.context_window,
							AnyModel::Classifier(model) => model.context_window,
							AnyModel::Image(_) => 0,
						})
						.collect::<Vec<_>>()
				})
				.unwrap_or_default()
		};
		assert_eq!(stored_windows(), [65536, 65536]);

		*state.lock() = (false, None);
		let second = create_llama_provider();
		let stored = store.lock().clone();
		refresh(&second, refresh_context(Some(credential.clone()), stored, publish_handle(Arc::clone(&store)), true))
			.await
			.expect("second refresh succeeds");
		let models = second.get_models();
		assert_eq!(models.len(), 1);
		assert_eq!(models[0].base.id, "qwen");
		assert_eq!(models[0].context_window, 65536);
		assert_eq!(stored_windows(), [65536, 65536]);

		*state.lock() = (false, Some(vec!["llama-server".to_string(), "--ctx-size".to_string(), "32768".to_string()]));
		let stored = store.lock().clone();
		refresh(&second, refresh_context(Some(credential), stored, publish_handle(Arc::clone(&store)), true))
			.await
			.expect("third refresh succeeds");
		assert_eq!(stored_windows(), [32768, 32768]);
	}

	#[tokio::test]
	async fn exposes_unloaded_presets_only_when_router_autoload_is_enabled() {
		let props_requests = Arc::new(parking_lot::Mutex::new(0usize));
		let request_counter = Arc::clone(&props_requests);
		let store: PersistStore = Arc::new(parking_lot::Mutex::new(None));
		let server = listen(handler(move |request| {
			let request_counter = Arc::clone(&request_counter);
			Box::pin(async move {
				assert_eq!(request.authorization.as_deref(), Some("Bearer local"));
				match request.target.as_str() {
					"/models" => TestResponse::Json(
						200,
						json!({
							"data": [
								{ "id": "preset", "status": { "value": "unloaded" }, "source": "preset", "meta": { "n_ctx": 65536 } },
								{ "id": "failed-preset", "status": { "value": "unloaded", "failed": true }, "source": "preset" },
								{ "id": "cache", "status": { "value": "unloaded" }, "source": "cache" },
								{ "id": "models-dir", "status": { "value": "unloaded" }, "source": "models_dir" },
							],
						}),
					),
					"/props" => {
						*request_counter.lock() += 1;
						TestResponse::Json(200, json!({ "role": "router", "models_autoload": true }))
					}
					_ => TestResponse::Empty(404),
				}
			})
		}))
		.await;

		let controller = create_llama_provider();
		refresh(&controller, refresh_context(Some(llama_credential("local", &server.url)), None, publish_handle(Arc::clone(&store)), true))
			.await
			.expect("refresh succeeds");

		assert_eq!(*props_requests.lock(), 1);
		assert_eq!(controller.get_models().iter().map(|model| model.base.id.clone()).collect::<Vec<_>>(), ["preset"]);
		let entry = store.lock().clone().expect("persisted entry");
		assert_eq!(
			entry.models.iter().map(|model| (model.id().to_string(), model.api().to_string())).collect::<Vec<_>>(),
			[
				("preset".to_string(), "openai-completions".to_string()),
				("preset".to_string(), "llama-cpp-classify".to_string()),
			]
		);
	}

	#[tokio::test]
	async fn hides_unloaded_presets_when_router_autoload_is_disabled() {
		let server = listen(handler(move |request| {
			Box::pin(async move {
				match request.target.as_str() {
					"/models" => {
						TestResponse::Json(200, json!({ "data": [{ "id": "preset", "status": { "value": "unloaded" }, "source": "preset" }] }))
					}
					"/props" => TestResponse::Json(200, json!({ "role": "router", "models_autoload": false })),
					_ => TestResponse::Empty(404),
				}
			})
		}))
		.await;

		let controller = create_llama_provider();
		refresh(
			&controller,
			refresh_context(
				Some(llama_credential("local", &server.url)),
				None,
				publish_handle(Arc::new(parking_lot::Mutex::new(None))),
				true,
			),
		)
		.await
		.expect("refresh succeeds");

		assert!(controller.get_models().is_empty());
	}

	#[tokio::test]
	async fn classifies_with_selectable_models_through_llama_server() {
		let paths: Arc<parking_lot::Mutex<Vec<String>>> = Arc::new(parking_lot::Mutex::new(Vec::new()));
		let path_sink = Arc::clone(&paths);
		let server = listen(handler(move |request| {
			let path_sink = Arc::clone(&path_sink);
			Box::pin(async move {
				path_sink.lock().push(request.target.clone());
				assert_eq!(request.body.get("model").and_then(|model| model.as_str()), Some("qwen"));
				match request.target.as_str() {
					"/tokenize" => {
						let content = request.body.get("content").and_then(|content| content.as_str()).unwrap_or("");
						let tokens: Vec<u32> = content.chars().map(|character| character as u32).collect();
						TestResponse::Json(200, json!({ "tokens": tokens }))
					}
					"/apply-template" => TestResponse::Json(200, json!({ "prompt": "<|im_start|>assistant\n" })),
					"/completion" => TestResponse::Json(
						200,
						json!({
							"completion_probabilities": [
								{
									"top_logprobs": [
										{ "id": 66, "token": "B", "logprob": -0.1 },
										{ "id": 65, "token": "A", "logprob": -2.4 },
									],
								},
							],
						}),
					),
					_ => TestResponse::Empty(404),
				}
			})
		}))
		.await;

		let controller = create_llama_provider();
		controller.set_catalog(&[model_info("qwen", LlamaModelStatus::Loaded, json!({}))], &server.url, false);
		let classifier = controller
			.get_all_models()
			.into_iter()
			.find_map(|model| match model {
				AnyModel::Classifier(classifier) => Some(classifier),
				_ => None,
			})
			.expect("classifier model");

		// Provider auth resolves the OpenAI-compatible /v1 URL, which replaces the model's base URL.
		let mut classifier = classifier;
		classifier.base.base_url = format!("{}/v1", server.url);
		let context = pi_ai::types::ClassifierContext {
			state: json!({ "message": "The build is red again." })
				.as_object()
				.expect("object")
				.clone(),
			questions: [(
				"kind".to_string(),
				pi_ai::types::ClassifierQuestion::Choice {
					instructions: "What is this about?".to_string(),
					criteria: [
						("billing".to_string(), String::new()),
						("ci".to_string(), String::new()),
					]
					.into_iter()
					.collect(),
				},
			)]
			.into_iter()
			.collect(),
		};
		let options = pi_ai::types::ClassifierOptions {
			base: pi_ai::types::ProviderRequestOptions { api_key: Some("local".to_string()), ..Default::default() },
			temperature: None,
		};
		let result = match controller.provider.classify(&classifier, &context, &options) {
			Some(future) => future.await,
			None => panic!("llama provider classify must be present"),
		};

		assert!(result.error_message.is_none(), "unexpected error: {:?}", result.error_message);
		match result.answers.get("kind") {
			Some(pi_ai::types::ClassifierAnswer::Choice { choice, .. }) => assert_eq!(choice, "ci"),
			other => panic!("expected a choice answer, got {other:?}"),
		}
		assert!(paths.lock().iter().any(|path| path == "/completion"));
	}

	#[tokio::test]
	async fn stays_dormant_until_configured_and_stores_url_plus_optional_key() {
		let controller = create_llama_provider();
		let auth = controller.provider.auth().api_key.as_ref().expect("api key auth");
		struct EmptyAuthContext;

		impl AuthContext for EmptyAuthContext {
			fn env<'a>(&'a self, _name: &'a str) -> BoxFuture<'a, Option<String>> {
				Box::pin(async { None })
			}

			fn file_exists<'a>(&'a self, _path: &'a str) -> BoxFuture<'a, bool> {
				Box::pin(async { false })
			}
		}

		let signal = AbortSignal::new();
		let empty_check = {
			let input = ApiKeyAuthInput { ctx: Arc::new(EmptyAuthContext), credential: None, signal: signal.clone() };
			(auth.check.as_ref().expect("check"))(input).await
		};
		assert!(matches!(empty_check, Ok(None)));
		let empty_resolve = {
			let input = ApiKeyAuthInput { ctx: Arc::new(EmptyAuthContext), credential: None, signal: signal.clone() };
			(auth.resolve)(input).await
		};
		assert!(matches!(empty_resolve, Ok(None)));

		let server = listen(handler(move |request| {
			Box::pin(async move {
				assert_eq!(request.authorization.as_deref(), Some("Bearer secret"));
				TestResponse::Json(200, json!({ "data": [] }))
			})
		}))
		.await;
		let answers = Arc::new(parking_lot::Mutex::new(VecDeque::from([server.url.clone(), "secret".to_string()])));
		let interaction = ProviderAuthInteraction {
			signal: signal.clone(),
			prompt: Box::new(move |_prompt| {
				let answers = Arc::clone(&answers);
				Box::pin(async move { answers.lock().pop_front().ok_or_else(|| ThrownValue::message("no answer")) })
			}),
			notify: Box::new(|_event| {}),
		};
		let login = auth.login.as_ref().expect("login");
		let credential = login(interaction).await.expect("login succeeds");
		assert_eq!(
			credential,
			ApiKeyCredential {
				key: Some("secret".to_string()),
				env: Some([(LLAMA_BASE_URL_KEY.to_string(), server.url.clone())].into_iter().collect()),
			}
		);
		let resolved = {
			let input = ApiKeyAuthInput { ctx: Arc::new(EmptyAuthContext), credential: Some(&credential), signal };
			(auth.resolve)(input).await.expect("resolve succeeds").expect("resolves")
		};
		assert_eq!(resolved.auth.api_key.as_deref(), Some("secret"));
		assert_eq!(resolved.auth.base_url.as_deref(), Some(format!("{}/v1", server.url).as_str()));
		assert_eq!(
			resolved.env,
			Some([(LLAMA_BASE_URL_KEY.to_string(), server.url.clone())].into_iter().collect())
		);
		assert_eq!(resolved.source.as_deref(), Some("stored credential"));
	}
}

