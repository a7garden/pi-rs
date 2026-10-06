//! Port of `packages/ai/src/models.ts` — the Provider abstraction.
//!
//! # The Provider contract (design spec for later provider waves)
//!
//! Upstream `Provider` is an object literal; the Rust port is a trait so the
//! ~38 remaining provider files attach mechanically. One upstream object
//! method maps to one trait method:
//!
//! | upstream                        | Rust                                          |
//! |---------------------------------|-----------------------------------------------|
//! | `id`, `name`, `baseUrl`, `headers`, `auth` | [`Provider::id`] etc. (plain getters) |
//! | `getModels()`                   | [`Provider::get_models`] (chat models)        |
//! | `getAllModels()`                | [`Provider::get_all_models`] (default: chat)  |
//! | `refreshModels(ctx)`            | [`Provider::refresh_models`] (`None` = absent)|
//! | `filterModels` / `filterAllModels` | [`Provider::filter_models`] / [`Provider::filter_all_models`] (`None` = absent) |
//! | `stream` / `streamSimple`       | [`Provider::stream`] / [`Provider::stream_simple`] |
//! | `fetchDeferred` / `cancelDeferred` | [`Provider::fetch_deferred`] / [`Provider::cancel_deferred`] (`None` = unsupported) |
//! | `generateImages` / `classify`   | [`Provider::generate_images`] / [`Provider::classify`] (`None` = unsupported) |
//!
//! ## Streaming model
//!
//! Upstream returns an `AssistantMessageEventStream` (an `AsyncIterable` of
//! [`AssistantMessageEvent`] with a `result()` promise). The Rust port returns
//! the concrete cloneable [`AssistantMessageEventStream`] handle: producers
//! call `push(event)` / `end(result)`, consumers `await` it as a
//! `futures_core::Stream` (single consumer, FIFO) and/or await `.result()`
//! independently — exactly the upstream observable contract. Adapters produce
//! events with `push` synchronously as parse work happens inside a spawned
//! task; request setup that must stay behind the returned stream uses
//! [`crate::api::lazy::lazy_stream`]. Error termination must produce an
//! `AssistantMessage` with `stopReason` `"error"`/`"aborted"` and
//! `errorMessage`, emitted via the event protocol — adapters never panic
//! across the trait boundary.
//!
//! Direct `stream_simple()` calls may fail synchronously when request auth is
//! missing upstream; in Rust those become the stream error event (same
//! protocol), because Rust functions cannot throw.
//!
//! ## API implementations
//!
//! An API implementation module (upstream `src/api/*.ts`) exports `stream`
//! and `streamSimple` (plus optional deferred methods); providers hold them as
//! values (`ProviderStreams`). The Rust port is the [`ProviderStreams`] trait;
//! each api module exposes `pub fn <api>_api() -> Arc<dyn ProviderStreams>`.
//! Per-API option interfaces (`AnthropicOptions`,
//! `OpenAICompletionsOptions`, `OpenAIResponsesOptions`) become nested option
//! groups on [`StreamOptions`] (`options.anthropic`,
//! `options.openai_completions`, `options.openai_responses`); the api module
//! functions read their group plus the shared base (crate docs, deviation
//! D2).
//!
//! `create_provider` builds a [`Provider`] from parts, dispatching a single
//! `api` to all chat models or a per-`model.api` map, matching upstream error
//! strings for missing implementations.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::types::{
	AnyModel, AssistantMessage, AssistantMessageEventStream, ClassifierContext, ClassifierModel, ClassifierOptions,
	ClassifierResult, Context, DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, ImagesContext,
	ImagesOptions, Model, ModelCompat, ModelCost, ModelCostRates, ModelCostTier, ModelThinkingLevel, ModelType,
	ProviderHeaders, SimpleStreamOptions, StreamOptions, TranscriptContext, Usage, UsageCost,
};
use crate::utils::models_error::{ModelsError, ModelsErrorCode};
use crate::utils::thrown::ThrownValue;

/// Error message used when a deferred method is not implemented by a wrapped
/// API (`"API does not support deferred responses"` upstream).
pub const API_MISSING_DEFERRED_FETCH: &str = "API does not support deferred responses";
/// `"API cannot cancel deferred responses"` upstream.
pub const API_MISSING_DEFERRED_CANCEL: &str = "API cannot cancel deferred responses";

/// The uniform stream contract of an API implementation module
/// (`ProviderStreams`). `stream`/`stream_simple` are required;
/// `fetch_deferred`/`cancel_deferred` are optional (`None` = not supported).
pub trait ProviderStreams: Send + Sync {
	fn stream(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream;

	fn stream_simple(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: &SimpleStreamOptions,
	) -> AssistantMessageEventStream;

	/// Presence probe for `fetchDeferred` (upstream checks
	/// `entry.fetchDeferred !== undefined`).
	fn has_fetch_deferred(&self) -> bool {
		false
	}

	/// Presence probe for `cancelDeferred`.
	fn has_cancel_deferred(&self) -> bool {
		false
	}

	fn fetch_deferred(
		&self,
		_model: &Model,
		_handle: &DeferredHandle,
		_options: &DeferredFetchOptions,
	) -> Option<AssistantMessageEventStream> {
		None
	}

	fn cancel_deferred(
		&self,
		_model: &Model,
		_handle: &DeferredHandle,
		_options: &DeferredCancelOptions,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		None
	}
}

/// The uniform contract of an image-generation API implementation module
/// (`ProviderImages`).
pub trait ProviderImages: Send + Sync {
	fn generate_images<'a>(
		&'a self,
		model: &'a crate::types::ImageModel,
		context: &'a ImagesContext,
		options: &'a ImagesOptions,
	) -> Pin<Box<dyn Future<Output = crate::types::AssistantImages> + Send + 'a>>;
}

/// The uniform contract implemented by classifier API modules
/// (`ProviderClassifier`).
pub trait ProviderClassifier: Send + Sync {
	fn classify<'a>(
		&'a self,
		model: &'a ClassifierModel,
		context: &'a ClassifierContext,
		options: &'a ClassifierOptions,
	) -> Pin<Box<dyn Future<Output = ClassifierResult> + Send + 'a>>;
}

/// Provider-selected persisted catalog publication (`ModelsPublication`).
pub struct ModelsPublication {
	/// Provider-selected persisted catalog. `None` leaves storage unchanged;
	/// [`ModelsStoreDelete`] deletes it.
	pub persist: Option<ModelsStoreWrite>,
	/// Optional synchronous update of provider-private in-memory catalog state.
	pub update: Option<Box<dyn FnOnce() + Send>>,
}

/// Persist mutation handed to [`RefreshModelsContext::publish`]: an entry to
/// write, or the delete marker (`persist: null` upstream).
pub enum ModelsStoreWrite {
	Write(crate::models_store::ModelsStoreEntry),
	Delete,
}

/// Shared, generation-checked publication callback
/// (`context.publish` upstream). Cloned into provider refresh tasks; the
/// `Models` collection owns the underlying closure.
#[derive(Clone)]
pub struct PublishHandle {
	inner: Arc<parking_lot::Mutex<dyn FnMut(ModelsPublication) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send>>,
}

impl PublishHandle {
	pub fn new(
		publish: impl FnMut(ModelsPublication) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + 'static,
	) -> PublishHandle {
		PublishHandle { inner: Arc::new(parking_lot::Mutex::new(Box::new(publish))) }
	}

	/// Run one publication through the generation-checked gate. Resolves to
	/// false when the publication was superseded or aborted.
	pub async fn publish(&self, publication: ModelsPublication) -> bool {
		let future = {
			let mut publish = self.inner.lock();
			(*publish)(publication)
		};
		future.await
	}
}

/// Context handed to [`Provider::refresh_models`] (`RefreshModelsContext`).
/// The `publish` callback is generation-checked by the `Models` collection.
pub struct RefreshModelsContext {
	/// Effective configured credential. OAuth credentials are refreshed before
	/// network access.
	pub credential: Option<crate::auth::types::Credential>,
	/// Immutable provider-scoped catalog snapshot captured before this refresh
	/// phase.
	pub stored: Option<crate::models_store::ModelsStoreEntry>,
	/// Generation-checked publication. Persistence policy remains
	/// provider-owned; the update runs synchronously only after the selected
	/// persistence mutation.
	pub publish: PublishHandle,
	/// False during offline/cache-only initialization.
	pub allow_network: bool,
	/// Bypass provider freshness checks and fetch immediately when network
	/// access is allowed.
	pub force: Option<bool>,
	/// Always present, including when the public refresh caller omits its
	/// optional signal.
	pub signal: crate::utils::abort::AbortSignal,
}

/// A provider is the concrete runtime unit (`Provider`): id/name/base
/// metadata, auth methods, model listing, and the operations its models
/// support. See the module docs for the full design contract.
pub trait Provider: Send + Sync {
	fn id(&self) -> &str;
	fn name(&self) -> &str;
	fn base_url(&self) -> Option<&str> {
		None
	}
	fn headers(&self) -> Option<&ProviderHeaders> {
		None
	}

	/// Required: at least one of `apiKey`/`oauth`. Every provider has auth
	/// semantics.
	fn auth(&self) -> &crate::auth::types::ProviderAuth;

	/// Current known chat models, sync. Static providers return their catalog;
	/// dynamic providers return the list as of the last `refresh_models()`
	/// (empty before the first). Must not fail; `Models` treats a failing
	/// implementation as having no models.
	fn get_models(&self) -> Vec<Model>;

	/// Current known models of every type, sync, with the same contract as
	/// `get_models()`. Default: the chat list.
	fn get_all_models(&self) -> Vec<AnyModel> {
		self.get_models().into_iter().map(AnyModel::Chat).collect()
	}

	/// Dynamic providers only: restore `context.stored` and optionally fetch a
	/// newer list using the effective credential. `None` = static provider.
	fn refresh_models(
		&self,
		_context: RefreshModelsContext,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		None
	}

	/// Optional provider policy for credential-specific chat model
	/// availability. `None` = absent.
	fn filter_models(&self, _models: &[Model], _credential: Option<&crate::auth::types::Credential>) -> Option<Vec<Model>> {
		None
	}

	/// Optional credential-specific availability policy across every model
	/// type. `None` = absent.
	fn filter_all_models(
		&self,
		_models: &[AnyModel],
		_credential: Option<&crate::auth::types::Credential>,
	) -> Option<Vec<AnyModel>> {
		None
	}

	/// Stream a normalized transcript. `Models` normalizes the caller's
	/// [`Context`] before dispatching here.
	fn stream(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream;

	fn stream_simple(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: &SimpleStreamOptions,
	) -> AssistantMessageEventStream;

	fn fetch_deferred(
		&self,
		_model: &Model,
		_handle: &DeferredHandle,
		_options: &DeferredFetchOptions,
	) -> Option<AssistantMessageEventStream> {
		None
	}

	fn cancel_deferred(
		&self,
		_model: &Model,
		_handle: &DeferredHandle,
		_options: &DeferredCancelOptions,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		None
	}

	/// Present when the provider supports dedicated image models. Never
	/// rejects (result carries the error).
	fn generate_images<'a>(
		&'a self,
		_model: &'a crate::types::ImageModel,
		_context: &'a ImagesContext,
		_options: &'a ImagesOptions,
	) -> Option<Pin<Box<dyn Future<Output = crate::types::AssistantImages> + Send + 'a>>> {
		None
	}

	/// Present when the provider supports structured classifier models. Never
	/// rejects.
	fn classify<'a>(
		&'a self,
		_model: &'a ClassifierModel,
		_context: &'a ClassifierContext,
		_options: &'a ClassifierOptions,
	) -> Option<Pin<Box<dyn Future<Output = ClassifierResult> + Send + 'a>>> {
		None
	}
}

/// How [`create_provider`] dispatches chat APIs (`api` /
/// `Partial<Record<TApi, ProviderStreams>>` upstream).
pub enum ApiDispatch {
	/// A single implementation for all chat models.
	Single(Arc<dyn ProviderStreams>),
	/// Implementations keyed by `model.api`.
	ByApi(BTreeMap<String, Arc<dyn ProviderStreams>>),
}

/// Builds a provider from parts (`CreateProviderOptions`). Built-in provider
/// factories and models.json custom providers both go through this.
pub struct CreateProviderOptions {
	pub id: String,
	/// Display name. Default: `id`.
	pub name: Option<String>,
	pub base_url: Option<String>,
	pub headers: Option<ProviderHeaders>,
	/// Required — every provider has auth semantics, even ambient/keyless ones.
	pub auth: crate::auth::types::ProviderAuth,
	/// Static baseline models of every type (empty for purely dynamic
	/// providers). Models without `type` are chat models.
	pub models: Vec<AnyModel>,
	/// Fetch a dynamic model overlay of every type. `create_provider` restores
	/// and publishes it transactionally and drops models of unknown types.
	pub fetch_models: Option<
		Box<
			dyn Fn(&RefreshModelsContext) -> Pin<Box<dyn Future<Output = Result<Vec<AnyModel>, ThrownValue>> + Send>>
				+ Send
				+ Sync,
		>,
	>,
	/// Credential-specific chat model availability (`Provider.filter_models`).
	pub filter_models: Option<
		Box<dyn Fn(&[Model], Option<&crate::auth::types::Credential>) -> Vec<Model> + Send + Sync>,
	>,
	/// Credential-specific availability across every model type
	/// (`Provider.filter_all_models`).
	pub filter_all_models: Option<
		Box<dyn Fn(&[AnyModel], Option<&crate::auth::types::Credential>) -> Vec<AnyModel> + Send + Sync>,
	>,
	/// Chat implementation: a single one for all chat models, or a map keyed by
	/// `model.api` for mixed-API providers.
	pub api: Option<ApiDispatch>,
	/// Image-generation implementations keyed by `model.api`.
	pub images: BTreeMap<String, Arc<dyn ProviderImages>>,
	/// Classifier implementations keyed by `model.api`.
	pub classifiers: BTreeMap<String, Arc<dyn ProviderClassifier>>,
}

/// Upstream error thrown when no implementation is configured.
const NO_IMPLEMENTATION_ERROR: &str =
	"Provider {}: at least one of \"api\", \"images\", or \"classifiers\" is required.";

/// Builds a provider from parts (`createProvider`). A single `api` streams all
/// chat models; an `api` map dispatches on `model.api`, and a model whose api
/// has no entry produces a stream error. One-shot operation maps dispatch on
/// `model.api` the same way. At least one concrete implementation across
/// `api`/`images`/`classifiers` is required; empty maps are rejected.
pub fn create_provider(input: CreateProviderOptions) -> Result<Arc<dyn Provider>, ModelsError> {
	let has_single_api = matches!(input.api, Some(ApiDispatch::Single(_)));
	let streams: Vec<Arc<dyn ProviderStreams>> = match &input.api {
		Some(ApiDispatch::Single(api)) => vec![api.clone()],
		Some(ApiDispatch::ByApi(map)) => map.values().cloned().collect(),
		None => Vec::new(),
	};
	if streams.is_empty() && input.images.is_empty() && input.classifiers.is_empty() {
		return Err(ModelsError::new(
			ModelsErrorCode::Provider,
			NO_IMPLEMENTATION_ERROR.replace("{}", &input.id),
		));
	}

	struct GeneratedProvider {
		id: String,
		name: String,
		base_url: Option<String>,
		headers: Option<ProviderHeaders>,
		auth: crate::auth::types::ProviderAuth,
		single_api: Option<Arc<dyn ProviderStreams>>,
		by_api: Option<BTreeMap<String, Arc<dyn ProviderStreams>>>,
		has_fetch_deferred: bool,
		has_cancel_deferred: bool,
		baseline_models: Vec<AnyModel>,
		dynamic_models: Arc<parking_lot::Mutex<Vec<AnyModel>>>,
		fetch_models: Option<
			Box<
				dyn Fn(&RefreshModelsContext)
					-> Pin<Box<dyn Future<Output = Result<Vec<AnyModel>, ThrownValue>> + Send>>
					+ Send
					+ Sync,
			>,
		>,
		filter_models: Option<Box<dyn Fn(&[Model], Option<&crate::auth::types::Credential>) -> Vec<Model> + Send + Sync>>,
		filter_all_models: Option<
			Box<dyn Fn(&[AnyModel], Option<&crate::auth::types::Credential>) -> Vec<AnyModel> + Send + Sync>,
		>,
		images: BTreeMap<String, Arc<dyn ProviderImages>>,
		classifiers: BTreeMap<String, Arc<dyn ProviderClassifier>>,
	}

	impl GeneratedProvider {
		fn current_models(&self) -> Vec<AnyModel> {
			let mut merged = self.baseline_models.clone();
			let dynamic = self.dynamic_models.lock();
			for model in dynamic.iter() {
				match merged.iter().position(|entry| entry.model_type() == model.model_type() && entry.id() == model.id()) {
					Some(index) => merged[index] = model.clone(),
					None => merged.push(model.clone()),
				}
			}
			merged
		}

		fn api_for(&self, model: &Model) -> Option<Arc<dyn ProviderStreams>> {
			match (&self.single_api, &self.by_api) {
				(Some(single), _) => Some(single.clone()),
				(None, Some(map)) => map.get(model.api()).cloned(),
				_ => None,
			}
		}

		fn dispatch(
			&self,
			model: &Model,
			run: impl FnOnce(&dyn ProviderStreams) -> AssistantMessageEventStream,
		) -> AssistantMessageEventStream {
			match self.api_for(model) {
				Some(streams) => run(streams.as_ref()),
				None => crate::api::lazy::lazy_stream(
					model,
					{
						let provider_id = self.id.clone();
						let api = model.api().to_string();
						async move {
							Err(ThrownValue::error(ModelsError::new(
								ModelsErrorCode::Stream,
								format!("Provider {provider_id} has no API implementation for \"{api}\""),
							)))
						}
					},
				),
			}
		}
	}

	impl Provider for GeneratedProvider {
		fn id(&self) -> &str {
			&self.id
		}

		fn name(&self) -> &str {
			&self.name
		}

		fn base_url(&self) -> Option<&str> {
			self.base_url.as_deref()
		}

		fn headers(&self) -> Option<&ProviderHeaders> {
			self.headers.as_ref()
		}

		fn auth(&self) -> &crate::auth::types::ProviderAuth {
			&self.auth
		}

		fn get_models(&self) -> Vec<Model> {
			self.current_models()
				.into_iter()
				.filter_map(|model| match model {
					AnyModel::Chat(model) => Some(model),
					_ => None,
				})
				.collect()
		}

		fn get_all_models(&self) -> Vec<AnyModel> {
			self.current_models()
		}

		fn refresh_models(
			&self,
			context: RefreshModelsContext,
		) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
			let fetch_models = self.fetch_models.as_ref()?;
			let provider_id = self.id.clone();
			let dynamic_models = Arc::clone(&self.dynamic_models);
			Some(Box::pin(async move {
				if let Some(stored) = &context.stored {
					let restored: Vec<AnyModel> = stored
						.models
						.iter()
						.filter(|model| model.provider() == provider_id)
						.cloned()
						.collect();
					let published = {
						let state = Arc::clone(&dynamic_models);
						let update = move || {
							*state.lock() = restored;
						};
						context
							.publish
							.publish(ModelsPublication { persist: None, update: Some(Box::new(update)) })
							.await
					};
					if !published {
						return Ok(());
					}
				}
				if !context.allow_network || context.signal.aborted() {
					return Ok(());
				}
				let fetched = fetch_models(&context).await?;
				if context.signal.aborted() {
					return Ok(());
				}
				let refreshed: Vec<AnyModel> = fetched
					.into_iter()
					.filter(|model| {
						matches!(
							model.model_type(),
							ModelType::Chat | ModelType::Image | ModelType::Classifier
						)
					})
					.collect();
				let checked_at = crate::utils::diagnostics::now_ms();
				let published = {
					let state = Arc::clone(&dynamic_models);
					let for_update = refreshed.clone();
					let update = move || {
						*state.lock() = for_update;
					};
					context
						.publish
						.publish(ModelsPublication {
							persist: Some(ModelsStoreWrite::Write(crate::models_store::ModelsStoreEntry {
								models: refreshed,
								last_modified: None,
								checked_at: Some(checked_at),
								etag: None,
							})),
							update: Some(Box::new(update)),
						})
						.await
				};
				let _ = published;
				Ok(())
			}))
		}

		fn filter_models(&self, models: &[Model], credential: Option<&crate::auth::types::Credential>) -> Option<Vec<Model>> {
			self.filter_models.as_ref().map(|filter| filter(models, credential))
		}

		fn filter_all_models(
			&self,
			models: &[AnyModel],
			credential: Option<&crate::auth::types::Credential>,
		) -> Option<Vec<AnyModel>> {
			self.filter_all_models.as_ref().map(|filter| filter(models, credential))
		}

		fn stream(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
			self.dispatch(model, |streams| streams.stream(model, context, options))
		}

		fn stream_simple(
			&self,
			model: &Model,
			context: &TranscriptContext,
			options: &SimpleStreamOptions,
		) -> AssistantMessageEventStream {
			self.dispatch(model, |streams| streams.stream_simple(model, context, options))
		}

		fn fetch_deferred(
			&self,
			model: &Model,
			handle: &DeferredHandle,
			options: &DeferredFetchOptions,
		) -> Option<AssistantMessageEventStream> {
			if !self.has_fetch_deferred {
				return None;
			}
			let provider_id = self.id.clone();
			let single_api = self.single_api.clone();
			let by_api = self.by_api.clone();
			let model_for_error = model.clone();
			let handle = handle.clone();
			let options = options.clone();
			Some(crate::api::lazy::lazy_stream(
				model,
				async move {
					let implementation = {
						let api = model_for_error.api();
						match (&single_api, &by_api) {
							(Some(single), _) => Some(single.clone()),
							(None, Some(map)) => map.get(api).cloned(),
							_ => None,
						}
					};
					let implementation = implementation.ok_or_else(|| {
						ThrownValue::error(ModelsError::new(
							ModelsErrorCode::Provider,
							format!(
								"Provider {} does not support deferred responses for \"{}\"",
								provider_id,
								model_for_error.api()
							),
						))
					})?;
					implementation
						.fetch_deferred(&model_for_error, &handle, &options)
						.ok_or_else(|| {
							ThrownValue::error(ModelsError::new(
								ModelsErrorCode::Provider,
								format!(
									"Provider {} does not support deferred responses for \"{}\"",
									provider_id,
									model_for_error.api()
								),
							))
						})
				},
			))
		}

		fn cancel_deferred(
			&self,
			model: &Model,
			handle: &DeferredHandle,
			options: &DeferredCancelOptions,
		) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
			if !self.has_cancel_deferred {
				return None;
			}
			let implementation = self.api_for(model)?;
			let provider_id = self.id.clone();
			let model = model.clone();
			let handle = handle.clone();
			let options = options.clone();
			Some(Box::pin(async move {
				let operation = implementation
					.cancel_deferred(&model, &handle, &options)
					.ok_or_else(|| {
						ThrownValue::error(ModelsError::new(
							ModelsErrorCode::Provider,
							format!(
								"Provider {} cannot cancel deferred responses for \"{}\"",
								provider_id,
								model.api()
							),
						))
					})?;
				operation.await
			}))
		}

		fn generate_images<'a>(
			&'a self,
			model: &'a crate::types::ImageModel,
			_context: &'a ImagesContext,
			_options: &'a ImagesOptions,
		) -> Option<Pin<Box<dyn Future<Output = crate::types::AssistantImages> + Send + 'a>>> {
			if self.images.is_empty() {
				return None;
			}
			let provider_id = self.id.clone();
			let implementation = self.images.get(model.api());
			let error = implementation.is_none().then(|| {
				ThrownValue::error(ModelsError::new(
					ModelsErrorCode::Provider,
					format!(
						"Provider {} has no image generation implementation for \"{}\"",
						provider_id,
						model.api()
					),
				))
			});
			Some(Box::pin(async move {
				match (implementation, error) {
					(Some(implementation), None) => {
						implementation.generate_images(model, _context, _options).await
					}
					(_, Some(error)) => crate::utils::model_operations::image_error_result(model, &error, false),
					_ => unreachable!("one of implementation/error is set"),
				}
			}))
		}

		fn classify<'a>(
			&'a self,
			model: &'a ClassifierModel,
			_context: &'a ClassifierContext,
			_options: &'a ClassifierOptions,
		) -> Option<Pin<Box<dyn Future<Output = ClassifierResult> + Send + 'a>>> {
			if self.classifiers.is_empty() {
				return None;
			}
			let provider_id = self.id.clone();
			let implementation = self.classifiers.get(model.api());
			let error = implementation.is_none().then(|| {
				ThrownValue::error(ModelsError::new(
					ModelsErrorCode::Provider,
					format!(
						"Provider {} has no classifier implementation for \"{}\"",
						provider_id,
						model.api()
					),
				))
			});
			Some(Box::pin(async move {
				match (implementation, error) {
					(Some(implementation), None) => implementation.classify(model, _context, _options).await,
					(_, Some(error)) => crate::utils::model_operations::classifier_error_result(model, &error, false),
					_ => unreachable!("one of implementation/error is set"),
				}
			}))
		}
	}

	let has_fetch_deferred = streams.iter().any(|entry| entry.has_fetch_deferred());
	let has_cancel_deferred = streams.iter().any(|entry| entry.has_cancel_deferred());
	let by_api = if has_single_api {
		None
	} else {
		match &input.api {
			Some(ApiDispatch::ByApi(map)) => Some(map.clone()),
			_ => None,
		}
	};

	let id = input.id.clone();
	let name = input.name.clone().unwrap_or_else(|| input.id.clone());

	Ok(Arc::new(GeneratedProvider {
		id,
		name,
		base_url: input.base_url,
		headers: input.headers,
		auth: input.auth,
		single_api: match input.api {
			Some(ApiDispatch::Single(api)) => Some(api),
			_ => None,
		},
		by_api,
		has_fetch_deferred,
		has_cancel_deferred,
		baseline_models: input.models,
		dynamic_models: Arc::new(parking_lot::Mutex::new(Vec::new())),
		fetch_models: input.fetch_models,
		filter_models: input.filter_models,
		filter_all_models: input.filter_all_models,
		images: input.images,
		classifiers: input.classifiers,
	}))
}

/// Runtime-checked narrowing for dynamically looked-up models (`hasApi`).
/// Non-chat models never match, even when their api id equals `api`.
pub fn has_api<'a>(model: &'a AnyModel, api: &str) -> Option<&'a Model> {
	match model {
		AnyModel::Chat(model) if model.api() == api => Some(model),
		_ => None,
	}
}

/// Upstream `calculateCost`: compute the cost breakdown for a usage record at
/// the model's catalog price, mutating `usage.cost` in place and returning a
/// copy (upstream mutates and returns `usage.cost`).
pub fn calculate_cost(model: &AnyModel, usage: &mut Usage) -> UsageCost {
	let cost_field = match model {
		AnyModel::Chat(model) => &model.base.cost,
		AnyModel::Image(model) => &model.base.cost,
		AnyModel::Classifier(model) => &model.base.cost,
	};
	calculate_cost_at(cost_field, usage)
}

fn calculate_cost_at(cost: &ModelCost, usage: &mut Usage) -> UsageCost {
	let input_tokens = usage.input + usage.cache_read + usage.cache_write;
	let mut rates: &ModelCostRates = &cost.rates;
	let mut matched_threshold = -1.0f64;
	for tier in cost.tiers.iter().flatten() {
		let ModelCostTier { rates: _, input_tokens_above } = tier;
		let threshold = *input_tokens_above;
		if input_tokens as f64 > threshold && threshold > matched_threshold {
			rates = &tier.rates;
			matched_threshold = threshold;
		}
	}

	// Anthropic charges 2x base input for 1h cache writes.
	let long_write = usage.cache_write_1h.unwrap_or(0);
	let short_write = usage.cache_write - long_write;
	usage.cost.input = (rates.input / 1_000_000.0) * usage.input as f64;
	usage.cost.output = (rates.output / 1_000_000.0) * usage.output as f64;
	usage.cost.cache_read = (rates.cache_read / 1_000_000.0) * usage.cache_read as f64;
	usage.cost.cache_write =
		(rates.cache_write * short_write as f64 + rates.input * 2.0 * long_write as f64) / 1_000_000.0;
	usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
	usage.cost.clone()
}

/// Upstream `getSupportedThinkingLevels`.
pub fn get_supported_thinking_levels(model: &Model) -> Vec<ModelThinkingLevel> {
	if !model.reasoning {
		return vec![ModelThinkingLevel::Off];
	}

	ModelThinkingLevel::LADDER
		.into_iter()
		.filter(|level| {
			let mapped = model.thinking_level_map.as_ref().and_then(|map| map.get(level));
			match mapped {
				// `null` marks a level as unsupported.
				Some(None) => false,
				// xhigh/max require an explicit mapping.
				Some(Some(_)) => true,
				None => !matches!(level, ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max),
			}
		})
		.collect()
}

/// Upstream `clampThinkingLevel`: snap a requested level onto the model's
/// supported ladder (nearest upward first, then downward).
pub fn clamp_thinking_level(model: &Model, level: ModelThinkingLevel) -> ModelThinkingLevel {
	let available_levels = get_supported_thinking_levels(model);
	if available_levels.contains(&level) {
		return level;
	}

	let ladder = ModelThinkingLevel::LADDER;
	let requested_index = ladder.iter().position(|entry| *entry == level);
	let Some(requested_index) = requested_index else {
		return available_levels.first().copied().unwrap_or(ModelThinkingLevel::Off);
	};

	for candidate in &ladder[requested_index..] {
		if available_levels.contains(candidate) {
			return *candidate;
		}
	}
	for candidate in ladder[..requested_index].iter().rev() {
		if available_levels.contains(candidate) {
			return *candidate;
		}
	}
	available_levels.first().copied().unwrap_or(ModelThinkingLevel::Off)
}

/// Check if two models are equal by comparing their type, id, and provider
/// (`modelsAreEqual`). `None` (null/undefined upstream) compares unequal.
pub fn models_are_equal(a: Option<&AnyModel>, b: Option<&AnyModel>) -> bool {
	match (a, b) {
		(Some(a), Some(b)) => a.model_type() == b.model_type() && a.id() == b.id() && a.provider() == b.provider(),
		_ => false,
	}
}

/// Merged model compatibility view for a chat model (`model.compat`).
pub fn get_compat(model: &Model) -> Option<&ModelCompat> {
	model.compat.as_ref()
}

/// Convenience constructor used by tests and provider factories.
pub fn models_error(code: ModelsErrorCode, message: impl Into<String>) -> ModelsError {
	ModelsError::new(code, message)
}

/// Alias so `lazy.rs` can reference the missing-deferred strings without a
/// models.rs cycle.
pub mod lazy_strings {
	pub use super::{API_MISSING_DEFERRED_CANCEL, API_MISSING_DEFERRED_FETCH};
}

/// The `Models` collection trait (`Models` upstream): runtime collection of
/// providers plus auth application and request convenience. Implemented by
/// [`create_models`](crate::create_models); see upstream `models.ts` for the
/// operation-by-operation contract.
#[async_trait::async_trait]
pub trait Models: Send + Sync {
	fn get_providers(&self) -> Vec<Arc<dyn Provider>>;
	fn get_provider(&self, id: &str) -> Option<Arc<dyn Provider>>;
	fn get_models(&self, provider: Option<&str>) -> Vec<Model>;
	fn get_model(&self, provider: &str, id: &str) -> Option<Model>;
	fn get_models_of_type(&self, type_: ModelType, provider: Option<&str>) -> Vec<AnyModel>;
	fn get_model_of_type(&self, type_: ModelType, provider: &str, id: &str) -> Option<AnyModel>;
	fn get_all_models(&self, provider: Option<&str>) -> Vec<AnyModel>;

	async fn refresh(
		&self,
		options: Option<ModelsRefreshOptions>,
	) -> ModelsRefreshResult;

	async fn check_auth(
		&self,
		provider_id: &str,
		options: Option<crate::auth::types::AuthOperationOptions>,
	) -> Result<Option<crate::auth::types::AuthCheck>, ThrownValue>;

	async fn get_available(
		&self,
		provider_id: Option<&str>,
		options: Option<crate::auth::types::AuthOperationOptions>,
	) -> Result<Vec<Model>, ThrownValue>;

	async fn get_all_available(
		&self,
		provider_id: Option<&str>,
		options: Option<crate::auth::types::AuthOperationOptions>,
	) -> Result<Vec<AnyModel>, ThrownValue>;

	async fn get_auth_by_provider(
		&self,
		provider_id: &str,
		overrides: Option<crate::auth::resolve::AuthResolutionOverrides>,
	) -> Result<Option<crate::auth::types::AuthResult>, ThrownValue>;

	async fn get_auth_by_model(
		&self,
		model: &AnyModel,
		overrides: Option<crate::auth::resolve::AuthResolutionOverrides>,
	) -> Result<Option<crate::auth::types::AuthResult>, ThrownValue>;

	async fn login(
		&self,
		provider_id: &str,
		type_: crate::auth::types::AuthType,
		interaction: crate::auth::types::AuthInteraction,
		options: Option<crate::auth::types::LoginOptions>,
	) -> Result<crate::auth::types::Credential, ThrownValue>;

	async fn logout(
		&self,
		provider_id: &str,
		options: Option<crate::auth::types::AuthOperationOptions>,
	) -> Result<(), ThrownValue>;

	fn stream(
		&self,
		model: &Model,
		context: &Context,
		options: Option<ModelsApiStreamOptions>,
	) -> AssistantMessageEventStream;

	async fn complete(
		&self,
		model: &Model,
		context: &Context,
		options: Option<ModelsApiStreamOptions>,
	) -> AssistantMessage;

	fn stream_simple(
		&self,
		model: &Model,
		context: &Context,
		options: Option<ModelsSimpleStreamOptions>,
	) -> AssistantMessageEventStream;

	async fn complete_simple(
		&self,
		model: &Model,
		context: &Context,
		options: Option<ModelsSimpleStreamOptions>,
	) -> AssistantMessage;

	fn stream_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: Option<ModelsDeferredFetchOptions>,
	) -> AssistantMessageEventStream;

	async fn fetch_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: Option<ModelsDeferredFetchOptions>,
	) -> AssistantMessage;

	async fn cancel_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: Option<ModelsDeferredCancelOptions>,
	) -> Result<(), ThrownValue>;

	async fn generate_images(
		&self,
		model: &crate::types::ImageModel,
		context: &ImagesContext,
		options: Option<ModelsImagesOptions>,
	) -> crate::types::AssistantImages;

	async fn classify(
		&self,
		model: &ClassifierModel,
		context: &ClassifierContext,
		options: Option<ModelsClassifierOptions>,
	) -> ClassifierResult;
}

/// `MutableModels` upstream: provider registry mutation.
pub trait MutableModels: Models {
	fn set_provider(&self, provider: Arc<dyn Provider>);
	fn delete_provider(&self, id: &str);
	fn clear_providers(&self);
}

/// `ModelsRequestTransforms` upstream: caller hooks applied just before
/// provider dispatch.
#[derive(Default, Clone)]
pub struct ModelsRequestTransforms {
	pub transform_headers: Option<
		Arc<
			dyn Fn(Option<ProviderHeaders>) -> Pin<Box<dyn Future<Output = Option<ProviderHeaders>> + Send>>
				+ Send
				+ Sync,
		>,
	>,
}

macro_rules! models_options {
	($name:ident, $base:ty) => {
		#[derive(Clone, Default)]
		pub struct $name {
			pub base: $base,
			pub transforms: ModelsRequestTransforms,
		}

		impl std::ops::Deref for $name {
			type Target = $base;
			fn deref(&self) -> &$base {
				&self.base
			}
		}

		impl From<$base> for $name {
			fn from(base: $base) -> Self {
				$name { base, transforms: Default::default() }
			}
		}
	};
}

models_options!(ModelsApiStreamOptions, StreamOptions);
models_options!(ModelsSimpleStreamOptions, SimpleStreamOptions);
models_options!(ModelsDeferredFetchOptions, DeferredFetchOptions);
models_options!(ModelsDeferredCancelOptions, DeferredCancelOptions);
models_options!(ModelsImagesOptions, ImagesOptions);
models_options!(ModelsClassifierOptions, ClassifierOptions);

/// `ModelsRefreshOptions` upstream.
#[derive(Default, Clone)]
pub struct ModelsRefreshOptions {
	pub allow_network: Option<bool>,
	/// Restrict refresh to these provider IDs. Unknown and static providers
	/// are ignored.
	pub providers: Option<Vec<String>>,
	/// Bypass provider freshness checks and fetch immediately when network
	/// access is allowed.
	pub force: Option<bool>,
	pub signal: Option<crate::utils::abort::AbortSignal>,
}

/// `ModelsRefreshResult` upstream.
#[derive(Default)]
pub struct ModelsRefreshResult {
	pub aborted: bool,
	pub errors: BTreeMap<String, ThrownValue>,
}

/// `CreateModelsOptions` upstream.
pub struct CreateModelsOptions {
	pub credentials: Option<Arc<dyn crate::auth::types::CredentialStore>>,
	pub models_store: Option<Arc<dyn crate::models_store::ModelsStore>>,
	pub auth_context: Option<Arc<dyn crate::auth::types::AuthContext>>,
}

impl Default for CreateModelsOptions {
	fn default() -> Self {
		CreateModelsOptions { credentials: None, models_store: None, auth_context: None }
	}
}

// ==== Models collection (upstream models.ts ModelsImpl) ====
//
// Runtime collection of providers plus auth application and request
// convenience (`models.ts` lines 333-1043). Providers own request behavior;
// `Models` resolves auth and delegates each request to the provider that owns
// the model. Read accessors come in three flavors: the unqualified ones
// (`get_models`, `get_model`, `get_available`) return chat models, the
// `*_of_type` accessors return one model type, and `get_all_models` /
// `get_all_available` return every type.
//
// Deviations (continuing the crate list):
// - **D19 — publication store failures resolve `false`.** Upstream a failed
//   `modelsStore.write/delete` inside `context.publish` rejects the publish
//   promise, surfacing as a refresh error. The pinned `PublishHandle` is
//   infallible (`Future<Output = bool>`), so the failure resolves to `false`
//   ("superseded") instead. No upstream test exercises the rejection.
// - **D20 — deferred/image/classifier support checks run after auth.**
//   Upstream probes `provider.fetchDeferred !== undefined` before resolving
//   auth; the Rust `Provider` trait has no presence probes for those methods
//   (calling them builds a lazy future), so `Models` calls them after
//   `apply_auth` and maps `None` to the identical upstream error strings.
// - **D21 — provider reads are infallible.** Rust `Provider::get_models`
//   returns `Vec` and cannot throw; upstream's try/catch around provider
//   listing (`"swallows provider source failures"`) is structurally
//   unreachable and its test is skipped.
// - **D22 — refresh presence probing.** `has_refresh_models` invokes
//   `Provider::refresh_models` with a probe context and drops the returned
//   future unpolled (Rust futures run lazily), mirroring upstream's
//   `refreshModels !== undefined` check.
// - **D23 — signal identity.** Rust `AbortSignal` has no `===`; tests (and
//   the store-signal test below) compare abort-state propagation instead.

use std::collections::BTreeSet;
use std::sync::{OnceLock, Weak};
use {parking_lot::Mutex, parking_lot::RwLock};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::api::lazy::lazy_stream;
use crate::auth::context::default_provider_auth_context;
use crate::auth::credential_store::InMemoryCredentialStore;
use crate::auth::resolve::{
	refresh_stored_oauth_credential, resolve_provider_auth, AuthResolutionOverrides, AuthResolutionProvider,
};
use crate::auth::types::{
	ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthCheck, AuthContext, AuthInteraction, AuthOperationOptions,
	AuthResult, AuthType, Credential, CredentialStore, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential,
	ProviderAuth, ProviderAuthInteraction,
};
use crate::model_catalog::OrderedMap;
use crate::models_store::{InMemoryModelsStore, ModelsStore, ModelsStoreEntry, ModelsStoreOperationOptions};
use crate::types::{
	AssistantImages, AssistantMessageEvent, ImageModel, Message, ProviderEnv, StopReason, StreamDoneReason,
	UserContentField, UserMessage,
};
use crate::utils::abort::{operation_signal, race_with_abort_signal, AbortController, AbortReason, AbortSignal};
use crate::utils::diagnostics::now_ms;
use crate::utils::model_operations::{assert_chat_model, classifier_error_result, image_error_result};
use crate::utils::transcript::normalize_context;

/// The DOM `AbortError` carried by [`abort_reason_to_thrown`] — the
/// `name: "AbortError"` of upstream rejected operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("This operation was aborted")]
pub struct AbortError;

/// Upstream rejection reason of a raced operation (`signal.reason`).
pub fn abort_reason_to_thrown(reason: AbortReason) -> ThrownValue {
	match reason {
		AbortReason::Aborted => ThrownValue::error(AbortError),
		AbortReason::Message(message) => ThrownValue::message(message),
		AbortReason::Error(error) => ThrownValue::Error(error),
	}
}

/// Upstream `signal.throwIfAborted()`.
pub fn throw_if_aborted(signal: &AbortSignal) -> Result<(), ThrownValue> {
	signal.check().map_err(abort_reason_to_thrown)
}

/// Upstream `rejects.toMatchObject({ name: "AbortError" })`.
pub fn is_abort_error(value: &ThrownValue) -> bool {
	matches!(value, ThrownValue::Error(error) if error.is::<AbortError>())
}

/// Upstream `error.message` for any thrown value.
pub fn thrown_message(value: &ThrownValue) -> String {
	match value {
		ThrownValue::Error(error) => error.to_string(),
		ThrownValue::Text(text) => text.clone(),
	}
}

/// [`ModelsErrorCode`] of a thrown [`ModelsError`], if any.
pub fn thrown_models_code(value: &ThrownValue) -> Option<ModelsErrorCode> {
	match value {
		ThrownValue::Error(error) if error.is::<ModelsError>() => {
			Some(error.downcast_ref::<ModelsError>().expect("checked downcast").code)
		}
		_ => None,
	}
}

/// Run `operation` on a detached task and race only the *wait* with `signal`.
/// Cancellation must not drop the operation itself: upstream promises keep
/// running after their racer resolves, which is load-bearing — a started
/// OAuth refresh must persist its rotated token, and a superseded provider
/// refresh must still reach (and be rejected by) the generation gate.
async fn awaited_with_abort<T, F>(operation: F, signal: AbortSignal) -> Result<T, AbortReason>
where
	T: Send + 'static,
	F: Future<Output = T> + Send + 'static,
{
	let handle = tokio::spawn(operation);
	race_with_abort_signal(
		async move { handle.await.unwrap_or_else(|join_error| std::panic::resume_unwind(join_error.into_panic())) },
		signal,
	)
	.await
}

/// Upstream `mergeHeaders`: case-insensitive header override. Override keys
/// replace every base key that differs only by case (keeping the override's
/// casing); `None` overrides leave `base` untouched.
fn merge_headers(
	base: Option<&ProviderHeaders>,
	override_headers: Option<&ProviderHeaders>,
) -> Option<ProviderHeaders> {
	if base.is_none() && override_headers.is_none() {
		return None;
	}
	let mut merged = base.cloned().unwrap_or_default();
	if let Some(override_headers) = override_headers {
		for (name, value) in override_headers {
			let lower_name = name.to_lowercase();
			let lowercase_keys: Vec<String> = merged
				.keys()
				.filter(|existing| existing.to_lowercase() == lower_name)
				.cloned()
				.collect();
			for existing in lowercase_keys {
				merged.remove(&existing);
			}
			merged.insert(name.clone(), value.clone());
		}
	}
	Some(merged)
}

/// Drops stored models whose type this version does not know
/// (`withKnownModelTypes`). The drop happens at the [`ModelsStoreEntry`]
/// deserialization boundary (unknown `type` strings fail [`AnyModel::from_json`]
/// and the entry parse filters them), so this is the identity; kept for spec
/// parity.
fn with_known_model_types(entry: ModelsStoreEntry) -> ModelsStoreEntry {
	entry
}

/// Presence probe for [`Provider::refresh_models`] (`refreshModels !==
/// undefined` upstream, deviation D22): Rust futures run lazily, so invoking
/// the method builds and drops the future without polling it.
fn has_refresh_models(provider: &dyn Provider) -> bool {
	let context = RefreshModelsContext {
		credential: None,
		stored: None,
		publish: PublishHandle::new(|_publication| Box::pin(async { false })),
		allow_network: false,
		force: None,
		signal: AbortController::new().signal(),
	};
	provider.refresh_models(context).is_some()
}

/// `model.headers` across every model type (no upstream equivalent needed —
/// TS reads the shared property). Model headers are `Record<string, string>`;
/// request headers carry `null` suppression values, so the conversion wraps
/// each value.
fn model_headers(model: &AnyModel) -> Option<ProviderHeaders> {
	let headers = match model {
		AnyModel::Chat(model) => model.base.headers.clone(),
		AnyModel::Image(model) => model.base.headers.clone(),
		AnyModel::Classifier(model) => model.base.headers.clone(),
	};
	headers.map(|headers| headers.into_iter().map(|(name, value)| (name, Some(value))).collect())
}

/// Field access into the merged request options of the six
/// `Models*Options` wrappers (upstream spreads
/// `Omit<TOptions, "transformHeaders"> & ProviderRequestOptions<TModel>`).
pub trait ModelsOptionsAccess {
	fn request_options(&self) -> &crate::types::ProviderRequestOptions;
	fn request_options_mut(&mut self) -> &mut crate::types::ProviderRequestOptions;
	fn transforms(&self) -> &ModelsRequestTransforms;
}

impl ModelsOptionsAccess for ModelsApiStreamOptions {
	fn request_options(&self) -> &crate::types::ProviderRequestOptions {
		&self.base.base
	}
	fn request_options_mut(&mut self) -> &mut crate::types::ProviderRequestOptions {
		&mut self.base.base
	}
	fn transforms(&self) -> &ModelsRequestTransforms {
		&self.transforms
	}
}

impl ModelsOptionsAccess for ModelsSimpleStreamOptions {
	fn request_options(&self) -> &crate::types::ProviderRequestOptions {
		&self.base.base.base
	}
	fn request_options_mut(&mut self) -> &mut crate::types::ProviderRequestOptions {
		&mut self.base.base.base
	}
	fn transforms(&self) -> &ModelsRequestTransforms {
		&self.transforms
	}
}

impl ModelsOptionsAccess for ModelsDeferredFetchOptions {
	fn request_options(&self) -> &crate::types::ProviderRequestOptions {
		&self.base.base
	}
	fn request_options_mut(&mut self) -> &mut crate::types::ProviderRequestOptions {
		&mut self.base.base
	}
	fn transforms(&self) -> &ModelsRequestTransforms {
		&self.transforms
	}
}

impl ModelsOptionsAccess for ModelsDeferredCancelOptions {
	fn request_options(&self) -> &crate::types::ProviderRequestOptions {
		&self.base.base
	}
	fn request_options_mut(&mut self) -> &mut crate::types::ProviderRequestOptions {
		&mut self.base.base
	}
	fn transforms(&self) -> &ModelsRequestTransforms {
		&self.transforms
	}
}

impl ModelsOptionsAccess for ModelsImagesOptions {
	fn request_options(&self) -> &crate::types::ProviderRequestOptions {
		&self.base.base
	}
	fn request_options_mut(&mut self) -> &mut crate::types::ProviderRequestOptions {
		&mut self.base.base
	}
	fn transforms(&self) -> &ModelsRequestTransforms {
		&self.transforms
	}
}

impl ModelsOptionsAccess for ModelsClassifierOptions {
	fn request_options(&self) -> &crate::types::ProviderRequestOptions {
		&self.base.base
	}
	fn request_options_mut(&mut self) -> &mut crate::types::ProviderRequestOptions {
		&mut self.base.base
	}
	fn transforms(&self) -> &ModelsRequestTransforms {
		&self.transforms
	}
}

/// The concrete `Models` collection (`ModelsImpl`). All mutable state sits
/// behind locks; provider order is insertion-ordered like the upstream
/// `Map`.
pub struct ModelsImpl {
	/// Set once by [`create_models`]; async paths that must survive
	/// cancellation (or run behind a returned stream) clone this handle.
	me: OnceLock<Weak<ModelsImpl>>,
	/// `provider.id → provider`, insertion-ordered like the upstream `Map`.
	providers: RwLock<OrderedMap<Arc<dyn Provider>>>,
	credentials: Arc<dyn CredentialStore>,
	models_store: Arc<dyn ModelsStore>,
	auth_context: Arc<dyn AuthContext>,
	refresh_generations: Mutex<std::collections::BTreeMap<String, u64>>,
	refresh_controllers: Mutex<std::collections::BTreeMap<String, (u64, AbortController)>>,
	publication_chains: Mutex<std::collections::BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl ModelsImpl {
	fn self_arc(&self) -> Arc<ModelsImpl> {
		self.me
			.get()
			.and_then(Weak::upgrade)
			.expect("ModelsImpl must be created through create_models")
	}

	fn supersede_provider_refresh(&self, provider_id: &str) -> u64 {
		let generation = {
			let mut generations = self.refresh_generations.lock();
			let generation = generations.entry(provider_id.to_string()).or_insert(0);
			*generation += 1;
			*generation
		};
		if let Some(previous) = self.refresh_controllers.lock().remove(provider_id) {
			previous.1.abort();
		}
		generation
	}

	fn begin_provider_refresh(&self, provider_id: &str) -> (u64, AbortSignal) {
		let generation = self.supersede_provider_refresh(provider_id);
		let controller = AbortController::new();
		self.refresh_controllers
			.lock()
			.insert(provider_id.to_string(), (generation, controller.clone()));
		(generation, controller.signal())
	}

	fn finish_provider_refresh(&self, provider_id: &str, generation: u64) {
		let mut controllers = self.refresh_controllers.lock();
		if controllers.get(provider_id).is_some_and(|(stored, _)| *stored == generation) {
			controllers.remove(provider_id);
		}
	}

	/// Upstream `publishProviderModels`: publications for one provider run
	/// serially through a generation-checked gate; the persistence survives
	/// caller cancellation (detached task) while the caller's wait races the
	/// abort signal.
	async fn publish_provider_models(
		&self,
		provider_id: &str,
		generation: u64,
		signal: AbortSignal,
		publication: ModelsPublication,
	) -> bool {
		let chain = {
			self.publication_chains
				.lock()
				.entry(provider_id.to_string())
				.or_default()
				.clone()
		};
		let this = self.self_arc();
		let provider_id = provider_id.to_string();
		let task_signal = signal.clone();
		let (sender, receiver) = tokio::sync::oneshot::channel::<bool>();
		tokio::spawn(async move {
			let _guard = chain.lock().await;
			let published = run_publication(&this, &provider_id, generation, &task_signal, publication).await;
			let _ = sender.send(published);
			let mut chains = this.publication_chains.lock();
			if chains.get(&provider_id).is_some_and(|existing| Arc::ptr_eq(existing, &chain)) {
				chains.remove(&provider_id);
			}
		});
		match race_with_abort_signal(async { receiver.await.unwrap_or(false) }, signal).await {
			Ok(published) => published,
			// Upstream rejects with the abort reason; the refresh operation is
			// itself raced with the same signal, so resolving `false` is
			// observably identical.
			Err(_aborted) => false,
		}
	}

	async fn run_provider_refresh_phase(
		&self,
		provider: &Arc<dyn Provider>,
		credential: Option<Credential>,
		allow_network: bool,
		force: Option<bool>,
		generation: u64,
		signal: AbortSignal,
	) -> Result<(), ThrownValue> {
		let stored = self
			.models_store
			.read(provider.id(), Some(ModelsStoreOperationOptions { signal: Some(signal.clone()) }))
			.await?;
		let stored = stored.map(with_known_model_types);
		let publish = publication_handle(self.self_arc(), provider.id().to_string(), generation, signal.clone());
		let context = RefreshModelsContext {
			credential,
			stored,
			publish,
			allow_network,
			force: if allow_network { force } else { None },
			signal: signal.clone(),
		};
		match provider.refresh_models(context) {
			Some(operation) => operation.await,
			None => Ok(()),
		}
	}

	async fn refresh_impl(self: Arc<Self>, options: Option<ModelsRefreshOptions>) -> ModelsRefreshResult {
		let options = options.unwrap_or_default();
		let allow_network = options.allow_network.unwrap_or(true);
		let force = options.force;
		let caller_signal = operation_signal(options.signal.as_ref());
		if caller_signal.aborted() {
			return ModelsRefreshResult::default();
		}
		let selected: Option<BTreeSet<String>> =
			options.providers.as_ref().map(|providers| providers.iter().cloned().collect());
		let refreshable: Vec<Arc<dyn Provider>> = self
			.get_providers()
			.into_iter()
			.filter(|provider| {
				has_refresh_models(provider.as_ref())
					&& selected.as_ref().map_or(true, |selected| selected.contains(provider.id()))
			})
			.collect();
		let errors: Arc<Mutex<std::collections::BTreeMap<String, ThrownValue>>> = Arc::default();
		let mut handles = Vec::with_capacity(refreshable.len());
		for provider in refreshable {
			let this = self.clone();
			let errors = errors.clone();
			let caller_signal = caller_signal.clone();
			handles.push(tokio::spawn(async move {
				let (generation, controller_signal) = this.begin_provider_refresh(provider.id());
				let signal = AbortSignal::any([caller_signal.clone(), controller_signal]);
				let operation = {
					let this = this.clone();
					let provider = provider.clone();
					let signal = signal.clone();
					async move {
						let credential_result = this.read_credential(provider.id(), signal.clone()).await;
						let credential_error = credential_result.as_ref().err().cloned();
						let stored_credential = credential_result.ok().flatten();
						this.run_provider_refresh_phase(&provider, stored_credential.clone(), false, None, generation, signal.clone())
							.await?;
						if let Some(error) = credential_error {
							return Err(error);
						}
						if !allow_network || signal.aborted() {
							return Ok(());
						}
						match this.resolve_refresh_credential(provider.as_ref(), stored_credential, signal.clone()).await? {
							Some(credential) => {
								this.run_provider_refresh_phase(&provider, Some(credential), true, force, generation, signal.clone())
									.await
							}
							None => Ok(()),
						}
					}
				};
				match awaited_with_abort(operation, signal.clone()).await {
					Ok(Ok(())) => {}
					Ok(Err(error)) => {
						if !signal.aborted() {
							// Upstream stores thrown `Error`s as-is and wraps
							// non-Error values in a `model_source` ModelsError.
							let error = match error {
								ThrownValue::Text(text) => ThrownValue::error(ModelsError::with_cause(
									ModelsErrorCode::ModelSource,
									format!("Model refresh failed for {}", provider.id()),
									Some(&ThrownValue::Text(text)),
								)),
								error => error,
							};
							errors.lock().insert(provider.id().to_string(), error);
						}
					}
					Err(_aborted) => {}
				}
				this.finish_provider_refresh(provider.id(), generation);
			}));
		}
		let _ = race_with_abort_signal(futures_util::future::join_all(handles), caller_signal.clone()).await;
		let errors = errors.lock().clone();
		ModelsRefreshResult { aborted: caller_signal.aborted(), errors }
	}

	async fn resolve_refresh_credential(
		&self,
		provider: &dyn Provider,
		stored: Option<Credential>,
		signal: AbortSignal,
	) -> Result<Option<Credential>, ThrownValue> {
		if let Some(Credential::OAuth(oauth_credential)) = &stored {
			let Some(oauth) = provider.auth().oauth.clone() else {
				return Ok(None);
			};
			if now_ms() < oauth_credential.credentials.expires {
				return Ok(stored);
			}
			if signal.aborted() {
				return Ok(None);
			}
			// A started refresh survives cancellation or a superseding model
			// refresh (upstream comment: a rotated refresh token is always
			// persisted); callers race this with their signal.
			return refresh_stored_oauth_credential(
				self.credentials.clone(),
				provider.id(),
				oauth,
				move |current: &OAuthCredential| now_ms() >= current.credentials.expires,
				signal,
			)
			.await
			.map(|credential| credential.map(Credential::OAuth));
		}

		let Some(api_key) = provider.auth().api_key.as_ref() else {
			return Ok(None);
		};
		let credential = match &stored {
			Some(Credential::ApiKey(api_key_credential)) => Some(api_key_credential),
			_ => None,
		};
		let resolution = (api_key.resolve)(ApiKeyAuthInput {
			ctx: self.auth_context.clone(),
			credential,
			signal: signal.clone(),
		})
		.await?;
		Ok(resolution.map(|resolution| Credential::ApiKey(ApiKeyCredential {
			key: resolution.auth.api_key.clone(),
			env: resolution.env.clone(),
		})))
	}

	async fn read_credential(&self, provider_id: &str, signal: AbortSignal) -> Result<Option<Credential>, ThrownValue> {
		self.credentials
			.read(provider_id, Some(AuthOperationOptions { signal: Some(signal) }))
			.await
			.map_err(|error| {
				ThrownValue::error(ModelsError::with_cause(
					ModelsErrorCode::Auth,
					format!("Credential store read failed for {provider_id}"),
					Some(&error),
				))
			})
	}

	async fn check_provider_auth(
		&self,
		provider: &dyn Provider,
		credential: Option<Credential>,
		signal: AbortSignal,
	) -> Result<Option<AuthCheck>, ThrownValue> {
		if matches!(credential, Some(Credential::OAuth(_))) {
			return Ok(provider.auth().oauth.is_some().then(|| AuthCheck {
				source: Some("OAuth".to_string()),
				check_type: AuthType::OAuth,
			}));
		}
		let Some(api_key) = provider.auth().api_key.as_ref() else {
			return Ok(None);
		};
		if let Some(check) = api_key.check.as_ref() {
			let credential = match &credential {
				Some(Credential::ApiKey(api_key_credential)) => Some(api_key_credential),
				_ => None,
			};
			return check(ApiKeyAuthInput { ctx: self.auth_context.clone(), credential, signal }).await.map_err(|error| {
				ThrownValue::error(ModelsError::with_cause(
					ModelsErrorCode::Auth,
					format!("API key auth check failed for provider {}", provider.id()),
					Some(&error),
				))
			});
		}

		let resolution = resolve_provider_auth_view(provider, self.credentials.clone(), self.auth_context.clone(), Some(AuthResolutionOverrides { signal: Some(signal), ..Default::default() })).await?;
		Ok(resolution.map(|resolution| AuthCheck { source: resolution.source, check_type: AuthType::ApiKey }))
	}

	async fn get_authenticated_providers(
		&self,
		provider_id: Option<&str>,
		signal: AbortSignal,
	) -> Result<Vec<(Arc<dyn Provider>, Option<Credential>, AuthCheck)>, ThrownValue> {
		throw_if_aborted(&signal)?;
		let providers: Vec<Arc<dyn Provider>> = match provider_id {
			Some(provider_id) => self.get_provider(provider_id).into_iter().collect(),
			None => self.get_providers(),
		};
		let mut authenticated = Vec::with_capacity(providers.len());
		for provider in providers {
			let credential = self.read_credential(provider.id(), signal.clone()).await?;
			let Some(check) = self.check_provider_auth(provider.as_ref(), credential.clone(), signal.clone()).await? else {
				continue;
			};
			authenticated.push((provider, credential, check));
		}
		Ok(authenticated)
	}

	async fn get_auth_inner(
		&self,
		provider_id: &str,
		model_headers: Option<ProviderHeaders>,
		overrides: Option<AuthResolutionOverrides>,
	) -> Result<Option<AuthResult>, ThrownValue> {
		let signal = operation_signal(overrides.as_ref().and_then(|entry| entry.signal.as_ref()));
		let mut overrides = overrides.unwrap_or_default();
		overrides.signal = Some(signal.clone());
		let provider = self.get_provider(provider_id);
		let operation = {
			let credentials = self.credentials.clone();
			let auth_context = self.auth_context.clone();
			async move {
				let Some(provider) = provider else {
					return Ok(None);
				};
				let view = AuthResolutionProvider {
					id: provider.id(),
					api_key: provider.auth().api_key.as_ref(),
					oauth: provider.auth().oauth.clone(),
				};
				resolve_provider_auth(view, credentials, auth_context, Some(overrides)).await
			}
		};
		let resolution = match awaited_with_abort(operation, signal).await {
			Ok(result) => result,
			Err(reason) => return Err(abort_reason_to_thrown(reason)),
		}?;
		let Some(mut result) = resolution else {
			return Ok(None);
		};
		let Some(headers) = model_headers else {
			return Ok(Some(result));
		};
		result.auth.headers = merge_headers(result.auth.headers.as_ref(), Some(&headers));
		Ok(Some(result))
	}

	fn require_provider(&self, model: &AnyModel) -> Result<Arc<dyn Provider>, ModelsError> {
		self.get_provider(model.provider())
			.ok_or_else(|| ModelsError::new(ModelsErrorCode::Provider, format!("Unknown provider: {}", model.provider())))
	}

	fn require_chat_provider(&self, model: &AnyModel) -> Result<Arc<dyn Provider>, ModelsError> {
		assert_chat_model(model)?;
		self.require_provider(model)
	}

	/// Upstream `applyAuth`: resolve provider auth for one request, merge the
	/// explicit per-request options over it per field, run the Models-only
	/// header transform last, and stamp the resolved `baseUrl` onto the
	/// request model.
	async fn apply_auth<O: ModelsOptionsAccess + Default>(
		&self,
		model: AnyModel,
		options: Option<O>,
	) -> Result<(AnyModel, O), ThrownValue> {
		self.require_provider(&model).map_err(ThrownValue::error)?;
		let overrides = options.as_ref().map(|options| AuthResolutionOverrides {
			api_key: options.request_options().api_key.clone(),
			env: options.request_options().env.clone(),
			signal: options.request_options().signal.clone(),
			..Default::default()
		});
		let resolution = self
			.get_auth_inner(model.provider(), model_headers(&model), overrides)
			.await?
			.ok_or_else(|| {
				ThrownValue::error(ModelsError::new(
					ModelsErrorCode::Auth,
					format!("Provider is not configured: {}", model.provider()),
				))
			})?;
		let auth = resolution.auth;

		let mut options = options.unwrap_or_default();
		let explicit = options.request_options().clone();
		// Explicit request options win per field; the Models-only transform
		// runs last.
		let api_key = explicit.api_key.clone().or_else(|| auth.api_key.clone());
		let mut headers = merge_headers(auth.headers.as_ref(), explicit.headers.as_ref());
		if let Some(transform) = options.transforms().transform_headers.as_ref() {
			headers = transform(Some(headers.unwrap_or_default())).await;
		}
		let env = match (resolution.env.clone(), explicit.env.clone()) {
			(Some(mut resolved), Some(explicit)) => {
				resolved.extend(explicit);
				Some(resolved)
			}
			(None, Some(explicit)) => Some(explicit),
			(resolved, None) => resolved,
		};
		let request_model = match (&auth.base_url, &model) {
			(Some(base_url), AnyModel::Chat(chat)) => AnyModel::Chat(Model {
				base: crate::types::BaseModelFields { base_url: base_url.clone(), ..chat.base.clone() },
				..chat.clone()
			}),
			(Some(base_url), AnyModel::Image(image)) => AnyModel::Image(ImageModel {
				base: crate::types::BaseModelFields { base_url: base_url.clone(), ..image.base.clone() },
				..image.clone()
			}),
			(Some(base_url), AnyModel::Classifier(classifier)) => AnyModel::Classifier(ClassifierModel {
				base: crate::types::BaseModelFields { base_url: base_url.clone(), ..classifier.base.clone() },
				..classifier.clone()
			}),
			(None, _) => model.clone(),
		};
		let request = options.request_options_mut();
		request.api_key = api_key;
		request.headers = headers;
		request.env = env;
		Ok((request_model, options))
	}
}

/// Generation-checked persistence body of [`ModelsImpl::publish_provider_models`].
async fn run_publication(
	this: &ModelsImpl,
	provider_id: &str,
	generation: u64,
	signal: &AbortSignal,
	publication: ModelsPublication,
) -> bool {
	let generation_current = || {
		*this.refresh_generations.lock().get(provider_id).unwrap_or(&0) == generation
	};
	if signal.aborted() || !generation_current() {
		return false;
	}
	let persisted = match publication.persist {
		Some(ModelsStoreWrite::Delete) => {
			this.models_store
				.delete(provider_id, Some(ModelsStoreOperationOptions { signal: Some(signal.clone()) }))
				.await
		}
		Some(ModelsStoreWrite::Write(entry)) => {
			this.models_store
				.write(provider_id, entry, Some(ModelsStoreOperationOptions { signal: Some(signal.clone()) }))
				.await
		}
		None => Ok(()),
	};
	// Upstream a failed persistence rejects `context.publish` (surfacing as a
	// refresh error); the pinned `PublishHandle` is infallible, so the
	// failure resolves to `false` (deviation D19).
	if persisted.is_err() {
		return false;
	}
	if signal.aborted() || !generation_current() {
		return false;
	}
	if let Some(update) = publication.update {
		update();
	}
	true
}

/// Shared, generation-checked publication callback handed to provider
/// refreshes (`context.publish`).
fn publication_handle(this: Arc<ModelsImpl>, provider_id: String, generation: u64, signal: AbortSignal) -> PublishHandle {
	PublishHandle::new(move |publication| {
		let this = this.clone();
		let provider_id = provider_id.clone();
		let signal = signal.clone();
		Box::pin(async move { this.publish_provider_models(&provider_id, generation, signal, publication).await })
	})
}

/// [`AuthResolutionProvider`] view of a provider, borrowed for the duration
/// of one resolution call.
fn resolve_provider_auth_view<'a>(
	provider: &'a dyn Provider,
	credentials: Arc<dyn CredentialStore>,
	auth_context: Arc<dyn AuthContext>,
	overrides: Option<AuthResolutionOverrides>,
) -> impl Future<Output = Result<Option<AuthResult>, ThrownValue>> + Send + 'a {
	let view = AuthResolutionProvider {
		id: provider.id(),
		api_key: provider.auth().api_key.as_ref(),
		oauth: provider.auth().oauth.clone(),
	};
	resolve_provider_auth(view, credentials, auth_context, overrides)
}

#[async_trait::async_trait]
impl Models for ModelsImpl {
	fn get_providers(&self) -> Vec<Arc<dyn Provider>> {
		self.providers.read().values().cloned().collect()
	}

	fn get_provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
		self.providers.read().get(id).cloned()
	}

	fn get_models(&self, provider: Option<&str>) -> Vec<Model> {
		match provider {
			Some(provider_id) => self.get_provider(provider_id).map(|provider| provider.get_models()).unwrap_or_default(),
			None => self.get_providers().into_iter().flat_map(|provider| provider.get_models()).collect(),
		}
	}

	fn get_model(&self, provider: &str, id: &str) -> Option<Model> {
		self.get_models(Some(provider)).into_iter().find(|model| model.base.id == id)
	}

	fn get_models_of_type(&self, type_: ModelType, provider: Option<&str>) -> Vec<AnyModel> {
		self.get_all_models(provider).into_iter().filter(|model| model.model_type() == type_).collect()
	}

	fn get_model_of_type(&self, type_: ModelType, provider: &str, id: &str) -> Option<AnyModel> {
		self.get_models_of_type(type_, Some(provider)).into_iter().find(|model| model.id() == id)
	}

	fn get_all_models(&self, provider: Option<&str>) -> Vec<AnyModel> {
		match provider {
			Some(provider_id) => {
				self.get_provider(provider_id).map(|provider| provider.get_all_models()).unwrap_or_default()
			}
			None => self.get_providers().into_iter().flat_map(|provider| provider.get_all_models()).collect(),
		}
	}

	async fn refresh(&self, options: Option<ModelsRefreshOptions>) -> ModelsRefreshResult {
		let arc = self.self_arc();
		ModelsImpl::refresh_impl(arc, options).await
	}

	async fn check_auth(
		&self,
		provider_id: &str,
		options: Option<AuthOperationOptions>,
	) -> Result<Option<AuthCheck>, ThrownValue> {
		let signal = operation_signal(options.as_ref().and_then(|entry| entry.signal.as_ref()));
		let this = self.self_arc();
		let provider_id = provider_id.to_string();
		let operation = {
			let signal = signal.clone();
			async move {
			throw_if_aborted(&signal)?;
			let Some(provider) = this.get_provider(&provider_id) else {
				return Ok(None);
			};
			let credential = this.read_credential(&provider_id, signal.clone()).await?;
			this.check_provider_auth(provider.as_ref(), credential, signal.clone()).await
			}
		};
		match awaited_with_abort(operation, signal).await {
			Ok(result) => result,
			Err(reason) => Err(abort_reason_to_thrown(reason)),
		}
	}

	async fn get_available(
		&self,
		provider_id: Option<&str>,
		options: Option<AuthOperationOptions>,
	) -> Result<Vec<Model>, ThrownValue> {
		let signal = operation_signal(options.as_ref().and_then(|entry| entry.signal.as_ref()));
		let this = self.self_arc();
		let provider_id = provider_id.map(str::to_string);
		let operation = {
			let signal = signal.clone();
			async move {
			let authenticated = this.get_authenticated_providers(provider_id.as_deref(), signal.clone()).await?;
			let mut models = Vec::new();
			for (provider, credential, _check) in authenticated {
				let provider_models = provider.get_models();
				match provider.filter_models(&provider_models, credential.as_ref()) {
					Some(filtered) => models.extend(filtered),
					None => models.extend(provider_models),
				}
			}
			Ok(models)
			}
		};
		match awaited_with_abort(operation, signal).await {
			Ok(result) => result,
			Err(reason) => Err(abort_reason_to_thrown(reason)),
		}
	}

	async fn get_all_available(
		&self,
		provider_id: Option<&str>,
		options: Option<AuthOperationOptions>,
	) -> Result<Vec<AnyModel>, ThrownValue> {
		let signal = operation_signal(options.as_ref().and_then(|entry| entry.signal.as_ref()));
		let this = self.self_arc();
		let provider_id = provider_id.map(str::to_string);
		let operation = {
			let signal = signal.clone();
			async move {
			let authenticated = this.get_authenticated_providers(provider_id.as_deref(), signal.clone()).await?;
			let mut models = Vec::new();
			for (provider, credential, _check) in authenticated {
				let provider_models = provider.get_all_models();
				if let Some(filtered) = provider.filter_all_models(&provider_models, credential.as_ref()) {
					models.extend(filtered);
					continue;
				}
				let chat_models = provider.get_models();
				let Some(available_chat) = provider.filter_models(&chat_models, credential.as_ref()) else {
					models.extend(provider_models);
					continue;
				};
				let available_chat_ids: BTreeSet<String> =
					available_chat.into_iter().map(|model| model.base.id).collect();
				models.extend(provider_models.into_iter().filter(|model| {
					model.model_type() != ModelType::Chat || available_chat_ids.contains(model.id())
				}));
			}
			Ok(models)
			}
		};
		match awaited_with_abort(operation, signal).await {
			Ok(result) => result,
			Err(reason) => Err(abort_reason_to_thrown(reason)),
		}
	}

	async fn get_auth_by_provider(
		&self,
		provider_id: &str,
		overrides: Option<AuthResolutionOverrides>,
	) -> Result<Option<AuthResult>, ThrownValue> {
		self.get_auth_inner(provider_id, None, overrides).await
	}

	async fn get_auth_by_model(
		&self,
		model: &AnyModel,
		overrides: Option<AuthResolutionOverrides>,
	) -> Result<Option<AuthResult>, ThrownValue> {
		let provider_id = model.provider().to_string();
		let headers = model_headers(model);
		self.get_auth_inner(&provider_id, headers, overrides).await
	}

	async fn login(
		&self,
		provider_id: &str,
		type_: AuthType,
		interaction: AuthInteraction,
		options: Option<LoginOptions>,
	) -> Result<Credential, ThrownValue> {
		let signal = operation_signal(interaction.signal.as_ref());
		throw_if_aborted(&signal)?;
		let provider = self.get_provider(provider_id).ok_or_else(|| {
			ThrownValue::error(ModelsError::new(
				ModelsErrorCode::Provider,
				format!("Unknown provider: {provider_id}"),
			))
		})?;
		let login_operation: std::pin::Pin<Box<dyn Future<Output = Result<Credential, ThrownValue>> + Send>> =
			match type_ {
				AuthType::OAuth => {
					let Some(oauth) = provider.auth().oauth.as_ref() else {
						return Err(ThrownValue::error(ModelsError::new(
							ModelsErrorCode::Auth,
							format!("{} does not support {} login", provider.name(), auth_type_name(type_)),
						)));
					};
					let interaction = ProviderAuthInteraction::from_interaction(interaction, signal.clone());
					let credential = (oauth.login)(interaction, options);
					Box::pin(async move { credential.await.map(Credential::OAuth) })
				}
				AuthType::ApiKey => {
					let method = provider.auth().api_key.as_ref().filter(|method| method.login.is_some());
					let Some(method) = method else {
						return Err(ThrownValue::error(ModelsError::new(
							ModelsErrorCode::Auth,
							format!("{} does not support {} login", provider.name(), auth_type_name(type_)),
						)));
					};
					let interaction = ProviderAuthInteraction::from_interaction(interaction, signal.clone());
					let login = method.login.as_ref().expect("filtered above");
					let credential = login(interaction);
					Box::pin(async move { credential.await.map(Credential::ApiKey) })
				}
			};
		let credential = match race_with_abort_signal(login_operation, signal.clone()).await {
			Ok(Ok(credential)) => credential,
			Ok(Err(error)) => return Err(error),
			Err(reason) => return Err(abort_reason_to_thrown(reason)),
		};

		// Persist through the store's serialized write path. The wait resolves
		// when the mutation started (or completed); an abort before the
		// mutation started cancels it, an abort afterwards lets it finish.
		let mutation_started = Arc::new(AtomicBool::new(false));
		let started = Arc::new(tokio::sync::Notify::new());
		let credential_for_store = credential.clone();
		let started_for_store = started.clone();
		let started_flag = mutation_started.clone();
		let mutation = self.credentials.modify(
			provider_id,
			Box::new(move |_current| {
				started_flag.store(true, Ordering::SeqCst);
				started_for_store.notify_one();
				Box::pin(async move { Ok(Some(credential_for_store)) })
			}),
			Some(AuthOperationOptions { signal: Some(signal.clone()) }),
		);
		futures_util::pin_mut!(mutation);
		let mut mutation_result: Option<Result<Option<Credential>, ThrownValue>> = None;
		// Wait until the mutation started (an abort cancels it only before
		// that point), then run it to completion — an abort afterwards must
		// not discard a credential the store already persisted.
		let aborted_before_start = loop {
			let started_already = mutation_started.load(Ordering::SeqCst);
			tokio::select! {
				biased;
				_ = signal.aborted_future(), if !started_already => break true,
				_ = started.notified() => {
					mutation_started.store(true, Ordering::SeqCst);
				}
				outcome = &mut mutation => {
					mutation_result = Some(outcome);
					break false;
				}
			}
		};
		if aborted_before_start {
			// Upstream rejects with the signal reason and rethrows after
			// `throwIfAborted`.
			return Err(abort_reason_to_thrown(signal.reason().unwrap_or(AbortReason::Aborted)));
		}
		match mutation_result {
			Some(Err(error)) => {
				throw_if_aborted(&signal)?;
				Err(ThrownValue::error(ModelsError::with_cause(
					ModelsErrorCode::Auth,
					format!("Credential store modify failed for {provider_id}"),
					Some(&error),
				)))
			}
			_ => Ok(credential),
		}
	}

	async fn logout(&self, provider_id: &str, options: Option<AuthOperationOptions>) -> Result<(), ThrownValue> {
		let signal = operation_signal(options.as_ref().and_then(|entry| entry.signal.as_ref()));
		throw_if_aborted(&signal)?;
		match self.credentials.delete(provider_id, Some(AuthOperationOptions { signal: Some(signal.clone()) })).await {
			Ok(()) => Ok(()),
			Err(error) => {
				throw_if_aborted(&signal)?;
				Err(ThrownValue::error(ModelsError::with_cause(
					ModelsErrorCode::Auth,
					format!("Credential store delete failed for {provider_id}"),
					Some(&error),
				)))
			}
		}
	}

	fn stream(
		&self,
		model: &Model,
		context: &Context,
		options: Option<ModelsApiStreamOptions>,
	) -> AssistantMessageEventStream {
		let transcript = normalize_context(context);
		let this = self.self_arc();
		let stream_model = model.clone();
		let model = model.clone();
		lazy_stream(&stream_model, async move {
			let any = AnyModel::Chat(model.clone());
			let provider = this.require_chat_provider(&any).map_err(ThrownValue::error)?;
			let (request_model, request_options) = this.apply_auth(any, options).await?;
			let AnyModel::Chat(request_model) = request_model else {
				return Err(ThrownValue::message("internal error: chat request narrowed to a non-chat model"));
			};
			Ok(provider.stream(&request_model, &transcript, &request_options.base))
		})
	}

	async fn complete(
		&self,
		model: &Model,
		context: &Context,
		options: Option<ModelsApiStreamOptions>,
	) -> AssistantMessage {
		self.stream(model, context, options).result().await
	}

	fn stream_simple(
		&self,
		model: &Model,
		context: &Context,
		options: Option<ModelsSimpleStreamOptions>,
	) -> AssistantMessageEventStream {
		let transcript = normalize_context(context);
		let this = self.self_arc();
		let stream_model = model.clone();
		let model = model.clone();
		lazy_stream(&stream_model, async move {
			let any = AnyModel::Chat(model.clone());
			let provider = this.require_chat_provider(&any).map_err(ThrownValue::error)?;
			let (request_model, request_options) = this.apply_auth(any, options).await?;
			let AnyModel::Chat(request_model) = request_model else {
				return Err(ThrownValue::message("internal error: chat request narrowed to a non-chat model"));
			};
			Ok(provider.stream_simple(&request_model, &transcript, &request_options.base))
		})
	}

	async fn complete_simple(
		&self,
		model: &Model,
		context: &Context,
		options: Option<ModelsSimpleStreamOptions>,
	) -> AssistantMessage {
		self.stream_simple(model, context, options).result().await
	}

	fn stream_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: Option<ModelsDeferredFetchOptions>,
	) -> AssistantMessageEventStream {
		let this = self.self_arc();
		let stream_model = model.clone();
		let model = model.clone();
		let handle = handle.clone();
		lazy_stream(&stream_model, async move {
			let any = AnyModel::Chat(model.clone());
			let provider = this.require_chat_provider(&any).map_err(ThrownValue::error)?;
			let (request_model, request_options) = this.apply_auth(any, options).await?;
			let AnyModel::Chat(request_model) = &request_model else {
				return Err(ThrownValue::message("internal error: chat request narrowed to a non-chat model"));
			};
			match provider.fetch_deferred(request_model, &handle, &request_options.base) {
				Some(stream) => Ok(stream),
				None => Err(ThrownValue::error(ModelsError::new(
					ModelsErrorCode::Provider,
					format!("Provider {} does not support deferred responses", model.base.provider),
				))),
			}
		})
	}

	async fn fetch_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: Option<ModelsDeferredFetchOptions>,
	) -> AssistantMessage {
		self.stream_deferred(model, handle, options).result().await
	}

	async fn cancel_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: Option<ModelsDeferredCancelOptions>,
	) -> Result<(), ThrownValue> {
		let any = AnyModel::Chat(model.clone());
		let provider = self.require_chat_provider(&any).map_err(ThrownValue::error)?;
		let (request_model, request_options) = self.apply_auth(any, options).await?;
		let AnyModel::Chat(request_model) = &request_model else {
			return Err(ThrownValue::message("internal error: chat request narrowed to a non-chat model"));
		};
		let operation = provider.cancel_deferred(request_model, handle, &request_options.base);
		match operation {
			Some(operation) => operation.await,
			None => Err(ThrownValue::error(ModelsError::new(
				ModelsErrorCode::Provider,
				format!("Provider {} does not support deferred responses", model.base.provider),
			))),
		}
	}

	async fn generate_images(
		&self,
		model: &ImageModel,
		context: &ImagesContext,
		options: Option<ModelsImagesOptions>,
	) -> AssistantImages {
		let signal = options.as_ref().and_then(|options| options.base.signal.clone());
		let result: Result<AssistantImages, ThrownValue> = async {
			let any = AnyModel::Image(model.clone());
			let provider = self.require_provider(&any).map_err(ThrownValue::error)?;
			let (request_model, request_options) = self.apply_auth(any, options).await?;
			let AnyModel::Image(request_model) = &request_model else {
				return Err(ThrownValue::message("internal error: image request narrowed to a non-image model"));
			};
			let Some(operation) = provider.generate_images(request_model, context, &request_options.base) else {
				return Err(ThrownValue::error(ModelsError::new(
					ModelsErrorCode::Provider,
					format!("Provider {} does not support image generation", model.base.provider),
				)));
			};
			Ok(operation.await)
		}
		.await;
		match result {
			Ok(images) => images,
			Err(error) => image_error_result(model, &error, signal.is_some_and(|signal| signal.aborted())),
		}
	}

	async fn classify(
		&self,
		model: &ClassifierModel,
		context: &ClassifierContext,
		options: Option<ModelsClassifierOptions>,
	) -> ClassifierResult {
		let signal = options.as_ref().and_then(|options| options.base.signal.clone());
		let result: Result<ClassifierResult, ThrownValue> = async {
			let any = AnyModel::Classifier(model.clone());
			let provider = self.require_provider(&any).map_err(ThrownValue::error)?;
			let (request_model, request_options) = self.apply_auth(any, options).await?;
			let AnyModel::Classifier(request_model) = &request_model else {
				return Err(ThrownValue::message("internal error: classifier request narrowed to a non-classifier model"));
			};
			let Some(operation) = provider.classify(request_model, context, &request_options.base) else {
				return Err(ThrownValue::error(ModelsError::new(
					ModelsErrorCode::Provider,
					format!("Provider {} does not support classification", model.base.provider),
				)));
			};
			Ok(operation.await)
		}
		.await;
		match result {
			Ok(result) => result,
			Err(error) => classifier_error_result(model, &error, signal.is_some_and(|signal| signal.aborted())),
		}
	}
}

impl ModelsImpl {
	/// Upstream `getAvailableOfType` — present on the upstream `Models`
	/// interface but missing from the pinned Rust `Models` trait, so it is
	/// exposed as an inherent method (reported to Main).
	pub async fn get_available_of_type(
		&self,
		type_: ModelType,
		provider_id: Option<&str>,
		options: Option<AuthOperationOptions>,
	) -> Result<Vec<AnyModel>, ThrownValue> {
		let models = Models::get_all_available(self, provider_id, options).await?;
		Ok(models.into_iter().filter(|model| model.model_type() == type_).collect())
	}
}

impl MutableModels for ModelsImpl {
	fn set_provider(&self, provider: Arc<dyn Provider>) {
		self.supersede_provider_refresh(provider.id());
		self.providers.write().insert(provider.id().to_string(), provider);
	}

	fn delete_provider(&self, id: &str) {
		self.supersede_provider_refresh(id);
		self.providers.write().remove(id);
	}

	fn clear_providers(&self) {
		let ids: Vec<String> = {
			let providers = self.providers.read();
			let controllers = self.refresh_controllers.lock();
			providers
				.keys()
				.map(str::to_string)
				.chain(controllers.keys().cloned())
				.collect()
		};
		for id in ids {
			self.supersede_provider_refresh(&id);
		}
		self.providers.write().clear();
	}
}

/// Upstream `AuthType` interpolation in login error messages
/// (`"api_key" | "oauth"`).
fn auth_type_name(type_: AuthType) -> &'static str {
	match type_ {
		AuthType::ApiKey => "api_key",
		AuthType::OAuth => "oauth",
	}
}

/// Upstream `createModels`.
pub fn create_models(options: Option<CreateModelsOptions>) -> Arc<ModelsImpl> {
	let options = options.unwrap_or_default();
	let credentials: Arc<dyn CredentialStore> = options
		.credentials
		.unwrap_or_else(|| Arc::new(InMemoryCredentialStore::new()));
	let models_store: Arc<dyn ModelsStore> =
		options.models_store.unwrap_or_else(crate::models_store::in_memory_models_store);
	let auth_context: Arc<dyn AuthContext> =
		options.auth_context.unwrap_or_else(default_provider_auth_context);
	let models = ModelsImpl {
		me: OnceLock::new(),
		providers: RwLock::default(),
		credentials,
		models_store,
		auth_context,
		refresh_generations: Mutex::default(),
		refresh_controllers: Mutex::default(),
		publication_chains: Mutex::default(),
	};
	let arc = Arc::new(models);
	let _ = arc.me.set(Arc::downgrade(&arc));
	arc
}

// ==== Tests (ported from upstream `test/models-runtime.test.ts`,
// `test/models-entry.test.ts` runtime subset, and
// `test/model-data-validation.test.ts` runtime-observable invariants) ====
//
// Skipped upstream cases and why (no silent drops):
// - models-runtime "swallows provider source failures for both all-provider
//   and single-provider listing": Rust `Provider::get_models` returns `Vec`
//   and cannot throw (deviation D21); the unknown-provider half of the
//   behavior is covered by `lists_and_finds_models_per_provider`.
// - models-runtime "keeps chat reads independent from the all-model catalog"
//   relies on `getAllModels` *throwing*; ported as the Rust-expressible
//   variant (an explicit empty `get_all_models` override) below.
// - models-entry's module-load hook assertions (no TypeBox/SDKs/catalog
//   imports in the entry) are Node module-resolution checks with no Rust
//   equivalent (static linking); its faux-completion runtime half is ported
//   using a local test provider while `providers/faux.rs` lands from its own
//   slice.
// - model-data-validation targets the build-time generator scripts
//   (`scripts/model-data.ts`, `scripts/hydrate-model-catalog.ts`), which are
//   not library code and are not part of the port; its runtime-observable
//   invariants (every embedded model parses to a known type, ids/providers
//   consistent) live in `models_generated.rs` tests.
#[cfg(test)]
mod models_collection_tests {
	use super::*;
	use crate::types::{BaseModelFields, ModelInput};
	use std::sync::atomic::AtomicUsize;

	// ---- fixtures ----

	fn test_model(provider: &str, id: &str) -> Model {
		Model {
			base: BaseModelFields {
				id: id.to_string(),
				name: id.to_string(),
				api: "test-api".to_string(),
				provider: provider.to_string(),
				base_url: "https://example.test/v1".to_string(),
				input: vec![ModelInput::Text],
				input_limits: None,
				cost: ModelCost { rates: ModelCostRates::default(), tiers: None },
				headers: None,
			},
			model_type: None,
			reasoning: false,
			thinking_level_map: None,
			prompt_cache: None,
			context_window: 10_000,
			max_tokens: 1_000,
			sampling_params: None,
			sampling_params_by_thinking_level: None,
			compat: None,
		}
	}

	fn done_message(model: &Model, text: &str) -> AssistantMessage {
		AssistantMessage {
			content: vec![crate::types::AssistantContent::Text(crate::types::TextContent::new(text))],
			api: model.base.api.clone(),
			provider: model.base.provider.clone(),
			model: model.base.id.clone(),
			response_model: None,
			response_id: None,
			provider_thinking_level: None,
			thinking_level: None,
			diagnostics: None,
			usage: Usage {
				input: 0,
				output: 0,
				cache_read: 0,
				cache_write: 0,
				cache_write_1h: None,
				reasoning: None,
				total_tokens: 0,
				cost: UsageCost::default(),
			},
			stop_reason: StopReason::Stop,
			deferred: None,
			error_message: None,
			raw_stop_reason: None,
			end_turn: None,
			timestamp: now_ms(),
		}
	}

	#[derive(Debug, Clone, Default)]
	struct RecordedOptions {
		api_key: Option<String>,
		headers: Option<ProviderHeaders>,
		env: Option<ProviderEnv>,
	}

	struct RecordedCall {
		model: Model,
		options: RecordedOptions,
	}

	type Calls = Arc<Mutex<Vec<RecordedCall>>>;

	fn record_options(base: &crate::types::ProviderRequestOptions) -> RecordedOptions {
		RecordedOptions { api_key: base.api_key.clone(), headers: base.headers.clone(), env: base.env.clone() }
	}

	/// Headers builder for merge assertions.
	fn headers(pairs: &[(&str, &str)]) -> ProviderHeaders {
		pairs.iter().map(|(name, value)| (name.to_string(), Some(value.to_string()))).collect()
	}

	fn assert_headers(actual: Option<&ProviderHeaders>, expected: &[(&str, &str)]) {
		assert_eq!(actual.unwrap_or(&ProviderHeaders::default()), &headers(expected));
	}

	/// Ambient auth for keyless test providers; reports "configured" with no
	/// auth values.
	fn ambient_auth() -> ApiKeyAuth {
		ApiKeyAuth {
			name: "Ambient".to_string(),
			login: None,
			check: None,
			resolve: Box::new(|_input| {
				Box::pin(async { Ok(Some(AuthResult { auth: ModelAuth::default(), env: None, source: None })) })
			}),
		}
	}

	fn env_key_auth(key: Option<&str>) -> ApiKeyAuth {
		let key = key.map(str::to_string);
		ApiKeyAuth {
			name: "Test API key".to_string(),
			login: None,
			check: None,
			resolve: Box::new(move |input| {
				let key = key.clone();
				Box::pin(async move {
					let resolved = input.credential.and_then(|credential| credential.key.clone()).or(key);
					let Some(resolved) = resolved else {
						return Ok(None);
					};
					Ok(Some(AuthResult {
						auth: ModelAuth { api_key: Some(resolved), ..Default::default() },
						env: None,
						source: Some(if input.credential.is_some() { "stored" } else { "env" }.to_string()),
					}))
				})
			}),
		}
	}

	type OAuthRefresh = Box<dyn Fn(OAuthCredential, AbortSignal) -> crate::auth::types::BoxFuture<'static, Result<OAuthCredential, ThrownValue>> + Send + Sync>;

	fn test_oauth() -> OAuthAuth {
		test_oauth_with_refresh(Box::new(|credential: OAuthCredential, _signal: AbortSignal| {
			Box::pin(async move { Ok(credential) })
		}))
	}

	fn test_oauth_with_refresh(refresh: OAuthRefresh) -> OAuthAuth {
		OAuthAuth {
			name: "Test OAuth".to_string(),
			is_subscription: false,
			login_label: None,
			login: Box::new(|_interaction, _options| Box::pin(async { Err(ThrownValue::message("not used")) })),
			refresh,
			to_auth: Box::new(|credential: &OAuthCredential| {
				Box::pin(async move {
					Ok(ModelAuth { api_key: Some(credential.credentials.access.clone()), ..Default::default() })
				})
			}),
		}
	}

	fn oauth_credential(access: &str, refresh: &str, expires: i64) -> Credential {
		Credential::OAuth(oauth_credentials(access, refresh, expires))
	}

	/// The stored-credential payload (`OAuthCredential`) — what
	/// `OAuthAuth::refresh` returns upstream.
	fn oauth_credentials(access: &str, refresh: &str, expires: i64) -> OAuthCredential {
		OAuthCredential {
			credentials: crate::auth::types::OAuthCredentials {
				refresh: refresh.to_string(),
				access: access.to_string(),
				expires,
				extra: Default::default(),
			},
		}
	}

	type RefreshFn = Arc<
		dyn Fn(RefreshModelsContext) -> Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send>> + Send + Sync,
	>;

	struct TestProvider {
		id: String,
		name: Option<String>,
		auth: ProviderAuth,
		models: Vec<Model>,
		get_models: Option<Box<dyn Fn() -> Vec<Model> + Send + Sync>>,
		get_all_models: Option<Box<dyn Fn() -> Vec<AnyModel> + Send + Sync>>,
		refresh: Option<RefreshFn>,
		calls: Option<Calls>,
	}

	impl TestProvider {
		fn new(id: &str) -> TestProvider {
			TestProvider {
				id: id.to_string(),
				name: None,
				auth: ProviderAuth { api_key: Some(ambient_auth()), oauth: None },
				models: vec![test_model(id, "model-a")],
				get_models: None,
				get_all_models: None,
				refresh: None,
				calls: None,
			}
		}

		fn with_auth(mut self, auth: ProviderAuth) -> TestProvider {
			self.auth = auth;
			self
		}

		fn with_models(mut self, models: Vec<Model>) -> TestProvider {
			self.models = models;
			self
		}

		fn with_calls(mut self, calls: Calls) -> TestProvider {
			self.calls = Some(calls);
			self
		}

		fn with_get_models(mut self, get_models: Box<dyn Fn() -> Vec<Model> + Send + Sync>) -> TestProvider {
			self.get_models = Some(get_models);
			self
		}

		fn with_refresh(mut self, refresh: RefreshFn) -> TestProvider {
			self.refresh = Some(refresh);
			self
		}

		fn build(self) -> Arc<dyn Provider> {
			Arc::new(self)
		}

		fn record(&self, model: &Model, options: RecordedOptions) {
			if let Some(calls) = &self.calls {
				calls.lock().push(RecordedCall { model: model.clone(), options });
			}
		}

		fn respond(&self, model: &Model) -> AssistantMessageEventStream {
			let stream = AssistantMessageEventStream::new();
			let message = done_message(model, "ok");
			stream.push(AssistantMessageEvent::Start { partial: message.clone() });
			stream.push(AssistantMessageEvent::Done { reason: StreamDoneReason::Stop, message: message.clone() });
			stream.end(Some(message));
			stream
		}
	}

	impl Provider for TestProvider {
		fn id(&self) -> &str {
			&self.id
		}

		fn name(&self) -> &str {
			self.name.as_deref().unwrap_or(&self.id)
		}

		fn auth(&self) -> &ProviderAuth {
			&self.auth
		}

		fn get_models(&self) -> Vec<Model> {
			match &self.get_models {
				Some(get_models) => get_models(),
				None => self.models.clone(),
			}
		}

		fn get_all_models(&self) -> Vec<AnyModel> {
			match &self.get_all_models {
				Some(get_all_models) => get_all_models(),
				None => self.get_models().into_iter().map(AnyModel::Chat).collect(),
			}
		}

		fn refresh_models(
			&self,
			context: RefreshModelsContext,
		) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
			let refresh = self.refresh.clone()?;
			Some(Box::pin(async move { refresh(context).await }))
		}

		fn stream(&self, model: &Model, _context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
			self.record(model, record_options(&options.base));
			self.respond(model)
		}

		fn stream_simple(
			&self,
			model: &Model,
			_context: &TranscriptContext,
			options: &SimpleStreamOptions,
		) -> AssistantMessageEventStream {
			self.record(model, record_options(&options.base.base));
			self.respond(model)
		}
	}

	fn test_context() -> Context {
		Context {
			system_prompt: None,
			messages: vec![Message::User(UserMessage {
				content: UserContentField::Text("hi".to_string()),
				timestamp: now_ms(),
			})],
			tools: None,
		}
	}

	fn no_op_streams() -> Arc<dyn ProviderStreams> {
		struct NoopStreams;
		impl ProviderStreams for NoopStreams {
			fn stream(
				&self,
				_model: &Model,
				_context: &TranscriptContext,
				_options: &StreamOptions,
			) -> AssistantMessageEventStream {
				AssistantMessageEventStream::new()
			}
			fn stream_simple(
				&self,
				_model: &Model,
				_context: &TranscriptContext,
				_options: &SimpleStreamOptions,
			) -> AssistantMessageEventStream {
				AssistantMessageEventStream::new()
			}
		}
		Arc::new(NoopStreams)
	}

	async fn store_credential(credentials: &Arc<dyn CredentialStore>, provider_id: &str, credential: Credential) {
		credentials
			.modify(provider_id, Box::new(move |_current| Box::pin(async move { Ok(Some(credential)) })), None)
			.await
			.expect("credential stored");
	}

	async fn stored_credential(credentials: &Arc<dyn CredentialStore>, provider_id: &str) -> Option<Credential> {
		credentials.read(provider_id, None).await.expect("credential read")
	}

	/// Poll until `condition` holds (upstream `vi.waitFor`).
	async fn wait_for_condition(mut condition: impl FnMut() -> bool) {
		for _ in 0..1_000 {
			if condition() {
				return;
			}
			tokio::time::sleep(std::time::Duration::from_millis(10)).await;
		}
		panic!("condition not reached");
	}

	// ---- cases ----

	#[tokio::test]
	async fn enumerates_credential_metadata_without_exposing_secrets() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		store_credential(&credentials, "api-provider", Credential::ApiKey(ApiKeyCredential {
			key: Some("secret".to_string()),
			env: None,
		}))
		.await;
		store_credential(&credentials, "oauth-provider", oauth_credential("access", "refresh", now_ms() + 60_000)).await;

		let listed = credentials.list(None).await.expect("list");
		assert_eq!(
			listed,
			vec![
				crate::auth::types::CredentialInfo {
					provider_id: "api-provider".to_string(),
					credential_type: "api_key".to_string()
				},
				crate::auth::types::CredentialInfo {
					provider_id: "oauth-provider".to_string(),
					credential_type: "oauth".to_string()
				},
			]
		);
	}

	#[test]
	fn applies_request_wide_pricing_tiers_above_the_configured_input_threshold() {
		let mut model = test_model("openai", "gpt-5.6-sol");
		model.base.cost = ModelCost {
			rates: ModelCostRates { input: 5.0, output: 30.0, cache_read: 0.5, cache_write: 6.25 },
			tiers: Some(vec![ModelCostTier {
				rates: ModelCostRates { input: 10.0, output: 45.0, cache_read: 1.0, cache_write: 12.5 },
				input_tokens_above: 272_000.0,
			}]),
		};
		let create_usage = |cache_write: u64| Usage {
			input: 200_000,
			output: 100_000,
			cache_read: 72_000,
			cache_write,
			cache_write_1h: None,
			reasoning: None,
			total_tokens: 372_000 + cache_write,
			cost: UsageCost::default(),
		};

		let short = calculate_cost(&AnyModel::Chat(model.clone()), &mut create_usage(0));
		assert_eq!(short.input, 1.0);
		assert_eq!(short.output, 3.0);
		assert_eq!(short.cache_read, 0.036);
		assert_eq!(short.cache_write, 0.0);

		let long = calculate_cost(&AnyModel::Chat(model), &mut create_usage(1));
		assert_eq!(long.input, 2.0);
		assert_eq!(long.output, 4.5);
		assert_eq!(long.cache_read, 0.072);
		assert_eq!(long.cache_write, 0.0000125);
	}

	#[tokio::test]
	async fn registers_replaces_and_deletes_providers() {
		let models = create_models(None);
		models.set_provider(TestProvider::new("p1").build());
		models.set_provider(TestProvider::new("p2").build());
		assert_eq!(
			models.get_providers().iter().map(|provider| provider.id().to_string()).collect::<Vec<_>>(),
			vec!["p1", "p2"]
		);

		let replacement = TestProvider::new("p1").build();
		models.set_provider(replacement.clone());
		assert!(Arc::ptr_eq(&models.get_provider("p1").expect("p1"), &replacement));
		assert_eq!(models.get_providers().len(), 2);

		models.delete_provider("p1");
		assert!(models.get_provider("p1").is_none());

		models.clear_providers();
		assert_eq!(models.get_providers().len(), 0);
	}

	#[tokio::test]
	async fn lists_and_finds_models_per_provider() {
		let models = create_models(None);
		models.set_provider(
			TestProvider::new("p1").with_models(vec![test_model("p1", "m1"), test_model("p1", "m2")]).build(),
		);
		models.set_provider(TestProvider::new("p2").with_models(vec![test_model("p2", "m3")]).build());

		assert_eq!(
			models.get_models(None).iter().map(|model| model.base.id.as_str()).collect::<Vec<_>>(),
			["m1", "m2", "m3"]
		);
		assert_eq!(
			models.get_models(Some("p1")).iter().map(|model| model.base.id.as_str()).collect::<Vec<_>>(),
			["m1", "m2"]
		);
		assert_eq!(models.get_models(Some("nope")).len(), 0);
		assert_eq!(models.get_model("p2", "m3").map(|model| model.base.id), Some("m3".to_string()));
		assert!(models.get_model("p2", "missing").is_none());

		// hasApi() narrows dynamically looked-up models with a runtime check
		let found = models.get_model("p2", "m3").expect("m3");
		assert!(has_api(&AnyModel::Chat(found.clone()), "openai-completions").is_none());
		let as_any = AnyModel::Chat(found);
		let narrowed = has_api(&as_any, "test-api").expect("narrowed");
		assert_eq!(narrowed.base.id, "m3");
	}

	/// Upstream variant of "keeps chat reads independent from the all-model
	/// catalog": upstream makes `getAllModels` *throw*; the Rust-expressible
	/// equivalent is an explicit empty override (deviation D21).
	#[tokio::test]
	async fn keeps_chat_reads_independent_from_the_all_model_catalog() {
		let provider = TestProvider::new("chat-only");
		let provider = TestProvider {
			get_all_models: Some(Box::new(|| Vec::new())),
			..provider
		};
		let models = create_models(None);
		models.set_provider(provider.build());

		assert_eq!(
			models.get_models(Some("chat-only")).iter().map(|model| model.base.id.as_str()).collect::<Vec<_>>(),
			["model-a"]
		);
		assert_eq!(models.get_model("chat-only", "model-a").map(|model| model.base.id), Some("model-a".to_string()));
		assert_eq!(
			Models::get_available(models.as_ref(), Some("chat-only"), None)
				.await
				.expect("available")
				.iter()
				.map(|model| model.base.id.as_str())
				.collect::<Vec<_>>(),
			["model-a"]
		);
		assert_eq!(models.get_all_models(Some("chat-only")), Vec::new());
	}

	#[tokio::test]
	async fn refresh_updates_every_configured_dynamic_provider_and_reports_failures() {
		let list: Arc<Mutex<Vec<Model>>> = Arc::new(Mutex::new(vec![test_model("dyn", "before")]));
		let refreshes = Arc::new(AtomicUsize::new(0));
		let models = create_models(None);
		let provider = {
			let list = list.clone();
			let refreshes = refreshes.clone();
			let get_models_list = list.clone();
			TestProvider::new("dyn")
				.with_get_models(Box::new(move || get_models_list.lock().clone()))
				.with_refresh(Arc::new(move |context: RefreshModelsContext| {
					let list = list.clone();
					let refreshes = refreshes.clone();
					Box::pin(async move {
						if !context.allow_network {
							return Ok(());
						}
						refreshes.fetch_add(1, Ordering::SeqCst);
						context
							.publish
							.publish(ModelsPublication {
								persist: None,
								update: Some(Box::new(move || {
									*list.lock() = vec![test_model("dyn", "after")];
								})),
							})
							.await;
						Ok(())
					})
				}))
		};
		models.set_provider(provider.build());
		models.set_provider(TestProvider::new("static").with_models(vec![test_model("static", "s1")]).build());

		assert!(models.get_model("dyn", "before").is_some());
		let first = models.refresh(None).await;
		assert_eq!(first.errors.len(), 0);
		assert_eq!(refreshes.load(Ordering::SeqCst), 1);
		assert!(models.get_model("dyn", "after").is_some());
		assert!(models.get_model("dyn", "before").is_none());

		let flaky = TestProvider::new("flaky").with_refresh(Arc::new(|context: RefreshModelsContext| {
			Box::pin(async move {
				if context.allow_network {
					// Upstream throws `new Error("fetch failed")` — an Error
					// value, so `refresh` stores it unwrapped.
					return Err(ThrownValue::error(std::io::Error::other("fetch failed")));
				}
				Ok(())
			})
		}));
		models.set_provider(flaky.build());
		let second = models.refresh(None).await;
		assert_eq!(refreshes.load(Ordering::SeqCst), 2);
		assert_eq!(thrown_message(second.errors.get("flaky").expect("flaky error")), "fetch failed");
	}

	#[tokio::test]
	async fn restricts_refresh_work_to_selected_providers() {
		let calls: Arc<Mutex<Vec<String>>> = Arc::default();
		let models = create_models(None);
		for id in ["one", "two"] {
			let recorded = calls.clone();
			models.set_provider(
				TestProvider::new(id)
					.with_refresh(Arc::new(move |context: RefreshModelsContext| {
						let recorded = recorded.clone();
						let id = id.to_string();
						Box::pin(async move {
							recorded
								.lock()
								.push(format!("{}:{}", id, if context.allow_network { "network" } else { "cache" }));
							Ok(())
						})
					}))
					.build(),
			);
		}

		let result = models
			.refresh(Some(ModelsRefreshOptions {
				providers: Some(vec!["two".to_string(), "unknown".to_string()]),
				..Default::default()
			}))
			.await;

		assert_eq!(result.errors.len(), 0);
		assert_eq!(*calls.lock(), vec!["two:cache".to_string(), "two:network".to_string()]);
	}

	#[tokio::test]
	async fn restores_cached_models_before_waiting_for_network_auth() {
		let models_store: Arc<dyn ModelsStore> = Arc::new(InMemoryModelsStore::new());
		models_store
			.write(
				"dynamic",
				ModelsStoreEntry { models: vec![AnyModel::Chat(test_model("dynamic", "cached"))], ..Default::default() },
				None,
			)
			.await
			.expect("cached write");
		let auth_started = Arc::new(tokio::sync::Notify::new());
		let finish_auth = Arc::new(tokio::sync::Notify::new());
		let blocked = ApiKeyAuth {
			name: "Blocked auth".to_string(),
			login: None,
			check: None,
			resolve: {
				let auth_started = auth_started.clone();
				let finish_auth = finish_auth.clone();
				Box::new(move |_input| {
					let auth_started = auth_started.clone();
					let finish_auth = finish_auth.clone();
					Box::pin(async move {
						auth_started.notify_one();
						finish_auth.notified().await;
						Ok(Some(AuthResult {
							auth: ModelAuth { api_key: Some("key".to_string()), ..Default::default() },
							..Default::default()
						}))
					})
				})
			},
		};
		let provider = create_provider(CreateProviderOptions {
			id: "dynamic".to_string(),
			name: None,
			base_url: None,
			headers: None,
			auth: ProviderAuth { api_key: Some(blocked), oauth: None },
			models: Vec::new(),
			fetch_models: Some(Box::new(|_context| {
				Box::pin(async { Err(ThrownValue::message("must not fetch")) })
			})),
			filter_models: None,
			filter_all_models: None,
			api: Some(ApiDispatch::Single(no_op_streams())),
			images: Default::default(),
			classifiers: Default::default(),
		})
		.expect("provider builds");
		let models = create_models(Some(CreateModelsOptions {
			models_store: Some(models_store),
			..Default::default()
		}));
		models.set_provider(provider);
		let controller = AbortController::new();
		let pending = tokio::spawn({
			let models = models.clone();
			let signal = controller.signal();
			async move { models.refresh(Some(ModelsRefreshOptions { signal: Some(signal), ..Default::default() })).await }
		});
		auth_started.notified().await;

		assert!(models.get_model("dynamic", "cached").is_some());
		controller.abort();
		let result = pending.await.expect("refresh task");
		assert!(result.aborted);
		finish_auth.notify_one();
	}

	#[tokio::test]
	async fn lets_providers_choose_persistent_deletion_and_ephemeral_publication_atomically() {
		#[derive(Default)]
		struct SharedStore {
			entry: Mutex<Option<ModelsStoreEntry>>,
		}
		impl SharedStore {
			fn entry_snapshot(&self) -> Option<ModelsStoreEntry> {
				self.entry.lock().clone()
			}
		}
		#[async_trait::async_trait]
		impl ModelsStore for SharedStore {
			async fn read(
				&self,
				_provider_id: &str,
				_options: Option<ModelsStoreOperationOptions>,
			) -> Result<Option<ModelsStoreEntry>, ThrownValue> {
				Ok(self.entry_snapshot())
			}
			async fn write(
				&self,
				_provider_id: &str,
				entry: ModelsStoreEntry,
				_options: Option<ModelsStoreOperationOptions>,
			) -> Result<(), ThrownValue> {
				*self.entry.lock() = Some(entry);
				Ok(())
			}
			async fn delete(
				&self,
				_provider_id: &str,
				_options: Option<ModelsStoreOperationOptions>,
			) -> Result<(), ThrownValue> {
				*self.entry.lock() = None;
				Ok(())
			}
		}
		let store = Arc::new(SharedStore::default());
		*store.entry.lock() = Some(ModelsStoreEntry {
			models: vec![AnyModel::Chat(test_model("dynamic", "stored"))],
			..Default::default()
		});
		let state = Arc::new(Mutex::new("initial".to_string()));
		let probe = store.clone();
		let models = create_models(Some(CreateModelsOptions { models_store: Some(probe), ..Default::default() }));
		let update_state = state.clone();
		let entry_probe = store.clone();
		models.set_provider(
			TestProvider::new("dynamic")
				.with_refresh(Arc::new(move |context: RefreshModelsContext| {
					let update_state = update_state.clone();
					let entry_probe = entry_probe.clone();
					Box::pin(async move {
						assert_eq!(
							context
								.stored
								.as_ref()
								.and_then(|entry| entry.models.first())
								.map(|model| model.id().to_string()),
							Some("stored".to_string())
						);
						let state = update_state.clone();
						let entry_probe = entry_probe.clone();
						let published = context
							.publish
							.publish(ModelsPublication {
								persist: Some(ModelsStoreWrite::Delete),
								update: Some(Box::new(move || {
									assert!(entry_probe.entry_snapshot().is_none());
									*state.lock() = "deleted".to_string();
								})),
							})
							.await;
						assert!(published);
						let state = update_state.clone();
						context
							.publish
							.publish(ModelsPublication {
								persist: None,
								update: Some(Box::new(move || {
									*state.lock() = "ephemeral".to_string();
								})),
							})
							.await;
						Ok(())
					})
				}))
				.build(),
		);

		let result = models
			.refresh(Some(ModelsRefreshOptions { allow_network: Some(false), ..Default::default() }))
			.await;

		assert_eq!(result.errors.len(), 0);
		assert!(store.entry_snapshot().is_none());
		assert_eq!(*state.lock(), "ephemeral");
	}

	#[tokio::test]
	async fn persists_dynamic_catalogs_and_restores_them_without_network_access() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		store_credential(&credentials, "dynamic", Credential::ApiKey(ApiKeyCredential {
			key: Some("key".to_string()),
			env: None,
		}))
		.await;
		let models_store: Arc<dyn ModelsStore> = Arc::new(InMemoryModelsStore::new());
		let create_dynamic = |fetch_models: bool| {
			let fetch: Option<
				Box<
					dyn Fn(&RefreshModelsContext)
						-> Pin<Box<dyn Future<Output = Result<Vec<AnyModel>, ThrownValue>> + Send>>
						+ Send
						+ Sync,
				>,
			> = if fetch_models {
				Some(Box::new(|_context| {
					Box::pin(async { Ok(vec![AnyModel::Chat(test_model("dynamic", "fetched"))]) })
				}))
			} else {
				Some(Box::new(|_context| Box::pin(async { Err(ThrownValue::message("must not fetch")) })))
			};
			create_provider(CreateProviderOptions {
				id: "dynamic".to_string(),
				name: None,
				base_url: None,
				headers: None,
				auth: ProviderAuth { api_key: Some(env_key_auth(None)), oauth: None },
				models: Vec::new(),
				fetch_models: fetch,
				filter_models: None,
				filter_all_models: None,
				api: Some(ApiDispatch::Single(no_op_streams())),
				images: Default::default(),
				classifiers: Default::default(),
			})
			.expect("provider builds")
		};

		let online = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			models_store: Some(models_store.clone()),
			..Default::default()
		}));
		online.set_provider(create_dynamic(true));
		assert_eq!(online.refresh(None).await.errors.len(), 0);
		assert!(online.get_model("dynamic", "fetched").is_some());

		let offline = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			models_store: Some(models_store.clone()),
			..Default::default()
		}));
		offline.set_provider(create_dynamic(false));
		assert_eq!(
			offline
				.refresh(Some(ModelsRefreshOptions { allow_network: Some(false), ..Default::default() }))
				.await
				.errors
				.len(),
			0
		);
		assert!(offline.get_model("dynamic", "fetched").is_some());
	}

	#[tokio::test]
	async fn passes_effective_api_key_credentials_and_refresh_options_while_skipping_unconfigured_providers() {
		let effective_credential: Arc<Mutex<Option<Credential>>> = Arc::default();
		let force_refresh: Arc<Mutex<Option<bool>>> = Arc::default();
		let unconfigured_refreshes = Arc::new(AtomicUsize::new(0));
		let models = create_models(None);
		models.set_provider(
			{
				let effective_credential = effective_credential.clone();
				let force_refresh = force_refresh.clone();
				TestProvider::new("configured")
					.with_auth(ProviderAuth { api_key: Some(env_key_auth(Some("ambient-key"))), oauth: None })
					.with_refresh(Arc::new(move |context: RefreshModelsContext| {
						let effective_credential = effective_credential.clone();
						let force_refresh = force_refresh.clone();
						Box::pin(async move {
							if !context.allow_network {
								return Ok(());
							}
							*effective_credential.lock() = context.credential.clone();
							*force_refresh.lock() = context.force;
							Ok(())
						})
					}))
					.build()
			},
		);
		models.set_provider(
			{
				let unconfigured_refreshes = unconfigured_refreshes.clone();
				TestProvider::new("unconfigured")
					.with_auth(ProviderAuth { api_key: Some(env_key_auth(None)), oauth: None })
					.with_refresh(Arc::new(move |context: RefreshModelsContext| {
						let unconfigured_refreshes = unconfigured_refreshes.clone();
						Box::pin(async move {
							if context.allow_network {
								unconfigured_refreshes.fetch_add(1, Ordering::SeqCst);
							}
							Ok(())
						})
					}))
					.build()
			},
		);

		models.refresh(Some(ModelsRefreshOptions { force: Some(true), ..Default::default() })).await;
		assert_eq!(
			effective_credential.lock().clone(),
			Some(Credential::ApiKey(ApiKeyCredential { key: Some("ambient-key".to_string()), env: None }))
		);
		assert_eq!(*force_refresh.lock(), Some(true));
		assert_eq!(unconfigured_refreshes.load(Ordering::SeqCst), 0);
	}

	#[tokio::test]
	async fn refreshes_expired_oauth_before_refreshing_models() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		store_credential(&credentials, "oauth-dynamic", oauth_credential("expired", "refresh", 0)).await;
		let model_refresh_credential: Arc<Mutex<Option<Credential>>> = Arc::default();
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(
			{
				let model_refresh_credential = model_refresh_credential.clone();
				TestProvider::new("oauth-dynamic")
					.with_auth(ProviderAuth {
						api_key: None,
						oauth: Some(Arc::new(test_oauth_with_refresh(Box::new(
							|_credential: OAuthCredential, _signal: AbortSignal| {
								Box::pin(async {
									Ok(oauth_credentials("fresh", "rotated", now_ms() + 60_000))
								})
							},
						)))),
					})
					.with_refresh(Arc::new(move |context: RefreshModelsContext| {
						let model_refresh_credential = model_refresh_credential.clone();
						Box::pin(async move {
							if context.allow_network {
								*model_refresh_credential.lock() = context.credential.clone();
							}
							Ok(())
						})
					}))
					.build()
			},
		);

		assert_eq!(models.refresh(None).await.errors.len(), 0);
		assert!(matches!(
			model_refresh_credential.lock().clone(),
			Some(Credential::OAuth(oauth)) if oauth.credentials.access == "fresh" && oauth.credentials.refresh == "rotated"
		));
		assert!(matches!(
			stored_credential(&credentials, "oauth-dynamic").await,
			Some(Credential::OAuth(oauth)) if oauth.credentials.access == "fresh" && oauth.credentials.refresh == "rotated"
		));
	}

	#[tokio::test]
	async fn persists_an_oauth_refresh_that_started_before_the_model_refresh_was_cancelled() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		store_credential(&credentials, "p1", oauth_credential("old", "old-refresh", 0)).await;
		let controller = AbortController::new();
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("p1")
				.with_auth(ProviderAuth {
					api_key: None,
					oauth: Some(Arc::new(test_oauth_with_refresh(Box::new({
						let controller = controller.clone();
						move |credential: OAuthCredential, _signal: AbortSignal| {
							let controller = controller.clone();
							Box::pin(async move {
								controller.abort();
								Ok(OAuthCredential {
									credentials: crate::auth::types::OAuthCredentials {
										refresh: "new-refresh".to_string(),
										access: "new".to_string(),
										expires: now_ms() + 60_000,
										extra: credential.credentials.extra.clone(),
									},
								})
							})
						}
					})))),
				})
				.with_refresh(Arc::new(|_context: RefreshModelsContext| Box::pin(async { Ok(()) })))
				.build(),
		);

		assert!(models
			.refresh(Some(ModelsRefreshOptions { signal: Some(controller.signal()), ..Default::default() }))
			.await
			.aborted);
		// The detached operation persists the rotated token after cancellation.
		for _ in 0..1_000 {
			if matches!(
				stored_credential(&credentials, "p1").await,
				Some(Credential::OAuth(oauth)) if oauth.credentials.refresh == "new-refresh"
			) {
				break;
			}
			tokio::time::sleep(std::time::Duration::from_millis(10)).await;
		}
		assert!(matches!(
			stored_credential(&credentials, "p1").await,
			Some(Credential::OAuth(oauth)) if oauth.credentials.refresh == "new-refresh"
		));
	}

	#[tokio::test]
	async fn always_gives_providers_a_concrete_signal() {
		let received: Arc<Mutex<Option<crate::utils::abort::AbortSignal>>> = Arc::default();
		let models = create_models(None);
		models.set_provider(
			{
				let received = received.clone();
				TestProvider::new("dynamic")
					.with_refresh(Arc::new(move |context: RefreshModelsContext| {
						let received = received.clone();
						Box::pin(async move {
							*received.lock() = Some(context.signal.clone());
							Ok(())
						})
					}))
					.build()
			},
		);

		let result = models.refresh(None).await;
		assert!(!result.aborted);
		let signal = received.lock().clone().expect("provider received a signal");
		assert!(!signal.aborted());
	}

	#[tokio::test]
	async fn binds_model_store_waits_to_the_provider_refresh_signal() {
		struct RecordingStore {
			signals: Mutex<Vec<crate::utils::abort::AbortSignal>>,
		}
		#[async_trait::async_trait]
		impl ModelsStore for RecordingStore {
			async fn read(
				&self,
				_provider_id: &str,
				options: Option<ModelsStoreOperationOptions>,
			) -> Result<Option<ModelsStoreEntry>, ThrownValue> {
				if let Some(signal) = options.and_then(|options| options.signal) {
					self.signals.lock().push(signal);
				}
				Ok(None)
			}
			async fn write(
				&self,
				_provider_id: &str,
				_entry: ModelsStoreEntry,
				options: Option<ModelsStoreOperationOptions>,
			) -> Result<(), ThrownValue> {
				if let Some(signal) = options.and_then(|options| options.signal) {
					self.signals.lock().push(signal);
				}
				Ok(())
			}
			async fn delete(
				&self,
				_provider_id: &str,
				_options: Option<ModelsStoreOperationOptions>,
			) -> Result<(), ThrownValue> {
				Ok(())
			}
		}
		let store = Arc::new(RecordingStore { signals: Mutex::new(Vec::new()) });
		let provider_signal: Arc<Mutex<Option<crate::utils::abort::AbortSignal>>> = Arc::default();
		let models = create_models(Some(CreateModelsOptions { models_store: Some(store.clone()), ..Default::default() }));
		models.set_provider(
			{
				let provider_signal = provider_signal.clone();
				TestProvider::new("dynamic")
					.with_auth(ProviderAuth { api_key: Some(env_key_auth(Some("key"))), oauth: None })
					.with_refresh(Arc::new(move |context: RefreshModelsContext| {
						let provider_signal = provider_signal.clone();
						Box::pin(async move {
							*provider_signal.lock() = Some(context.signal.clone());
							if !context.allow_network {
								return Ok(());
							}
							context
								.publish
								.publish(ModelsPublication {
									persist: Some(ModelsStoreWrite::Write(ModelsStoreEntry {
										models: vec![AnyModel::Chat(test_model("dynamic", "fresh"))],
										..Default::default()
									})),
									update: None,
								})
								.await;
							Ok(())
						})
					}))
					.build()
			},
		);

		let result = models
			.refresh(Some(ModelsRefreshOptions { providers: Some(vec!["dynamic".to_string()]), ..Default::default() }))
			.await;

		assert_eq!(result.errors.len(), 0);
		let signals = store.signals.lock();
		assert_eq!(signals.len(), 3);
		// Abort-state propagation stands in for `AbortSignal ===` identity
		// (deviation D23): every store wait observes the provider's signal.
		let provider_signal = provider_signal.lock().clone().expect("provider signal");
		assert!(!signals.is_empty() && signals.iter().all(|signal| !signal.aborted()));
		drop(signals);
		crate::utils::abort::AbortSignal::abort(&provider_signal);
		wait_for_condition(|| store.signals.lock().iter().all(|signal| signal.aborted())).await;
	}

	#[tokio::test]
	async fn returns_aborted_state_without_reporting_cancellation_as_a_provider_error() {
		let controller = AbortController::new();
		let models = create_models(None);
		models.set_provider(
			{
				let controller = controller.clone();
				TestProvider::new("dynamic")
					.with_refresh(Arc::new(move |context: RefreshModelsContext| {
						let controller = controller.clone();
						Box::pin(async move {
							controller.abort();
							if context.signal.aborted() {
								return Ok(());
							}
							Ok(())
						})
					}))
					.build()
			},
		);

		let result = models
			.refresh(Some(ModelsRefreshOptions { signal: Some(controller.signal()), ..Default::default() }))
			.await;
		assert!(result.aborted);
		assert_eq!(result.errors.len(), 0);
	}

	#[tokio::test]
	async fn stops_waiting_on_abort_when_a_provider_ignores_its_signal() {
		let controller = AbortController::new();
		let started = Arc::new(tokio::sync::Notify::new());
		let release = Arc::new(tokio::sync::Notify::new());
		let calls = Arc::new(AtomicUsize::new(0));
		let models = create_models(None);
		models.set_provider(
			{
				let started = started.clone();
				let release = release.clone();
				let calls = calls.clone();
				TestProvider::new("dynamic")
					.with_refresh(Arc::new(move |_context: RefreshModelsContext| {
						let started = started.clone();
						let release = release.clone();
						let calls = calls.clone();
						Box::pin(async move {
							if calls.fetch_add(1, Ordering::SeqCst) != 0 {
								return Ok(());
							}
							started.notify_one();
							release.notified().await;
							// Upstream settles the stalled promise with a late
							// failure that must NOT be recorded (the signal
							// already aborted).
							Err(ThrownValue::error(std::io::Error::other("late provider failure")))
						})
					}))
					.build()
			},
		);

		let pending = tokio::spawn({
			let models = models.clone();
			let signal = controller.signal();
			async move { models.refresh(Some(ModelsRefreshOptions { signal: Some(signal), ..Default::default() })).await }
		});
		started.notified().await;
		controller.abort();

		let result = pending.await.expect("refresh task");
		assert!(result.aborted);
		assert_eq!(result.errors.len(), 0);

		release.notify_one();
		tokio::time::sleep(std::time::Duration::from_millis(20)).await;
		assert_eq!(result.errors.len(), 0);
	}

	#[tokio::test]
	async fn rejects_late_publication_from_a_superseded_non_cooperative_provider() {
		let models_store: Arc<dyn ModelsStore> = Arc::new(InMemoryModelsStore::new());
		let state = Arc::new(Mutex::new("initial".to_string()));
		let calls = Arc::new(AtomicUsize::new(0));
		let first_started = Arc::new(tokio::sync::Notify::new());
		let release_first = Arc::new(tokio::sync::Notify::new());
		let models = create_models(Some(CreateModelsOptions { models_store: Some(models_store.clone()), ..Default::default() }));
		models.set_provider(
			{
				let state = state.clone();
				let calls = calls.clone();
				let first_started = first_started.clone();
				let release_first = release_first.clone();
				TestProvider::new("dynamic")
					.with_refresh(Arc::new(move |context: RefreshModelsContext| {
						let state = state.clone();
						let calls = calls.clone();
						let first_started = first_started.clone();
						let release_first = release_first.clone();
						Box::pin(async move {
							if !context.allow_network {
								return Ok(());
							}
							let current = calls.fetch_add(1, Ordering::SeqCst) + 1;
							if current == 1 {
								first_started.notify_one();
								release_first.notified().await;
							}
							let value = format!("generation-{current}");
							context
								.publish
								.publish(ModelsPublication {
									persist: Some(ModelsStoreWrite::Write(ModelsStoreEntry {
										models: vec![AnyModel::Chat(test_model("dynamic", &value))],
										..Default::default()
									})),
									update: Some(Box::new(move || {
										*state.lock() = value.clone();
									})),
								})
								.await;
							Ok(())
						})
					}))
					.build()
			},
		);

		let first = tokio::spawn({
			let models = models.clone();
			async move { models.refresh(Some(ModelsRefreshOptions { providers: Some(vec!["dynamic".to_string()]), ..Default::default() })).await }
		});
		first_started.notified().await;
		let second = models
			.refresh(Some(ModelsRefreshOptions { providers: Some(vec!["dynamic".to_string()]), ..Default::default() }))
			.await;
		assert_eq!(second.errors.len(), 0);
		let _ = first.await;
		release_first.notify_one();
		tokio::time::sleep(std::time::Duration::from_millis(50)).await;

		assert_eq!(*state.lock(), "generation-2");
		let stored = models_store.read("dynamic", None).await.expect("stored read");
		assert_eq!(stored.and_then(|entry| entry.models.first().map(|model| model.id().to_string())), Some("generation-2".to_string()));
	}

	#[tokio::test]
	async fn passes_caller_signals_to_provider_auth_callbacks() {
		let controller = AbortController::new();
		let received: Arc<Mutex<Vec<crate::utils::abort::AbortSignal>>> = Arc::default();
		let resolve_received = received.clone();
		let check_received = received.clone();
		let login_received = received.clone();
		let signal = controller.signal();
		let api_key = ApiKeyAuth {
			name: "Signal auth".to_string(),
			login: Some(Box::new(move |interaction: ProviderAuthInteraction| {
				let login_received = login_received.clone();
				Box::pin(async move {
					login_received.lock().push(interaction.signal.clone());
					Ok(ApiKeyCredential { key: Some("saved".to_string()), env: None })
				})
			})),
			check: Some(Box::new(move |input: ApiKeyAuthInput<'_>| {
				let check_received = check_received.clone();
				Box::pin(async move {
					check_received.lock().push(input.signal.clone());
					Ok(Some(AuthCheck { source: None, check_type: AuthType::ApiKey }))
				})
			})),
			resolve: Box::new(move |input: ApiKeyAuthInput<'_>| {
				let resolve_received = resolve_received.clone();
				Box::pin(async move {
					resolve_received.lock().push(input.signal.clone());
					Ok(Some(AuthResult {
						auth: ModelAuth { api_key: Some("resolved".to_string()), ..Default::default() },
						..Default::default()
					}))
				})
			}),
		};
		let models = create_models(None);
		models.set_provider(TestProvider::new("p1").with_auth(ProviderAuth { api_key: Some(api_key), oauth: None }).build());

		let _ = Models::check_auth(models.as_ref(), "p1", Some(AuthOperationOptions { signal: Some(signal.clone()) })).await;
		let _ = models
			.get_auth_by_provider("p1", Some(crate::auth::resolve::AuthResolutionOverrides { signal: Some(signal.clone()), ..Default::default() }))
			.await;
		let _ = models
			.login(
				"p1",
				AuthType::ApiKey,
				AuthInteraction {
					signal: Some(signal.clone()),
					prompt: Box::new(|_prompt| Box::pin(async { Ok("unused".to_string()) })),
					notify: Box::new(|_event| {}),
				},
				None,
			)
			.await;

		let received = received.lock();
		assert_eq!(received.len(), 3);
		// Abort-state propagation stands in for signal identity (D23).
		crate::utils::abort::AbortSignal::abort(&signal);
		assert!(received.iter().all(|received| received.aborted()));
	}

	#[tokio::test]
	async fn stops_waiting_for_non_cooperative_auth_callbacks() {
		let check_started = Arc::new(tokio::sync::Notify::new());
		let finish_check = Arc::new(tokio::sync::Notify::new());
		let resolve_started = Arc::new(tokio::sync::Notify::new());
		let finish_resolve = Arc::new(tokio::sync::Notify::new());
		let models = create_models(None);
		models.set_provider(
			TestProvider::new("p1")
				.with_auth(ProviderAuth {
					api_key: Some(ApiKeyAuth {
						name: "Blocked auth".to_string(),
						login: None,
						check: Some({
							let check_started = check_started.clone();
							let finish_check = finish_check.clone();
							Box::new(move |_input: ApiKeyAuthInput<'_>| {
								let check_started = check_started.clone();
								let finish_check = finish_check.clone();
								Box::pin(async move {
									check_started.notify_one();
									finish_check.notified().await;
									Ok(Some(AuthCheck { source: None, check_type: AuthType::ApiKey }))
								})
							})
						}),
						resolve: {
							let resolve_started = resolve_started.clone();
							let finish_resolve = finish_resolve.clone();
							Box::new(move |_input: ApiKeyAuthInput<'_>| {
								let resolve_started = resolve_started.clone();
								let finish_resolve = finish_resolve.clone();
								Box::pin(async move {
									resolve_started.notify_one();
									finish_resolve.notified().await;
									Ok(Some(AuthResult {
										auth: ModelAuth { api_key: Some("key".to_string()), ..Default::default() },
										..Default::default()
									}))
								})
							})
						},
					}),
					oauth: None,
				})
				.build(),
		);

		let available_controller = AbortController::new();
		let available = tokio::spawn({
			let models = models.clone();
			let signal = available_controller.signal();
			async move { Models::get_available(models.as_ref(), None, Some(AuthOperationOptions { signal: Some(signal) })).await }
		});
		check_started.notified().await;
		available_controller.abort();
		assert!(is_abort_error(&available.await.expect("available task").expect_err("aborted get_available")));

		let auth_controller = AbortController::new();
		let auth = tokio::spawn({
			let models = models.clone();
			let signal = auth_controller.signal();
			async move {
				models
					.get_auth_by_provider("p1", Some(crate::auth::resolve::AuthResolutionOverrides { signal: Some(signal), ..Default::default() }))
					.await
			}
		});
		resolve_started.notified().await;
		auth_controller.abort();
		assert!(is_abort_error(&auth.await.expect("auth task").expect_err("aborted get_auth")));

		finish_check.notify_one();
		finish_resolve.notify_one();
	}

	/// Radius bug 01a10855-43d8-7447-a710-2acf5ee0b2ee: refresh_token_invalidated
	/// after a cancelled request — the started OAuth refresh must persist.
	#[tokio::test]
	async fn persists_an_oauth_refresh_that_started_before_the_request_was_cancelled() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		store_credential(&credentials, "p1", oauth_credential("old", "old-refresh", 0)).await;
		let controller = AbortController::new();
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("p1")
				.with_auth(ProviderAuth {
					api_key: None,
					oauth: Some(Arc::new(test_oauth_with_refresh(Box::new({
						let controller = controller.clone();
						move |credential: OAuthCredential, _signal: AbortSignal| {
							let controller = controller.clone();
							Box::pin(async move {
								// The provider has rotated old-refresh by the time
								// the request is cancelled.
								controller.abort();
								Ok(OAuthCredential {
									credentials: crate::auth::types::OAuthCredentials {
										refresh: "new-refresh".to_string(),
										access: "new".to_string(),
										expires: now_ms() + 60_000,
										extra: credential.credentials.extra.clone(),
									},
								})
							})
						}
					})))),
				})
				.build(),
		);

		let result = models
			.get_auth_by_provider(
				"p1",
				Some(crate::auth::resolve::AuthResolutionOverrides { signal: Some(controller.signal()), ..Default::default() }),
			)
			.await;
		assert!(is_abort_error(&result.expect_err("aborted get_auth")));
		for _ in 0..1_000 {
			if matches!(
				stored_credential(&credentials, "p1").await,
				Some(Credential::OAuth(oauth)) if oauth.credentials.refresh == "new-refresh"
			) {
				break;
			}
			tokio::time::sleep(std::time::Duration::from_millis(10)).await;
		}
		assert!(matches!(
			stored_credential(&credentials, "p1").await,
			Some(Credential::OAuth(oauth)) if oauth.credentials.refresh == "new-refresh"
		));
	}

	#[tokio::test]
	async fn resolves_auth_stored_credential_owns_provider_ambient_only_when_nothing_stored() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("p1")
				.with_auth(ProviderAuth { api_key: Some(env_key_auth(Some("env-key"))), oauth: Some(Arc::new(test_oauth())) })
				.build(),
		);
		let model = test_model("p1", "model-a");

		// model and provider-id overloads resolve the same provider-scoped auth
		let by_model = models.get_auth_by_model(&AnyModel::Chat(model.clone()), None).await.expect("model auth").expect("resolved");
		assert_eq!(by_model.auth.api_key.as_deref(), Some("env-key"));
		let by_provider = models.get_auth_by_provider("p1", None).await.expect("provider auth").expect("resolved");
		assert_eq!(by_provider.auth.api_key.as_deref(), Some("env-key"));
		let explicit = models
			.get_auth_by_model(
				&AnyModel::Chat(model.clone()),
				Some(crate::auth::resolve::AuthResolutionOverrides {
					api_key: Some("explicit-key".to_string()),
					..Default::default()
				}),
			)
			.await
			.expect("explicit auth")
			.expect("resolved");
		assert_eq!(explicit.auth.api_key.as_deref(), Some("explicit-key"));

		// stored oauth credential (persisted via the single write path): beats ambient env
		store_credential(&credentials, "p1", oauth_credential("oauth-token", "r", now_ms() + 10 * 60_000)).await;
		let oauth_resolution = models.get_auth_by_provider("p1", None).await.expect("oauth auth").expect("resolved");
		assert_eq!(oauth_resolution.auth.api_key.as_deref(), Some("oauth-token"));
		assert_eq!(oauth_resolution.source.as_deref(), Some("OAuth"));

		// stored api-key credential resolves through apiKey auth, beats env
		store_credential(&credentials, "p1", Credential::ApiKey(ApiKeyCredential {
			key: Some("stored-key".to_string()),
			env: None,
		}))
		.await;
		let api_key_resolution = models.get_auth_by_provider("p1", None).await.expect("stored auth").expect("resolved");
		assert_eq!(api_key_resolution.auth.api_key.as_deref(), Some("stored-key"));
		assert_eq!(api_key_resolution.source.as_deref(), Some("stored"));
	}

	#[tokio::test]
	async fn checks_provider_auth_without_refreshing_oauth_and_filters_available_models() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		let refreshes = Arc::new(AtomicUsize::new(0));
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("ambient").with_auth(ProviderAuth { api_key: Some(env_key_auth(Some("env-key"))), oauth: None }).build(),
		);
		models.set_provider(
			TestProvider::new("missing").with_auth(ProviderAuth { api_key: Some(env_key_auth(None)), oauth: None }).build(),
		);
		models.set_provider(
			{
				let refreshes = refreshes.clone();
				TestProvider::new("oauth")
					.with_auth(ProviderAuth {
						api_key: None,
						oauth: Some(Arc::new(test_oauth_with_refresh(Box::new(move |credential: OAuthCredential, _signal: AbortSignal| {
							let refreshes = refreshes.clone();
							Box::pin(async move {
								refreshes.fetch_add(1, Ordering::SeqCst);
								Ok(credential)
							})
						})))),
					})
					.build()
			},
		);
		store_credential(&credentials, "oauth", oauth_credential("expired", "refresh", 0)).await;

		assert_eq!(
			Models::check_auth(models.as_ref(), "ambient", None).await.expect("ambient check"),
			Some(AuthCheck { source: Some("env".to_string()), check_type: AuthType::ApiKey })
		);
		assert_eq!(Models::check_auth(models.as_ref(), "missing", None).await.expect("missing check"), None);
		assert_eq!(
			Models::check_auth(models.as_ref(), "oauth", None).await.expect("oauth check"),
			Some(AuthCheck { source: Some("OAuth".to_string()), check_type: AuthType::OAuth })
		);
		assert_eq!(refreshes.load(Ordering::SeqCst), 0);
		assert_eq!(
			Models::get_available(models.as_ref(), None, None)
				.await
				.expect("available")
				.iter()
				.map(|model| model.base.provider.as_str())
				.collect::<Vec<_>>(),
			["ambient", "oauth"]
		);
		assert_eq!(
			Models::get_available(models.as_ref(), Some("ambient"), None)
				.await
				.expect("available")
				.iter()
				.map(|model| model.base.provider.as_str())
				.collect::<Vec<_>>(),
			["ambient"]
		);
	}

	#[tokio::test]
	async fn runs_provider_login_and_logout_through_the_credential_store() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		let mut api_key = env_key_auth(None);
		api_key.login = Some(Box::new(|_interaction: ProviderAuthInteraction| {
			Box::pin(async {
				Ok(ApiKeyCredential { key: Some("logged-in".to_string()), env: None })
			})
		}));
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(TestProvider::new("p1").with_auth(ProviderAuth { api_key: Some(api_key), oauth: None }).build());

		let credential = models
			.login(
				"p1",
				AuthType::ApiKey,
				AuthInteraction {
					signal: None,
					prompt: Box::new(|_prompt| Box::pin(async { Ok("unused".to_string()) })),
					notify: Box::new(|_event| {}),
				},
				None,
			)
			.await
			.expect("login");
		assert_eq!(
			credential,
			Credential::ApiKey(ApiKeyCredential { key: Some("logged-in".to_string()), env: None })
		);
		assert_eq!(stored_credential(&credentials, "p1").await, Some(credential.clone()));

		models.logout("p1", None).await.expect("logout");
		assert_eq!(stored_credential(&credentials, "p1").await, None);
	}

	#[tokio::test]
	async fn a_stored_credential_without_a_matching_handler_blocks_ambient_fallback() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		// provider has only apiKey auth, but an oauth credential is stored (stale config)
		models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: Some(env_key_auth(Some("env-key"))), oauth: None }).build(),
		);
		store_credential(&credentials, "p1", oauth_credential("a", "r", 0)).await;

		assert_eq!(models.get_auth_by_provider("p1", None).await.expect("get_auth"), None);
	}

	#[tokio::test]
	async fn refreshes_expired_oauth_credentials_and_persists_the_rotated_credential() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		let oauth = test_oauth_with_refresh(Box::new(|credential: OAuthCredential, _signal: AbortSignal| {
			Box::pin(async move {
				Ok(OAuthCredential {
					credentials: crate::auth::types::OAuthCredentials {
						refresh: credential.credentials.refresh,
						access: "new-token".to_string(),
						expires: now_ms() + 60 * 60_000,
						extra: credential.credentials.extra,
					},
				})
			})
		}));
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: None, oauth: Some(Arc::new(oauth)) }).build(),
		);
		store_credential(&credentials, "p1", oauth_credential("old-token", "r", 0)).await;

		let resolution = models.get_auth_by_provider("p1", None).await.expect("get_auth").expect("resolved");
		assert_eq!(resolution.auth.api_key.as_deref(), Some("new-token"));
		assert!(matches!(
			stored_credential(&credentials, "p1").await,
			Some(Credential::OAuth(oauth)) if oauth.credentials.access == "new-token"
		));
	}

	#[tokio::test]
	async fn refreshes_oauth_credentials_with_less_than_five_minutes_remaining() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		let refreshes = Arc::new(AtomicUsize::new(0));
		let oauth = test_oauth_with_refresh(Box::new({
			let refreshes = refreshes.clone();
			move |credential: OAuthCredential, _signal: AbortSignal| {
				let refreshes = refreshes.clone();
				Box::pin(async move {
					refreshes.fetch_add(1, Ordering::SeqCst);
					Ok(OAuthCredential {
						credentials: crate::auth::types::OAuthCredentials {
							refresh: credential.credentials.refresh,
							access: "new-token".to_string(),
							expires: now_ms() + 60 * 60_000,
							extra: credential.credentials.extra,
						},
					})
				})
			}
		}));
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: None, oauth: Some(Arc::new(oauth)) }).build(),
		);
		store_credential(&credentials, "p1", oauth_credential("old-token", "r", now_ms() + 60_000)).await;

		assert_eq!(
			models.get_auth_by_provider("p1", None).await.expect("get_auth").expect("resolved").auth.api_key.as_deref(),
			Some("new-token")
		);
		assert_eq!(refreshes.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn honors_a_callers_longer_oauth_minimum_validity() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		let refreshes = Arc::new(AtomicUsize::new(0));
		let oauth = test_oauth_with_refresh(Box::new({
			let refreshes = refreshes.clone();
			move |credential: OAuthCredential, _signal: AbortSignal| {
				let refreshes = refreshes.clone();
				Box::pin(async move {
					refreshes.fetch_add(1, Ordering::SeqCst);
					Ok(OAuthCredential {
						credentials: crate::auth::types::OAuthCredentials {
							refresh: credential.credentials.refresh,
							access: "new-token".to_string(),
							expires: now_ms() + 60 * 60_000,
							extra: credential.credentials.extra,
						},
					})
				})
			}
		}));
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: None, oauth: Some(Arc::new(oauth)) }).build(),
		);
		store_credential(&credentials, "p1", oauth_credential("old-token", "r", now_ms() + 10 * 60_000)).await;

		let resolution = models
			.get_auth_by_provider(
				"p1",
				Some(crate::auth::resolve::AuthResolutionOverrides {
					min_oauth_validity_ms: Some(30 * 60_000),
					..Default::default()
				}),
			)
			.await
			.expect("get_auth")
			.expect("resolved");
		assert_eq!(resolution.auth.api_key.as_deref(), Some("new-token"));
		assert_eq!(refreshes.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn rejects_with_code_oauth_when_refresh_fails_preserving_the_stored_credential() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		let oauth = test_oauth_with_refresh(Box::new(|_credential: OAuthCredential, _signal: AbortSignal| {
			Box::pin(async { Err(ThrownValue::message("invalid_grant")) })
		}));
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: None, oauth: Some(Arc::new(oauth)) }).build(),
		);
		store_credential(&credentials, "p1", oauth_credential("old", "r", 0)).await;

		let error = models.get_auth_by_provider("p1", None).await.expect_err("oauth failure");
		assert_eq!(thrown_models_code(&error), Some(ModelsErrorCode::OAuth));
		// credential preserved for retry / re-login
		assert!(matches!(
			stored_credential(&credentials, "p1").await,
			Some(Credential::OAuth(oauth)) if oauth.credentials.access == "old"
		));
	}

	#[tokio::test]
	async fn serializes_concurrent_oauth_refreshes_through_store_modify() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		store_credential(&credentials, "p1", oauth_credential("old", "r1", 0)).await;

		let refreshes = Arc::new(AtomicUsize::new(0));
		let oauth = test_oauth_with_refresh(Box::new({
			let refreshes = refreshes.clone();
			move |_credential: OAuthCredential, _signal: AbortSignal| {
				let refreshes = refreshes.clone();
				Box::pin(async move {
					refreshes.fetch_add(1, Ordering::SeqCst);
					tokio::time::sleep(std::time::Duration::from_millis(10)).await;
					Ok(oauth_credentials("new-1", "r2", now_ms() + 60 * 60_000))
				})
			}
		}));
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: None, oauth: Some(Arc::new(oauth)) }).build(),
		);
		let (a, b) = tokio::join!(models.get_auth_by_provider("p1", None), models.get_auth_by_provider("p1", None));
		assert_eq!(refreshes.load(Ordering::SeqCst), 1);
		assert_eq!(a.expect("a").expect("resolved").auth.api_key.as_deref(), Some("new-1"));
		assert_eq!(b.expect("b").expect("resolved").auth.api_key.as_deref(), Some("new-1"));
	}

	#[tokio::test]
	async fn valid_oauth_tokens_resolve_without_touching_modify() {
		let modifies = Arc::new(AtomicUsize::new(0));
		let base = Arc::new(InMemoryCredentialStore::new());
		let base_dyn: Arc<dyn CredentialStore> = base.clone();
		store_credential(&base_dyn, "p1", oauth_credential("valid", "r", now_ms() + 10 * 60_000)).await;
		struct CountingStore {
			base: Arc<InMemoryCredentialStore>,
			modifies: Arc<AtomicUsize>,
		}
		#[async_trait::async_trait]
		impl CredentialStore for CountingStore {
			async fn read(
				&self,
				provider_id: &str,
				options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				self.base.read(provider_id, options).await
			}
			async fn list(&self, options: Option<AuthOperationOptions>) -> Result<Vec<crate::auth::types::CredentialInfo>, ThrownValue> {
				self.base.list(options).await
			}
			async fn modify(
				&self,
				provider_id: &str,
				modify_fn: crate::auth::types::CredentialModifyFn,
				options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				self.modifies.fetch_add(1, Ordering::SeqCst);
				self.base.modify(provider_id, modify_fn, options).await
			}
			async fn delete(&self, provider_id: &str, options: Option<AuthOperationOptions>) -> Result<(), ThrownValue> {
				self.base.delete(provider_id, options).await
			}
		}
		let credentials: Arc<dyn CredentialStore> =
			Arc::new(CountingStore { base, modifies: modifies.clone() });
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone()),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: None, oauth: Some(Arc::new(test_oauth())) }).build(),
		);

		assert_eq!(
			models.get_auth_by_provider("p1", None).await.expect("get_auth").expect("resolved").auth.api_key.as_deref(),
			Some("valid")
		);
		assert_eq!(modifies.load(Ordering::SeqCst), 0);
	}

	#[tokio::test]
	async fn wraps_credential_store_failures_in_models_error() {
		struct ReadFailingStore;
		#[async_trait::async_trait]
		impl CredentialStore for ReadFailingStore {
			async fn read(
				&self,
				_provider_id: &str,
				_options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				Err(ThrownValue::message("disk on fire"))
			}
			async fn list(&self, _options: Option<AuthOperationOptions>) -> Result<Vec<crate::auth::types::CredentialInfo>, ThrownValue> {
				Ok(Vec::new())
			}
			async fn modify(
				&self,
				_provider_id: &str,
				_modify_fn: crate::auth::types::CredentialModifyFn,
				_options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				Ok(None)
			}
			async fn delete(&self, _provider_id: &str, _options: Option<AuthOperationOptions>) -> Result<(), ThrownValue> {
				Ok(())
			}
		}
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(Arc::new(ReadFailingStore)),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: Some(env_key_auth(Some("env-key"))), oauth: None }).build(),
		);
		let error = models.get_auth_by_provider("p1", None).await.expect_err("read failure");
		assert_eq!(thrown_models_code(&error), Some(ModelsErrorCode::Auth));

		// modify failure during refresh
		struct ModifyFailingStore;
		#[async_trait::async_trait]
		impl CredentialStore for ModifyFailingStore {
			async fn read(
				&self,
				_provider_id: &str,
				_options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				Ok(Some(oauth_credential("old", "r", 0)))
			}
			async fn list(&self, _options: Option<AuthOperationOptions>) -> Result<Vec<crate::auth::types::CredentialInfo>, ThrownValue> {
				Ok(Vec::new())
			}
			async fn modify(
				&self,
				_provider_id: &str,
				_modify_fn: crate::auth::types::CredentialModifyFn,
				_options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				Err(ThrownValue::message("disk on fire"))
			}
			async fn delete(&self, _provider_id: &str, _options: Option<AuthOperationOptions>) -> Result<(), ThrownValue> {
				Ok(())
			}
		}
		let oauth_models = create_models(Some(CreateModelsOptions {
			credentials: Some(Arc::new(ModifyFailingStore)),
			..Default::default()
		}));
		oauth_models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: None, oauth: Some(Arc::new(test_oauth())) }).build(),
		);
		let error = oauth_models.get_auth_by_provider("p1", None).await.expect_err("modify failure");
		assert_eq!(thrown_models_code(&error), Some(ModelsErrorCode::Auth));
	}

	#[tokio::test]
	async fn keeps_the_underlying_reason_in_wrapped_oauth_refresh_errors() {
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		store_credential(&credentials, "p1", oauth_credential("old", "r", 0)).await;
		let oauth = test_oauth_with_refresh(Box::new(|_credential: OAuthCredential, _signal: AbortSignal| {
			Box::pin(async { Err(ThrownValue::message("token refresh failed (400): invalid_grant")) })
		}));
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials),
			..Default::default()
		}));
		models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: None, oauth: Some(Arc::new(oauth)) }).build(),
		);

		let error = models.get_auth_by_provider("p1", None).await.expect_err("refresh failure");
		assert_eq!(
			thrown_message(&error),
			"OAuth refresh failed for p1: token refresh failed (400): invalid_grant"
		);
	}

	#[tokio::test]
	async fn wraps_api_key_auth_failures_in_models_error() {
		let failing = ApiKeyAuth {
			name: "Failing".to_string(),
			login: None,
			check: None,
			resolve: Box::new(|_input: ApiKeyAuthInput<'_>| Box::pin(async { Err(ThrownValue::message("nope")) })),
		};
		let models = create_models(None);
		models.set_provider(TestProvider::new("p1").with_auth(ProviderAuth { api_key: Some(failing), oauth: None }).build());
		let error = models.get_auth_by_provider("p1", None).await.expect_err("resolve failure");
		assert_eq!(thrown_models_code(&error), Some(ModelsErrorCode::Auth));
	}

	#[tokio::test]
	async fn uses_explicit_request_api_key_and_env_during_provider_auth_resolution() {
		let calls: Calls = Arc::default();
		let scoped = ApiKeyAuth {
			name: "Scoped".to_string(),
			login: None,
			check: None,
			resolve: Box::new(|input: ApiKeyAuthInput<'_>| {
				Box::pin(async move {
					let account = input
						.credential
						.and_then(|credential| credential.env.as_ref())
						.and_then(|env| env.get("ACCOUNT_ID").cloned());
					let account = match account {
						Some(account) => Some(account),
						None => input.ctx.env("ACCOUNT_ID").await,
					};
					let key = input.credential.and_then(|credential| credential.key.clone());
					let (Some(key), Some(account)) = (key, account) else {
						return Ok(None);
					};
					Ok(Some(AuthResult {
						auth: ModelAuth {
							api_key: Some(key),
							base_url: Some(format!("https://example.test/{account}")),
							..Default::default()
						},
						env: Some(ProviderEnv::from([("ACCOUNT_ID".to_string(), account)])),
						source: None,
					}))
				})
			}),
		};
		let models = create_models(None);
		models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: Some(scoped), oauth: None }).with_calls(calls.clone()).build(),
		);
		let model = test_model("p1", "model-a");

		let mut options = ModelsSimpleStreamOptions::default();
		options.base.base.base.api_key = Some("explicit-key".to_string());
		options.base.base.base.env =
			Some(ProviderEnv::from([("ACCOUNT_ID".to_string(), "acct".to_string())]));
		let result = models.complete_simple(&model, &test_context(), Some(options)).await;
		assert_eq!(result.stop_reason, StopReason::Stop);

		let calls = calls.lock();
		assert_eq!(calls[0].model.base.base_url, "https://example.test/acct");
		assert_eq!(calls[0].options.api_key.as_deref(), Some("explicit-key"));
		assert_eq!(calls[0].options.env, Some(ProviderEnv::from([("ACCOUNT_ID".to_string(), "acct".to_string())])));
	}

	#[tokio::test]
	async fn merges_resolved_auth_into_stream_options_explicit_options_win_per_field() {
		let calls: Calls = Arc::default();
		let resolved = ApiKeyAuth {
			name: "Test".to_string(),
			login: None,
			check: None,
			resolve: Box::new(|_input: ApiKeyAuthInput<'_>| {
				Box::pin(async move {
					Ok(Some(AuthResult {
						auth: ModelAuth {
							api_key: Some("resolved-key".to_string()),
							headers: Some(headers(&[
								("Authorization", "Bearer resolved-key"),
								("x-a", "auth"),
								("x-b", "auth"),
							])),
							base_url: Some("https://auth.test/v1".to_string()),
						},
						env: None,
						source: None,
					}))
				})
			}),
		};
		let models = create_models(None);
		models.set_provider(
			TestProvider::new("p1").with_auth(ProviderAuth { api_key: Some(resolved), oauth: None }).with_calls(calls.clone()).build(),
		);
		let model = test_model("p1", "model-a");

		let mut options = ModelsSimpleStreamOptions::default();
		options.base.base.base.api_key = Some("explicit-key".to_string());
		options.base.base.base.headers =
			Some(headers(&[("authorization", "Explicit token"), ("x-b", "explicit")]));
		let result = models.complete_simple(&model, &test_context(), Some(options)).await;
		assert_eq!(result.stop_reason, StopReason::Stop);
		{
			let calls = calls.lock();
			assert_eq!(calls.len(), 1);
			assert_eq!(calls[0].options.api_key.as_deref(), Some("explicit-key"));
			assert_headers(calls[0].options.headers.as_ref(), &[
				("authorization", "Explicit token"),
				("x-a", "auth"),
				("x-b", "explicit"),
			]);
			assert_eq!(calls[0].model.base.base_url, "https://auth.test/v1");
		}

		// without explicit options, resolved auth applies
		let result = models.complete_simple(&model, &test_context(), None).await;
		assert_eq!(result.stop_reason, StopReason::Stop);
		let calls = calls.lock();
		assert_eq!(calls[1].options.api_key.as_deref(), Some("resolved-key"));
	}

	#[tokio::test]
	async fn adds_model_headers_only_for_model_auth_and_transforms_assembled_headers_once() {
		let calls: Calls = Arc::default();
		let models = create_models(None);
		models.set_provider(
			TestProvider::new("p1")
				.with_auth(ProviderAuth { api_key: Some(env_key_auth(Some("key"))), oauth: None })
				.with_calls(calls.clone())
				.build(),
		);
		let mut model = test_model("p1", "model-a");
		model.base.headers = Some([("x-model", "model"), ("x-shared", "model")]
			.iter()
			.map(|(name, value)| (name.to_string(), value.to_string()))
			.collect());

		assert_eq!(
			models.get_auth_by_provider("p1", None).await.expect("provider auth").expect("resolved").auth.headers,
			None
		);
		assert_headers(
			models
				.get_auth_by_model(&AnyModel::Chat(model.clone()), None)
				.await
				.expect("model auth")
				.expect("resolved")
				.auth
				.headers
				.as_ref(),
			&[("x-model", "model"), ("x-shared", "model")],
		);

		let transforms = Arc::new(AtomicUsize::new(0));
		let mut options = ModelsSimpleStreamOptions::default();
		options.base.base.base.headers = Some(headers(&[("x-explicit", "explicit"), ("X-Shared", "explicit")]));
		options.transforms = ModelsRequestTransforms {
			transform_headers: Some({
				let transforms = transforms.clone();
				Arc::new(move |assembled: Option<ProviderHeaders>| {
					let transforms = transforms.clone();
					Box::pin(async move {
						transforms.fetch_add(1, Ordering::SeqCst);
						assert_headers(assembled.as_ref(), &[
							("x-model", "model"),
							("x-explicit", "explicit"),
							("X-Shared", "explicit"),
						]);
						let mut merged = assembled.unwrap_or_default();
						merged.insert("x-transformed".to_string(), Some("yes".to_string()));
						Some(merged)
					})
				})
			}),
		};
		let result = models.complete_simple(&model, &test_context(), Some(options)).await;
		assert_eq!(result.stop_reason, StopReason::Stop);

		assert_eq!(transforms.load(Ordering::SeqCst), 1);
		let calls = calls.lock();
		assert_headers(calls[0].options.headers.as_ref(), &[
			("x-model", "model"),
			("x-explicit", "explicit"),
			("X-Shared", "explicit"),
			("x-transformed", "yes"),
		]);
	}

	#[tokio::test]
	async fn produces_an_error_stream_for_unknown_providers_instead_of_throwing() {
		let models = create_models(None);
		let result = models.complete_simple(&test_model("ghost", "model-a"), &test_context(), None).await;
		assert_eq!(result.stop_reason, StopReason::Error);
		assert!(result.error_message.expect("error message").contains("Unknown provider: ghost"));
	}

	#[tokio::test]
	async fn streams_through_the_provider() {
		let models = create_models(None);
		models.set_provider(TestProvider::new("p1").build());
		let model = test_model("p1", "model-a");

		let events: Arc<Mutex<Vec<&'static str>>> = Arc::default();
		let recorded = events.clone();
		let stream = models.stream_simple(&model, &test_context(), None);
		let collector = tokio::spawn(async move {
			use futures_util::StreamExt;
			let mut stream = stream;
			let mut types = Vec::new();
			while let Some(event) = stream.next().await {
				types.push(match event {
					AssistantMessageEvent::Start { .. } => "start",
					AssistantMessageEvent::Done { .. } => "done",
					AssistantMessageEvent::Error { .. } => "error",
					_ => "other",
				});
			}
			let result = stream.result().await;
			(types, result)
		});
		let (types, result) = collector.await.expect("collector task");
		assert_eq!(types, vec!["start", "done"]);
		assert_eq!(result.stop_reason, StopReason::Stop);
		let _ = recorded;
		let _ = events;
	}

	/// Runtime half of upstream `test/models-entry.test.ts` (the module-load
	/// hook assertions are Node-specific and unportable — see the module
	/// comment). A faux-style provider completes through the `Models` entry.
	#[tokio::test]
	async fn runs_a_faux_completion_through_the_models_entry() {
		let models = create_models(None);
		assert_eq!(models.get_models(None), Vec::new());
		assert!(models.get_providers().is_empty());
		models.set_provider(TestProvider::new("faux").build());
		let model = test_model("faux", "faux-model");
		let response = models.complete_simple(&model, &test_context(), None).await;
		assert_eq!(response.stop_reason, StopReason::Stop);
		assert!(matches!(
			response.content.first(),
			Some(crate::types::AssistantContent::Text(text)) if text.text == "ok"
		));
	}
}
