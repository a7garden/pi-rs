//! Port of `packages/coding-agent/src/core/model-runtime.ts`.
//!
//! Configured pi-ai Models collection used by coding-agent and SDK consumers
//! ([`ModelRuntime`]). Wraps a pi-ai [`ModelsImpl`] with:
//! - models.json composition ([`crate::core::model_config`]) and extension
//!   provider overlays ([`crate::core::provider_composer`]),
//! - runtime (non-persistent) API-key overrides ([`RuntimeCredentials`]),
//! - a synchronous availability/auth snapshot with sequence-guarded refreshes,
//! - credential lifecycle operations serialized per provider.
//!
//! Deviations (crate docs D63):
//! - **D63a — construction is async but infallible steps are sync.**
//!   `ModelConfig::load` is synchronous in the port (file read only), so
//!   `create()` is `async` solely to run the initial refresh.
//! - **D63b — catalog `generatedAt` is always `None`.** The embedded pi-ai
//!   catalog has no `generatedAt` manifest field (pi-ai `providers/all.rs`),
//!   so remote-catalog requests never send it.
//! - **D63c — credential-operation queue.** Upstream serializes operations
//!   per provider through a promise chain and races the "started" promise
//!   with the caller signal. The port serializes through a per-provider
//!   async mutex and checks the signal after acquiring it; an operation that
//!   is already running is awaited rather than abandoned.
//! - **D63d — create-time refresh timeout** cancels the refresh future
//!   directly (`tokio::time::timeout`) instead of aborting a linked signal.
//! - **D63e — typing replaces runtime asserts.** `stream`/`complete` and the
//!   image/classifier entry points take the concrete model type, so the
//!   upstream `assertChatModel`/`assertImageModel`/`assertClassifierModel`
//!   guards are compile-time facts.
//! - **D63f — snapshot maps are ordered.** `BTreeMap`/`BTreeSet` replace JS
//!   `Map`/`Set` insertion order (crate-wide convention, cf. pi-ai D16).

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, OnceLock, Weak};

use async_trait::async_trait;
use futures_util::future::BoxFuture;
use parking_lot::Mutex;

use pi_ai::auth::resolve::AuthResolutionOverrides;
use pi_ai::auth::types::{
	AuthCheck, AuthInteraction, AuthResult, AuthType, Credential, CredentialStore, LoginOptions,
};
use pi_ai::models::{
	clamp_thinking_level, create_models, CreateModelsOptions, Models, ModelsApiStreamOptions,
	ModelsClassifierOptions, ModelsDeferredCancelOptions, ModelsDeferredFetchOptions,
	ModelsImagesOptions, ModelsRefreshOptions, ModelsRefreshResult,
	ModelsSimpleStreamOptions, MutableModels, Provider,
};
use pi_ai::models_store::ModelsStore;
use pi_ai::types::{
	AnyModel, AssistantMessage, AssistantMessageEventStream, ClassifierContext, ClassifierModel, ClassifierResult,
	Context, DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, ImageModel, ImagesContext, ImagesOptions,
	Model, ModelThinkingLevel, ModelType, ProviderEnv, ProviderHeaders, SimpleStreamOptions, StreamOptions,
};
use pi_ai::utils::abort::{operation_signal, AbortSignal};
use pi_ai::utils::event_stream::AssistantMessageEventStream as _;
use pi_ai::models::models_error;
use pi_ai::utils::models_error::{ModelsError, ModelsErrorCode};
use pi_ai::utils::thrown::ThrownValue;
use pi_ai::utils::transcript::normalize_context;

use crate::core::model_config::ModelConfig;
use crate::core::provider_composer::{
	compose_model_provider, configured_request_auth_status, resolve_compatibility_request_config,
	resolve_configured_model_headers, AuthStatus, AuthStatusSource, CompatibilityRequestConfig, ProviderConfigInput,
};
use crate::core::runtime_credentials::RuntimeCredentials;

/// Credential operation kind (`CredentialSynchronizationOperation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialSynchronizationOperation {
	Login,
	Logout,
	SetRuntimeApiKey,
	RemoveRuntimeApiKey,
}

impl CredentialSynchronizationOperation {
	pub fn as_str(&self) -> &'static str {
		match self {
			CredentialSynchronizationOperation::Login => "login",
			CredentialSynchronizationOperation::Logout => "logout",
			CredentialSynchronizationOperation::SetRuntimeApiKey => "setRuntimeApiKey",
			CredentialSynchronizationOperation::RemoveRuntimeApiKey => "removeRuntimeApiKey",
		}
	}
}

/// Credentials changed successfully, but the local model/auth snapshot could
/// not be synchronized (`CredentialSynchronizationError`).
#[derive(Debug)]
pub struct CredentialSynchronizationError {
	pub provider_id: String,
	pub operation: CredentialSynchronizationOperation,
	pub cause: ThrownValue,
}

impl std::fmt::Display for CredentialSynchronizationError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"Credential {} committed for {}, but local synchronization failed",
			self.operation.as_str(),
			self.provider_id
		)
	}
}

impl std::error::Error for CredentialSynchronizationError {}

/// Auth overrides accepted by `getAuth` (`ModelRuntimeAuthOverrides`).
#[derive(Default, Clone)]
pub struct ModelRuntimeAuthOverrides {
	pub api_key: Option<String>,
	pub env: Option<ProviderEnv>,
	/// Require this much remaining OAuth-token validity; defaults to five
	/// minutes.
	pub min_oauth_validity_ms: Option<i64>,
	pub signal: Option<AbortSignal>,
}

impl ModelRuntimeAuthOverrides {
	fn to_resolution_overrides(&self) -> AuthResolutionOverrides {
		AuthResolutionOverrides {
			api_key: self.api_key.clone(),
			env: self.env.clone(),
			min_oauth_validity_ms: self.min_oauth_validity_ms,
			signal: self.signal.clone(),
		}
	}
}

/// `CreateModelRuntimeOptions`.
pub struct CreateModelRuntimeOptions {
	/// Credential storage. Defaults to the file at `auth_path`.
	pub credentials: Option<Arc<dyn CredentialStore>>,
	pub auth_path: Option<String>,
	/// `None` = default `get_agent_dir()/models.json`; `Some(None)` = upstream
	/// `modelsPath: null` (no file backing).
	pub models_path: Option<Option<String>>,
	pub models_store: Option<Arc<dyn ModelsStore>>,
	pub models_store_path: Option<String>,
	/// Allow `create()` to refresh model catalogs over the network. Defaults
	/// to false.
	pub allow_model_network: bool,
	/// Timeout for the create-time network model refresh.
	pub model_refresh_timeout_ms: Option<u64>,
	pub catalog_base_url: Option<String>,
	/// Optional caller cancellation for initial cache restoration and
	/// availability checks.
	pub signal: Option<AbortSignal>,
	/// Skip initial catalog and availability refresh. Static models remain
	/// available. Defaults to true.
	pub refresh_on_create: bool,
}

/// Upstream `create()` refreshes unless `refreshOnCreate === false`
/// (`options.refreshOnCreate !== false`), so the default constructs with
/// `refresh_on_create: true`. A derived `Default` silently produced `false`,
/// leaving freshly created runtimes without a synchronous availability pass —
/// register/prompt flows then raced the fire-and-forget refreshes
/// (`No API key found for <provider>` at first prompt).
impl Default for CreateModelRuntimeOptions {
	fn default() -> Self {
		CreateModelRuntimeOptions {
			credentials: None,
			auth_path: None,
			models_path: None,
			models_store: None,
			models_store_path: None,
			allow_model_network: false,
			model_refresh_timeout_ms: None,
			catalog_base_url: None,
			signal: None,
			refresh_on_create: true,
		}
	}
}

/// A virtual model registered under a provider (`RegisteredVirtualModel`).
pub struct RegisteredVirtualModel {
	pub model: Model,
	pub definition: crate::core::virtual_models::VirtualModelDefinition,
}

/// Immutable availability snapshot (`ModelRuntimeSnapshot`).
#[derive(Clone, Default)]
struct Snapshot {
	all: Vec<Model>,
	available: Vec<Model>,
	configured_providers: BTreeSet<String>,
	stored_providers: BTreeSet<String>,
	auth: BTreeMap<String, AuthCheck>,
}

/// Provider maps rebuilt on composition changes.
#[derive(Default)]
struct ProviderMaps {
	builtins: BTreeMap<String, Arc<dyn Provider>>,
	native_extension_providers: BTreeMap<String, Arc<dyn Provider>>,
	extension_providers: BTreeMap<String, ProviderConfigInput>,
	composition_errors: BTreeMap<String, String>,
}

/// Config-derived and mutable runtime state.
struct RuntimeState {
	config: ModelConfig,
	/// Backing models.json path (`None` = no file), used by `refresh`.
	models_path: Option<String>,
	snapshot: Snapshot,
	availability_error: Option<String>,
}

/// Configured pi-ai Models collection (`ModelRuntime`). Implements
/// [`Models`]; coding-agent internals consume it directly, extensions see
/// the [`crate::core::model_registry::ModelRegistry`] facade.
pub struct ModelRuntime {
	me: OnceLock<Weak<ModelRuntime>>,
	models: Arc<pi_ai::models::ModelsImpl>,
	credentials: Arc<RuntimeCredentials>,
	default_builtins: BTreeMap<String, Arc<dyn Provider>>,
	maps: Mutex<ProviderMaps>,
	virtual_models: Mutex<BTreeMap<String, BTreeMap<String, RegisteredVirtualModel>>>,
	state: Mutex<RuntimeState>,
	model_network_enabled: bool,
	availability_refresh_seq: AtomicU64,
	availability_error_seq: AtomicU64,
	provider_availability_seq: Mutex<BTreeMap<String, u64>>,
	credential_chains: Mutex<BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl ModelRuntime {
	fn new(
		credentials: Arc<RuntimeCredentials>,
		config: ModelConfig,
		models_path: Option<String>,
		models_store: Arc<dyn ModelsStore>,
		providers: Vec<Arc<dyn Provider>>,
		model_network_enabled: bool,
	) -> ModelRuntime {
		let default_builtins: BTreeMap<String, Arc<dyn Provider>> =
			providers.into_iter().map(|provider| (provider.id().to_string(), provider)).collect();
		let models = create_models(Some(CreateModelsOptions {
			credentials: Some(credentials.clone() as Arc<dyn CredentialStore>),
			models_store: Some(models_store),
			auth_context: None,
		}));
		let runtime = ModelRuntime {
			me: OnceLock::new(),
			models,
			credentials,
			default_builtins,
			maps: Mutex::new(ProviderMaps::default()),
			virtual_models: Mutex::new(BTreeMap::new()),
			state: Mutex::new(RuntimeState {
				config,
				models_path,
				snapshot: Snapshot::default(),
				availability_error: None,
			}),
			model_network_enabled,
			availability_refresh_seq: AtomicU64::new(0),
			availability_error_seq: AtomicU64::new(0),
			provider_availability_seq: Mutex::new(BTreeMap::new()),
			credential_chains: Mutex::new(BTreeMap::new()),
		};
		{
			let mut maps = runtime.maps.lock();
			for (id, provider) in runtime.default_builtins.iter() {
				maps.builtins.insert(id.clone(), provider.clone());
			}
		}
		runtime.rebuild_providers();
		runtime
	}

	/// `Arc` handle for spawned stream setups. Only `create` roots the
	/// runtime in an `Arc`.
	fn self_arc(&self) -> Option<Arc<ModelRuntime>> {
		self.me.get().and_then(|weak| weak.upgrade())
	}

	/// `ModelRuntime.create`.
	pub async fn create(options: CreateModelRuntimeOptions) -> Result<Arc<ModelRuntime>, ThrownValue> {
		let credentials = Arc::new(RuntimeCredentials::new(match options.credentials {
			Some(store) => store,
			None => Arc::new(crate::core::auth_storage::AuthStorage::new(options.auth_path)),
		}));
		let models_path = match options.models_path {
			Some(None) => None,
			Some(Some(path)) => Some(path),
			None => Some(
				std::path::Path::new(&crate::config::get_agent_dir())
					.join("models.json")
					.to_string_lossy()
					.into_owned(),
			),
		};
		let config = ModelConfig::load(models_path.as_deref());
		let models_store: Arc<dyn ModelsStore> = match options.models_store {
			Some(store) => store,
			None => match &models_path {
				Some(path) => {
					let store_path = match &options.models_store_path {
						Some(path) => path.clone(),
						None => {
							std::path::Path::new(path)
								.parent()
								.map(|dir| dir.join("models-store.json"))
								.unwrap_or_else(|| std::path::PathBuf::from("models-store.json"))
								.to_string_lossy()
								.into_owned()
						}
					};
					Arc::new(crate::core::models_store_wrapper::FileModelsStore::new(Some(&store_path)))
				}
				None => Arc::new(crate::core::models_store_wrapper::InMemoryCodingAgentModelsStore::new()),
			},
		};
		// D63b: the embedded pi-ai catalog has no `generatedAt` manifest field
		// (pi-ai `providers/all.rs`), so remote-catalog requests never send it.
		let generated_at: Option<u64> = None;
		let providers: Vec<Arc<dyn Provider>> = pi_ai::providers::all::builtin_providers()
			.into_iter()
			.map(|provider| {
				if provider.id() == "radius" {
					provider
				} else {
					crate::core::remote_catalog_provider::with_remote_catalog(
						provider,
						options.catalog_base_url.as_deref(),
						generated_at,
					)
				}
			})
			.collect();
		let model_network_enabled = std::env::var("PI_OFFLINE").is_err();
		let runtime = Arc::new(ModelRuntime::new(
			credentials,
			config,
			models_path,
			models_store,
			providers,
			model_network_enabled,
		));
		let _ = runtime.me.set(Arc::downgrade(&runtime));
		runtime.configure_radius_providers();
		runtime.rebuild_providers();
		let refresh_from_network = runtime.model_network_enabled && options.allow_model_network;
		if options.refresh_on_create {
			let refresh = runtime.refresh(Some(ModelsRefreshOptions {
				allow_network: Some(refresh_from_network),
				signal: options.signal.clone(),
				..Default::default()
			}));
			let result = match (refresh, options.model_refresh_timeout_ms) {
				(future, Some(ms)) if refresh_from_network => {
					tokio::time::timeout(std::time::Duration::from_millis(ms), future) // D63d
						.await
						.unwrap_or(ModelsRefreshResult { aborted: true, errors: BTreeMap::new() })
				}
				(future, _) => future.await,
			};
			let _ = result;
		}
		Ok(runtime)
	}

	/// Radius oauth providers in models.json become builtin-backed providers
	/// (`configureRadiusProviders`).
	fn configure_radius_providers(&self) {
		// Lock order: `maps` before `state` — canonical crate-wide order
		// (compose_provider takes maps→state); the reverse order here
		// deadlocked against concurrent recomposition.
		let radius_providers: Vec<(String, String, Option<String>)> = {
			let state = self.state.lock();
			state
				.config
				.get_provider_ids()
				.into_iter()
				.filter_map(|provider_id| {
					let config = state.config.get_provider(provider_id)?;
					if config.oauth.as_deref() != Some("radius") {
						return None;
					}
					let base_url = config.base_url.clone()?;
					Some((provider_id.to_string(), config.name.clone().unwrap_or_else(|| provider_id.to_string()), Some(base_url)))
				})
				.collect()
		};
		let mut maps = self.maps.lock();
		maps.builtins.clear();
		for (provider_id, provider) in self.default_builtins.iter() {
			maps.builtins.insert(provider_id.clone(), provider.clone());
		}
		for (provider_id, name, base_url) in radius_providers {
			let Some(base_url) = base_url else { continue };
			let gateway = base_url.trim_end_matches("/v1").trim_end_matches('/').to_string();
			maps.builtins.insert(
				provider_id.clone(),
				pi_ai::providers::radius::radius_provider(pi_ai::providers::radius::RadiusProviderOptions {
					id: Some(provider_id),
					name: Some(name),
					gateway: Some(gateway),
					fetch: None,
				}),
			);
		}
	}

	fn provider_ids(&self) -> BTreeSet<String> {
		let maps = self.maps.lock();
		let state = self.state.lock();
		let mut ids = BTreeSet::new();
		ids.extend(maps.builtins.keys().cloned());
		ids.extend(maps.native_extension_providers.keys().cloned());
		ids.extend(state.config.get_provider_ids().into_iter().map(|id| id.to_string()));
		ids.extend(maps.extension_providers.keys().cloned());
		ids.extend(self.virtual_models.lock().keys().cloned());
		ids
	}

	/// The provider without virtual models, or `None` when only virtual models
	/// define it (`recomposeProvider`).
	/// The composed provider including virtual models (no registry mutation).
	fn composed_provider(&self, provider_id: &str) -> Option<Arc<dyn Provider>> {
		let provider = self.compose_provider(provider_id);
		let virtual_models: Vec<Model> = self
			.virtual_models
			.lock()
			.get(provider_id)
			.map(|entries| entries.values().map(|entry| entry.model.clone()).collect())
			.unwrap_or_default();
		if !virtual_models.is_empty() {
			Some(crate::core::virtual_models::with_virtual_models(
				provider_id,
				provider,
				virtual_models,
			))
		} else {
			provider
		}
	}

	fn recompose_provider(&self, provider_id: &str) -> Option<Arc<dyn Provider>> {
		let provider = self.composed_provider(provider_id);
		if let Some(provider) = &provider {
			self.models.set_provider(provider.clone());
		} else {
			self.models.delete_provider(provider_id);
		}
		provider
	}

	/// The provider without virtual models, or `None` when nothing defines it
	/// (`composeProvider`).
	fn compose_provider(&self, provider_id: &str) -> Option<Arc<dyn Provider>> {
		let maps = self.maps.lock();
		let state = self.state.lock();
		let base = maps
			.native_extension_providers
			.get(provider_id)
			.or_else(|| maps.builtins.get(provider_id))
			.cloned();
		let extension = maps.extension_providers.get(provider_id).cloned();
		if state.config.get_provider(provider_id).is_none() && extension.is_none() {
			// No overlays: use the builtin untouched so its auth/login/stream
			// behavior is exact.
			drop(maps);
			drop(state);
			self.maps.lock().composition_errors.remove(provider_id);
			return base;
		}
		match compose_model_provider(provider_id, base.clone(), &state.config, extension) {
			Ok(provider) => {
				drop(maps);
				drop(state);
				self.maps.lock().composition_errors.remove(provider_id);
				Some(provider)
			}
			Err(error) => {
				drop(maps);
				drop(state);
				self.maps.lock().composition_errors.insert(provider_id.to_string(), error);
				base
			}
		}
	}

	fn rebuild_providers(&self) {
		self.maps.lock().composition_errors.clear();
		// Compose every provider first, then swap the set atomically so a
		// concurrent reader never observes the clear-then-re-add window
		// (single-threaded upstream JS cannot observe it; see crate docs).
		let mut composed: Vec<Arc<dyn Provider>> = Vec::new();
		for provider_id in self.provider_ids() {
			if let Some(provider) = self.composed_provider(&provider_id) {
				composed.push(provider);
			}
		}
		self.models.replace_providers(composed);
		self.update_model_snapshot();
	}

	fn update_model_snapshot(&self) {
		let all = self.models.get_models(None);
		let mut state = self.state.lock();
		let configured = state.snapshot.configured_providers.clone();
		state.snapshot.all = all.clone();
		state.snapshot.available = all.into_iter().filter(|model| configured.contains(model.provider())).collect();
	}

	fn snapshot(&self) -> Snapshot {
		self.state.lock().snapshot.clone()
	}

	async fn run_availability_refresh(&self, seq: u64, error_seq: u64, signal: AbortSignal) -> Result<(), ThrownValue> {
		let options = Some(pi_ai::auth::types::AuthOperationOptions { signal: Some(signal.clone()) });
		let providers = self.models.get_providers();
		let mut checks = Vec::with_capacity(providers.len());
		for provider in &providers {
			let check = self.models.check_auth(provider.id(), options.clone()).await?;
			checks.push((provider.id().to_string(), check));
		}
		let credentials = self.credentials.list(options.clone()).await?;
		if self.availability_refresh_seq.load(Ordering::SeqCst) != seq {
			return Ok(());
		}
		let auth: BTreeMap<String, AuthCheck> = checks
			.iter()
			.filter_map(|(provider_id, check)| check.clone().map(|check| (provider_id.clone(), check)))
			.collect();
		let configured_providers: BTreeSet<String> =
			checks.iter().filter(|(_, check)| check.is_some()).map(|(id, _)| id.clone()).collect();
		let snapshot = Snapshot {
			all: self.models.get_models(None),
			available: self.models.get_available(None, options).await?,
			configured_providers,
			stored_providers: credentials.into_iter().map(|entry| entry.provider_id).collect(),
			auth,
		};
		{
			let mut state = self.state.lock();
			state.snapshot = snapshot;
			if error_seq == self.availability_error_seq.load(Ordering::SeqCst) {
				state.availability_error = None;
			}
		}
		Ok(())
	}

	async fn queue_availability_refresh(&self, signal: Option<AbortSignal>) -> Result<(), ThrownValue> {
		let seq = self.availability_refresh_seq.fetch_add(1, Ordering::SeqCst) + 1;
		{
			let mut provider_seq = self.provider_availability_seq.lock();
			for (_, value) in provider_seq.iter_mut() {
				*value += 1;
			}
		}
		let error_seq = self.availability_error_seq.fetch_add(1, Ordering::SeqCst) + 1;
		let effective_signal = operation_signal(signal.as_ref());
		if let Err(error) = self.run_availability_refresh(seq, error_seq, effective_signal.clone()).await {
			if error_seq == self.availability_error_seq.load(Ordering::SeqCst) && !effective_signal.aborted() {
				self.state.lock().availability_error = Some(thrown_message(&error));
			}
			return Err(error);
		}
		Ok(())
	}

	async fn refresh_provider_availability(&self, provider_id: &str, signal: AbortSignal) -> Result<(), ThrownValue> {
		// Invalidate any full availability pass that started before this
		// credential change.
		let seq = self.availability_refresh_seq.fetch_add(1, Ordering::SeqCst) + 1;
		let provider_seq = {
			let mut map = self.provider_availability_seq.lock();
			let entry = map.entry(provider_id.to_string()).or_insert(0);
			*entry += 1;
			*entry
		};
		let error_seq = self.availability_error_seq.fetch_add(1, Ordering::SeqCst) + 1;
		let options = Some(pi_ai::auth::types::AuthOperationOptions { signal: Some(signal.clone()) });
		let result = async {
			let available = self.models.get_available(Some(provider_id), options.clone()).await?;
			let auth = self.models.check_auth(provider_id, options.clone()).await?;
			let credential = self.credentials.read(provider_id, options.clone()).await?;
			signal.check().map_err(ThrownValue::error)?;
			// D63h: upstream drops the whole update when a newer availability
			// pass for this provider started (`providerSeq` mismatch). The
			// facts gathered here are read fresh from the provider and store,
			// and the write below re-reads the current snapshot and merges
			// only this provider's entries — strictly newer information — so
			// the port always writes instead of dropping. This keeps the
			// post-credential-change visibility guarantee when a fire-and-
			// forget registration refresh overlaps the synchronization.
			let mut snapshot = self.snapshot();
			match &auth {
				Some(check) => {
					snapshot.configured_providers.insert(provider_id.to_string());
					snapshot.auth.insert(provider_id.to_string(), check.clone());
				}
				None => {
					snapshot.configured_providers.remove(provider_id);
					snapshot.auth.remove(provider_id);
				}
			}
			if credential.is_some() {
				snapshot.stored_providers.insert(provider_id.to_string());
			} else {
				snapshot.stored_providers.remove(provider_id);
			}
			let all = self.models.get_models(None);
			let mut available_by_key: BTreeMap<String, Model> = snapshot
				.available
				.iter()
				.filter(|model| model.provider() != provider_id)
				.map(|model| (model_key(model), model.clone()))
				.collect();
			for model in available {
				available_by_key.insert(model_key(&model), model);
			}
			snapshot.all = all.clone();
			snapshot.available =
				all.into_iter().filter_map(|model| available_by_key.remove(&model_key(&model))).collect();
			self.state.lock().snapshot = snapshot;
			let mut state = self.state.lock();
			if error_seq == self.availability_error_seq.load(Ordering::SeqCst) {
				state.availability_error = None;
			}
			Ok(())
		}
		.await;
		match result {
			Ok(value) => Ok(value),
			Err(error) => {
				if self.provider_availability_seq.lock().get(provider_id).copied() == Some(provider_seq)
					&& error_seq == self.availability_error_seq.load(Ordering::SeqCst)
					&& !signal.aborted()
				{
					self.state.lock().availability_error = Some(thrown_message(&error));
				}
				Err(error)
			}
		}
	}

	/// Last-known available chat models (`getAvailableSnapshot`).
	pub fn get_available_snapshot(&self) -> Vec<Model> {
		self.state.lock().snapshot.available.clone()
	}

	/// Aggregate configuration/availability errors (`getError`).
	pub fn get_error(&self) -> Option<String> {
		let maps = self.maps.lock();
		let state = self.state.lock();
		let mut errors = Vec::new();
		if let Some(config_error) = state.config.get_error() {
			errors.push(config_error.to_string());
		}
		for (provider_id, error) in &maps.composition_errors {
			errors.push(format!("Provider \"{provider_id}\": {error}"));
		}
		if let Some(availability_error) = &state.availability_error {
			errors.push(format!("Availability refresh: {availability_error}"));
		}
		if errors.is_empty() {
			None
		} else {
			Some(errors.join("\n\n"))
		}
	}

	/// `getRegisteredProviderConfig`.
	pub fn get_registered_provider_config(&self, provider_id: &str) -> Option<ProviderConfigInput> {
		self.maps.lock().extension_providers.get(provider_id).cloned()
	}

	/// `getRegisteredProviderIds`.
	pub fn get_registered_provider_ids(&self) -> Vec<String> {
		let maps = self.maps.lock();
		let mut ids: BTreeSet<String> = maps.extension_providers.keys().cloned().collect();
		ids.extend(maps.native_extension_providers.keys().cloned());
		ids.into_iter().collect()
	}

	/// `getRegisteredNativeProvider`.
	pub fn get_registered_native_provider(&self, provider_id: &str) -> Option<Arc<dyn Provider>> {
		self.maps.lock().native_extension_providers.get(provider_id).cloned()
	}

	/// Compatibility fallback for [`ModelRegistry`](crate::core::model_registry::ModelRegistry)
	/// when provider auth is unconfigured (`getCompatibilityRequestConfig`).
	pub fn get_compatibility_request_config(&self, model: &AnyModel) -> CompatibilityRequestConfig {
		let maps = self.maps.lock();
		let state = self.state.lock();
		resolve_compatibility_request_config(
			model,
			state.config.get_provider(model.provider()),
			maps.extension_providers.get(model.provider()),
		)
		.unwrap_or(CompatibilityRequestConfig { headers: None, auth_header: false })
	}

	/// `isUsingOAuth`.
	pub fn is_using_oauth(&self, provider_id: &str) -> bool {
		self.state
			.lock()
			.snapshot
			.auth
			.get(provider_id)
			.map(|check| check.check_type == AuthType::OAuth)
			.unwrap_or(false)
	}

	/// `isUsingSubscription`.
	pub fn is_using_subscription(&self, provider_id: &str) -> bool {
		let oauth = self.is_using_oauth(provider_id);
		oauth
			&& self
				.models
				.get_provider(provider_id)
				.map(|provider| provider.auth().oauth.as_ref().map(|oauth| oauth.is_subscription).unwrap_or(false))
				.unwrap_or(false)
	}

	/// `hasConfiguredAuth`.
	pub fn has_configured_auth(&self, provider_id: &str) -> bool {
		self.state.lock().snapshot.configured_providers.contains(provider_id)
	}

	/// `getAuth(providerId, overrides)`.
	pub async fn get_auth_for_provider(
		&self,
		provider_id: &str,
		overrides: Option<ModelRuntimeAuthOverrides>,
	) -> Result<Option<AuthResult>, ThrownValue> {
		let overrides = overrides.unwrap_or_default();
		self.models.get_auth_by_provider(provider_id, Some(overrides.to_resolution_overrides())).await
	}

	/// `getAuth(model, overrides)` — merges configured provider headers.
	pub async fn get_auth_for_model(
		&self,
		model: &AnyModel,
		overrides: Option<ModelRuntimeAuthOverrides>,
	) -> Result<Option<AuthResult>, ThrownValue> {
		let overrides = overrides.unwrap_or_default();
		let Some(mut resolution) =
			self.models.get_auth_by_model(model, Some(overrides.to_resolution_overrides())).await?
		else {
			return Ok(None);
		};
		let maps = self.maps.lock();
		let state = self.state.lock();
		let env: ProviderEnv = resolution
			.env
			.clone()
			.unwrap_or_default()
			.into_iter()
			.chain(overrides.env.clone().unwrap_or_default())
			.collect();
		let configured_headers = resolve_configured_model_headers(model, state.config.get_provider(model.provider()), maps.extension_providers.get(model.provider()), Some(&env))
			.map_err(ThrownValue::message)?;
		drop(maps);
		drop(state);
		let configured_headers = configured_headers.map(|headers| headers.into_iter().map(|(name, value)| (name, Some(value))).collect());
		resolution.auth.headers = merge_headers(resolution.auth.headers.clone(), configured_headers);
		Ok(Some(resolution))
	}

	/// Per-provider credential operation serialization (`enqueueCredentialOperation`).
	async fn enqueue_credential_operation<T, F>(&self, provider_id: &str, signal: &AbortSignal, task: F) -> Result<T, ThrownValue>
	where
		F: Future<Output = Result<T, ThrownValue>> + Send,
	{
		let chain = {
			let mut chains = self.credential_chains.lock();
			chains
				.entry(provider_id.to_string())
				.or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
				.clone()
		};
		let _guard = chain.lock().await;
		signal.check().map_err(ThrownValue::error)?; // D63c
		task.await
	}

	async fn synchronize_credential_state(
		&self,
		provider_id: &str,
		operation: CredentialSynchronizationOperation,
		credential: Option<Credential>,
		signal: AbortSignal,
	) -> Result<(), ThrownValue> {
		let inner = async {
			signal.check().map_err(ThrownValue::error)?;
			self.recompose_provider(provider_id);
			let composition_error = self.maps.lock().composition_errors.get(provider_id).cloned();
			if let Some(error) = composition_error {
				return Err(ThrownValue::message(error));
			}
			let result = self
				.models
				.refresh(Some(ModelsRefreshOptions {
					allow_network: Some(false),
					providers: Some(vec![provider_id.to_string()]),
					signal: Some(signal.clone()),
					..Default::default()
				}))
				.await;
			if result.aborted {
				signal.check().map_err(ThrownValue::error)?;
			}
			if let Some(error) = result.errors.get(provider_id) {
				return Err(error.clone());
			}
			self.update_model_snapshot();
			self.refresh_provider_availability(provider_id, signal).await
		};
		inner.await.map_err(|cause| {
			ThrownValue::error(CredentialSynchronizationError {
				provider_id: provider_id.to_string(),
				operation,
				cause,
			})
		})
	}

	/// `setRuntimeApiKey` (non-persistent override).
	pub async fn set_runtime_api_key(
		&self,
		provider_id: &str,
		api_key: &str,
		options: Option<pi_ai::auth::types::AuthOperationOptions>,
	) -> Result<(), ThrownValue> {
		let signal = operation_signal(options.as_ref().and_then(|options| options.signal.as_ref()));
		let provider_id = provider_id.to_string();
		let api_key = api_key.to_string();
		self.enqueue_credential_operation(&provider_id, &signal, async {
			self.credentials.set_runtime_api_key(&provider_id, &api_key);
			self.synchronize_credential_state(
				&provider_id,
				CredentialSynchronizationOperation::SetRuntimeApiKey,
				Some(Credential::ApiKey(pi_ai::auth::types::ApiKeyCredential { key: Some(api_key.clone()), env: None })),
				signal.clone(),
			)
			.await
		})
		.await
	}

	/// `removeRuntimeApiKey`.
	pub async fn remove_runtime_api_key(
		&self,
		provider_id: &str,
		options: Option<pi_ai::auth::types::AuthOperationOptions>,
	) -> Result<(), ThrownValue> {
		let signal = operation_signal(options.as_ref().and_then(|options| options.signal.as_ref()));
		let provider_id = provider_id.to_string();
		self.enqueue_credential_operation(&provider_id, &signal, async {
			self.credentials.remove_runtime_api_key(&provider_id);
			self.synchronize_credential_state(&provider_id, CredentialSynchronizationOperation::RemoveRuntimeApiKey, None, signal.clone())
				.await
		})
		.await
	}

	/// `listCredentials`.
	pub async fn list_credentials(
		&self,
		options: Option<pi_ai::auth::types::AuthOperationOptions>,
	) -> Result<Vec<pi_ai::auth::types::CredentialInfo>, ThrownValue> {
		self.credentials.list(options).await
	}

	/// `getProviderAuthStatus`.
	pub fn get_provider_auth_status(&self, provider_id: &str) -> AuthStatus {
		if self.credentials.has_runtime_api_key(provider_id) {
			return AuthStatus { configured: true, source: Some(AuthStatusSource::Runtime), label: None };
		}
		if self.state.lock().snapshot.stored_providers.contains(provider_id) {
			return AuthStatus { configured: true, source: Some(AuthStatusSource::Stored), label: None };
		}
		let maps = self.maps.lock();
		let state = self.state.lock();
		if let Some(configured) = configured_request_auth_status(
			state.config.get_provider(provider_id),
			maps.extension_providers.get(provider_id),
		) {
			return configured;
		}
		match state.snapshot.auth.get(provider_id) {
			Some(check) => AuthStatus {
				configured: true,
				source: Some(AuthStatusSource::Environment),
				label: check.source.clone(),
			},
			None => AuthStatus { configured: false, source: None, label: None },
		}
	}

	/// Request preparation shared by every dispatch path (`prepareRequest`).
	async fn prepare_request(
		&self,
		model: AnyModel,
		api_key: Option<String>,
		headers: Option<ProviderHeaders>,
		env: Option<ProviderEnv>,
		signal: Option<AbortSignal>,
	) -> Result<PreparedRequest, ThrownValue> {
		let provider = self
			.models
			.get_provider(model.provider())
			.ok_or_else(|| ThrownValue::error(models_error(ModelsErrorCode::Provider, format!("Unknown provider: {}", model.provider()))))?;
		let resolution = self
			.get_auth_for_model(
				&model,
				Some(ModelRuntimeAuthOverrides {
					api_key,
					env,
					min_oauth_validity_ms: None,
					signal,
				}),
			)
			.await?
			.ok_or_else(|| {
				ThrownValue::error(models_error(ModelsErrorCode::Auth, format!("Provider is not configured: {}", model.provider())))
			})?;
		Ok(PreparedRequest {
			provider,
			model,
			api_key: resolution.auth.api_key.clone(),
			headers: resolution.auth.headers.clone(),
			env: resolution.env.clone(),
			base_url: resolution.auth.base_url.clone(),
		})
	}

	/// Fire-and-forget refresh after registration mutations (`void this.refresh(...)`).
	fn spawn_refresh(&self, allow_network: bool) {
		let Some(runtime) = self.me.get().and_then(|weak| weak.upgrade()) else { return };
		tokio::spawn(async move {
			let _ = runtime.refresh(Some(ModelsRefreshOptions { allow_network: Some(allow_network), ..Default::default() })).await;
		});
	}

	/// `registerNativeProvider`.
	pub fn register_native_provider(&self, provider: Arc<dyn Provider>) -> Result<(), ThrownValue> {
		if provider.id().trim().is_empty() {
			return Err(ThrownValue::message("Provider id must not be empty."));
		}
		{
			let mut maps = self.maps.lock();
			maps.extension_providers.remove(provider.id());
			maps.native_extension_providers.insert(provider.id().to_string(), provider.clone());
		}
		self.recompose_provider(provider.id());
		self.update_model_snapshot();
		let provisional = {
			let state = self.state.lock();
			configured_request_auth_status(state.config.get_provider(provider.id()), None)
		};
		let auth_type = if provider.auth().oauth.is_some() && provider.auth().api_key.is_none() {
			AuthType::OAuth
		} else {
			AuthType::ApiKey
		};
		self.mark_provisionally_configured(provider.id(), provisional, auth_type);
		self.spawn_refresh(false);
		Ok(())
	}

	/// Mark a newly registered provider as configured when it has a stored
	/// credential or a configured API key (`markProvisionallyConfigured`).
	fn mark_provisionally_configured(&self, provider_id: &str, configured_status: Option<AuthStatus>, auth_type: AuthType) {
		let mut state = self.state.lock();
		let status_configured = configured_status.as_ref().map(|status| status.configured).unwrap_or(false);
		if !state.snapshot.stored_providers.contains(provider_id) && !status_configured {
			return;
		}
		state.snapshot.configured_providers.insert(provider_id.to_string());
		state
			.snapshot
			.auth
			.entry(provider_id.to_string())
			.or_insert_with(|| AuthCheck { check_type: auth_type, source: Some("configured provider".to_string()) });
		state.snapshot.available =
			state.snapshot.all.iter().filter(|model| state.snapshot.configured_providers.contains(model.provider())).cloned().collect();
	}

	/// `registerProvider(name, config)`.
	pub fn register_provider(&self, provider_id: &str, config: ProviderConfigInput) -> Result<(), ThrownValue> {
		// Validate the incoming registration on its own, like the legacy
		// registry: a broken re-registration must throw without touching the
		// stored config.
		{
			let maps = self.maps.lock();
			let state = self.state.lock();
			crate::core::provider_composer::validate_extension_provider(
				provider_id,
				maps.builtins.get(provider_id),
				state.config.get_provider(provider_id),
				&config,
			)
			.map_err(ThrownValue::message)?;
		}
		{
			let mut maps = self.maps.lock();
			maps.native_extension_providers.remove(provider_id);
			// Re-registration merges defined values over the previous
			// registration and preserves undefined ones, matching the legacy
			// ModelRegistry contract.
			let previous = maps.extension_providers.get(provider_id).cloned().unwrap_or_default();
			maps.extension_providers.insert(provider_id.to_string(), merge_provider_config(previous, config));
		}
		self.recompose_provider(provider_id);
		self.update_model_snapshot();
		let (provisional, auth_type) = {
			let maps = self.maps.lock();
			let state = self.state.lock();
			let effective = maps.extension_providers.get(provider_id).cloned().unwrap_or_default();
			let status = configured_request_auth_status(state.config.get_provider(provider_id), Some(&effective));
			let auth_type = if effective.oauth.is_some() && effective.api_key.is_none() {
				AuthType::OAuth
			} else {
				AuthType::ApiKey
			};
			(status, auth_type)
		};
		self.mark_provisionally_configured(provider_id, provisional, auth_type);
		self.spawn_refresh(false);
		Ok(())
	}

	/// `unregisterProvider`.
	pub fn unregister_provider(&self, provider_id: &str) {
		{
			let mut maps = self.maps.lock();
			maps.extension_providers.remove(provider_id);
			maps.native_extension_providers.remove(provider_id);
		}
		self.recompose_provider(provider_id);
		self.update_model_snapshot();
		self.spawn_refresh(false);
	}

	/// `registerVirtualModel`.
	pub fn register_virtual_model(&self, definition: crate::core::virtual_models::VirtualModelDefinition) -> Result<(), ThrownValue> {
		let provider_id = definition.provider.clone();
		let id = definition.id.clone();
		if provider_id.trim().is_empty() || id.trim().is_empty() {
			return Err(ThrownValue::message("Virtual model provider and id must not be empty."));
		}
		if let Some(existing) = self.models.get_model(&provider_id, &id) {
			if !crate::core::virtual_models::is_virtual_model(&existing) {
				return Err(ThrownValue::message(format!(
					"Virtual model {provider_id}/{id} conflicts with a physical model."
				)));
			}
		}
		let model = crate::core::virtual_models::create_virtual_model(&definition);
		{
			let mut virtual_models = self.virtual_models.lock();
			virtual_models
				.entry(provider_id.clone())
				.or_default()
				.insert(id, RegisteredVirtualModel { model, definition });
		}
		let recomposed = self.recompose_provider(&provider_id);
		if recomposed.is_none() && !self.state.lock().snapshot.configured_providers.contains(&provider_id) {
			// A provider of only virtual models needs no credentials. Mark it
			// configured now: session restore checks auth before the refresh
			// below lands.
			let mut state = self.state.lock();
			state.snapshot.auth.insert(
				provider_id.clone(),
				AuthCheck { check_type: AuthType::ApiKey, source: Some("virtual".to_string()) },
			);
			state.snapshot.configured_providers.insert(provider_id);
		}
		self.update_model_snapshot();
		self.spawn_refresh(false);
		Ok(())
	}

	/// `registerVirtualModel` over an extension-queued definition
	/// ([`crate::core::extensions::types::VirtualModelDefinition`], D27/D31):
	/// adapts the extension route shape (`BoxFuture<ModelRoute>`) to the
	/// virtual-models router shape (`BoxFuture<Result<ModelRoute,
	/// ThrownValue>>`).
	pub fn register_extension_virtual_model(
		&self,
		definition: crate::core::extensions::types::VirtualModelDefinition,
	) -> Result<(), ThrownValue> {
		let crate::core::extensions::types::VirtualModelDefinition {
			provider,
			id,
			name,
			thinking_levels,
			context_window,
			max_tokens,
			input,
			route,
		} = definition;
		let adapted_route: crate::core::virtual_models::ModelRouteFn = Arc::new(move |request| {
			let route = Arc::clone(&route);
			Box::pin(async move {
				// virtual_models::RouteRequest -> extension ModelRouteRequest
				let request = crate::core::extensions::types::ModelRouteRequest {
					model: request.model,
					thinking_level: request.thinking_level,
					reason: match request.reason {
						crate::core::virtual_models::ModelRouteReason::User => {
							crate::core::extensions::types::ModelRouteReason::User
						}
						crate::core::virtual_models::ModelRouteReason::Continuation => {
							crate::core::extensions::types::ModelRouteReason::Continuation
						}
						crate::core::virtual_models::ModelRouteReason::Retry => {
							crate::core::extensions::types::ModelRouteReason::Retry
						}
						crate::core::virtual_models::ModelRouteReason::Direct => {
							crate::core::extensions::types::ModelRouteReason::Direct
						}
					},
					previous: request.previous.map(|previous| crate::core::extensions::types::RouteModelRef {
						model: previous.model,
						thinking_level: previous.thinking_level,
					}),
					failed: request.failed.map(|failed| crate::core::extensions::types::RouteFailedRequest {
						model: failed.model,
						thinking_level: failed.thinking_level,
						message: failed.message,
					}),
					state: request.state,
					messages: request.messages,
					signal: {
						// The loader stub uses crate utils::abort; the pi-ai
						// signal is carried by re-wrapping through the bridge
						// controller when present. Extension routes observe
						// cancellation through the crate signal (D30).
						Some(crate::utils::abort::AbortSignal::default())
					},
				};
				let route = route(request).await;
				Ok(crate::core::virtual_models::ModelRoute {
					model: route.model,
					thinking_level: route.thinking_level,
					state: route.state,
				})
			})
		});
		self.register_virtual_model(crate::core::virtual_models::VirtualModelDefinition {
			provider,
			id,
			name,
			thinking_levels,
			context_window,
			max_tokens,
			input,
			route: adapted_route,
		})
	}

	/// `unregisterVirtualModel`.
	pub fn unregister_virtual_model(&self, provider_id: &str, id: &str) {
		let removed = {
			let mut virtual_models = self.virtual_models.lock();
			let mut removed = false;
			if let Some(entries) = virtual_models.get_mut(provider_id) {
				removed = entries.remove(id).is_some();
				if entries.is_empty() {
					virtual_models.remove(provider_id);
				}
			}
			removed
		};
		if !removed {
			return;
		}
		self.recompose_provider(provider_id);
		self.update_model_snapshot();
		self.spawn_refresh(false);
	}

	/// Ask a virtual model's router for the model and thinking level of one
	/// request (`resolveModel`). See [`crate::core::virtual_models`] D62 for
	/// the route-callable seam.
	pub async fn resolve_model(
		&self,
		model: &Model,
		messages: &[pi_ai::types::Message],
		options: crate::core::virtual_models::RouteRequestOptions,
	) -> Result<crate::core::virtual_models::ModelRoute, ThrownValue> {
		let name = format!("Virtual model {}/{}", model.provider(), model.id());
		let virtual_entry = self
			.virtual_models
			.lock()
			.get(model.provider())
			.and_then(|entries| entries.get(model.id()))
			.map(|entry| entry.definition.clone());
		let Some(definition) = virtual_entry else {
			return Err(ThrownValue::message(format!("{name} is not registered.")));
		};
		let latest = crate::core::virtual_models::find_latest_response(messages);
		let previous = latest.and_then(|latest| {
			self.get_physical_model(&latest.provider, &latest.model).map(|model| {
				crate::core::virtual_models::PreviousRoute {
					model,
					thinking_level: latest.thinking_level,
				}
			})
		});
		// A failed routing attempt names the virtual model; there is no
		// physical request to report.
		let failed = options.failed.as_ref().and_then(|failed| {
			self.get_physical_model(&failed.provider, &failed.model).map(|model| {
				crate::core::virtual_models::FailedRoute {
					model,
					thinking_level: failed.thinking_level,
					message: failed.clone(),
				}
			})
		});
		let request = crate::core::virtual_models::RouteRequest {
			model: model.clone(),
			thinking_level: options.thinking_level,
			reason: options.reason,
			previous,
			failed,
			state: options.state,
			messages: messages.to_vec(),
			signal: options.signal,
		};
		let route = crate::core::virtual_models::route(&definition, &request).await?;
		let routed = format!("{name} routed to {}/{}", route.model.provider(), route.model.id());
		let target = self.get_physical_model(route.model.provider(), route.model.id());
		let Some(target) = target else {
			return Err(ThrownValue::message(format!("{routed}, which is not a physical model.")));
		};
		if !self.has_configured_auth(target.provider()) {
			return Err(ThrownValue::message(format!("{routed}, which has no credentials.")));
		}
		let thinking_level = clamp_thinking_level(&target, route.thinking_level);
		Ok(crate::core::virtual_models::ModelRoute {
			model: target,
			thinking_level,
			state: route.state,
		})
	}

	/// A catalog chat model that is not virtual (`getPhysicalModel`).
	pub fn get_physical_model(&self, provider_id: &str, model_id: &str) -> Option<Model> {
		let model = self.models.get_model(provider_id, model_id)?;
		if crate::core::virtual_models::is_virtual_model(&model) {
			None
		} else {
			Some(model)
		}
	}
}

/// Prepared request facts (`prepareRequest` result).
struct PreparedRequest {
	provider: Arc<dyn Provider>,
	model: AnyModel,
	api_key: Option<String>,
	headers: Option<ProviderHeaders>,
	env: Option<ProviderEnv>,
	base_url: Option<String>,
}

fn model_key(model: &Model) -> String {
	format!("{}\0{}", model.provider(), model.id())
}

fn thrown_message(value: &ThrownValue) -> String {
	match value {
		ThrownValue::Text(text) => text.clone(),
		ThrownValue::Error(error) => error.to_string(),
	}
}

/// Case-insensitive header merge (`mergeHeaders`). `None` values in the
/// override suppress base headers.
pub(crate) fn merge_headers(
	base: Option<ProviderHeaders>,
	override_headers: Option<ProviderHeaders>,
) -> Option<ProviderHeaders> {
	if base.is_none() && override_headers.is_none() {
		return None;
	}
	let mut merged = base.unwrap_or_default();
	if let Some(override_headers) = override_headers {
		for (name, value) in override_headers {
			let lower = name.to_lowercase();
			let existing: Vec<String> = merged
				.keys()
				.filter(|existing| existing.to_lowercase() == lower)
				.cloned()
				.collect();
			for existing in existing {
				merged.remove(&existing);
			}
			merged.insert(name, value);
		}
	}
	Some(merged)
}

/// Defined values over previous, preserving undefined
/// (`registerProvider` merge).
fn merge_provider_config(mut previous: ProviderConfigInput, config: ProviderConfigInput) -> ProviderConfigInput {
	if config.name.is_some() {
		previous.name = config.name;
	}
	if config.base_url.is_some() {
		previous.base_url = config.base_url;
	}
	if config.api_key.is_some() {
		previous.api_key = config.api_key;
	}
	if config.api.is_some() {
		previous.api = config.api;
	}
	if config.stream_simple.is_some() {
		previous.stream_simple = config.stream_simple;
	}
	if config.images.is_some() {
		previous.images = config.images;
	}
	if config.classifiers.is_some() {
		previous.classifiers = config.classifiers;
	}
	if config.headers.is_some() {
		previous.headers = config.headers;
	}
	if config.auth_header.is_some() {
		previous.auth_header = config.auth_header;
	}
	if config.models.is_some() {
		previous.models = config.models;
	}
	if config.refresh_models.is_some() {
		previous.refresh_models = config.refresh_models;
	}
	if config.oauth.is_some() {
		previous.oauth = config.oauth;
	}
	previous
}

#[async_trait::async_trait]
impl Models for ModelRuntime {
	fn get_providers(&self) -> Vec<Arc<dyn Provider>> {
		self.models.get_providers()
	}

	fn get_provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
		self.models.get_provider(id)
	}

	fn get_models(&self, provider: Option<&str>) -> Vec<Model> {
		self.models.get_models(provider)
	}

	fn get_model(&self, provider: &str, id: &str) -> Option<Model> {
		self.models.get_model(provider, id)
	}

	fn get_models_of_type(&self, type_: ModelType, provider: Option<&str>) -> Vec<AnyModel> {
		self.models.get_models_of_type(type_, provider)
	}

	fn get_model_of_type(&self, type_: ModelType, provider: &str, id: &str) -> Option<AnyModel> {
		self.models.get_model_of_type(type_, provider, id)
	}

	fn get_all_models(&self, provider: Option<&str>) -> Vec<AnyModel> {
		self.models.get_all_models(provider)
	}

	async fn refresh(&self, options: Option<ModelsRefreshOptions>) -> ModelsRefreshResult {
		let options = options.unwrap_or_default();
		{
			let models_path = self.state.lock().models_path.clone();
			let config = ModelConfig::load(models_path.as_deref());
			self.state.lock().config = config;
		}
		self.configure_radius_providers();
		if let Some(providers) = &options.providers {
			for provider_id in providers.iter().collect::<BTreeSet<_>>() {
				self.recompose_provider(provider_id);
			}
			self.update_model_snapshot();
		} else {
			self.rebuild_providers();
		}
		let refresh_options = ModelsRefreshOptions {
			allow_network: options.allow_network.or(Some(self.model_network_enabled)),
			..options.clone()
		};
		let result = self.models.refresh(Some(refresh_options.clone())).await;
		let mut errors = result.errors.clone();
		self.update_model_snapshot();
		let signal = operation_signal(options.signal.as_ref());
		if let Some(providers) = &options.providers {
			for provider_id in providers.iter().collect::<BTreeSet<_>>() {
				if let Err(error) = self.refresh_provider_availability(provider_id, signal.clone()).await {
					if !signal.aborted() {
						errors.insert(provider_id.clone(), error);
					}
				}
			}
		} else if let Err(_error) = self.queue_availability_refresh(options.signal.clone()).await {
			// Availability errors are recorded by the latest pass; refreshed
			// models remain usable.
		}
		ModelsRefreshResult { aborted: result.aborted || signal.aborted(), errors }
	}

	async fn check_auth(
		&self,
		provider_id: &str,
		options: Option<pi_ai::auth::types::AuthOperationOptions>,
	) -> Result<Option<AuthCheck>, ThrownValue> {
		self.models.check_auth(provider_id, options).await
	}

	async fn get_available(
		&self,
		provider_id: Option<&str>,
		options: Option<pi_ai::auth::types::AuthOperationOptions>,
	) -> Result<Vec<Model>, ThrownValue> {
		if let Some(provider_id) = provider_id {
			let error_seq = self.availability_error_seq.fetch_add(1, Ordering::SeqCst) + 1;
			let result = self.models.get_available(Some(provider_id), options.clone()).await;
			match result {
				Ok(available) => {
					if error_seq == self.availability_error_seq.load(Ordering::SeqCst) {
						self.state.lock().availability_error = None;
					}
					Ok(available)
				}
				Err(error) => {
					let aborted = options.as_ref().and_then(|options| options.signal.as_ref()).map(|signal| signal.aborted()).unwrap_or(false);
					if error_seq == self.availability_error_seq.load(Ordering::SeqCst) && !aborted {
						self.state.lock().availability_error = Some(thrown_message(&error));
					}
					Err(error)
				}
			}
		} else {
			self.queue_availability_refresh(options.as_ref().and_then(|options| options.signal.clone())).await?;
			Ok(self.state.lock().snapshot.available.clone())
		}
	}

	async fn get_all_available(
		&self,
		provider_id: Option<&str>,
		options: Option<pi_ai::auth::types::AuthOperationOptions>,
	) -> Result<Vec<AnyModel>, ThrownValue> {
		self.models.get_all_available(provider_id, options).await
	}

	async fn get_auth_by_provider(
		&self,
		provider_id: &str,
		overrides: Option<AuthResolutionOverrides>,
	) -> Result<Option<AuthResult>, ThrownValue> {
		let overrides = overrides.map(|overrides| ModelRuntimeAuthOverrides {
			api_key: overrides.api_key,
			env: overrides.env,
			min_oauth_validity_ms: overrides.min_oauth_validity_ms,
			signal: overrides.signal,
		});
		self.get_auth_for_provider(provider_id, overrides).await
	}

	async fn get_auth_by_model(
		&self,
		model: &AnyModel,
		overrides: Option<AuthResolutionOverrides>,
	) -> Result<Option<AuthResult>, ThrownValue> {
		let overrides = overrides.map(|overrides| ModelRuntimeAuthOverrides {
			api_key: overrides.api_key,
			env: overrides.env,
			min_oauth_validity_ms: overrides.min_oauth_validity_ms,
			signal: overrides.signal,
		});
		self.get_auth_for_model(model, overrides).await
	}

	async fn login(
		&self,
		provider_id: &str,
		type_: AuthType,
		interaction: AuthInteraction,
		options: Option<LoginOptions>,
	) -> Result<Credential, ThrownValue> {
		let signal = operation_signal(interaction.signal.as_ref());
		let provider_id = provider_id.to_string();
		self.enqueue_credential_operation(&provider_id, &signal, async {
			let credential = self.models.login(&provider_id, type_, interaction, options).await?;
			self.synchronize_credential_state(&provider_id, CredentialSynchronizationOperation::Login, Some(credential.clone()), signal.clone())
				.await?;
			Ok(credential)
		})
		.await
	}

	async fn logout(
		&self,
		provider_id: &str,
		options: Option<pi_ai::auth::types::AuthOperationOptions>,
	) -> Result<(), ThrownValue> {
		let signal = operation_signal(options.as_ref().and_then(|options| options.signal.as_ref()));
		let provider_id = provider_id.to_string();
		self.enqueue_credential_operation(&provider_id, &signal, async {
			self.models.logout(&provider_id, Some(pi_ai::auth::types::AuthOperationOptions { signal: Some(signal.clone()) })).await?;
			self.synchronize_credential_state(&provider_id, CredentialSynchronizationOperation::Logout, None, signal.clone()).await
		})
		.await
	}

	fn stream(
		&self,
		model: &Model,
		context: &Context,
		options: Option<ModelsApiStreamOptions>,
	) -> AssistantMessageEventStream {
		let transcript = normalize_context(context);
		let model = model.clone();
		let options = options.unwrap_or_default();
		let this = match ModelRuntime::self_arc(self) {
			Some(this) => this,
			None => {
				return pi_ai::api::lazy::lazy_stream(&model, async {
					Err(ThrownValue::message("ModelRuntime must be Arc-rooted to stream"))
				})
			}
		};
		let outer_model = model.clone();
		pi_ai::api::lazy::lazy_stream(&outer_model, async move {
			let prepared = this
				.prepare_request(
					AnyModel::Chat(model.clone()),
					options.base.base.api_key.clone(),
					options.base.base.headers.clone(),
					options.base.base.env.clone(),
					options.base.base.signal.clone(),
				)
				.await?;
			let mut stream_options = options;
			stream_options.base.base.api_key =
				Some(stream_options.base.base.api_key.clone().or(prepared.api_key).unwrap_or_default());
			stream_options.base.base.headers =
				merge_headers(prepared.headers, stream_options.base.base.headers.take());
			stream_options.base.base.env = match (prepared.env, stream_options.base.base.env.take()) {
				(None, None) => None,
				(resolved, caller) => Some(resolved.unwrap_or_default().into_iter().chain(caller.unwrap_or_default()).collect()),
			};
			// Caller transforms apply once to the fully assembled headers and
			// are not forwarded to the provider (`ModelsRequestTransforms`).
			if let Some(transform_headers) = stream_options.transforms.transform_headers.clone() {
				stream_options.base.base.headers = transform_headers(stream_options.base.base.headers.take()).await;
			}
			let request_model = match &prepared.model {
				AnyModel::Chat(model) => {
					let mut model = model.clone();
					if let Some(base_url) = &prepared.base_url {
						model.base.base_url = base_url.clone();
					}
					model
				}
				_ => model.clone(),
			};
			Ok(prepared.provider.stream(&request_model, &transcript, &stream_options.base))
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
		let model = model.clone();
		let options = options.unwrap_or_default();
		// Requests outside the agent loop are routed here. Callers sized them
		// before routing, so cap the output budget to the routed model.
		// (Virtual-model routing is applied by `resolve_model` for callers
		// that need it; `streamSimple` dispatches the physical request, D63g.)
		let this = match ModelRuntime::self_arc(self) {
			Some(this) => this,
			None => {
				return pi_ai::api::lazy::lazy_stream(&model, async {
					Err(ThrownValue::message("ModelRuntime must be Arc-rooted to stream"))
				})
			}
		};
		let outer_model = model.clone();
		pi_ai::api::lazy::lazy_stream(&outer_model, async move {
			let prepared = this
				.prepare_request(
					AnyModel::Chat(model.clone()),
					options.base.base.api_key.clone(),
					options.base.base.headers.clone(),
					options.base.base.env.clone(),
					options.base.base.signal.clone(),
				)
				.await?;
			let mut simple_options = options;
			simple_options.base.base.api_key =
				Some(simple_options.base.base.api_key.clone().or(prepared.api_key).unwrap_or_default());
			simple_options.base.base.headers = merge_headers(prepared.headers, simple_options.base.base.headers.take());
			simple_options.base.base.env = match (prepared.env, simple_options.base.base.env.take()) {
				(None, None) => None,
				(resolved, caller) => Some(resolved.unwrap_or_default().into_iter().chain(caller.unwrap_or_default()).collect()),
			};
			// Caller transforms apply once to the fully assembled headers and
			// are not forwarded to the provider (`ModelsRequestTransforms`).
			if let Some(transform_headers) = simple_options.transforms.transform_headers.clone() {
				simple_options.base.base.headers = transform_headers(simple_options.base.base.headers.take()).await;
			}
			let mut request_model = model.clone();
			if let AnyModel::Chat(prepared_model) = &prepared.model {
				if let Some(base_url) = &prepared.base_url {
					request_model.base.base_url = base_url.clone();
				}
			}
			Ok(prepared.provider.stream_simple(&request_model, &transcript, &simple_options.base))
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
		let model = model.clone();
		let handle = handle.clone();
		let options = options.unwrap_or_default();
		let this = match ModelRuntime::self_arc(self) {
			Some(this) => this,
			None => {
				return pi_ai::api::lazy::lazy_stream(&model, async {
					Err(ThrownValue::message("ModelRuntime must be Arc-rooted to stream"))
				})
			}
		};
		let outer_model = model.clone();
		pi_ai::api::lazy::lazy_stream(&outer_model, async move {
			let prepared = this
				.prepare_request(
					AnyModel::Chat(model.clone()),
					options.base.base.api_key.clone(),
					options.base.base.headers.clone(),
					options.base.base.env.clone(),
					options.base.base.signal.clone(),
				)
				.await?;
			let request_model = match &prepared.model {
				AnyModel::Chat(model) => {
					let mut model = model.clone();
					if let Some(base_url) = &prepared.base_url {
						model.base.base_url = base_url.clone();
					}
					model
				}
				_ => model.clone(),
			};
			match prepared.provider.fetch_deferred(&request_model, &handle, &options.base) {
				Some(stream) => Ok(stream),
				None => Err(ThrownValue::Error(Arc::new(ModelsError::new(
					ModelsErrorCode::Provider,
					format!("Provider {} does not support deferred responses", request_model.provider()),
				)))),
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
		let options = options.unwrap_or_default();
		let prepared = self
			.prepare_request(
				AnyModel::Chat(model.clone()),
				options.base.base.api_key.clone(),
				options.base.base.headers.clone(),
				options.base.base.env.clone(),
				options.base.base.signal.clone(),
			)
			.await?;
		let future = prepared.provider.cancel_deferred(model, handle, &options.base);
		match future {
			Some(future) => future.await,
			None => Err(ThrownValue::Error(Arc::new(ModelsError::new(
				ModelsErrorCode::Provider,
				format!("Provider {} does not support deferred responses", model.provider()),
			)))),
		}
	}

	async fn generate_images(
		&self,
		model: &pi_ai::types::ImageModel,
		context: &pi_ai::types::ImagesContext,
		options: Option<ModelsImagesOptions>,
	) -> pi_ai::types::AssistantImages {
		let options = options.unwrap_or_default();
		let aborted = options.base.base.signal.as_ref().map(|signal| signal.aborted()).unwrap_or(false);
		let prepared = self
			.prepare_request(
				AnyModel::Image(model.clone()),
				options.base.base.api_key.clone(),
				options.base.base.headers.clone(),
				options.base.base.env.clone(),
				options.base.base.signal.clone(),
			)
			.await;
		let prepared = match prepared {
			Ok(prepared) => prepared,
			Err(error) => return pi_ai::utils::model_operations::image_error_result(model, &error, aborted),
		};
		let AnyModel::Image(request_model) = &prepared.model else {
			return pi_ai::utils::model_operations::image_error_result(model, &ThrownValue::message("model is not an image model"), aborted);
		};
		let mut request_model = request_model.clone();
		if let Some(base_url) = &prepared.base_url {
			request_model.base.base_url = base_url.clone();
		}
		let future = prepared.provider.generate_images(&request_model, context, &options.base);
		match future {
			Some(future) => future.await,
			None => pi_ai::utils::model_operations::image_error_result(
				model,
				&ThrownValue::message(format!("Provider {} does not support image generation", model.provider())),
				aborted,
			),
		}
	}

	async fn classify(
		&self,
		model: &pi_ai::types::ClassifierModel,
		context: &pi_ai::types::ClassifierContext,
		options: Option<ModelsClassifierOptions>,
	) -> ClassifierResult {
		let options = options.unwrap_or_default();
		let aborted = options.base.base.signal.as_ref().map(|signal| signal.aborted()).unwrap_or(false);
		let prepared = self
			.prepare_request(
				AnyModel::Classifier(model.clone()),
				options.base.base.api_key.clone(),
				options.base.base.headers.clone(),
				options.base.base.env.clone(),
				options.base.base.signal.clone(),
			)
			.await;
		let prepared = match prepared {
			Ok(prepared) => prepared,
			Err(error) => return pi_ai::utils::model_operations::classifier_error_result(model, &error, aborted),
		};
		let AnyModel::Classifier(request_model) = &prepared.model else {
			return pi_ai::utils::model_operations::classifier_error_result(model, &ThrownValue::message("model is not a classifier model"), aborted);
		};
		let mut request_model = request_model.clone();
		if let Some(base_url) = &prepared.base_url {
			request_model.base.base_url = base_url.clone();
		}
		let future = prepared.provider.classify(&request_model, context, &options.base);
		match future {
			Some(future) => future.await,
			None => pi_ai::utils::model_operations::classifier_error_result(
				model,
				&ThrownValue::message(format!("Provider {} does not support classification", model.provider())),
				aborted,
			),
		}
	}
}

/// The read surface consumed by `core/model_resolver` (seam trait, crate
/// docs D47) — implemented by the real runtime now.
impl crate::core::model_resolver::ModelRuntimeSource for ModelRuntime {
	fn get_models(&self) -> Vec<Model> {
		Models::get_models(self, None)
	}

	fn get_available_snapshot(&self) -> Vec<Model> {
		ModelRuntime::get_available_snapshot(self)
	}

	fn get_available<'a>(
		&'a self,
		provider: Option<&'a str>,
		options: Option<pi_ai::auth::types::AuthOperationOptions>,
	) -> BoxFuture<'a, Result<Vec<Model>, ThrownValue>> {
		Box::pin(async move { Models::get_available(self, provider, options).await })
	}

	fn get_model(&self, provider: &str, model_id: &str) -> Option<Model> {
		self.models.get_model(provider, model_id)
	}

	fn has_configured_auth(&self, provider: &str) -> bool {
		ModelRuntime::has_configured_auth(self, provider)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn merge_headers_is_case_insensitive_and_none_preserving() {
		let mut base = ProviderHeaders::new();
		base.insert("X-Api-Key".to_string(), Some("a".to_string()));
		base.insert("X-Keep".to_string(), Some("keep".to_string()));
		let mut override_headers = ProviderHeaders::new();
		override_headers.insert("x-api-key".to_string(), Some("b".to_string()));
		override_headers.insert("X-Drop".to_string(), None);
		let merged = merge_headers(Some(base), Some(override_headers)).unwrap();
		assert_eq!(merged.get("X-Api-Key"), None);
		assert_eq!(merged.get("x-api-key"), Some(&Some("b".to_string())));
		assert_eq!(merged.get("X-Keep"), Some(&Some("keep".to_string())));
		assert_eq!(merged.get("X-Drop"), Some(&None));
		assert!(merge_headers(None, None).is_none());
	}

	#[test]
	fn merge_provider_config_preserves_undefined() {
		let mut previous = ProviderConfigInput::default();
		previous.name = Some("old".to_string());
		previous.api_key = Some("key".to_string());
		let config = ProviderConfigInput {
			name: Some("new".to_string()),
			..Default::default()
		};
		let merged = merge_provider_config(previous, config);
		assert_eq!(merged.name.as_deref(), Some("new"));
		assert_eq!(merged.api_key.as_deref(), Some("key"));
	}

	#[test]
	fn runtime_credentials_override_read_and_list() {
		let inner: Arc<dyn CredentialStore> = Arc::new(pi_ai::auth::credential_store::InMemoryCredentialStore::new());
		let runtime = RuntimeCredentials::new(inner);
		assert!(!runtime.has_runtime_api_key("p1"));
		runtime.set_runtime_api_key("p1", "sk-test");
		assert!(runtime.has_runtime_api_key("p1"));
		let read = tokio::runtime::Runtime::new().unwrap().block_on(runtime.read("p1", None)).unwrap();
		match read {
			Some(Credential::ApiKey(key)) => assert_eq!(key.key.as_deref(), Some("sk-test")),
			other => panic!("expected api key credential, got {other:?}"),
		}
		let listed = tokio::runtime::Runtime::new().unwrap().block_on(runtime.list(None)).unwrap();
		assert_eq!(listed.len(), 1);
		assert_eq!(listed[0].provider_id, "p1");
		assert_eq!(listed[0].credential_type, "api_key");
		runtime.remove_runtime_api_key("p1");
		assert!(!runtime.has_runtime_api_key("p1"));
		let read = tokio::runtime::Runtime::new().unwrap().block_on(runtime.read("p1", None)).unwrap();
		assert!(read.is_none());
	}
}

// ---------------------------------------------------------------------------
// Ported upstream suites:
//   packages/coding-agent/test/model-runtime-auth-options.test.ts
//   packages/coding-agent/test/model-runtime-credential-sync.test.ts
// (1:1 ports; JS-specific seams are noted per test.)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod ported_tests {
	use super::*;
	use futures_util::future::BoxFuture;
	use std::future::Future;
	use std::pin::Pin;
	use pi_ai::auth::credential_store::InMemoryCredentialStore;
	use pi_ai::auth::types::{
		ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthCheck, AuthInteraction, AuthOperationOptions, AuthPrompt,
		AuthResult, AuthType, Credential, CredentialInfo, CredentialStore, CredentialModifyFn, ModelAuth, OAuthCredential,
		OAuthCredentials, ProviderAuth, ProviderAuthInteraction,
	};
	use pi_ai::models::{Models as _, Provider as _, RefreshModelsContext};
	use pi_ai::types::{BaseModelFields, ModelCost, ModelCostRates, ModelInput};
	use pi_ai::utils::abort::AbortController;
	use crate::core::extensions::types::{
		ProviderChatModelConfig, ProviderModelConfig, ProviderModelConfigBase, ProviderOAuthConfig,
	};

	fn now_ms() -> i64 {
		std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.unwrap()
			.as_millis() as i64
	}

	fn oauth_credential(access: &str, refresh: &str, expires: i64) -> Credential {
		Credential::OAuth(OAuthCredential {
			credentials: OAuthCredentials {
				access: access.to_string(),
				refresh: refresh.to_string(),
				expires,
				extra: Default::default(),
			},
		})
	}

	fn preload_modify_fn(
		credential: Credential,
	) -> Box<dyn FnOnce(Option<&Credential>) -> BoxFuture<'_, Result<Option<Credential>, ThrownValue>> + Send> {
		Box::new(move |_| Box::pin(async move { Ok(Some(credential)) }))
	}

	async fn preload(store: &InMemoryCredentialStore, provider_id: &str, credential: Credential) {
		store.modify(provider_id, preload_modify_fn(credential), None).await.unwrap();
	}

	fn create_options(credentials: Arc<dyn CredentialStore>) -> CreateModelRuntimeOptions {
		CreateModelRuntimeOptions {
			credentials: Some(credentials),
			models_path: Some(None),
			// Upstream `create()` refreshes unless told otherwise; the network
			// stays off because `allow_model_network` defaults to false.
			refresh_on_create: true,
			..Default::default()
		}
	}

	async fn create_runtime(credentials: Arc<dyn CredentialStore>) -> Arc<ModelRuntime> {
		ModelRuntime::create(create_options(credentials)).await.unwrap()
	}

	fn in_memory() -> Arc<InMemoryCredentialStore> {
		Arc::new(InMemoryCredentialStore::new())
	}

	fn interaction() -> AuthInteraction {
		AuthInteraction {
			signal: None,
			prompt: Box::new(|_prompt: AuthPrompt| Box::pin(async { Ok("unused".to_string()) })),
			notify: Box::new(|_| {}),
		}
	}

	fn interaction_with(signal: pi_ai::utils::abort::AbortSignal) -> AuthInteraction {
		AuthInteraction { signal: Some(signal), ..interaction() }
	}

	fn thrown_text(error: &ThrownValue) -> String {
		thrown_message(error)
	}

	fn assert_sync_error(error: ThrownValue, provider_id: &str) {
		// Port deviation (D63c struct shape): the port's
		// `CredentialSynchronizationError` carries `provider_id`/`operation`/
		// `cause`; the upstream `credential` field has no port seam, so the
		// committed credential is asserted through the credential store.
		match &error {
			ThrownValue::Error(shared) if shared.is::<CredentialSynchronizationError>() => {
				let sync = shared.downcast_ref::<CredentialSynchronizationError>().unwrap();
				assert_eq!(sync.provider_id, provider_id);
				assert_eq!(sync.operation, CredentialSynchronizationOperation::Login);
			}
			other => panic!("expected CredentialSynchronizationError, got {other:?}"),
		}
	}

	fn credential_api_key(credential: &Option<Credential>) -> Option<String> {
		match credential {
			Some(Credential::ApiKey(key)) => key.key.clone(),
			_ => None,
		}
	}

	// ----- auth options projection --------------------------------------

	pub struct AuthOptionSummary {
		pub provider_id: String,
		pub provider_name: String,
		pub auth_type: AuthType,
		pub method_name: String,
		pub is_subscription: bool,
		pub has_login: bool,
	}

	fn auth_options(runtime: &ModelRuntime, type_filter: Option<AuthType>) -> Vec<AuthOptionSummary> {
		let mut out = Vec::new();
		for provider in Models::get_providers(runtime) {
			let auth = provider.auth();
			if type_filter.is_none() || type_filter == Some(AuthType::OAuth) {
				if let Some(oauth) = &auth.oauth {
					out.push(AuthOptionSummary {
						provider_id: provider.id().to_string(),
						provider_name: provider.name().to_string(),
						auth_type: AuthType::OAuth,
						method_name: oauth.name.clone(),
						is_subscription: oauth.is_subscription,
						has_login: true,
					});
				}
			}
			if type_filter.is_none() || type_filter == Some(AuthType::ApiKey) {
				if let Some(api_key) = &auth.api_key {
					out.push(AuthOptionSummary {
						provider_id: provider.id().to_string(),
						provider_name: provider.name().to_string(),
						auth_type: AuthType::ApiKey,
						method_name: api_key.name.clone(),
						is_subscription: false,
						has_login: api_key.login.is_some(),
					});
				}
			}
		}
		out
	}

	fn find_option<'a>(options: &'a [AuthOptionSummary], provider_id: &str, auth_type: AuthType) -> &'a AuthOptionSummary {
		options
			.iter()
			.find(|option| option.provider_id == provider_id && option.auth_type == auth_type)
			.unwrap_or_else(|| panic!("no {auth_type:?} option for {provider_id}"))
	}

	fn chat_model_config(id: &str) -> ProviderModelConfig {
		ProviderModelConfig::Chat(ProviderChatModelConfig {
			base: ProviderModelConfigBase {
				id: id.to_string(),
				name: id.to_string(),
				input: vec![ModelInput::Text],
				..Default::default()
			},
			model_type: None,
			api: Some("openai-completions".to_string()),
			reasoning: false,
			thinking_level_map: None,
			prompt_cache: None,
			context_window: 10_000,
			max_tokens: 1_000,
			sampling_params: None,
			compat: None,
		})
	}

	// ----- native test provider (upstream `provider()` helper) -----------

	type LoginHook =
		Arc<dyn Fn(ProviderAuthInteraction) -> BoxFuture<'static, Result<ApiKeyCredential, ThrownValue>> + Send + Sync>;
	type ApiCheckHook =
		Arc<dyn Fn(Option<ApiKeyCredential>) -> BoxFuture<'static, Result<Option<AuthCheck>, ThrownValue>> + Send + Sync>;
	type RefreshHook = Arc<dyn Fn(RefreshModelsContext) -> BoxFuture<'static, Result<(), ThrownValue>> + Send + Sync>;

	#[derive(Clone, Default)]
	struct TestProviderHooks {
		login: Option<LoginHook>,
		check: Option<ApiCheckHook>,
		refresh_models: Option<RefreshHook>,
	}

	struct TestProvider {
		id: String,
		name: String,
		auth: ProviderAuth,
		models: Vec<Model>,
		refresh_models: Option<RefreshHook>,
	}

	fn default_stored_check(
	) -> Arc<dyn Fn(Option<ApiKeyCredential>) -> BoxFuture<'static, Result<Option<AuthCheck>, ThrownValue>> + Send + Sync>
	{
		Arc::new(|credential| {
			Box::pin(async move {
				Ok(credential.map(|_| AuthCheck {
					source: Some("stored".to_string()),
					check_type: AuthType::ApiKey,
				}))
			})
		})
	}

	fn native_api_key_provider(id: &str, hooks: TestProviderHooks) -> Arc<TestProvider> {
		let check = hooks.check.clone().unwrap_or_else(default_stored_check);
		let login = hooks.login.clone().unwrap_or_else(|| default_login(id));
		let resolve: Arc<
			dyn Fn(Option<ApiKeyCredential>) -> BoxFuture<'static, Result<Option<AuthResult>, ThrownValue>> + Send + Sync,
		> = Arc::new(|credential| {
			Box::pin(async move {
				Ok(credential.map(|credential| AuthResult {
					auth: ModelAuth { api_key: credential.key.clone(), headers: None, base_url: None },
					env: None,
					source: Some("stored".to_string()),
				}))
			})
		});
		let auth = ProviderAuth {
			api_key: Some(ApiKeyAuth {
				name: "API key".to_string(),
				login: Some(Box::new(move |interaction: ProviderAuthInteraction| login(interaction)) as pi_ai::auth::types::ApiLoginFn),
				check: Some(Box::new(move |input: ApiKeyAuthInput<'_>| {
					let credential = input.credential.cloned();
					let check = check.clone();
					Box::pin(async move { check(credential).await })
				})),
				resolve: Box::new(move |input: ApiKeyAuthInput<'_>| {
					let credential = input.credential.cloned();
					let resolve = resolve.clone();
					Box::pin(async move { resolve(credential).await })
				}),
			}),
			oauth: None,
		};
		Arc::new(TestProvider {
			id: id.to_string(),
			name: id.to_string(),
			auth,
			models: vec![Model {
				base: BaseModelFields {
					id: "dynamic".to_string(),
					name: "Dynamic".to_string(),
					api: "openai-completions".to_string(),
					provider: id.to_string(),
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
				context_window: 1_000,
				max_tokens: 100,
				sampling_params: None,
				sampling_params_by_thinking_level: None,
				compat: None,
			}],
			refresh_models: hooks.refresh_models,
		})
	}

	impl pi_ai::models::Provider for TestProvider {
		fn id(&self) -> &str {
			&self.id
		}

		fn name(&self) -> &str {
			&self.name
		}

		fn auth(&self) -> &ProviderAuth {
			&self.auth
		}

		fn get_models(&self) -> Vec<Model> {
			self.models.clone()
		}

		fn refresh_models(
			&self,
			context: RefreshModelsContext,
		) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
			let hook = self.refresh_models.clone()?;
			Some(Box::pin(async move { hook(context).await }))
		}

		fn stream(
			&self,
			model: &Model,
			_context: &pi_ai::types::TranscriptContext,
			_options: &pi_ai::types::StreamOptions,
		) -> pi_ai::types::AssistantMessageEventStream {
			pi_ai::api::lazy::lazy_stream(model, async { Err(ThrownValue::message("unused")) })
		}

		fn stream_simple(
			&self,
			model: &Model,
			_context: &pi_ai::types::TranscriptContext,
			_options: &pi_ai::types::SimpleStreamOptions,
		) -> pi_ai::types::AssistantMessageEventStream {
			pi_ai::api::lazy::lazy_stream(model, async { Err(ThrownValue::message("unused")) })
		}
	}

	async fn runtime_with_provider(provider: Arc<dyn pi_ai::models::Provider>) -> Arc<ModelRuntime> {
		runtime_with_credentials(in_memory() as Arc<dyn CredentialStore>, provider).await
	}

	async fn runtime_with_credentials(
		credentials: Arc<dyn CredentialStore>,
		provider: Arc<dyn pi_ai::models::Provider>,
	) -> Arc<ModelRuntime> {
		let runtime = create_runtime(credentials).await;
		runtime.register_native_provider(provider.clone()).unwrap();
		let id = provider.id().to_string();
		Models::refresh(
			&*runtime,
			Some(ModelsRefreshOptions {
				allow_network: Some(false),
				providers: Some(vec![id]),
				..Default::default()
			}),
		)
		.await;
		runtime
	}

	async fn read_stored(store: &dyn CredentialStore, provider_id: &str) -> Option<Credential> {
		store.read(provider_id, None).await.unwrap()
	}

	// =====================================================================
	// model-runtime-auth-options.test.ts
	// =====================================================================

	#[tokio::test]
	async fn accepts_a_pi_ai_credential_store() {
		let credentials = in_memory();
		preload(&credentials, "anthropic", Credential::ApiKey(ApiKeyCredential { key: Some("stored-key".to_string()), env: None })).await;
		let runtime = create_runtime(credentials as Arc<dyn CredentialStore>).await;

		let auth = runtime.get_auth_for_provider("anthropic", None).await.unwrap().expect("auth resolved");
		assert_eq!(auth.auth.api_key.as_deref(), Some("stored-key"));
	}

	#[tokio::test]
	async fn scopes_provider_availability_reads_and_records_refresh_failures() {
		struct RecordingStore {
			base: InMemoryCredentialStore,
			reads: parking_lot::Mutex<Vec<String>>,
			fail: std::sync::atomic::AtomicBool,
		}
		#[async_trait::async_trait]
		impl CredentialStore for RecordingStore {
			async fn read(
				&self,
				provider_id: &str,
				_options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				self.reads.lock().push(provider_id.to_string());
				if self.fail.load(std::sync::atomic::Ordering::SeqCst) {
					return Err(ThrownValue::message(format!("read failed for {provider_id}")));
				}
				self.base.read(provider_id, None).await
			}

			async fn list(&self, _options: Option<AuthOperationOptions>) -> Result<Vec<CredentialInfo>, ThrownValue> {
				self.base.list(None).await
			}

			async fn modify(
				&self,
				provider_id: &str,
				modify_fn: CredentialModifyFn,
				_options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				self.base.modify(provider_id, modify_fn, None).await
			}

			async fn delete(&self, provider_id: &str, _options: Option<AuthOperationOptions>) -> Result<(), ThrownValue> {
				self.base.delete(provider_id, None).await
			}
		}

		let store = Arc::new(RecordingStore {
			base: InMemoryCredentialStore::new(),
			reads: parking_lot::Mutex::new(Vec::new()),
			fail: std::sync::atomic::AtomicBool::new(false),
		});
		let runtime = create_runtime(store.clone() as Arc<dyn CredentialStore>).await;

		store.reads.lock().clear();
		Models::get_available(&*runtime, Some("anthropic"), None).await.unwrap();
		let reads: Vec<String> = store.reads.lock().clone();
		let unique: std::collections::BTreeSet<String> = reads.into_iter().collect();
		assert_eq!(unique, std::collections::BTreeSet::from(["anthropic".to_string()]));

		store.fail.store(true, std::sync::atomic::Ordering::SeqCst);
		let error = Models::get_available(&*runtime, Some("anthropic"), None).await.unwrap_err();
		assert!(
			thrown_text(&error).contains("Credential store read failed for anthropic"),
			"{}",
			thrown_text(&error)
		);
		assert!(
			runtime.get_error().unwrap().contains("Availability refresh: Credential store read failed for anthropic"),
			"{:?}",
			runtime.get_error()
		);

		store.fail.store(false, std::sync::atomic::Ordering::SeqCst);
		Models::get_available(&*runtime, None, None).await.unwrap();
		assert_eq!(runtime.get_error(), None);
	}

	#[tokio::test]
	async fn projects_provider_owned_methods_names_and_status() {
		let runtime = create_runtime(in_memory() as Arc<dyn CredentialStore>).await;
		let options = auth_options(&runtime, None);

		let bedrock = find_option(&options, "amazon-bedrock", AuthType::ApiKey);
		assert_eq!(bedrock.provider_name, "Amazon Bedrock");
		assert_eq!(bedrock.method_name, "AWS credentials or bearer token");
		let vertex = find_option(&options, "google-vertex", AuthType::ApiKey);
		assert_eq!(vertex.provider_name, "Google Vertex AI");
		assert_eq!(vertex.method_name, "Google Cloud credentials");
		find_option(&options, "anthropic", AuthType::OAuth);
		find_option(&options, "cloudflare-ai-gateway", AuthType::ApiKey);
		find_option(&options, "cloudflare-workers-ai", AuthType::ApiKey);

		assert!(auth_options(&runtime, Some(AuthType::ApiKey)).iter().all(|option| option.auth_type == AuthType::ApiKey));
		assert!(auth_options(&runtime, Some(AuthType::OAuth)).iter().all(|option| option.auth_type == AuthType::OAuth));
		assert!(!options.iter().any(|option| option.provider_id == "openai-codex" && option.auth_type == AuthType::ApiKey));
	}

	#[tokio::test]
	async fn attaches_the_providers_active_auth_status_to_every_method_option() {
		let credentials = in_memory();
		preload(&credentials, "anthropic", oauth_credential("access", "refresh", now_ms() + 60_000)).await;
		let runtime = create_runtime(credentials as Arc<dyn CredentialStore>).await;

		let options: Vec<AuthOptionSummary> =
			auth_options(&runtime, None).into_iter().filter(|option| option.provider_id == "anthropic").collect();
		assert_eq!(options.len(), 2);
		let check = Models::check_auth(&*runtime, "anthropic", None).await.unwrap().expect("auth check");
		assert_eq!(check.check_type, AuthType::OAuth);
	}

	#[tokio::test]
	async fn distinguishes_subscription_oauth_from_generic_oauth_sign_in() {
		let credentials = in_memory();
		preload(&credentials, "anthropic", oauth_credential("anthropic-access", "anthropic-refresh", now_ms() + 60 * 60_000)).await;
		preload(&credentials, "openrouter", oauth_credential("openrouter-key", "", i64::MAX)).await;
		preload(&credentials, "radius", oauth_credential("radius-access", "radius-refresh", now_ms() + 60 * 60_000)).await;
		let runtime = create_runtime(credentials as Arc<dyn CredentialStore>).await;

		assert!(runtime.is_using_oauth("anthropic"));
		assert!(runtime.is_using_subscription("anthropic"));
		assert!(runtime.is_using_oauth("openrouter"));
		assert!(!runtime.is_using_subscription("openrouter"));
		assert!(runtime.is_using_oauth("radius"));
		assert!(!runtime.is_using_subscription("radius"));
	}

	#[tokio::test]
	async fn constructs_an_api_key_method_for_an_extension_api_key_provider() {
		let runtime = create_runtime(in_memory() as Arc<dyn CredentialStore>).await;
		runtime
			.register_provider(
				"extension-api-key",
				ProviderConfigInput {
					name: Some("Extension API Key".to_string()),
					base_url: Some("https://example.test/v1".to_string()),
					api_key: Some("$EXTENSION_TEST_API_KEY".to_string()),
					api: Some("openai-completions".to_string()),
					models: Some(vec![chat_model_config("extension-model")]),
					..Default::default()
				},
			)
			.unwrap();

		let options: Vec<AuthOptionSummary> =
			auth_options(&runtime, None).into_iter().filter(|option| option.provider_id == "extension-api-key").collect();
		assert_eq!(options.len(), 1);
		assert_eq!(options[0].auth_type, AuthType::ApiKey);
		assert_eq!(options[0].provider_name, "Extension API Key");
		assert_eq!(options[0].method_name, "API key");
		assert!(options[0].has_login);
	}

	#[tokio::test]
	async fn resolves_configured_auth_from_request_scoped_environment_overrides() {
		let runtime = create_runtime(in_memory() as Arc<dyn CredentialStore>).await;
		runtime
			.register_provider(
				"request-env-provider",
				ProviderConfigInput {
					base_url: Some("https://example.test/v1".to_string()),
					api_key: Some("$REQUEST_SCOPED_API_KEY".to_string()),
					headers: Some({
						let mut headers = std::collections::BTreeMap::new();
						headers.insert("x-request-value".to_string(), "$REQUEST_SCOPED_HEADER".to_string());
						headers
					}),
					api: Some("openai-completions".to_string()),
					models: Some(vec![chat_model_config("request-env-model")]),
					..Default::default()
				},
			)
			.unwrap();

		let auth = runtime
			.get_auth_for_provider(
				"request-env-provider",
				Some(ModelRuntimeAuthOverrides {
					env: Some(ProviderEnv::from([
						("REQUEST_SCOPED_API_KEY".to_string(), "request-key".to_string()),
						("REQUEST_SCOPED_HEADER".to_string(), "request-header".to_string()),
					])),
					..Default::default()
				}),
			)
			.await
			.unwrap()
			.expect("auth resolved");

		assert_eq!(auth.auth.api_key.as_deref(), Some("request-key"));
		let headers = auth.auth.headers.expect("headers");
		assert_eq!(headers.get("x-request-value"), Some(&Some("request-header".to_string())));
	}

	// Upstream drives this through `completeSimple` with a throwing
	// `streamSimple` that captures the dispatched options. The port uses the
	// same seam: `ProviderConfig.streamSimple` receives the fully prepared
	// `SimpleStreamOptions` and records `options.headers`.
	#[tokio::test]
	async fn lets_an_explicit_authorization_header_override_authheader_case_insensitively() {
		let runtime = create_runtime(in_memory() as Arc<dyn CredentialStore>).await;
		let captured: Arc<parking_lot::Mutex<Option<ProviderHeaders>>> = Arc::default();
		let captured_sink = captured.clone();
		runtime
			.register_provider(
				"auth-header-provider",
				ProviderConfigInput {
					base_url: Some("https://example.test/v1".to_string()),
					api_key: Some("generated-key".to_string()),
					auth_header: Some(true),
					api: Some("openai-completions".to_string()),
					stream_simple: Some(Arc::new(move |_model, _context, options| {
						*captured_sink.lock() = options.as_ref().and_then(|options| options.base.base.headers.clone());
						pi_ai::api::lazy::lazy_stream(&_model, async { Err(ThrownValue::message("captured")) })
					})),
					models: Some(vec![chat_model_config("auth-header-model")]),
					..Default::default()
				},
			)
			.unwrap();
		let model = Models::get_model(&*runtime, "auth-header-provider", "auth-header-model").expect("model");

		let _ = Models::complete_simple(
			&*runtime,
			&model,
			&Context::default(),
			Some(pi_ai::models::ModelsSimpleStreamOptions {
				base: SimpleStreamOptions {
					base: StreamOptions {
						base: pi_ai::types::ProviderRequestOptions {
							headers: Some(ProviderHeaders::from([(
								"authorization".to_string(),
								Some("Explicit token".to_string()),
							)])),
							..Default::default()
						},
						..Default::default()
					},
					..Default::default()
				},
				transforms: Default::default(),
			}),
		)
		.await;

		let headers = captured.lock().clone().expect("streamSimple invoked");
		assert_eq!(headers.len(), 1);
		assert_eq!(headers.get("authorization"), Some(&Some("Explicit token".to_string())));
	}

	#[tokio::test]
	async fn transforms_fully_assembled_headers_once_without_forwarding_the_transform() {
		let runtime = create_runtime(in_memory() as Arc<dyn CredentialStore>).await;
		let captured: Arc<parking_lot::Mutex<Option<ProviderHeaders>>> = Arc::default();
		let transforms = Arc::new(std::sync::atomic::AtomicUsize::new(0));
		let captured_sink = captured.clone();
		runtime
			.register_provider(
				"header-provider",
				ProviderConfigInput {
					base_url: Some("https://example.test/v1".to_string()),
					api_key: Some("generated-key".to_string()),
					auth_header: Some(true),
					headers: Some({
						let mut headers = std::collections::BTreeMap::new();
						headers.insert("x-provider".to_string(), "provider".to_string());
						headers
					}),
					api: Some("openai-completions".to_string()),
					stream_simple: Some(Arc::new(move |_model, _context, options| {
						*captured_sink.lock() = options.as_ref().and_then(|options| options.base.base.headers.clone());
						pi_ai::api::lazy::lazy_stream(&_model, async { Err(ThrownValue::message("captured")) })
					})),
					models: Some(vec![ProviderModelConfig::Chat(ProviderChatModelConfig {
						base: ProviderModelConfigBase {
							id: "header-model".to_string(),
							name: "header-model".to_string(),
							input: vec![ModelInput::Text],
							headers: Some({
								let mut headers = std::collections::BTreeMap::new();
								headers.insert("x-model".to_string(), "model".to_string());
								headers
							}),
							..Default::default()
						},
						model_type: None,
						api: Some("openai-completions".to_string()),
						reasoning: false,
						thinking_level_map: None,
						prompt_cache: None,
						context_window: 10_000,
						max_tokens: 1_000,
						sampling_params: None,
						compat: None,
					})]),
					..Default::default()
				},
			)
			.unwrap();
		let model = Models::get_model(&*runtime, "header-provider", "header-model").expect("model");

		let transform_counter = transforms.clone();
		let _ = Models::complete_simple(
			&*runtime,
			&model,
			&Context::default(),
			Some(pi_ai::models::ModelsSimpleStreamOptions {
				base: SimpleStreamOptions {
					base: StreamOptions {
						base: pi_ai::types::ProviderRequestOptions {
							headers: Some(ProviderHeaders::from([(
								"x-explicit".to_string(),
								Some("explicit".to_string()),
							)])),
							..Default::default()
						},
						..Default::default()
					},
					..Default::default()
				},
				transforms: pi_ai::models::ModelsRequestTransforms {
					transform_headers: Some(Arc::new(move |headers: Option<ProviderHeaders>| {
						let counter = transform_counter.clone();
						Box::pin(async move {
							counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
							let headers = headers.expect("assembled headers");
							assert_eq!(headers.get("Authorization"), Some(&Some("Bearer generated-key".to_string())));
							assert_eq!(headers.get("x-provider"), Some(&Some("provider".to_string())));
							assert_eq!(headers.get("x-model"), Some(&Some("model".to_string())));
							assert_eq!(headers.get("x-explicit"), Some(&Some("explicit".to_string())));
							let mut transformed = headers;
							transformed.insert("x-transformed".to_string(), Some("yes".to_string()));
							Some(transformed)
						})
					})),
				},
			}),
		)
		.await;

		assert_eq!(transforms.load(std::sync::atomic::Ordering::SeqCst), 1);
		let headers = captured.lock().clone().expect("streamSimple invoked");
		assert_eq!(headers.get("Authorization"), Some(&Some("Bearer generated-key".to_string())));
		assert_eq!(headers.get("x-provider"), Some(&Some("provider".to_string())));
		assert_eq!(headers.get("x-model"), Some(&Some("model".to_string())));
		assert_eq!(headers.get("x-explicit"), Some(&Some("explicit".to_string())));
		assert_eq!(headers.get("x-transformed"), Some(&Some("yes".to_string())));
	}

	#[tokio::test]
	async fn does_not_cancel_an_extension_oauth_refresh_with_the_request() {
		let credentials = in_memory();
		preload(&credentials, "extension-oauth", oauth_credential("expired", "refresh", 0)).await;
		let runtime = create_runtime(credentials as Arc<dyn CredentialStore>).await;
		let refresh_signal: Arc<parking_lot::Mutex<Option<crate::utils::abort::AbortSignal>>> = Arc::default();
		let signal_sink = refresh_signal.clone();
		runtime
			.register_provider(
				"extension-oauth",
				ProviderConfigInput {
					name: Some("Extension OAuth".to_string()),
					base_url: Some("https://example.test/v1".to_string()),
					api: Some("openai-completions".to_string()),
					oauth: Some(ProviderOAuthConfig {
						name: "Extension subscription".to_string(),
						is_subscription: None,
						uses_callback_server: None,
						login: Arc::new(|_callbacks| {
							Box::pin(async {
								Ok(OAuthCredentials {
									access: "access".to_string(),
									refresh: "refresh".to_string(),
									expires: now_ms() + 60_000,
									extra: Default::default(),
								})
							})
						}),
						refresh_token: Arc::new(move |credential, signal| {
							*signal_sink.lock() = Some(signal);
							Box::pin(async move {
								Ok(OAuthCredentials { expires: now_ms() + 60_000, ..credential })
							})
						}),
						get_api_key: Arc::new(|credential| credential.access.clone()),
						modify_models: None,
					}),
					models: Some(vec![chat_model_config("extension-model")]),
					..Default::default()
				},
			)
			.unwrap();

		let controller = AbortController::new();
		runtime
			.get_auth_for_provider(
				"extension-oauth",
				Some(ModelRuntimeAuthOverrides { signal: Some(controller.signal()), ..Default::default() }),
			)
			.await
			.unwrap();
		let signal = refresh_signal.lock().clone().expect("refresh started with a signal");
		controller.abort();
		// A started refresh may already have rotated the refresh token and
		// must not be cancelled with the request.
		assert!(!signal.aborted());
	}

	#[tokio::test]
	async fn does_not_fabricate_an_api_key_method_for_an_extension_oauth_only_provider() {
		let runtime = create_runtime(in_memory() as Arc<dyn CredentialStore>).await;
		runtime
			.register_provider(
				"extension-oauth",
				ProviderConfigInput {
					name: Some("Extension OAuth".to_string()),
					base_url: Some("https://example.test/v1".to_string()),
					api: Some("openai-completions".to_string()),
					oauth: Some(ProviderOAuthConfig {
						name: "Extension subscription".to_string(),
						is_subscription: Some(true),
						uses_callback_server: None,
						login: Arc::new(|_callbacks| {
							Box::pin(async {
								Ok(OAuthCredentials {
									access: "access".to_string(),
									refresh: "refresh".to_string(),
									expires: now_ms() + 60_000,
									extra: Default::default(),
								})
							})
						}),
						refresh_token: Arc::new(|credential, _signal| Box::pin(async move { Ok(credential) })),
						get_api_key: Arc::new(|credential| credential.access.clone()),
						modify_models: None,
					}),
					models: Some(vec![chat_model_config("extension-model")]),
					..Default::default()
				},
			)
			.unwrap();

		let options: Vec<AuthOptionSummary> =
			auth_options(&runtime, None).into_iter().filter(|option| option.provider_id == "extension-oauth").collect();
		assert_eq!(options.len(), 1);
		assert_eq!(options[0].auth_type, AuthType::OAuth);
		assert_eq!(options[0].provider_name, "Extension OAuth");
		assert_eq!(options[0].method_name, "Extension subscription");
		assert!(options[0].is_subscription);
	}

	// =====================================================================
	// model-runtime-credential-sync.test.ts
	// =====================================================================

	fn default_login(id: &str) -> LoginHook {
		let key = format!("{id}-key");
		Arc::new(move |_interaction| {
			let key = key.clone();
			Box::pin(async move { Ok(ApiKeyCredential { key: Some(key), env: None }) })
		})
	}

	/// Channel used to observe hook start; the hook then blocks until the
	/// paired [`Gate`] is opened (upstream promise gymnastics).
	fn gate() -> (tokio::sync::mpsc::UnboundedSender<()>, tokio::sync::mpsc::UnboundedReceiver<()>, Arc<tokio::sync::Notify>) {
		let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
		(tx, rx, Arc::new(tokio::sync::Notify::new()))
	}

	fn blocking_login(tx: tokio::sync::mpsc::UnboundedSender<()>, released: Arc<tokio::sync::Notify>, key: &str) -> LoginHook {
		let key = key.to_string();
		Arc::new(move |_interaction| {
			let _ = tx.send(());
			let released = released.clone();
			let key = key.clone();
			Box::pin(async move {
				released.notified().await;
				Ok(ApiKeyCredential { key: Some(key), env: None })
			})
		})
	}

	#[tokio::test]
	async fn publishes_locally_consistent_availability_before_login_and_logout_resolve() {
		let credentials = in_memory();
		let runtime = runtime_with_credentials(
			credentials.clone() as Arc<dyn CredentialStore>,
			native_api_key_provider("dynamic", TestProviderHooks {
				login: Some(default_login("dynamic")),
				..Default::default()
			}),
		)
		.await;

		Models::login(&*runtime, "dynamic", AuthType::ApiKey, interaction(), None).await.unwrap();
		assert!(runtime.has_configured_auth("dynamic"));
		assert!(runtime.get_available_snapshot().iter().any(|model| model.provider() == "dynamic"));
		let stored_credential = read_stored(credentials.as_ref(), "dynamic").await;
		assert_eq!(stored_credential, Some(Credential::ApiKey(ApiKeyCredential { key: Some("dynamic-key".to_string()), env: None })));

		Models::logout(&*runtime, "dynamic", None).await.unwrap();
		assert!(!runtime.has_configured_auth("dynamic"));
		assert!(!runtime.get_available_snapshot().iter().any(|model| model.provider() == "dynamic"));
		assert!(read_stored(credentials.as_ref(), "dynamic").await.is_none());
	}

	#[tokio::test]
	async fn orders_same_provider_credential_operations_through_local_synchronization() {
		let credentials = in_memory();
		let runtime = create_runtime(credentials.clone() as Arc<dyn CredentialStore>).await;
		let (tx, mut started, released) = gate();
		let provider = native_api_key_provider("ordered", TestProviderHooks {
			login: Some(blocking_login(tx, released.clone(), "ordered-key")),
			..Default::default()
		});
		runtime.register_native_provider(provider).unwrap();
		Models::refresh(
			&*runtime,
			Some(ModelsRefreshOptions {
				allow_network: Some(false),
				providers: Some(vec!["ordered".to_string()]),
				..Default::default()
			}),
		)
		.await;

		let login_runtime = runtime.clone();
		let login = tokio::spawn(async move {
			Models::login(&*login_runtime, "ordered", AuthType::ApiKey, interaction(), None).await
		});
		started.recv().await;
		let logout_runtime = runtime.clone();
		let logout = tokio::spawn(async move { Models::logout(&*logout_runtime, "ordered", None).await });
		tokio::task::yield_now().await;
		assert!(read_stored(credentials.as_ref(), "ordered").await.is_none());

		released.notify_one();
		login.await.unwrap().unwrap();
		logout.await.unwrap().unwrap();
		assert!(read_stored(credentials.as_ref(), "ordered").await.is_none());
		assert!(!runtime.has_configured_auth("ordered"));
	}

	#[tokio::test]
	async fn allows_different_providers_to_run_credential_flows_concurrently() {
		let runtime = create_runtime(in_memory() as Arc<dyn CredentialStore>).await;
		let (tx_one, mut one_started, one_released) = gate();
		let (tx_two, mut two_started, two_released) = gate();
		runtime
			.register_native_provider(native_api_key_provider("one", TestProviderHooks {
				login: Some(blocking_login(tx_one, one_released.clone(), "one")),
				..Default::default()
			}))
			.unwrap();
		runtime
			.register_native_provider(native_api_key_provider("two", TestProviderHooks {
				login: Some(blocking_login(tx_two, two_released.clone(), "two")),
				..Default::default()
			}))
			.unwrap();
		Models::refresh(
			&*runtime,
			Some(ModelsRefreshOptions {
				allow_network: Some(false),
				providers: Some(vec!["one".to_string(), "two".to_string()]),
				..Default::default()
			}),
		)
		.await;

		let one_runtime = runtime.clone();
		let one = tokio::spawn(async move {
			Models::login(&*one_runtime, "one", AuthType::ApiKey, interaction(), None).await
		});
		let two_runtime = runtime.clone();
		let two = tokio::spawn(async move {
			Models::login(&*two_runtime, "two", AuthType::ApiKey, interaction(), None).await
		});
		one_started.recv().await;
		two_started.recv().await;
		two_released.notify_one();
		one_released.notify_one();
		one.await.unwrap().unwrap();
		two.await.unwrap().unwrap();
	}

	#[tokio::test]
	async fn does_not_wait_for_unrelated_provider_availability_during_local_synchronization() {
		let stall_unrelated = Arc::new(std::sync::atomic::AtomicBool::new(false));
		let runtime = create_runtime(in_memory() as Arc<dyn CredentialStore>).await;
		runtime
			.register_native_provider(native_api_key_provider("target", TestProviderHooks {
				login: Some(default_login("target")),
				..Default::default()
			}))
			.unwrap();
		let stall = stall_unrelated.clone();
		runtime
			.register_native_provider(native_api_key_provider("unrelated", TestProviderHooks {
				check: Some(Arc::new(move |_credential| {
					let stall = stall.clone();
					Box::pin(async move {
						if stall.load(std::sync::atomic::Ordering::SeqCst) {
							std::future::pending::<()>().await;
						}
						Ok(None)
					})
				})),
				..Default::default()
			}))
			.unwrap();
		Models::refresh(
			&*runtime,
			Some(ModelsRefreshOptions {
				allow_network: Some(false),
				providers: Some(vec!["target".to_string(), "unrelated".to_string()]),
				..Default::default()
			}),
		)
		.await;
		stall_unrelated.store(true, std::sync::atomic::Ordering::SeqCst);

		Models::login(&*runtime, "target", AuthType::ApiKey, interaction(), None).await.unwrap();
		assert!(runtime.has_configured_auth("target"));
		let result = Models::refresh(
			&*runtime,
			Some(ModelsRefreshOptions {
				allow_network: Some(false),
				providers: Some(vec!["target".to_string()]),
				..Default::default()
			}),
		)
		.await;
		assert!(!result.aborted);
	}

	#[tokio::test]
	async fn reports_cancellation_that_occurs_during_provider_scoped_availability() {
		let block = Arc::new(std::sync::atomic::AtomicBool::new(false));
		let (tx, mut started) = tokio::sync::mpsc::unbounded_channel();
		let provider = native_api_key_provider("cancelled-availability", TestProviderHooks {
			login: Some(default_login("cancelled-availability")),
			check: Some({
				let block = block.clone();
				Arc::new(move |credential| {
					let block = block.clone();
					let tx = tx.clone();
					Box::pin(async move {
						if block.load(std::sync::atomic::Ordering::SeqCst) {
							let _ = tx.send(());
							std::future::pending::<()>().await;
						}
						Ok(credential.map(|_| AuthCheck {
							source: Some("stored".to_string()),
							check_type: AuthType::ApiKey,
						}))
					})
				})
			}),
			..Default::default()
		});
		let runtime = runtime_with_provider(provider).await;
		runtime.set_runtime_api_key("cancelled-availability", "key", None).await.unwrap();
		block.store(true, std::sync::atomic::Ordering::SeqCst);

		let controller = AbortController::new();
		let refresh_runtime = runtime.clone();
		let signal = controller.signal();
		let refresh = tokio::spawn(async move {
			Models::refresh(
				&*refresh_runtime,
				Some(ModelsRefreshOptions {
					allow_network: Some(false),
					providers: Some(vec!["cancelled-availability".to_string()]),
					signal: Some(signal),
					..Default::default()
				}),
			)
			.await
		});
		started.recv().await;
		controller.abort();
		assert!(refresh.await.unwrap().aborted);
	}

	#[tokio::test]
	async fn does_not_run_network_refresh_inside_the_credential_operation_chain() {
		let network_refresh_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
		let counter = network_refresh_count.clone();
		let runtime = runtime_with_provider(native_api_key_provider("local-only", TestProviderHooks {
			login: Some(default_login("local-only")),
			refresh_models: Some(Arc::new(move |context: RefreshModelsContext| {
				let counter = counter.clone();
				Box::pin(async move {
					if context.allow_network {
						counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
						std::future::pending::<()>().await;
					}
					Ok(())
				})
			})),
			..Default::default()
		}))
		.await;

		Models::login(&*runtime, "local-only", AuthType::ApiKey, interaction(), None).await.unwrap();
		assert_eq!(network_refresh_count.load(std::sync::atomic::Ordering::SeqCst), 0);
		assert!(runtime.has_configured_auth("local-only"));
	}

	#[tokio::test]
	async fn keeps_provider_scoped_refreshes_from_superseding_unrelated_providers() {
		let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
			credentials: Some(in_memory() as Arc<dyn CredentialStore>),
			models_path: Some(None),
			refresh_on_create: true,
			allow_model_network: true,
			..Default::default()
		})
		.await
		.unwrap();
		let (tx, mut started) = tokio::sync::mpsc::unbounded_channel();
		let first_signal: Arc<parking_lot::Mutex<Option<pi_ai::utils::abort::AbortSignal>>> = Arc::default();
		let signal_sink = first_signal.clone();
		let finish = Arc::new(tokio::sync::Notify::new());
		let hook_finish = finish.clone();
		runtime
			.register_native_provider(native_api_key_provider("one", TestProviderHooks {
				refresh_models: Some(Arc::new(move |context: RefreshModelsContext| {
					let signal_sink = signal_sink.clone();
					let tx = tx.clone();
					let finish = hook_finish.clone();
					Box::pin(async move {
						if !context.allow_network {
							return Ok(());
						}
						*signal_sink.lock() = Some(context.signal.clone());
						let _ = tx.send(());
						finish.notified().await;
						Ok(())
					})
				})),
				..Default::default()
			}))
			.unwrap();
		runtime
			.register_native_provider(native_api_key_provider("two", TestProviderHooks::default()))
			.unwrap();
		Models::refresh(
			&*runtime,
			Some(ModelsRefreshOptions {
				allow_network: Some(false),
				providers: Some(vec!["one".to_string(), "two".to_string()]),
				..Default::default()
			}),
		)
		.await;
		runtime.set_runtime_api_key("one", "one-key", None).await.unwrap();
		runtime.set_runtime_api_key("two", "two-key", None).await.unwrap();

		let first_runtime = runtime.clone();
		let first = tokio::spawn(async move {
			Models::refresh(
				&*first_runtime,
				Some(ModelsRefreshOptions {
					allow_network: Some(true),
					providers: Some(vec!["one".to_string()]),
					..Default::default()
				}),
			)
			.await
		});
		started.recv().await;
		Models::refresh(
			&*runtime,
			Some(ModelsRefreshOptions {
				allow_network: Some(true),
				providers: Some(vec!["two".to_string()]),
				..Default::default()
			}),
		)
		.await;
		let signal = first_signal.lock().clone().expect("network refresh started");
		assert!(!signal.aborted());

		finish.notify_one();
		first.await.unwrap();
	}

	#[tokio::test]
	async fn waits_for_a_committed_credential_mutation_to_settle_before_reporting_cancellation() {
		struct DelayedCommitStore {
			stored: parking_lot::Mutex<Option<Credential>>,
			committed: tokio::sync::mpsc::UnboundedSender<()>,
			finish: Arc<tokio::sync::Notify>,
		}
		#[async_trait::async_trait]
		impl CredentialStore for DelayedCommitStore {
			async fn read(
				&self,
				_provider_id: &str,
				_options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				Ok(self.stored.lock().clone())
			}

			async fn list(&self, _options: Option<AuthOperationOptions>) -> Result<Vec<CredentialInfo>, ThrownValue> {
				Ok(self
					.stored
					.lock()
					.clone()
					.map(|stored| CredentialInfo { provider_id: "delayed-commit".to_string(), credential_type: stored.type_name().to_string() })
					.into_iter()
					.collect())
			}

			async fn modify(
				&self,
				_provider_id: &str,
				modify_fn: CredentialModifyFn,
				_options: Option<AuthOperationOptions>,
			) -> Result<Option<Credential>, ThrownValue> {
				let current = self.stored.lock().clone();
				let next = modify_fn(current.as_ref()).await?;
				if let Some(next) = next {
					*self.stored.lock() = Some(next);
				}
				let _ = self.committed.send(());
				self.finish.notified().await;
				Ok(self.stored.lock().clone())
			}

			async fn delete(&self, _provider_id: &str, _options: Option<AuthOperationOptions>) -> Result<(), ThrownValue> {
				*self.stored.lock() = None;
				Ok(())
			}
		}

		let (committed_tx, mut committed_rx) = tokio::sync::mpsc::unbounded_channel();
		let finish = Arc::new(tokio::sync::Notify::new());
		let credentials = Arc::new(DelayedCommitStore {
			stored: parking_lot::Mutex::new(None),
			committed: committed_tx,
			finish: finish.clone(),
		});
		let runtime = create_runtime(credentials.clone() as Arc<dyn CredentialStore>).await;
		runtime
			.register_native_provider(native_api_key_provider("delayed-commit", TestProviderHooks::default()))
			.unwrap();
		Models::refresh(
			&*runtime,
			Some(ModelsRefreshOptions {
				allow_network: Some(false),
				providers: Some(vec!["delayed-commit".to_string()]),
				..Default::default()
			}),
		)
		.await;
		let controller = AbortController::new();
		let (outcome_tx, mut outcome_rx) = tokio::sync::mpsc::unbounded_channel::<Result<Credential, ThrownValue>>();
		let login_runtime = runtime.clone();
		let signal = controller.signal();
		tokio::spawn(async move {
			let result =
				Models::login(
					&*login_runtime,
					"delayed-commit",
					AuthType::ApiKey,
					interaction_with(signal.clone()),
					None,
				)
				.await;
			let _ = outcome_tx.send(result);
		});
		committed_rx.recv().await;
		controller.abort();
		// The committed mutation must be awaited before cancellation is
		// reported: the outcome is still pending here.
		if let Ok(Some(early)) = tokio::time::timeout(std::time::Duration::from_millis(50), outcome_rx.recv()).await {
			panic!("login settled early: {early:?}");
		}

		finish.notify_one();
		let outcome = outcome_rx.recv().await.unwrap();
		assert_sync_error(outcome.unwrap_err(), "delayed-commit");
		assert_eq!(
			read_stored(credentials.as_ref(), "delayed-commit").await,
			Some(Credential::ApiKey(ApiKeyCredential { key: Some("delayed-commit-key".to_string()), env: None }))
		);
	}

	#[tokio::test]
	async fn reports_a_typed_error_when_cancellation_interrupts_post_commit_synchronization() {
		let block = Arc::new(std::sync::atomic::AtomicBool::new(false));
		let (tx, mut started) = tokio::sync::mpsc::unbounded_channel();
		let block_hook = block.clone();
		let credentials = in_memory();
		let runtime = runtime_with_credentials(credentials.clone() as Arc<dyn CredentialStore>, native_api_key_provider("cancelled-sync", TestProviderHooks {
			refresh_models: Some(Arc::new(move |context: RefreshModelsContext| {
				let block = block_hook.clone();
				let tx = tx.clone();
				Box::pin(async move {
					if !context.allow_network && block.load(std::sync::atomic::Ordering::SeqCst) {
						let _ = tx.send(());
						std::future::pending::<()>().await;
					}
					Ok(())
				})
			})),
			..Default::default()
		}))
		.await;
		block.store(true, std::sync::atomic::Ordering::SeqCst);

		let controller = AbortController::new();
		let login_runtime = runtime.clone();
		let signal = controller.signal();
		let login = tokio::spawn(async move {
			Models::login(
				&*login_runtime,
				"cancelled-sync",
				AuthType::ApiKey,
				interaction_with(signal.clone()),
				None,
			)
			.await
		});
		started.recv().await;
		controller.abort();

		let error = login.await.unwrap().unwrap_err();
		assert_sync_error(error, "cancelled-sync");
		assert_eq!(
			read_stored(credentials.as_ref(), "cancelled-sync").await,
			Some(Credential::ApiKey(ApiKeyCredential { key: Some("cancelled-sync-key".to_string()), env: None }))
		);
	}

	#[tokio::test]
	async fn reports_committed_credentials_when_local_synchronization_fails() {
		let fail = Arc::new(std::sync::atomic::AtomicBool::new(false));
		let fail_hook = fail.clone();
		let credentials = in_memory();
		let runtime = runtime_with_credentials(credentials.clone() as Arc<dyn CredentialStore>, native_api_key_provider("broken-sync", TestProviderHooks {
			refresh_models: Some(Arc::new(move |context: RefreshModelsContext| {
				let fail = fail_hook.clone();
				Box::pin(async move {
					if !context.allow_network && fail.load(std::sync::atomic::Ordering::SeqCst) {
						return Err(ThrownValue::message("cache restore failed"));
					}
					Ok(())
				})
			})),
			..Default::default()
		}))
		.await;
		fail.store(true, std::sync::atomic::Ordering::SeqCst);

		let error = Models::login(&*runtime, "broken-sync", AuthType::ApiKey, interaction(), None)
			.await
			.unwrap_err();
		assert_sync_error(error, "broken-sync");
		assert_eq!(
			read_stored(credentials.as_ref(), "broken-sync").await,
			Some(Credential::ApiKey(ApiKeyCredential { key: Some("broken-sync-key".to_string()), env: None }))
		);
	}
}
