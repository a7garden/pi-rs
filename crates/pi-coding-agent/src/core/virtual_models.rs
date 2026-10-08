//! Port of `packages/coding-agent/src/core/virtual-models.ts`.
//!
//! Virtual models are catalog entries that route each request to a physical
//! model. The selection (`model_change`, agent state, ...) may name a virtual
//! model; everything below the routing step only sees physical models —
//! providers stream them and assistant messages record them. A virtual model
//! never reaches a provider.
//!
//! Deviations (crate docs D62):
//! - **D62** — upstream's `TState` generic (default `unknown`, contract
//!   "must be JSON-serializable") collapses to [`serde_json::Value`]; the
//!   router state crosses the session-record boundary as JSON either way.
//!   The `route(request)` callback (sync value or promise) is
//!   [`ModelRouteFn`]: an owned [`RouteRequest`], returning a [`BoxFuture`]
//!   of `Result<ModelRoute, ThrownValue>` (thrown errors), with [`route`]
//!   as the single await seam.
//! - **D62a** — `Omit<VirtualModelDefinition, "route">` is not expressible;
//!   [`create_virtual_model`] takes the full definition and ignores `route`.
//! - **D62b** — `isVirtualModel({ api })` splits into
//!   [`is_virtual_model_api`] (any `api` string) and [`is_virtual_model`]
//!   (chat [`Model`]); assistant messages test `is_virtual_model_api` on
//!   `message.api`.
//! - **D62c** — `withVirtualModels` object-spread becomes an explicit
//!   [`Provider`] wrapper. Behavior is identical, including that the
//!   `providerId` argument only names the keyless provider built when no
//!   base provider is given (an existing provider keeps its own id, exactly
//!   like the upstream spread).
//! - **D62d** — [`find_latest_response`] and [`get_branch_selection`] treat
//!   only typed LLM assistant messages as responses; upstream's structural
//!   check would also admit custom `{role: "assistant"}` JSON values, whose
//!   `provider`/`model`/`api` would be `undefined`. [`find_latest_response`]
//!   walks the typed transcript (`&[Message]`) the runtime carries; upstream
//!   walks `AgentMessage[]`, whose custom entries are never
//!   assistant-typed.

use std::collections::{BTreeMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use pi_agent::AgentMessage;
use pi_ai::auth::types::{ApiKeyAuth, ApiKeyAuthInput, AuthResult, ModelAuth, ProviderAuth};
use pi_ai::types::{
	AnyModel, AssistantMessage, BaseModelFields, ClassifierContext, ClassifierModel, ClassifierOptions,
	ClassifierResult, DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, ImageModel, ImagesContext,
	ImagesOptions, Message, Model, ModelCost, ModelInput, ModelThinkingLevel, ModelType, SimpleStreamOptions,
	StreamOptions, ThinkingLevelMap, TranscriptContext,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::thrown::ThrownValue;
use pi_ai::{api::lazy::lazy_stream, Provider};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::session_manager::SessionEntry;

/// API id of virtual catalog entries. Requests for it fail unless routed first
/// (`VIRTUAL_MODEL_API`).
pub const VIRTUAL_MODEL_API: &str = "pi-virtual";

/// Custom entry type that stores router state on the session branch
/// (`VIRTUAL_MODEL_STATE_ENTRY`).
pub const VIRTUAL_MODEL_STATE_ENTRY: &str = "pi.virtual-model-state";

/// Data of a `pi.virtual-model-state` custom entry (`VirtualModelStateData`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VirtualModelStateData<TState = Value> {
	pub provider: String,
	#[serde(rename = "modelId")]
	pub model_id: String,
	pub state: TState,
}

/// Why a request is being routed (`ModelRouteReason`):
/// - [`ModelRouteReason::User`]: first request after a message the user wrote
///   (prompt, steering, or follow-up)
/// - [`ModelRouteReason::Continuation`]: any other request in the agent loop,
///   e.g. after tool results or extension messages
/// - [`ModelRouteReason::Retry`]: automatic retry after a failed request,
///   including after compaction for a context overflow
/// - [`ModelRouteReason::Direct`]: a request outside the agent loop, e.g. a
///   compaction summary or an extension call
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelRouteReason {
	User,
	Continuation,
	Retry,
	Direct,
}

/// Physical model and thinking level of another request, carried on
/// [`RouteRequest`] (`previous` inline `{ model, thinkingLevel }` shape,
/// `PreviousRoute`).
#[derive(Debug, Clone, PartialEq)]
pub struct PreviousRoute {
	pub model: Model,
	pub thinking_level: Option<ModelThinkingLevel>,
}

/// The failed request of a [`ModelRouteReason::Retry`] routing
/// (`RouteRequest.failed`).
#[derive(Debug, Clone, PartialEq)]
pub struct FailedRoute {
	pub model: Model,
	pub thinking_level: Option<ModelThinkingLevel>,
	/// The failed response; carries its `stopReason` and `errorMessage`.
	pub message: AssistantMessage,
}

/// Request handed to a router (`ModelRouteRequest`).
#[derive(Clone)]
pub struct RouteRequest {
	/// The selected virtual model.
	pub model: Model,
	/// The selected thinking level. Its meaning is up to the router.
	pub thinking_level: ModelThinkingLevel,
	pub reason: ModelRouteReason,
	/// Physical model and thinking level of the latest successful response in
	/// `messages`.
	pub previous: Option<PreviousRoute>,
	/// For `retry`: the failed request, which `messages` no longer contains.
	/// `message` carries its `stopReason` and `errorMessage`. Absent when the
	/// router itself failed.
	pub failed: Option<FailedRoute>,
	/// Router state last returned on this session branch. `None` before the
	/// first state and for `direct` requests.
	pub state: Option<Value>,
	/// Conversation for this request, including system messages.
	pub messages: Vec<Message>,
	pub signal: Option<pi_ai::utils::abort::AbortSignal>,
}

/// Selection facts of one `resolveModel` call, the non-request inputs of the
/// routing (upstream `resolveModel` options: `reason`, `thinkingLevel`,
/// `failed`; plus the threaded-through `state`/`signal`).
#[derive(Clone)]
pub struct RouteRequestOptions {
	pub reason: ModelRouteReason,
	/// The selected thinking level. Its meaning is up to the router.
	pub thinking_level: ModelThinkingLevel,
	/// For `retry`: the failed response (names the failed request's model and
	/// carries its `stopReason`/`errorMessage`).
	pub failed: Option<AssistantMessage>,
	/// Router state last returned on this session branch.
	pub state: Option<Value>,
	pub signal: Option<pi_ai::utils::abort::AbortSignal>,
}

/// Physical model and thinking level for one request (`ModelRoute`).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRoute {
	pub model: Model,
	pub thinking_level: ModelThinkingLevel,
	/// New router state, stored on the session branch unless it is
	/// `request.state` itself. Return `request.state` or `None` to keep the
	/// current state. Must be JSON-serializable. Ignored for `direct`
	/// requests.
	pub state: Option<Value>,
}

/// `route(request)` of a virtual model: sync value or promise upstream
/// (see crate docs D62).
pub type ModelRouteFn = Arc<dyn Fn(RouteRequest) -> ModelRouteOutcome + Send + Sync>;

/// Resolved outcome of one router call (upstream `ModelRoute | Promise`).
pub type ModelRouteOutcome = BoxFuture<'static, Result<ModelRoute, ThrownValue>>;

/// Run a virtual model's router for one request (upstream
/// `definition.route(request)`; a free function here so the sync-or-promise
/// callback has one await seam).
pub async fn route(definition: &VirtualModelDefinition, request: &RouteRequest) -> Result<ModelRoute, ThrownValue> {
	(definition.route)(request.clone()).await
}

/// A virtual catalog entry (`VirtualModelDefinition`).
#[derive(Clone)]
pub struct VirtualModelDefinition {
	/// Provider the virtual model is listed under. May be a provider with
	/// physical models.
	pub provider: String,
	/// Model id. Must not be the id of a physical model of `provider`.
	pub id: String,
	pub name: String,
	/// Thinking levels offered for selection. Defaults to `["off"]`.
	pub thinking_levels: Option<Vec<ModelThinkingLevel>>,
	/// Limits shown before the first response. Afterwards, Pi uses the limits
	/// of the physical model that answered. Unset limits are unknown (0).
	pub context_window: Option<u64>,
	pub max_tokens: Option<u64>,
	/// Input types accepted for selection. Defaults to text and images;
	/// routed models without image support get placeholders.
	pub input: Option<Vec<ModelInput>>,
	/// Pick the physical model, which must have credentials, and thinking
	/// level for one request.
	pub route: ModelRouteFn,
}

/// Ordered ladder used to build the catalog entry's thinking-level map
/// (upstream `THINKING_LEVELS`).
const THINKING_LEVELS: [ModelThinkingLevel; 7] = ModelThinkingLevel::LADDER;

/// Serde id of a thinking level (the identity mapping value of
/// `thinkingLevelMap`).
fn thinking_level_id(level: ModelThinkingLevel) -> &'static str {
	match level {
		ModelThinkingLevel::Off => "off",
		ModelThinkingLevel::Minimal => "minimal",
		ModelThinkingLevel::Low => "low",
		ModelThinkingLevel::Medium => "medium",
		ModelThinkingLevel::High => "high",
		ModelThinkingLevel::Xhigh => "xhigh",
		ModelThinkingLevel::Max => "max",
	}
}

/// Whether an `api` id names a virtual model (`isVirtualModel` on any
/// `{ api }` shape, D62b).
pub fn is_virtual_model_api(api: &str) -> bool {
	api == VIRTUAL_MODEL_API
}

/// Whether a model names a virtual model (`isVirtualModel`). Failed routing
/// leaves the virtual model on its message.
pub fn is_virtual_model(model: &Model) -> bool {
	is_virtual_model_api(model.api())
}

/// Latest successful response (`findLatestResponse`). Its model is physical:
/// failed or aborted requests, including failed routing, are skipped (D62d).
/// Takes the typed transcript (`Message[]`); upstream walks `AgentMessage[]`,
/// whose custom entries are never assistant-typed (D62d).
pub fn find_latest_response(messages: &[Message]) -> Option<&AssistantMessage> {
	messages.iter().rev().find_map(|message| match message {
		Message::Assistant(assistant)
			if assistant.stop_reason != pi_ai::types::StopReason::Error
				&& assistant.stop_reason != pi_ai::types::StopReason::Aborted =>
		{
			Some(assistant)
		}
		_ => None,
	})
}

/// The model selection a session branch records (`getBranchSelection`).
/// A virtual `model_change` holds until the next `model_change`, because
/// responses name the physical models it routed to. Otherwise the latest
/// physical response wins, as in sessions without virtual models. A virtual
/// model that is no longer registered does not hold, so the selection falls
/// back to the physical model that answered last.
///
/// Only the last `model_change` can hold, so this looks up at most one model
/// in the catalog.
pub fn get_branch_selection(
	branch: &[SessionEntry],
	get_model: &mut dyn FnMut(&str, &str) -> Option<Model>,
) -> Option<BranchSelection> {
	for (index, entry) in branch.iter().enumerate().rev() {
		match entry {
			SessionEntry::ModelChange(change) => {
				return Some(BranchSelection { provider: change.provider.clone(), model_id: change.model_id.clone() });
			}
			SessionEntry::Message(message_entry) => {
				let AgentMessage::Llm(Message::Assistant(assistant)) = &message_entry.message else {
					continue;
				};
				if is_virtual_model_api(&assistant.api) {
					continue;
				}
				let response =
					BranchSelection { provider: assistant.provider.clone(), model_id: assistant.model.clone() };
				let change = find_last_model_change(branch, index);
				let model = change.as_ref().and_then(|change| get_model(&change.provider, &change.model_id));
				let holds = change.is_some() && model.is_some() && is_virtual_model(&model.unwrap());
				return Some(if holds { change.unwrap() } else { response });
			}
			_ => {}
		}
	}
	None
}

/// A `{ provider, modelId }` selection (`getBranchSelection` result shape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchSelection {
	pub provider: String,
	pub model_id: String,
}

fn find_last_model_change(branch: &[SessionEntry], before: usize) -> Option<BranchSelection> {
	branch[..before].iter().rev().find_map(|entry| match entry {
		SessionEntry::ModelChange(change) => Some(BranchSelection {
			provider: change.provider.clone(),
			model_id: change.model_id.clone(),
		}),
		_ => None,
	})
}

/// Latest router state a session branch stores for a virtual model
/// (`getVirtualModelState`).
pub fn get_virtual_model_state(branch: &[SessionEntry], provider: &str, model_id: &str) -> Option<Value> {
	for entry in branch.iter().rev() {
		let SessionEntry::Custom(custom) = entry else { continue };
		if custom.custom_type != VIRTUAL_MODEL_STATE_ENTRY {
			continue;
		}
		let Some(data) = &custom.data else { continue };
		let Ok(data) = serde_json::from_value::<VirtualModelStateData>(data.clone()) else { continue };
		if data.provider == provider && data.model_id == model_id {
			return Some(data.state);
		}
	}
	None
}

/// Build the catalog entry of a virtual model (`createVirtualModel`). The
/// `route` callback is not part of a catalog entry and is ignored here
/// (upstream takes `Omit<VirtualModelDefinition, "route">`, D62a).
pub fn create_virtual_model(definition: &VirtualModelDefinition) -> Model {
	let levels = definition.thinking_levels.clone().unwrap_or_else(|| vec![ModelThinkingLevel::Off]);
	let mut thinking_level_map: ThinkingLevelMap = BTreeMap::new();
	for level in THINKING_LEVELS {
		let selected = levels.contains(&level);
		thinking_level_map.insert(level, selected.then(|| thinking_level_id(level).to_string()));
	}
	Model {
		base: BaseModelFields {
			id: definition.id.clone(),
			name: definition.name.clone(),
			api: VIRTUAL_MODEL_API.to_string(),
			provider: definition.provider.clone(),
			base_url: String::new(),
			input: definition.input.clone().unwrap_or_else(|| vec![ModelInput::Text, ModelInput::Image]),
			input_limits: None,
			cost: ModelCost::default(),
			headers: None,
		},
		model_type: None,
		reasoning: levels.iter().any(|level| *level != ModelThinkingLevel::Off),
		thinking_level_map: Some(thinking_level_map),
		prompt_cache: None,
		context_window: definition.context_window.unwrap_or(0),
		max_tokens: definition.max_tokens.unwrap_or(0),
		sampling_params: None,
		sampling_params_by_thinking_level: None,
		compat: None,
	}
}

/// Stream for a virtual model that was not routed, e.g. `stream()` with
/// API-specific options (`unroutedStream`).
fn unrouted_stream(model: &Model) -> AssistantMessageEventStream {
	let message =
		format!("Virtual model {}/{} must be routed before streaming", model.provider(), model.id());
	lazy_stream(model, async move { Err(ThrownValue::message(message)) })
}

/// The api-key `resolve` of the keyless provider: always resolves to an empty
/// auth labelled `"virtual"` (upstream
/// `{ apiKey: { name: "Virtual model", resolve: async () => ({ auth: {}, source: "virtual" }) } }`).
fn resolve_virtual<'a>(_input: ApiKeyAuthInput<'a>) -> BoxFuture<'a, Result<Option<AuthResult>, ThrownValue>> {
	Box::pin(async move {
		Ok(Some(AuthResult {
			auth: ModelAuth::default(),
			env: None,
			source: Some("virtual".to_string()),
		}))
	})
}

/// Keyless provider listing only the virtual models (`withVirtualModels`
/// without a provider).
struct VirtualOnlyProvider {
	id: String,
	auth: ProviderAuth,
	models: Vec<Model>,
}

impl VirtualOnlyProvider {
	fn new(id: &str, models: Vec<Model>) -> Self {
		VirtualOnlyProvider {
			id: id.to_string(),
			auth: ProviderAuth {
				api_key: Some(ApiKeyAuth {
					name: "Virtual model".to_string(),
					login: None,
					check: None,
					resolve: Box::new(resolve_virtual),
				}),
				oauth: None,
			},
			models,
		}
	}
}

impl Provider for VirtualOnlyProvider {
	fn id(&self) -> &str {
		&self.id
	}

	fn name(&self) -> &str {
		&self.id
	}

	fn auth(&self) -> &ProviderAuth {
		&self.auth
	}

	fn get_models(&self) -> Vec<Model> {
		self.models.clone()
	}

	fn get_all_models(&self) -> Vec<AnyModel> {
		self.models.iter().cloned().map(AnyModel::Chat).collect()
	}

	fn stream(&self, model: &Model, _context: &TranscriptContext, _options: &StreamOptions) -> AssistantMessageEventStream {
		unrouted_stream(model)
	}

	fn stream_simple(
		&self,
		model: &Model,
		_context: &TranscriptContext,
		_options: &SimpleStreamOptions,
	) -> AssistantMessageEventStream {
		unrouted_stream(model)
	}
}

/// A provider whose catalog gains the virtual models (`withVirtualModels`
/// with a provider, D62c).
struct VirtualModelsProvider {
	inner: Arc<dyn Provider>,
	/// Ids of the registered virtual models; these hide physical chat models.
	virtual_ids: HashSet<String>,
	virtual_models: Vec<Model>,
}

impl VirtualModelsProvider {
	fn new(inner: Arc<dyn Provider>, virtual_models: Vec<Model>) -> Self {
		let virtual_ids = virtual_models.iter().map(|model| model.id().to_string()).collect();
		VirtualModelsProvider { inner, virtual_ids, virtual_models }
	}

	/// Upstream `physical`: drops virtual catalog entries and physical chat
	/// models hidden by a same-id virtual model.
	fn physical_chat(&self, models: Vec<Model>) -> Vec<Model> {
		models
			.into_iter()
			.filter(|model| !is_virtual_model(model) && !self.virtual_ids.contains(model.id()))
			.collect()
	}

	fn physical_any(&self, models: Vec<AnyModel>) -> Vec<AnyModel> {
		models
			.into_iter()
			.filter(|model| {
				!is_virtual_model_api(model.api())
					&& !(model.model_type() == ModelType::Chat && self.virtual_ids.contains(model.id()))
			})
			.collect()
	}

	fn virtual_of(&self, models: &[AnyModel]) -> Vec<AnyModel> {
		models.iter().filter(|model| is_virtual_model_api(model.api())).cloned().collect()
	}
}

impl Provider for VirtualModelsProvider {
	fn id(&self) -> &str {
		self.inner.id()
	}

	fn name(&self) -> &str {
		self.inner.name()
	}

	fn base_url(&self) -> Option<&str> {
		self.inner.base_url()
	}

	fn headers(&self) -> Option<&pi_ai::types::ProviderHeaders> {
		self.inner.headers()
	}

	fn auth(&self) -> &ProviderAuth {
		self.inner.auth()
	}

	fn get_models(&self) -> Vec<Model> {
		let mut models = self.physical_chat(self.inner.get_models());
		models.extend(self.virtual_models.iter().cloned());
		models
	}

	fn get_all_models(&self) -> Vec<AnyModel> {
		let mut models = self.physical_any(self.inner.get_all_models());
		models.extend(self.virtual_models.iter().cloned().map(AnyModel::Chat));
		models
	}

	fn refresh_models(
		&self,
		context: pi_ai::models::RefreshModelsContext,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		self.inner.refresh_models(context)
	}

	fn filter_models(
		&self,
		models: &[Model],
		credential: Option<&pi_ai::auth::types::Credential>,
	) -> Option<Vec<Model>> {
		let real = self.physical_chat(models.to_vec());
		let mut filtered = self.inner.filter_models(&real, credential).unwrap_or(real);
		// Only virtual models that are part of the input survive here; the
		// full registered list is appended by `get_models`.
		filtered.extend(models.iter().filter(|model| is_virtual_model(model)).cloned());
		Some(filtered)
	}

	fn filter_all_models(
		&self,
		models: &[AnyModel],
		credential: Option<&pi_ai::auth::types::Credential>,
	) -> Option<Vec<AnyModel>> {
		let real = self.physical_any(models.to_vec());
		// Upstream `filterAllModels && (...)`: absent stays absent.
		self.inner.filter_all_models(&real, credential).map(|mut filtered| {
			filtered.extend(self.virtual_of(models));
			filtered
		})
	}

	fn stream(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
		if is_virtual_model(model) {
			unrouted_stream(model)
		} else {
			self.inner.stream(model, context, options)
		}
	}

	fn stream_simple(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: &SimpleStreamOptions,
	) -> AssistantMessageEventStream {
		if is_virtual_model(model) {
			unrouted_stream(model)
		} else {
			self.inner.stream_simple(model, context, options)
		}
	}

	fn fetch_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: &DeferredFetchOptions,
	) -> Option<AssistantMessageEventStream> {
		self.inner.fetch_deferred(model, handle, options)
	}

	fn cancel_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: &DeferredCancelOptions,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		self.inner.cancel_deferred(model, handle, options)
	}

	fn generate_images<'a>(
		&'a self,
		model: &'a ImageModel,
		context: &'a ImagesContext,
		options: &'a ImagesOptions,
	) -> Option<Pin<Box<dyn Future<Output = pi_ai::types::AssistantImages> + Send + 'a>>> {
		self.inner.generate_images(model, context, options)
	}

	fn classify<'a>(
		&'a self,
		model: &'a ClassifierModel,
		context: &'a ClassifierContext,
		options: &'a ClassifierOptions,
	) -> Option<Pin<Box<dyn Future<Output = ClassifierResult> + Send + 'a>>> {
		self.inner.classify(model, context, options)
	}
}

/// Add virtual models to a provider's catalog (`withVirtualModels`). Without
/// a provider, the result is a keyless provider that only lists the virtual
/// models. A virtual model hides a physical chat model with the same id,
/// which a catalog refresh can add after registration. Availability follows
/// the provider's auth.
pub fn with_virtual_models(
	provider_id: &str,
	provider: Option<Arc<dyn Provider>>,
	virtual_models: Vec<Model>,
) -> Arc<dyn Provider> {
	match provider {
		None => Arc::new(VirtualOnlyProvider::new(provider_id, virtual_models)),
		Some(provider) => Arc::new(VirtualModelsProvider::new(provider, virtual_models)),
	}
}

#[cfg(test)]
#[path = "virtual_models_tests.rs"]
mod virtual_models_tests;
