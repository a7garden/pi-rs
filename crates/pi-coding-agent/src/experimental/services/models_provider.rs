//! Port of `experimental/services/models-provider.ts` — the Models service
//! over one durable conversation plus its owning facet.
//!
//! Deviations (D144):
//! - The upstream structural `Models` object becomes [`ModelsService`]; the
//!   facet exposes it as a remote singleton whose members are the serialized
//!   `state` plus the `cycleThinking`/`getThinkingLevels`/`refresh`/`select`/
//!   `selectThinking` methods.
//! - `MutableReplicatedState<ModelsState>`/`DocumentState<AgentState>` map to
//!   chord's untyped JSON states; `AgentState` is (de)serialized at the seam.
//! - `refresh({ signal })` maps to the `pi_ai::Models::refresh` trait call;
//!   the context abort signal is not threaded into the model refresh options
//!   yet (the Rust `ModelsRefreshOptions` carries a `pi_ai` abort signal).
//! - `env.replicatedState` maps to [`pi_chord::api::replicated_state`].

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use pi_ai::models::{clamp_thinking_level, get_supported_thinking_levels, Models as _};
use pi_ai::types::ModelThinkingLevel;
use pi_ai::utils::thrown::ThrownValue;
use pi_chord::context::{Context, BACKGROUND_CONTEXT};
use pi_chord::facets::host::disposal_from_unsubscribe;
use pi_chord::json::JsonValue;
use pi_chord::services::errors::{type_error, AnyError};
use pi_chord::services::provider::RemoteServiceObject;
use pi_chord::services::state::{AttachedReplicatedState, MutableReplicatedState};
use pi_chord::types::Facet;
use pi_chord::types::ServiceMode;
use pi_durable::harness::agent_doc;
use pi_durable::harness::harness::Conversation;
use pi_durable::harness::types::{AgentChange, AgentState, ModelRef, SetOrClear};

use crate::core::model_runtime::ModelRuntime;
use crate::core::settings_manager::SettingsManager;
use crate::experimental::services::models::{
    ModelRef as DtoModelRef, ModelSummary, ModelsCatalog, ModelsConfiguration, ModelsRefreshState,
    ModelsState,
};

/// Shared settings handle (upstream `SettingsManager | undefined`).
pub type SharedSettingsManager = Arc<Mutex<SettingsManager>>;

/// Upstream `ModelsServiceRuntime` — the service plus activation hooks.
pub struct ModelsServiceRuntime {
    pub service: Arc<ModelsService>,
}

/// The Models service over one conversation (upstream `createModelsService`).
pub struct ModelsService {
    conversation: Arc<Conversation>,
    agent: Arc<AttachedReplicatedState>,
    model_runtime: Option<Arc<ModelRuntime>>,
    settings_manager: Option<SharedSettingsManager>,
    state: Arc<MutableReplicatedState>,
    catalog_revision: AtomicU64,
}

impl ModelsService {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        conversation: Arc<Conversation>,
        agent: Arc<AttachedReplicatedState>,
        model_runtime: Option<Arc<ModelRuntime>>,
        settings_manager: Option<SharedSettingsManager>,
        initial: ModelsState,
    ) -> Arc<ModelsService> {
        let state = pi_chord::api::replicated_state(
            serde_json::to_value(&initial).expect("ModelsState serializes"),
        );
        Arc::new(ModelsService {
            conversation,
            agent,
            model_runtime,
            settings_manager,
            state,
            catalog_revision: AtomicU64::new(initial.catalog.revision),
        })
    }

    /// Read-only replicated state (upstream `Models.state`).
    pub fn state(&self) -> Arc<MutableReplicatedState> {
        Arc::clone(&self.state)
    }

    fn agent_state(&self) -> Option<AgentState> {
        let value = self.agent.value();
        if value.is_null() {
            return None;
        }
        Some(serde_json::from_value(value.clone()).unwrap_or_default())
    }

    fn configuration_of(value: Option<&AgentState>) -> ModelsConfiguration {
        ModelsConfiguration {
            model: value
                .and_then(|state| state.model.as_ref())
                .map(|model| DtoModelRef {
                    provider: model.provider.clone(),
                    model_id: model.model_id.clone(),
                }),
            thinking_level: value
                .and_then(|state| state.thinking_level)
                .unwrap_or(ModelThinkingLevel::Off),
        }
    }

    fn configuration(&self) -> ModelsConfiguration {
        Self::configuration_of(self.agent_state().as_ref())
    }

    fn read_state(&self) -> Option<ModelsState> {
        serde_json::from_value(self.state.value().clone()).ok()
    }

    fn selected_model(&self) -> Option<pi_ai::types::Model> {
        let reference = self.agent_state().and_then(|state| state.model)?;
        let runtime = self.model_runtime.as_ref()?;
        runtime.get_model(&reference.provider, &reference.model_id)
    }

    fn read_thinking_levels(&self) -> Vec<ModelThinkingLevel> {
        match self.selected_model() {
            Some(model) => get_supported_thinking_levels(&model),
            None => vec![ModelThinkingLevel::Off],
        }
    }

    fn read_catalog(&self) -> ModelsCatalog {
        let selected = self.selected_model();
        let available = self
            .model_runtime
            .as_ref()
            .map(|runtime| runtime.get_available_snapshot())
            .unwrap_or_default();
        let mut models: Vec<pi_ai::types::Model> = available;
        if let Some(selected) = &selected {
            if !models.iter().any(|model| {
                model.base.provider == selected.base.provider && model.base.id == selected.base.id
            }) {
                models.push(selected.clone());
            }
        }
        let revision = self.catalog_revision.fetch_add(1, Ordering::SeqCst) + 1;
        ModelsCatalog {
            revision,
            available_models: models
                .into_iter()
                .map(|model| ModelSummary {
                    provider: model.base.provider,
                    model_id: model.base.id,
                    name: model.base.name,
                    reasoning: model.reasoning,
                })
                .collect(),
        }
    }

    fn change(&self, context: &Context, change: impl FnOnce(&mut ModelsState)) {
        let mut current: ModelsState = self.read_state().unwrap_or_else(|| ModelsState {
            catalog: ModelsCatalog {
                revision: 0,
                available_models: Vec::new(),
            },
            configuration: ModelsConfiguration {
                model: None,
                thinking_level: ModelThinkingLevel::Off,
            },
            refresh: ModelsRefreshState::Idle,
        });
        change(&mut current);
        let next = serde_json::to_value(&current).expect("ModelsState serializes");
        let _ = self.state.replace(context, next);
    }

    /// Upstream `Models.cycleThinking(context)`.
    pub async fn cycle_thinking(&self, context: Context) -> Result<(), String> {
        let levels = self.read_thinking_levels();
        let current = self
            .agent_state()
            .and_then(|state| state.thinking_level)
            .unwrap_or(ModelThinkingLevel::Off);
        let index = levels
            .iter()
            .position(|level| *level == current)
            .map_or(0, |index| index + 1);
        let next = levels
            .get(index % levels.len().max(1))
            .copied()
            .unwrap_or(ModelThinkingLevel::Off);
        self.conversation
            .configure(
                AgentChange {
                    thinking_level: Some(SetOrClear::Set(next)),
                    ..AgentChange::default()
                },
                &context,
            )
            .await
    }

    /// Upstream `Models.getThinkingLevels(context)`.
    pub fn get_thinking_levels(&self) -> Vec<ModelThinkingLevel> {
        self.read_thinking_levels()
    }

    /// Upstream `Models.refresh(context)`.
    pub async fn refresh(&self, context: Context) -> Result<(), String> {
        self.change(&context, |draft| {
            draft.refresh = ModelsRefreshState::Refreshing
        });
        let errors: BTreeMap<String, String> = match &self.model_runtime {
            Some(runtime) => runtime
                .refresh(None)
                .await
                .errors
                .into_iter()
                .map(|(id, thrown)| (id, thrown_value_text(thrown)))
                .collect(),
            None => BTreeMap::new(),
        };
        let catalog = self.read_catalog();
        self.change(&context, |draft| {
            draft.catalog = catalog;
            draft.refresh = if errors.is_empty() {
                ModelsRefreshState::Done
            } else {
                ModelsRefreshState::Warning { errors }
            };
        });
        Ok(())
    }

    /// Upstream `Models.select(model, context)`.
    pub async fn select(&self, model: DtoModelRef, context: Context) -> Result<(), String> {
        let selected = self
            .model_runtime
            .as_ref()
            .and_then(|runtime| runtime.get_model(&model.provider, &model.model_id));
        let Some(selected) = selected else {
            return Err(format!(
                "Unknown model: {}/{}",
                model.provider, model.model_id
            ));
        };
        let thinking_level = clamp_thinking_level(
            &selected,
            self.agent_state()
                .and_then(|state| state.thinking_level)
                .unwrap_or(ModelThinkingLevel::Off),
        );
        self.conversation
            .configure(
                AgentChange {
                    model: Some(SetOrClear::Set(ModelRef {
                        provider: selected.base.provider.clone(),
                        model_id: selected.base.id.clone(),
                    })),
                    thinking_level: Some(SetOrClear::Set(thinking_level)),
                    ..AgentChange::default()
                },
                &context,
            )
            .await?;
        if let Some(settings) = &self.settings_manager {
            let mut settings = settings.lock();
            settings.set_default_model_and_provider(&selected.base.provider, &selected.base.id);
            settings.flush();
        }
        Ok(())
    }

    /// Upstream `Models.selectThinking(level, context)`.
    pub async fn select_thinking(
        &self,
        level: ModelThinkingLevel,
        context: Context,
    ) -> Result<(), String> {
        let levels = self.read_thinking_levels();
        if !levels.contains(&level) {
            let joined = levels
                .iter()
                .map(|level| thinking_level_name(*level))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "Thinking level {} is unavailable; choose one of: {joined}",
                thinking_level_name(level)
            ));
        }
        self.conversation
            .configure(
                AgentChange {
                    thinking_level: Some(SetOrClear::Set(level)),
                    ..AgentChange::default()
                },
                &context,
            )
            .await
    }

    /// Upstream `runtime.activate(context)`.
    pub async fn activate(&self, context: Context) -> Result<(), String> {
        let catalog = self.read_catalog();
        self.change(&context, |draft| {
            draft.catalog = catalog;
            draft.refresh = ModelsRefreshState::Idle;
        });
        Ok(())
    }

    /// Upstream `runtime.syncConfiguration(context)` — publish the agent
    /// document's model and thinking level when they changed.
    pub fn sync_configuration(&self, context: Context) {
        let next = Self::configuration_of(self.agent_state().as_ref());
        let current = self.configuration();
        if current == next {
            return;
        }
        self.change(&context, |draft| draft.configuration = next);
    }
}

/// Wire name of a thinking level (upstream string literals).
fn thinking_level_name(level: ModelThinkingLevel) -> String {
    match level {
        ModelThinkingLevel::Off => "off",
        ModelThinkingLevel::Minimal => "minimal",
        ModelThinkingLevel::Low => "low",
        ModelThinkingLevel::Medium => "medium",
        ModelThinkingLevel::High => "high",
        ModelThinkingLevel::Xhigh => "xhigh",
        ModelThinkingLevel::Max => "max",
    }
    .to_string()
}

/// Upstream `error.message` for refresh errors.
fn thrown_value_text(thrown: ThrownValue) -> String {
    match thrown {
        ThrownValue::Text(text) => text,
        ThrownValue::Error(error) => error.to_string(),
    }
}

/// Remote facade members of the `pi.models` service (upstream object keys).
pub mod members {
    pub const STATE: &str = "state";
    pub const CYCLE_THINKING: &str = "cycleThinking";
    pub const GET_THINKING_LEVELS: &str = "getThinkingLevels";
    pub const REFRESH: &str = "refresh";
    pub const SELECT: &str = "select";
    pub const SELECT_THINKING: &str = "selectThinking";
}

/// Build the remote singleton implementation for the Models service.
pub fn models_service_object(runtime: &Arc<ModelsService>) -> Arc<RemoteServiceObject> {
    let cycle = Arc::clone(runtime);
    let levels = Arc::clone(runtime);
    let refresh = Arc::clone(runtime);
    let select = Arc::clone(runtime);
    let select_thinking = Arc::clone(runtime);
    RemoteServiceObject::builder()
        .state(members::STATE, runtime.state().internals())
        .method(members::CYCLE_THINKING, move |_args, context| {
            let runtime = Arc::clone(&cycle);
            Box::pin(async move {
                runtime.cycle_thinking(context).await?;
                Ok(None)
            })
        })
        .method(members::GET_THINKING_LEVELS, move |_args, _context| {
            let runtime = Arc::clone(&levels);
            Box::pin(async move {
                let names = runtime
                    .get_thinking_levels()
                    .into_iter()
                    .map(|level| JsonValue::String(thinking_level_name(level)))
                    .collect();
                Ok(Some(JsonValue::Array(names)))
            })
        })
        .method(members::REFRESH, move |_args, context| {
            let runtime = Arc::clone(&refresh);
            Box::pin(async move {
                runtime.refresh(context).await?;
                Ok(None)
            })
        })
        .method(members::SELECT, move |args, context| {
            let runtime = Arc::clone(&select);
            Box::pin(async move {
                let model = decode_arg::<DtoModelRef>(&args, members::SELECT)?;
                runtime.select(model, context).await?;
                Ok(None)
            })
        })
        .method(members::SELECT_THINKING, move |args, context| {
            let runtime = Arc::clone(&select_thinking);
            Box::pin(async move {
                let level = decode_arg::<ModelThinkingLevel>(&args, members::SELECT_THINKING)?;
                runtime.select_thinking(level, context).await?;
                Ok(None)
            })
        })
        .build()
}

fn decode_arg<T: serde::de::DeserializeOwned>(
    args: &[JsonValue],
    member: &str,
) -> Result<T, AnyError> {
    let value = args.first().cloned().unwrap_or(JsonValue::Null);
    serde_json::from_value(value).map_err(|error| type_error(format!("{member}: {error}")))
}

/// Upstream `createModelsServiceFacet` — acquire the conversation's agent
/// document, then build the facet that owns it.
pub async fn create_models_service_facet(
    options: ModelsServiceFacetOptions,
) -> Result<Facet, String> {
    let agent = options
        .harness
        .session
        .document_state(
            &agent_doc(),
            Some(options.conversation.id.0),
            None,
            &options.context(),
        )
        .await
        .map_err(|error| error.to_string())?;
    let Some(agent) = agent else {
        return Err(format!(
            "Conversation {} has no agent document",
            options.conversation.id
        ));
    };
    let initial = ModelsState {
        catalog: ModelsCatalog {
            revision: 0,
            available_models: Vec::new(),
        },
        configuration: ModelsService::configuration_of(
            serde_json::from_value(agent.value().clone()).ok().as_ref(),
        ),
        refresh: ModelsRefreshState::Idle,
    };
    let runtime = ModelsService::new(
        Arc::clone(&options.conversation),
        agent.clone(),
        options.model_runtime,
        options.settings_manager,
        initial,
    );
    let object = models_service_object(&runtime);
    let sync_runtime = Arc::clone(&runtime);
    let dispose_agent = agent.clone();
    Ok(pi_chord::api::define_facet(pi_chord::types::Facet::new(
        "@pi/models",
        move |env| {
            env.own({
                let dispose_agent = dispose_agent.clone();
                Arc::new(move || {
                    dispose_agent.dispose();
                    Box::pin(async { Ok(()) })
                })
            })?;
            env.provide_remote(
                super::models::MODELS.id(),
                ServiceMode::Singleton,
                object.clone(),
            )?;
            env.on_activate({
                let runtime = Arc::clone(&sync_runtime);
                let agent = dispose_agent.clone();
                Arc::new(move || {
                    let runtime = Arc::clone(&runtime);
                    let agent = agent.clone();
                    Box::pin(async move {
                        runtime.activate(BACKGROUND_CONTEXT.clone()).await?;
                        let unsubscribe = agent.subscribe({
                            let runtime = Arc::clone(&runtime);
                            Arc::new(move |_value, context, _delivery| {
                                runtime.sync_configuration(context.clone());
                                Box::pin(async {})
                            })
                        });
                        let _ = disposal_from_unsubscribe(unsubscribe);
                        Ok(())
                    })
                })
            })?;
            Ok(())
        },
    )))
}

/// Upstream `createModelsServiceFacet` options.
pub struct ModelsServiceFacetOptions {
    pub harness: Arc<pi_durable::harness::harness::Harness>,
    pub conversation: Arc<Conversation>,
    pub model_runtime: Option<Arc<ModelRuntime>>,
    pub settings_manager: Option<SharedSettingsManager>,
    /// Context used to acquire the agent document (upstream `context`).
    pub context: Context,
}

impl ModelsServiceFacetOptions {
    fn context(&self) -> Context {
        self.context.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_state_value(state: &AgentState) -> JsonValue {
        serde_json::to_value(state).unwrap()
    }

    #[test]
    fn configuration_of_maps_agent_state() {
        let state = AgentState {
            model: Some(ModelRef {
                provider: "p".into(),
                model_id: "m".into(),
            }),
            thinking_level: Some(ModelThinkingLevel::High),
            ..AgentState::default()
        };
        let configuration = ModelsService::configuration_of(Some(&state));
        assert_eq!(configuration.model.as_ref().unwrap().provider, "p");
        assert_eq!(configuration.thinking_level, ModelThinkingLevel::High);
        assert_eq!(
            serde_json::to_value(&configuration).unwrap()["thinkingLevel"],
            "high"
        );
        let empty = ModelsService::configuration_of(None);
        assert_eq!(empty.model, None);
        assert_eq!(empty.thinking_level, ModelThinkingLevel::Off);
    }

    #[test]
    fn agent_state_round_trips_through_json() {
        let state = AgentState::default();
        let value = agent_state_value(&state);
        let parsed: AgentState = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, AgentState::default());
    }
}
