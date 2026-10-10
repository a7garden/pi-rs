//! Port of `experimental/services/models.ts` — session-scoped model catalog /
//! configuration DTOs and the remote service token.
//!
//! Deviations (D144):
//! - `ReplicatedState<ModelsState>` becomes chord's untyped
//!   [`pi_chord::MutableReplicatedState`] over the serialized [`ModelsState`].
//! - Upstream `ModelThinkingLevel` maps to [`pi_ai::types::ModelThinkingLevel`]
//!   with its wire names (`off`/`minimal`/…/`xhigh`/`max`).

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use pi_ai::types::ModelThinkingLevel;
use pi_chord::api::define_service;

/// Upstream `ModelRef`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef {
    pub provider: String,
    pub model_id: String,
}

/// Upstream `ModelSummary`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSummary {
    pub provider: String,
    pub model_id: String,
    pub name: String,
    pub reasoning: bool,
}

/// Upstream `ModelsState.refresh` union.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum ModelsRefreshState {
    Idle,
    Refreshing,
    Done,
    Warning {
        errors: std::collections::BTreeMap<String, String>,
    },
}

/// Upstream `ModelsState`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsState {
    pub catalog: ModelsCatalog,
    pub configuration: ModelsConfiguration,
    pub refresh: ModelsRefreshState,
}

/// Upstream `ModelsState["catalog"]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsCatalog {
    pub revision: u64,
    pub available_models: Vec<ModelSummary>,
}

/// Upstream `ModelsState["configuration"]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsConfiguration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelRef>,
    pub thinking_level: ModelThinkingLevel,
}

/// `defineService<Models>("pi.models")`; provided remotely by the Session
/// worker's models facet and consumed remotely by presentations (D144 marker
/// token).
pub struct ModelsToken;

pub static MODELS: LazyLock<pi_chord::Service<ModelsToken>> =
    LazyLock::new(|| define_service("pi.models", false).expect("service id is valid"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_state_serializes_camel_case() {
        let state = ModelsState {
            catalog: ModelsCatalog {
                revision: 1,
                available_models: vec![],
            },
            configuration: ModelsConfiguration {
                model: Some(ModelRef {
                    provider: "p".into(),
                    model_id: "m".into(),
                }),
                thinking_level: ModelThinkingLevel::Off,
            },
            refresh: ModelsRefreshState::Idle,
        };
        let value = serde_json::to_value(&state).unwrap();
        assert_eq!(value["catalog"]["availableModels"], serde_json::json!([]));
        assert_eq!(value["configuration"]["thinkingLevel"], "off");
        assert_eq!(value["configuration"]["model"]["modelId"], "m");
        assert_eq!(value["refresh"]["status"], "idle");
    }

    #[test]
    fn refresh_warning_carries_errors() {
        let state = ModelsRefreshState::Warning {
            errors: [("p".to_string(), "boom".to_string())]
                .into_iter()
                .collect(),
        };
        let value = serde_json::to_value(&state).unwrap();
        assert_eq!(value["status"], "warning");
        assert_eq!(value["errors"]["p"], "boom");
    }
}
