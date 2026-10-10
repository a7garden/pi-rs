//! Port of `packages/coding-agent/src/core/model-config.ts` — the immutable,
//! credential-blind `models.json` snapshot ([`ModelConfig`]) and its data
//! types ([`ModelsJsonProvider`], [`ModelsJsonModel`],
//! [`ModelsJsonModelOverride`]) consumed by `provider-composer.ts`.
//!
//! # Deviations
//!
//! - **TypeBox validation becomes serde + explicit checks.** The upstream
//!   TypeBox schemas compile to one validator whose errors are rendered as
//!   `  - <path>: <message>` lines; the port deserializes with serde and
//!   checks the `minLength: 1` / `oauth: "radius"` constraints afterwards,
//!   producing the same error bucket (`Invalid models.json schema:` …) with
//!   serde-style messages. TypeBox `Type.Number()` fields the ported data
//!   model carries as `u64` (context windows, token counts, cache
//!   lifetimes, image limits) reject floats/negatives where upstream would
//!   pass them through. The `ProviderCompat` union becomes pi-ai's merged
//!   [`pi_ai::types::ModelCompat`] (pi-ai deviation D7 precedent), so its
//!   deeply-variant routing fields accept the same JSON with looser
//!   validation.
//! - **Immutability by sharing.** `deepFreeze(structuredClone(provider))`
//!   becomes plain owned storage; [`ModelConfig::get_provider`] hands out
//!   shared references, which are immutable.
//! - **Sync file IO.** `readFile` becomes [`std::fs::read_to_string`]
//!   (settings-manager D20 precedent).
//! - **Provider order.** `Object.entries` insertion order is preserved with
//!   an [`IndexMap`] (upstream `Map`).

use std::collections::BTreeMap;

use indexmap::IndexMap;
use pi_ai::types::{
    ModelCompat, ModelCost, ModelCostTier, ModelInput, ModelInputLimits, ModelPromptCache,
    SamplingParams, SamplingParamsByThinkingLevel, ThinkingLevelMap,
};
use serde::Deserialize;

use crate::utils::paths::{normalize_path, PathInputOptions};
use crate::utils::text::{strip_bom, strip_json_comments};

/// A model defined under `providers.<id>.models[]` (`ModelsJsonModel`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsJsonModel {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<ModelInput>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<ModelInputLimits>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache: Option<ModelPromptCache>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<SamplingParams>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params_by_thinking_level: Option<SamplingParamsByThinkingLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompat>,
}

/// Partial per-rate cost override (`ModelOverrideSchema.cost`).
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsJsonCostOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<f64>,
    #[serde(rename = "cacheRead", default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    #[serde(
        rename = "cacheWrite",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_write: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiers: Option<Vec<ModelCostTier>>,
}

/// A per-model customization under `providers.<id>.modelOverrides`
/// (`ModelsJsonModelOverride`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsJsonModelOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<ModelInput>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<ModelInputLimits>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelsJsonCostOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache: Option<ModelPromptCache>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<SamplingParams>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params_by_thinking_level: Option<SamplingParamsByThinkingLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompat>,
}

/// A provider entry under `providers` (`ModelsJsonProvider`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsJsonProvider {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// Only the literal `"radius"` passes upstream validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oauth: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<ModelsJsonModel>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_overrides: Option<BTreeMap<String, ModelsJsonModelOverride>>,
}

#[derive(Debug, Deserialize)]
struct ModelsConfig {
    providers: IndexMap<String, ModelsJsonProvider>,
}

/// Schema constraints serde cannot express (`minLength: 1`, the `"radius"`
/// literal); rendered like upstream validator errors.
fn validate_config(config: &ModelsConfig) -> Vec<String> {
    let mut errors = Vec::new();
    for (provider_id, provider) in &config.providers {
        let base = format!("providers.{provider_id}");
        check_min_length(&mut errors, &base, "name", provider.name.as_deref());
        check_min_length(&mut errors, &base, "baseUrl", provider.base_url.as_deref());
        check_min_length(&mut errors, &base, "apiKey", provider.api_key.as_deref());
        check_min_length(&mut errors, &base, "api", provider.api.as_deref());
        if let Some(oauth) = &provider.oauth {
            if oauth != "radius" {
                errors.push(format!("{base}.oauth: invalid enum value"));
            }
        }
        for (index, model) in provider.models.iter().flatten().enumerate() {
            let base = format!("{base}.models.{index}");
            check_min_length(&mut errors, &base, "id", Some(&model.id));
            check_min_length(&mut errors, &base, "name", model.name.as_deref());
            check_min_length(&mut errors, &base, "api", model.api.as_deref());
            check_min_length(&mut errors, &base, "baseUrl", model.base_url.as_deref());
        }
    }
    errors
}

fn check_min_length(errors: &mut Vec<String>, base: &str, field: &str, value: Option<&str>) {
    if value.is_some_and(str::is_empty) {
        errors.push(format!("{base}.{field}: must not be empty"));
    }
}

/// One immutable load of models.json (`ModelConfig`). Errors from the last
/// load are carried in the instance (`getError`) instead of failing the
/// constructor.
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    providers: IndexMap<String, ModelsJsonProvider>,
    error: Option<String>,
}

impl ModelConfig {
    /// Load and validate a models.json snapshot (`ModelConfig.load`).
    /// `None` or a missing file yields an empty config; parse/read/schema
    /// failures yield an empty config with [`ModelConfig::get_error`] set.
    pub fn load(models_json_path: Option<&str>) -> ModelConfig {
        let Some(models_json_path) = models_json_path else {
            return ModelConfig::default();
        };
        let path = normalize_path(models_json_path, &PathInputOptions::default());
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return ModelConfig::default()
            }
            Err(error) => {
                return ModelConfig {
                    providers: IndexMap::new(),
                    error: Some(format!(
                        "Failed to load models.json: {error}\n\nFile: {path}"
                    )),
                };
            }
        };

        let parsed: serde_json::Result<serde_json::Value> =
            serde_json::from_str(&strip_json_comments(&strip_bom(&content)));
        let parsed = match parsed {
            Ok(parsed) => parsed,
            Err(error) => {
                return ModelConfig {
                    providers: IndexMap::new(),
                    error: Some(format!(
                        "Failed to parse models.json: {error}\n\nFile: {path}"
                    )),
                };
            }
        };

        let config: Result<ModelsConfig, _> = serde_json::from_value(parsed);
        let config = match config {
            Ok(config) => config,
            Err(error) => {
                return ModelConfig {
                    providers: IndexMap::new(),
                    error: Some(format!(
                        "Invalid models.json schema:\n  - {error}\n\nFile: {path}"
                    )),
                };
            }
        };

        let errors = validate_config(&config);
        if !errors.is_empty() {
            return ModelConfig {
                providers: IndexMap::new(),
                error: Some(format!(
                    "Invalid models.json schema:\n{}\n\nFile: {path}",
                    errors
                        .iter()
                        .map(|error| format!("  - {error}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                )),
            };
        }

        ModelConfig {
            providers: config.providers,
            error: None,
        }
    }

    /// The provider entry for `providerId` (`getProvider`).
    pub fn get_provider(&self, provider_id: &str) -> Option<&ModelsJsonProvider> {
        self.providers.get(provider_id)
    }

    /// Provider ids in document order (`getProviderIds`).
    pub fn get_provider_ids(&self) -> Vec<&str> {
        self.providers.keys().map(String::as_str).collect()
    }

    /// The load error, if any (`getError`).
    pub fn get_error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

#[cfg(test)]
mod model_config_tests {
    use super::*;

    #[test]
    fn empty_path_yields_empty_config() {
        let config = ModelConfig::load(None);
        assert_eq!(config.get_provider_ids(), Vec::<&str>::new());
        assert_eq!(config.get_error(), None);
    }

    #[test]
    fn missing_file_yields_empty_config() {
        let config = ModelConfig::load(Some("/nonexistent/pi/models.json"));
        assert_eq!(config.get_provider_ids(), Vec::<&str>::new());
        assert_eq!(config.get_error(), None);
    }

    #[test]
    fn malformed_json_reports_parse_error_with_file_path() {
        let dir = std::env::temp_dir().join(format!("pi-model-config-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("malformed.json");
        std::fs::write(&path, "{\n  \"providers\": {\n").unwrap();
        let config = ModelConfig::load(Some(path.to_str().unwrap()));
        let error = config.get_error().unwrap();
        assert!(error.contains("Failed to parse models.json"), "{error}");
        assert!(
            error.contains(&format!("File: {}", path.display())),
            "{error}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn schema_violations_report_invalid_schema() {
        let dir = std::env::temp_dir().join(format!("pi-model-config-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("invalid.json");
        std::fs::write(
            &path,
            r#"{"providers": {"custom": {"models": [{"id": ""}]}}}"#,
        )
        .unwrap();
        let config = ModelConfig::load(Some(path.to_str().unwrap()));
        let error = config.get_error().unwrap();
        assert!(error.contains("Invalid models.json schema"), "{error}");
        assert!(error.contains("providers.custom.models.0.id"), "{error}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn loads_providers_in_document_order() {
        let dir = std::env::temp_dir().join(format!("pi-model-config-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ordered.json");
        std::fs::write(
			&path,
			r#"{"providers": {"zeta": {"baseUrl": "https://z.example.com"}, "alpha": {"baseUrl": "https://a.example.com"}}}"#,
		)
		.unwrap();
        let config = ModelConfig::load(Some(path.to_str().unwrap()));
        assert_eq!(config.get_provider_ids(), vec!["zeta", "alpha"]);
        assert_eq!(
            config.get_provider("alpha").unwrap().base_url.as_deref(),
            Some("https://a.example.com")
        );
        assert_eq!(config.get_error(), None);
        let _ = std::fs::remove_file(&path);
    }
}
