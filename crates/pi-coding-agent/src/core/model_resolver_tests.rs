//! Port of `test/model-resolver.test.ts` (1:1 for the suites that do not
//! construct a full `AgentSession`): `parseModelPattern`,
//! `resolveModelScopeWithDiagnostics`, `resolveCliModel`, and
//! `default model selection`. The "persisted default model scoping" suite
//! stays unported (AgentSession + AuthStorage + ModelRuntime); the
//! `defaultModelPerProvider` catalog test uses pi-ai's generated catalog.

use pi_agent::AgentThinkingLevel;
use pi_ai::compat::{get_builtin_models, get_builtin_providers};

use super::{
    default_model_for_provider, find_initial_model, parse_model_pattern, resolve_cli_model,
    resolve_model_scope_with_diagnostics, FindInitialModelOptions, ModelRuntimeSource,
    ResolveCliModelOptions,
};

fn level(name: &str) -> AgentThinkingLevel {
    match name {
        "off" => AgentThinkingLevel::Off,
        "minimal" => AgentThinkingLevel::Minimal,
        "low" => AgentThinkingLevel::Low,
        "medium" => AgentThinkingLevel::Medium,
        "high" => AgentThinkingLevel::High,
        "xhigh" => AgentThinkingLevel::Xhigh,
        "max" => AgentThinkingLevel::Max,
        _ => panic!("unknown level {name}"),
    }
}

fn chat_model(
    id: &str,
    name: &str,
    provider: &str,
    base_url: &str,
    reasoning: bool,
    input_image: bool,
) -> pi_ai::types::Model {
    pi_ai::types::Model {
        base: pi_ai::types::BaseModelFields {
            id: id.to_string(),
            name: name.to_string(),
            api: "anthropic-messages".to_string(),
            provider: provider.to_string(),
            base_url: base_url.to_string(),
            input: if input_image {
                vec![
                    pi_ai::types::ModelInput::Text,
                    pi_ai::types::ModelInput::Image,
                ]
            } else {
                vec![pi_ai::types::ModelInput::Text]
            },
            input_limits: None,
            cost: pi_ai::types::ModelCost {
                rates: pi_ai::types::ModelCostRates {
                    input: if provider == "openai" { 5.0 } else { 1.0 },
                    output: 15.0,
                    cache_read: if provider == "openai" { 0.5 } else { 0.1 },
                    cache_write: if provider == "openai" { 5.0 } else { 1.0 },
                },
                tiers: None,
            },
            headers: None,
        },
        model_type: None,
        reasoning,
        thinking_level_map: None,
        prompt_cache: None,
        context_window: if provider == "anthropic" {
            200000
        } else {
            128000
        },
        max_tokens: if id == "gpt-4o" { 4096 } else { 8192 },
        sampling_params: None,
        sampling_params_by_thinking_level: None,
        compat: None,
    }
}

// Mock models for testing (upstream `mockModels`).
fn mock_models() -> Vec<pi_ai::types::Model> {
    vec![
        chat_model(
            "claude-sonnet-4-5",
            "Claude Sonnet 4.5",
            "anthropic",
            "https://api.anthropic.com",
            true,
            true,
        ),
        chat_model(
            "gpt-4o",
            "GPT-4o",
            "openai",
            "https://api.openai.com",
            false,
            true,
        ),
    ]
}

// Mock OpenRouter models with colons in IDs (upstream `mockOpenRouterModels`).
fn mock_openrouter_models() -> Vec<pi_ai::types::Model> {
    vec![
        chat_model(
            "qwen/qwen3-coder:exacto",
            "Qwen3 Coder Exacto",
            "openrouter",
            "https://openrouter.ai/api/v1",
            true,
            false,
        ),
        chat_model(
            "openai/gpt-4o:extended",
            "GPT-4o Extended",
            "openrouter",
            "https://openrouter.ai/api/v1",
            false,
            true,
        ),
    ]
}

fn all_models() -> Vec<pi_ai::types::Model> {
    let mut models = mock_models();
    models.extend(mock_openrouter_models());
    models
}

/// Duck-typed fake registry mirroring the upstream test stubs: only the
/// members a test provides are meaningful.
#[derive(Default)]
struct FakeRuntime {
    models: Vec<pi_ai::types::Model>,
    available: Option<Vec<pi_ai::types::Model>>,
    available_snapshot: Option<Vec<pi_ai::types::Model>>,
    authenticated_providers: Vec<String>,
    model_lookup: Option<Box<dyn Fn(&str, &str) -> Option<pi_ai::types::Model> + Send + Sync>>,
}

impl FakeRuntime {
    fn of(models: Vec<pi_ai::types::Model>) -> Self {
        FakeRuntime {
            models,
            ..FakeRuntime::default()
        }
    }

    fn with_available(mut self, available: Vec<pi_ai::types::Model>) -> Self {
        self.available = Some(available);
        self
    }

    fn with_snapshot(mut self, snapshot: Vec<pi_ai::types::Model>) -> Self {
        self.available_snapshot = Some(snapshot);
        self
    }

    fn with_auth(mut self, providers: &[&str]) -> Self {
        self.authenticated_providers = providers.iter().map(|p| p.to_string()).collect();
        self
    }

    fn with_lookup(
        mut self,
        lookup: impl Fn(&str, &str) -> Option<pi_ai::types::Model> + Send + Sync + 'static,
    ) -> Self {
        self.model_lookup = Some(Box::new(lookup));
        self
    }
}

impl ModelRuntimeSource for FakeRuntime {
    fn get_models(&self) -> Vec<pi_ai::types::Model> {
        self.models.clone()
    }

    fn get_available_snapshot(&self) -> Vec<pi_ai::types::Model> {
        self.available_snapshot
            .clone()
            .unwrap_or_else(|| self.available.clone().unwrap_or_default())
    }

    fn get_available<'a>(
        &'a self,
        _provider: Option<&'a str>,
        _options: Option<pi_ai::auth::types::AuthOperationOptions>,
    ) -> super::BoxFuture<'a, Result<Vec<pi_ai::types::Model>, pi_ai::utils::thrown::ThrownValue>>
    {
        Box::pin(async move {
            Ok(self
                .available
                .clone()
                .unwrap_or_else(|| self.models.clone()))
        })
    }

    fn get_model(&self, provider: &str, model_id: &str) -> Option<pi_ai::types::Model> {
        if let Some(lookup) = &self.model_lookup {
            return lookup(provider, model_id);
        }
        self.models
            .iter()
            .find(|m| m.provider() == provider && m.id() == model_id)
            .cloned()
    }

    fn has_configured_auth(&self, provider: &str) -> bool {
        self.authenticated_providers.iter().any(|p| p == provider)
    }
}

// ==== parseModelPattern ====

#[test]
fn parse_exact_match_returns_model_with_undefined_thinking_level() {
    let result = parse_model_pattern("claude-sonnet-4-5", &all_models(), None);
    assert_eq!(result.model.as_ref().unwrap().id(), "claude-sonnet-4-5");
    assert_eq!(result.thinking_level, None);
    assert_eq!(result.warning, None);
}

#[test]
fn parse_partial_match_returns_best_model_with_undefined_thinking_level() {
    let result = parse_model_pattern("sonnet", &all_models(), None);
    assert_eq!(result.model.as_ref().unwrap().id(), "claude-sonnet-4-5");
    assert_eq!(result.thinking_level, None);
    assert_eq!(result.warning, None);
}

#[test]
fn parse_no_match_returns_undefined_model_and_thinking_level() {
    let result = parse_model_pattern("nonexistent", &all_models(), None);
    assert_eq!(result.model, None);
    assert_eq!(result.thinking_level, None);
    assert_eq!(result.warning, None);
}

#[test]
fn parse_sonnet_high_returns_sonnet_with_high_thinking_level() {
    let result = parse_model_pattern("sonnet:high", &all_models(), None);
    assert_eq!(result.model.as_ref().unwrap().id(), "claude-sonnet-4-5");
    assert_eq!(result.thinking_level, Some(AgentThinkingLevel::High));
    assert_eq!(result.warning, None);
}

#[test]
fn parse_gpt4o_medium_returns_gpt4o_with_medium_thinking_level() {
    let result = parse_model_pattern("gpt-4o:medium", &all_models(), None);
    assert_eq!(result.model.as_ref().unwrap().id(), "gpt-4o");
    assert_eq!(result.thinking_level, Some(AgentThinkingLevel::Medium));
    assert_eq!(result.warning, None);
}

#[test]
fn parse_all_valid_thinking_levels_work() {
    for name in ["off", "minimal", "low", "medium", "high", "xhigh", "max"] {
        let result = parse_model_pattern(&format!("sonnet:{name}"), &all_models(), None);
        assert_eq!(
            result.model.as_ref().unwrap().id(),
            "claude-sonnet-4-5",
            "level {name}"
        );
        assert_eq!(result.thinking_level, Some(level(name)), "level {name}");
        assert_eq!(result.warning, None, "level {name}");
    }
}

#[test]
fn parse_sonnet_random_returns_sonnet_with_warning() {
    let result = parse_model_pattern("sonnet:random", &all_models(), None);
    assert_eq!(result.model.as_ref().unwrap().id(), "claude-sonnet-4-5");
    assert_eq!(result.thinking_level, None);
    assert!(result
        .warning
        .as_deref()
        .unwrap()
        .contains("Invalid thinking level"));
    assert!(result.warning.as_deref().unwrap().contains("random"));
}

#[test]
fn parse_gpt4o_invalid_returns_gpt4o_with_warning() {
    let result = parse_model_pattern("gpt-4o:invalid", &all_models(), None);
    assert_eq!(result.model.as_ref().unwrap().id(), "gpt-4o");
    assert_eq!(result.thinking_level, None);
    assert!(result
        .warning
        .as_deref()
        .unwrap()
        .contains("Invalid thinking level"));
}

#[test]
fn parse_openrouter_exacto_matches_the_model() {
    let result = parse_model_pattern("qwen/qwen3-coder:exacto", &all_models(), None);
    assert_eq!(
        result.model.as_ref().unwrap().id(),
        "qwen/qwen3-coder:exacto"
    );
    assert_eq!(result.thinking_level, None);
    assert_eq!(result.warning, None);
}

#[test]
fn parse_openrouter_exacto_matches_with_provider_prefix() {
    let result = parse_model_pattern("openrouter/qwen/qwen3-coder:exacto", &all_models(), None);
    assert_eq!(
        result.model.as_ref().unwrap().id(),
        "qwen/qwen3-coder:exacto"
    );
    assert_eq!(result.model.as_ref().unwrap().provider(), "openrouter");
    assert_eq!(result.thinking_level, None);
    assert_eq!(result.warning, None);
}

#[test]
fn parse_exacto_high_matches_model_with_high_thinking_level() {
    let result = parse_model_pattern("qwen/qwen3-coder:exacto:high", &all_models(), None);
    assert_eq!(
        result.model.as_ref().unwrap().id(),
        "qwen/qwen3-coder:exacto"
    );
    assert_eq!(result.thinking_level, Some(AgentThinkingLevel::High));
    assert_eq!(result.warning, None);
}

#[test]
fn parse_exacto_high_matches_with_provider_and_thinking_level() {
    let result = parse_model_pattern(
        "openrouter/qwen/qwen3-coder:exacto:high",
        &all_models(),
        None,
    );
    assert_eq!(
        result.model.as_ref().unwrap().id(),
        "qwen/qwen3-coder:exacto"
    );
    assert_eq!(result.model.as_ref().unwrap().provider(), "openrouter");
    assert_eq!(result.thinking_level, Some(AgentThinkingLevel::High));
    assert_eq!(result.warning, None);
}

#[test]
fn parse_gpt4o_extended_matches_the_extended_model() {
    let result = parse_model_pattern("openai/gpt-4o:extended", &all_models(), None);
    assert_eq!(
        result.model.as_ref().unwrap().id(),
        "openai/gpt-4o:extended"
    );
    assert_eq!(result.thinking_level, None);
    assert_eq!(result.warning, None);
}

#[test]
fn parse_exacto_random_returns_model_with_warning() {
    let result = parse_model_pattern("qwen/qwen3-coder:exacto:random", &all_models(), None);
    assert_eq!(
        result.model.as_ref().unwrap().id(),
        "qwen/qwen3-coder:exacto"
    );
    assert_eq!(result.thinking_level, None);
    assert!(result
        .warning
        .as_deref()
        .unwrap()
        .contains("Invalid thinking level"));
    assert!(result.warning.as_deref().unwrap().contains("random"));
}

#[test]
fn parse_exacto_high_random_returns_model_with_warning() {
    let result = parse_model_pattern("qwen/qwen3-coder:exacto:high:random", &all_models(), None);
    assert_eq!(
        result.model.as_ref().unwrap().id(),
        "qwen/qwen3-coder:exacto"
    );
    assert_eq!(result.thinking_level, None);
    assert!(result
        .warning
        .as_deref()
        .unwrap()
        .contains("Invalid thinking level"));
    assert!(result.warning.as_deref().unwrap().contains("random"));
}

#[test]
fn parse_empty_pattern_matches_via_partial_matching() {
    // Empty string is included in all model IDs, so partial matching finds a match
    let result = parse_model_pattern("", &all_models(), None);
    assert!(result.model.is_some());
    assert_eq!(result.thinking_level, None);
}

#[test]
fn parse_trailing_colon_treats_empty_suffix_as_invalid() {
    let result = parse_model_pattern("sonnet:", &all_models(), None);
    // Empty string after colon is not a valid thinking level
    // So it tries to match "sonnet:" which won't match, then tries "sonnet"
    assert_eq!(result.model.as_ref().unwrap().id(), "claude-sonnet-4-5");
    assert!(result
        .warning
        .as_deref()
        .unwrap()
        .contains("Invalid thinking level"));
}

// ==== resolveModelScopeWithDiagnostics ====

#[tokio::test]
async fn scope_returns_scoped_models_and_structured_diagnostics_without_console_warnings() {
    let runtime = FakeRuntime::of(all_models()).with_available(all_models());
    let patterns: Vec<String> = ["sonnet:high", "gpt-4o:invalid", "missing"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let result = resolve_model_scope_with_diagnostics(&patterns, &runtime, None)
        .await
        .unwrap();

    let ids: Vec<&str> = result
        .scoped_models
        .iter()
        .map(|scoped| scoped.model.id())
        .collect();
    assert_eq!(ids, vec!["claude-sonnet-4-5", "gpt-4o"]);
    assert_eq!(
        result.scoped_models[0].thinking_level,
        Some(AgentThinkingLevel::High)
    );
    assert_eq!(result.scoped_models[1].thinking_level, None);
    assert_eq!(result.diagnostics.len(), 2);
    assert_eq!(
        result.diagnostics[0].message,
        "Invalid thinking level \"invalid\" in pattern \"gpt-4o:invalid\". Using default instead."
    );
    assert_eq!(result.diagnostics[0].pattern, "gpt-4o:invalid");
    assert_eq!(
        result.diagnostics[1].message,
        "No models match pattern \"missing\""
    );
    assert_eq!(result.diagnostics[1].pattern, "missing");
}

#[tokio::test]
async fn scope_bracketed_model_ids_resolve_as_exact_references_before_glob_matching() {
    let mut models = all_models();
    models.push(chat_model(
        "bracketed-model[1m]",
        "Bracketed Model",
        "custom",
        "https://example.invalid",
        true,
        false,
    ));
    let runtime = FakeRuntime::of(models.clone()).with_available(models);
    let patterns = vec!["custom/bracketed-model[1m]".to_string()];

    let result = resolve_model_scope_with_diagnostics(&patterns, &runtime, None)
        .await
        .unwrap();

    assert_eq!(
        result
            .scoped_models
            .iter()
            .map(|s| s.model.id().to_string())
            .collect::<Vec<_>>(),
        vec!["bracketed-model[1m]"]
    );
    assert_eq!(result.diagnostics, Vec::new());
}

#[tokio::test]
async fn scope_bracketed_model_ids_with_thinking_levels_resolve_before_glob_matching() {
    let mut models = all_models();
    models.push(chat_model(
        "bracketed-model[1m]",
        "Bracketed Model",
        "custom",
        "https://example.invalid",
        true,
        false,
    ));
    let runtime = FakeRuntime::of(models.clone()).with_available(models);
    let patterns = vec!["custom/bracketed-model[1m]:high".to_string()];

    let result = resolve_model_scope_with_diagnostics(&patterns, &runtime, None)
        .await
        .unwrap();

    assert_eq!(
        result
            .scoped_models
            .iter()
            .map(|s| s.model.id().to_string())
            .collect::<Vec<_>>(),
        vec!["bracketed-model[1m]"]
    );
    assert_eq!(
        result.scoped_models[0].thinking_level,
        Some(AgentThinkingLevel::High)
    );
    assert_eq!(result.diagnostics, Vec::new());
}

// ==== resolveCliModel ====

#[test]
fn cli_resolves_model_provider_id_without_provider() {
    let runtime = FakeRuntime::of(all_models());
    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: None,
        cli_model: Some("openai/gpt-4o"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.error, None);
    assert_eq!(result.model.as_ref().unwrap().provider(), "openai");
    assert_eq!(result.model.as_ref().unwrap().id(), "gpt-4o");
}

#[test]
fn cli_resolves_fuzzy_patterns_within_an_explicit_provider() {
    let runtime = FakeRuntime::of(all_models());
    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: Some("openai"),
        cli_model: Some("4o"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.error, None);
    assert_eq!(result.model.as_ref().unwrap().provider(), "openai");
    assert_eq!(result.model.as_ref().unwrap().id(), "gpt-4o");
}

#[test]
fn cli_supports_model_pattern_thinking_suffix_without_explicit_thinking() {
    let runtime = FakeRuntime::of(all_models());
    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: None,
        cli_model: Some("sonnet:high"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.error, None);
    assert_eq!(result.model.as_ref().unwrap().id(), "claude-sonnet-4-5");
    assert_eq!(result.thinking_level, Some(AgentThinkingLevel::High));
}

#[test]
fn cli_prefers_exact_model_id_match_over_provider_inference() {
    let runtime = FakeRuntime::of(all_models());
    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: None,
        cli_model: Some("openai/gpt-4o:extended"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.error, None);
    assert_eq!(result.model.as_ref().unwrap().provider(), "openrouter");
    assert_eq!(
        result.model.as_ref().unwrap().id(),
        "openai/gpt-4o:extended"
    );
}

#[test]
fn cli_does_not_strip_invalid_suffix_as_thinking_level() {
    let runtime = FakeRuntime::of(all_models());
    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: Some("openai"),
        cli_model: Some("gpt-4o:extended"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.error, None);
    assert_eq!(result.model.as_ref().unwrap().provider(), "openai");
    assert_eq!(result.model.as_ref().unwrap().id(), "gpt-4o:extended");
}

#[test]
fn cli_allows_custom_model_ids_without_double_prefixing() {
    let runtime = FakeRuntime::of(all_models());
    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: Some("openrouter"),
        cli_model: Some("openrouter/openai/ghost-model"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.error, None);
    assert_eq!(result.model.as_ref().unwrap().provider(), "openrouter");
    assert_eq!(result.model.as_ref().unwrap().id(), "openai/ghost-model");
}

#[test]
fn cli_returns_clear_error_when_there_are_no_models() {
    let runtime = FakeRuntime::of(Vec::new());
    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: Some("openai"),
        cli_model: Some("gpt-4o"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.model, None);
    assert!(result
        .error
        .as_deref()
        .unwrap()
        .contains("No models available"));
}

#[test]
fn cli_prefers_sole_authenticated_provider_for_ambiguous_bare_exact_id() {
    let mut azure = chat_model(
        "gpt-5.6-sol",
        "GPT 5.6 Sol",
        "azure",
        "https://api.openai.com",
        false,
        true,
    );
    azure.base.api = "anthropic-messages".to_string();
    let mut codex = azure.clone();
    codex.base.provider = "openai-codex".to_string();
    let runtime = FakeRuntime::of(vec![azure, codex]).with_auth(&["openai-codex"]);

    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: None,
        cli_model: Some("gpt-5.6-sol"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.error, None);
    assert_eq!(result.model.as_ref().unwrap().provider(), "openai-codex");
    assert_eq!(result.model.as_ref().unwrap().id(), "gpt-5.6-sol");
}

#[test]
fn cli_requires_explicit_provider_for_ambiguous_unauthenticated_id() {
    let mut azure = chat_model(
        "gpt-5.6-sol",
        "GPT 5.6 Sol",
        "azure",
        "https://api.openai.com",
        false,
        true,
    );
    let mut codex = azure.clone();
    codex.base.provider = "openai-codex".to_string();
    let _ = &mut azure;
    let runtime = FakeRuntime::of(vec![azure, codex]);

    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: None,
        cli_model: Some("gpt-5.6-sol"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.model, None);
    let error = result.error.as_deref().unwrap();
    assert!(
        error.contains("Model \"gpt-5.6-sol\" is ambiguous across providers"),
        "{error}"
    );
    assert!(error.contains("azure/gpt-5.6-sol"), "{error}");
    assert!(error.contains("openai-codex/gpt-5.6-sol"), "{error}");
    assert!(
        error.contains("Use --provider or provider/model"),
        "{error}"
    );
}

#[test]
fn cli_prefers_provider_model_split_over_gateway_model_with_matching_id() {
    // When a user writes "zai/glm-5", and both a zai provider model (id: "glm-5")
    // and a gateway model (id: "zai/glm-5") exist, prefer the zai provider model.
    let mut models = all_models();
    models.push(chat_model(
        "glm-5",
        "GLM-5",
        "zai",
        "https://open.bigmodel.cn/api/paas/v4",
        true,
        false,
    ));
    models.push(chat_model(
        "zai/glm-5",
        "GLM-5",
        "vercel-ai-gateway",
        "https://ai-gateway.vercel.sh",
        true,
        false,
    ));
    let runtime = FakeRuntime::of(models).with_auth(&[
        "zai",
        "vercel-ai-gateway",
        "anthropic",
        "openai",
        "openrouter",
    ]);

    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: None,
        cli_model: Some("zai/glm-5"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.error, None);
    assert_eq!(result.model.as_ref().unwrap().provider(), "zai");
    assert_eq!(result.model.as_ref().unwrap().id(), "glm-5");
}

#[test]
fn cli_prefers_authenticated_exact_raw_model_id_over_unauthenticated_inferred_provider() {
    let mut models = all_models();
    models.push(chat_model(
        "xiaomi/mimo-v2.5-pro",
        "Xiaomi MiMo via Commandcode",
        "commandcode",
        "https://example.invalid",
        false,
        false,
    ));
    models.push(chat_model(
        "mimo-v2.5-pro",
        "Xiaomi MiMo",
        "xiaomi",
        "https://api.xiaomimimo.com",
        false,
        false,
    ));
    let runtime = FakeRuntime::of(models).with_auth(&["commandcode"]);

    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: None,
        cli_model: Some("xiaomi/mimo-v2.5-pro"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.error, None);
    assert_eq!(result.model.as_ref().unwrap().provider(), "commandcode");
    assert_eq!(result.model.as_ref().unwrap().id(), "xiaomi/mimo-v2.5-pro");
}

#[test]
fn cli_resolves_provider_prefixed_fuzzy_patterns() {
    let runtime = FakeRuntime::of(all_models());
    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider: None,
        cli_model: Some("openrouter/qwen"),
        cli_thinking: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.error, None);
    assert_eq!(result.model.as_ref().unwrap().provider(), "openrouter");
    assert_eq!(
        result.model.as_ref().unwrap().id(),
        "qwen/qwen3-coder:exacto"
    );
}

// Custom model fallback with :thinking suffix (#5552)
mod custom_model_fallback {
    use super::*;

    fn models_with_neuralwatt() -> Vec<pi_ai::types::Model> {
        let mut models = all_models();
        models.push(chat_model(
            "some-base-model",
            "Some Base Model",
            "neuralwatt",
            "https://api.neuralwatt.com",
            false,
            false,
        ));
        models
    }

    #[test]
    fn strips_thinking_suffix_from_custom_model_id_in_fallback_path() {
        let runtime = FakeRuntime::of(models_with_neuralwatt());
        let result = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("neuralwatt/zai-org/GLM-5.1-FP8:high"),
            cli_thinking: None,
            model_runtime: &runtime,
        });

        assert_eq!(result.error, None);
        assert_eq!(result.model.as_ref().unwrap().provider(), "neuralwatt");
        // The :high suffix must NOT leak into the model id sent to the API
        assert_eq!(result.model.as_ref().unwrap().id(), "zai-org/GLM-5.1-FP8");
        assert!(result.model.as_ref().unwrap().reasoning);
        assert_eq!(result.thinking_level, Some(AgentThinkingLevel::High));
    }

    #[test]
    fn custom_model_without_thinking_suffix_works_normally() {
        let runtime = FakeRuntime::of(models_with_neuralwatt());
        let result = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("neuralwatt/zai-org/GLM-5.1-FP8"),
            cli_thinking: None,
            model_runtime: &runtime,
        });

        assert_eq!(result.error, None);
        assert_eq!(result.model.as_ref().unwrap().provider(), "neuralwatt");
        assert_eq!(result.model.as_ref().unwrap().id(), "zai-org/GLM-5.1-FP8");
        assert_eq!(result.thinking_level, None);
    }

    #[test]
    fn all_valid_thinking_levels_work_in_fallback_path() {
        for name in ["off", "minimal", "low", "medium", "high", "xhigh", "max"] {
            let runtime = FakeRuntime::of(models_with_neuralwatt());
            let result = resolve_cli_model(ResolveCliModelOptions {
                cli_provider: None,
                cli_model: Some(&format!("neuralwatt/zai-org/GLM-5.1-FP8:{name}")),
                cli_thinking: None,
                model_runtime: &runtime,
            });

            assert_eq!(result.error, None, "level {name}");
            assert_eq!(
                result.model.as_ref().unwrap().id(),
                "zai-org/GLM-5.1-FP8",
                "level {name}"
            );
            assert_eq!(result.thinking_level, Some(level(name)), "level {name}");
        }
    }

    #[test]
    fn invalid_thinking_suffix_is_treated_as_part_of_model_id() {
        let runtime = FakeRuntime::of(models_with_neuralwatt());
        let result = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("neuralwatt/zai-org/GLM-5.1-FP8:banana"),
            cli_thinking: None,
            model_runtime: &runtime,
        });

        assert_eq!(result.error, None);
        assert_eq!(result.model.as_ref().unwrap().provider(), "neuralwatt");
        // Invalid suffix stays in the id (it's not a thinking level)
        assert_eq!(
            result.model.as_ref().unwrap().id(),
            "zai-org/GLM-5.1-FP8:banana"
        );
        assert_eq!(result.thinking_level, None);
    }

    #[test]
    fn explicit_provider_with_custom_model_thinking_strips_suffix() {
        let runtime = FakeRuntime::of(models_with_neuralwatt());
        let result = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: Some("neuralwatt"),
            cli_model: Some("zai-org/GLM-5.1-FP8:high"),
            cli_thinking: None,
            model_runtime: &runtime,
        });

        assert_eq!(result.error, None);
        assert_eq!(result.model.as_ref().unwrap().provider(), "neuralwatt");
        assert_eq!(result.model.as_ref().unwrap().id(), "zai-org/GLM-5.1-FP8");
        assert_eq!(result.thinking_level, Some(AgentThinkingLevel::High));
    }

    #[test]
    fn with_explicit_thinking_suffix_is_kept_as_part_of_model_id() {
        let runtime = FakeRuntime::of(models_with_neuralwatt());
        let result = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("neuralwatt/zai-org/GLM-5.1-FP8:high"),
            cli_thinking: Some(AgentThinkingLevel::Medium),
            model_runtime: &runtime,
        });

        assert_eq!(result.error, None);
        assert_eq!(result.model.as_ref().unwrap().provider(), "neuralwatt");
        // :high is kept as part of the model id since --thinking was explicit
        assert_eq!(
            result.model.as_ref().unwrap().id(),
            "zai-org/GLM-5.1-FP8:high"
        );
        assert_eq!(result.thinking_level, None);
    }
}

// ==== default model selection ====

#[test]
fn openai_defaults_track_current_models() {
    assert_eq!(default_model_for_provider("openai"), Some("gpt-5.5"));
    assert_eq!(
        default_model_for_provider("openai-codex"),
        Some("gpt-6.1-sol")
    );
}

#[test]
fn zai_minimax_cerebras_and_ant_ling_defaults_track_current_models() {
    assert_eq!(default_model_for_provider("zai"), Some("glm-5.3"));
    assert_eq!(default_model_for_provider("zai-coding-cn"), Some("glm-5.3"));
    assert_eq!(default_model_for_provider("minimax"), Some("MiniMax-M2.7"));
    assert_eq!(
        default_model_for_provider("minimax-cn"),
        Some("MiniMax-M2.7")
    );
    assert_eq!(default_model_for_provider("cerebras"), Some("gpt-oss-120b"));
    assert_eq!(default_model_for_provider("ant-ling"), Some("Ring-2.6-1T"));
}

#[test]
fn builtin_chat_providers_have_defaults_in_their_generated_catalogs() {
    for provider in get_builtin_providers() {
        let chat_models = get_builtin_models(Some(provider.id()));
        let default_id = default_model_for_provider(provider.id());
        if chat_models.is_empty() {
            assert_eq!(
                default_id,
                None,
                "{} has no chat models and should have no chat default",
                provider.id()
            );
            continue;
        }
        assert!(
            chat_models
                .iter()
                .any(|model| model.id() == default_id.unwrap_or_default()),
            "{} default {:?} should exist in its generated catalog",
            provider.id(),
            default_id
        );
    }
}

#[test]
fn ai_gateway_default_tracks_current_model() {
    assert_eq!(
        default_model_for_provider("vercel-ai-gateway"),
        Some("zai/glm-5.1")
    );
}

#[test]
fn xai_default_tracks_current_model() {
    assert_eq!(default_model_for_provider("xai"), Some("grok-4.7"));
}

#[test]
fn qwen_token_plan_individual_default_tracks_current_model() {
    assert_eq!(
        default_model_for_provider("qwen-token-plan-individual"),
        Some("qwen3.8-max")
    );
}

#[test]
fn find_initial_model_accepts_explicit_provider_custom_model_ids() {
    let runtime = FakeRuntime::of(all_models());
    let result = find_initial_model(FindInitialModelOptions {
        cli_provider: Some("openrouter"),
        cli_model: Some("openrouter/openai/ghost-model"),
        scoped_models: &[],
        is_continuing: false,
        default_provider: None,
        default_model_id: None,
        default_thinking_level: None,
        model_thinking_levels: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.model.as_ref().unwrap().provider(), "openrouter");
    assert_eq!(result.model.as_ref().unwrap().id(), "openai/ghost-model");
}

#[test]
fn find_initial_model_selects_ai_gateway_default_when_available() {
    let mut gateway_model = chat_model(
        "anthropic/claude-opus-4-6",
        "Claude Opus 4.6",
        "vercel-ai-gateway",
        "https://ai-gateway.vercel.sh",
        true,
        true,
    );
    gateway_model.base.cost.rates.input = 5.0;
    let runtime = FakeRuntime::default().with_snapshot(vec![gateway_model]);

    let result = find_initial_model(FindInitialModelOptions {
        cli_provider: None,
        cli_model: None,
        scoped_models: &[],
        is_continuing: false,
        default_provider: None,
        default_model_id: None,
        default_thinking_level: None,
        model_thinking_levels: None,
        model_runtime: &runtime,
    });

    assert_eq!(
        result.model.as_ref().unwrap().provider(),
        "vercel-ai-gateway"
    );
    assert_eq!(
        result.model.as_ref().unwrap().id(),
        "anthropic/claude-opus-4-6"
    );
}

#[test]
fn find_initial_model_ignores_an_unauthenticated_saved_default() {
    let saved_deepseek = chat_model(
        "deepseek-v4-flash",
        "DeepSeek V4 Flash",
        "deepseek",
        "https://api.deepseek.com",
        true,
        false,
    );
    let mut local_deepseek = saved_deepseek.clone();
    local_deepseek.base.provider = "spark-two".to_string();
    local_deepseek.base.base_url = "http://spark-two:8000/v1".to_string();

    let saved_clone = saved_deepseek.clone();
    let runtime = FakeRuntime {
        authenticated_providers: vec!["spark-two".to_string()],
        available_snapshot: Some(vec![local_deepseek]),
        model_lookup: Some(Box::new(move |provider: &str, model_id: &str| {
            if provider == saved_clone.provider() && model_id == saved_clone.id() {
                Some(saved_clone.clone())
            } else {
                None
            }
        })),
        ..FakeRuntime::default()
    };

    let result = find_initial_model(FindInitialModelOptions {
        cli_provider: None,
        cli_model: None,
        scoped_models: &[],
        is_continuing: false,
        default_provider: Some("deepseek"),
        default_model_id: Some("deepseek-v4-flash"),
        default_thinking_level: None,
        model_thinking_levels: None,
        model_runtime: &runtime,
    });

    assert_eq!(result.model.as_ref().unwrap().provider(), "spark-two");
    assert_eq!(result.model.as_ref().unwrap().id(), "deepseek-v4-flash");
}

// restoreModelFromSession has no dedicated upstream suite (exercised via
// agent-session tests); basic fallback behavior is covered here.
#[test]
fn restore_falls_back_to_available_default_when_saved_model_missing() {
    let gateway = chat_model(
        "anthropic/claude-opus-4-6",
        "Claude Opus 4.6",
        "vercel-ai-gateway",
        "https://ai-gateway.vercel.sh",
        true,
        true,
    );
    let runtime = FakeRuntime::default().with_snapshot(vec![gateway]);

    let result =
        super::restore_model_from_session("anthropic", "claude-sonnet-4-5", None, false, &runtime);
    assert_eq!(
        result.model.as_ref().unwrap().provider(),
        "vercel-ai-gateway"
    );
    assert!(result
        .fallback_message
        .as_deref()
        .unwrap()
        .contains("Could not restore model anthropic/claude-sonnet-4-5 (model no longer exists)"));
}

#[test]
fn restore_keeps_current_model_when_saved_model_lost_auth() {
    let current = chat_model(
        "glm-5",
        "GLM-5",
        "zai",
        "https://open.bigmodel.cn/api/paas/v4",
        true,
        false,
    );
    let saved = chat_model(
        "deepseek-v4-flash",
        "DeepSeek V4 Flash",
        "deepseek",
        "https://api.deepseek.com",
        true,
        false,
    );
    let runtime = FakeRuntime::of(vec![saved]);

    let result = super::restore_model_from_session(
        "deepseek",
        "deepseek-v4-flash",
        Some(&current),
        false,
        &runtime,
    );
    assert_eq!(result.model.as_ref().unwrap().provider(), "zai");
    assert!(result
        .fallback_message
        .as_deref()
        .unwrap()
        .contains("(no auth configured). Using zai/glm-5."));
}
