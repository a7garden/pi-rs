//! Port of `packages/ai/src/providers/all.ts` — the built-in provider
//! registry: every factory, the generated-catalog accessors, and the
//! all-providers `Models` collection.
//!
//! Upstream registers built-ins at module load; Rust exposes them as explicit
//! constructors ([`builtin_provider_factories`], [`builtin_models`]) consumed
//! by `compat` on first use (D42, static linking D10).

use std::sync::Arc;

use super::*;

use crate::models::{create_models, CreateModelsOptions, Models, MutableModels, Provider};

/// Every built-in provider factory in upstream `builtinProviders()` order.
/// Ids match the generated catalog keys.
pub fn builtin_provider_factories() -> Vec<(&'static str, fn() -> Arc<dyn Provider>)> {
    vec![
        ("amazon-bedrock", amazon_bedrock::amazon_bedrock_provider),
        ("ant-ling", ant_ling::ant_ling_provider),
        ("anthropic", anthropic::anthropic_provider),
        ("azure", azure::azure_provider),
        ("baseten", baseten::baseten_provider),
        ("cerebras", cerebras::cerebras_provider),
        (
            "cloudflare-ai-gateway",
            cloudflare_ai_gateway::cloudflare_ai_gateway_provider,
        ),
        (
            "cloudflare-workers-ai",
            cloudflare_workers_ai::cloudflare_workers_ai_provider,
        ),
        ("deepseek", deepseek::deepseek_provider),
        ("fireworks", fireworks::fireworks_provider),
        ("github-copilot", github_copilot::github_copilot_provider),
        ("google", google::google_provider),
        ("google-vertex", google_vertex::google_vertex_provider),
        ("groq", groq::groq_provider),
        ("huggingface", huggingface::huggingface_provider),
        ("kimi-coding", kimi_coding::kimi_coding_provider),
        ("meta", meta::meta_provider),
        ("minimax", minimax::minimax_provider),
        ("minimax-cn", minimax_cn::minimax_cn_provider),
        ("mistral", mistral::mistral_provider),
        ("moonshotai", moonshotai::moonshotai_provider),
        ("moonshotai-cn", moonshotai_cn::moonshotai_cn_provider),
        ("nvidia", nvidia::nvidia_provider),
        ("openai", openai::openai_provider),
        ("openai-codex", openai_codex::openai_codex_provider),
        ("opencode", opencode::opencode_provider),
        ("opencode-go", opencode_go::opencode_go_provider),
        ("openrouter", openrouter::openrouter_provider),
        ("qwen-token-plan", qwen_token_plan::qwen_token_plan_provider),
        (
            "qwen-token-plan-cn",
            qwen_token_plan_cn::qwen_token_plan_cn_provider,
        ),
        (
            "qwen-token-plan-individual",
            qwen_token_plan_individual::qwen_token_plan_individual_provider,
        ),
        ("radius", || {
            radius::radius_provider(radius::RadiusProviderOptions::default())
        }),
        ("together", together::together_provider),
        ("typesafe", typesafe::typesafe_provider),
        (
            "vercel-ai-gateway",
            vercel_ai_gateway::vercel_ai_gateway_provider,
        ),
        ("xai", xai::xai_provider),
        ("xiaomi", xiaomi::xiaomi_provider),
        (
            "xiaomi-token-plan-ams",
            xiaomi_token_plan_ams::xiaomi_token_plan_ams_provider,
        ),
        (
            "xiaomi-token-plan-cn",
            xiaomi_token_plan_cn::xiaomi_token_plan_cn_provider,
        ),
        (
            "xiaomi-token-plan-sgp",
            xiaomi_token_plan_sgp::xiaomi_token_plan_sgp_provider,
        ),
        ("zai", zai::zai_provider),
        ("zai-coding-cn", zai_coding_cn::zai_coding_cn_provider),
    ]
}

/// Upstream `builtinProviders()` — freshly constructed built-ins.
pub fn builtin_providers() -> Vec<Arc<dyn Provider>> {
    builtin_provider_factories()
        .into_iter()
        .map(|(_, factory)| factory())
        .collect()
}

/// Upstream `builtinModels()` — a `Models` collection with every built-in
/// provider registered.
pub fn builtin_models(options: Option<CreateModelsOptions>) -> Arc<dyn MutableModels> {
    let models = create_models(options);
    for provider in builtin_providers() {
        models.set_provider(provider);
    }
    models
}

/// Upstream `getBuiltinProviderIds`.
pub fn get_builtin_provider_ids() -> Vec<String> {
    crate::models_generated::models()
        .keys()
        .map(str::to_string)
        .collect()
}

fn catalog_model(
    catalog: &crate::models_generated::GeneratedCatalog,
    provider: &str,
    model_id: &str,
) -> Option<crate::types::AnyModel> {
    catalog.get(provider)?.get(model_id).cloned()
}

/// Upstream `getBuiltinModel`.
pub fn get_builtin_model(provider: &str, model_id: &str) -> Option<crate::types::Model> {
    match catalog_model(crate::models_generated::models(), provider, model_id)? {
        crate::types::AnyModel::Chat(model) => Some(model),
        _ => None,
    }
}

/// Upstream `getBuiltinImageModel`.
pub fn get_builtin_image_model(provider: &str, model_id: &str) -> Option<crate::types::ImageModel> {
    match catalog_model(crate::models_generated::image_models(), provider, model_id)? {
        crate::types::AnyModel::Image(model) => Some(model),
        _ => None,
    }
}

/// Upstream `getBuiltinClassifierModel`.
pub fn get_builtin_classifier_model(
    provider: &str,
    model_id: &str,
) -> Option<crate::types::ClassifierModel> {
    match catalog_model(
        crate::models_generated::classifier_models(),
        provider,
        model_id,
    )? {
        crate::types::AnyModel::Classifier(model) => Some(model),
        _ => None,
    }
}

/// Upstream `getBuiltinModelDataGeneratedAt`: the embedded catalog data has
/// no `generatedAt` manifest field (see `models_generated.rs`), so this is
/// always `None` today; the signature keeps callers (remote catalog overlay,
/// model runtime) shaped like upstream.
pub fn get_builtin_model_data_generated_at() -> Option<u64> {
    None
}

/// Upstream `all.ts` also re-exports `radiusProvider`; kept additive here so
/// callers can reach it next to the other built-in factories.
pub use crate::providers::radius::radius_provider;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_every_builtin_in_catalog_order() {
        let factories = builtin_provider_factories();
        assert_eq!(factories.len(), 42);
        let ids: Vec<&'static str> = factories.iter().map(|(id, _)| *id).collect();
        let catalog_ids = get_builtin_provider_ids();
        assert_eq!(ids.len(), catalog_ids.len());
        for (id, catalog_id) in ids.iter().zip(catalog_ids.iter()) {
            assert_eq!(id, catalog_id, "factory list must match catalog keys");
        }
    }

    #[test]
    fn every_factory_constructs() {
        for (id, factory) in builtin_provider_factories() {
            let provider = factory();
            assert_eq!(provider.id(), id, "provider id matches registry key");
        }
    }

    #[test]
    fn builtin_models_registers_all_providers() {
        let models = builtin_models(None);
        assert_eq!(models.get_providers().len(), 42);
    }

    #[test]
    fn catalog_accessors_resolve_first_entries() {
        let chat_first = crate::models_generated::models()
            .get("anthropic")
            .and_then(|c| c.keys().next().map(str::to_string))
            .expect("anthropic has models");
        let model = get_builtin_model("anthropic", &chat_first).expect("builtin chat model");
        assert_eq!(model.base.provider, "anthropic");
        assert!(get_builtin_model("anthropic", "no-such-model").is_none());

        let image_first = crate::models_generated::image_models()
            .get("openai")
            .and_then(|c| c.keys().next())
            .map(str::to_string);
        if let Some(id) = image_first {
            assert!(get_builtin_image_model("openai", &id).is_some());
        }

        let classifier_first = crate::models_generated::classifier_models()
            .get("openai")
            .and_then(|c| c.keys().next())
            .map(str::to_string);
        if let Some(id) = classifier_first {
            assert!(get_builtin_classifier_model("openai", &id).is_some());
        }
    }
}
