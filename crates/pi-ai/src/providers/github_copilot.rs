//! Port of `packages/ai/src/providers/github-copilot.ts`.
//!
//! Deviations:
//! - D17: upstream factories return a non-failing `Provider`; Rust
//!   [`create_provider`] returns `Result`, unwrapped with `expect` because
//!   the configuration here is static and valid.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::auth::helpers::{env_api_key_auth, lazy_oauth, LazyOAuthInput};
use crate::auth::types::{Credential, ProviderAuth};
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};
use crate::types::Model;

/// Upstream `githubCopilotProvider()`.
pub fn github_copilot_provider() -> Arc<dyn Provider> {
    let mut api = BTreeMap::new();
    api.insert(
        "anthropic-messages".to_string(),
        crate::api::anthropic_messages::anthropic_messages_api(),
    );
    api.insert(
        "openai-completions".to_string(),
        crate::api::openai_completions::openai_completions_api(),
    );
    api.insert(
        "openai-responses".to_string(),
        crate::api::openai_responses::openai_responses_api(),
    );
    create_provider(CreateProviderOptions {
        id: "github-copilot".to_string(),
        name: Some("GitHub Copilot".to_string()),
        base_url: Some("https://api.individual.githubcopilot.com".to_string()),
        headers: None,
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "GitHub Copilot token",
                &["COPILOT_GITHUB_TOKEN"],
            )),
            oauth: Some(Arc::new(lazy_oauth(LazyOAuthInput {
                name: "GitHub Copilot".to_string(),
                is_subscription: true,
                login_label: None,
                load: crate::auth::oauth::load::load_github_copilot_oauth,
            }))),
        },
        models: super::github_copilot_models::github_copilot_any_models(),
        fetch_models: None,
        filter_models: Some(Box::new(filter_models)),
        filter_all_models: None,
        api: Some(ApiDispatch::ByApi(api)),
        images: Default::default(),
        classifiers: Default::default(),
    })
    .expect("static github-copilot provider configuration is valid")
}

/// Upstream `filterModels`: OAuth credentials carry `availableModelIds`
/// (extra index-signature keys are preserved verbatim); non-OAuth credentials
/// and malformed lists keep every model.
fn filter_models(models: &[Model], credential: Option<&Credential>) -> Vec<Model> {
    let models = models.to_vec();
    let Some(Credential::OAuth(oauth)) = credential else {
        return models;
    };
    let Some(available) = oauth.credentials.extra.get("availableModelIds") else {
        return models;
    };
    let Some(available) = available.as_array() else {
        return models;
    };
    if !available.iter().all(|id| id.is_string()) {
        return models;
    }
    let available: std::collections::BTreeSet<&str> =
        available.iter().filter_map(|id| id.as_str()).collect();
    models
        .into_iter()
        .filter(|model| available.contains(model.id()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::types::{OAuthCredential, OAuthCredentials};
    use std::collections::BTreeMap;

    fn test_model(id: &str) -> Model {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "name": id,
            "api": "anthropic-messages",
            "provider": "github-copilot",
            "baseUrl": "https://api.individual.githubcopilot.com",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 1000,
            "maxTokens": 100,
        }))
        .expect("test model parses")
    }

    fn oauth_credential(extra: serde_json::Value) -> Credential {
        Credential::OAuth(OAuthCredential {
            credentials: OAuthCredentials {
                refresh: "r".to_string(),
                access: "a".to_string(),
                expires: 0,
                extra: [("availableModelIds".to_string(), extra)]
                    .into_iter()
                    .collect::<BTreeMap<_, _>>(),
            },
        })
    }

    // Port of the upstream providers.test.ts github-copilot filterModels case.
    #[test]
    fn filters_models_to_oauth_available_ids() {
        let models = vec![test_model("a"), test_model("b"), test_model("c")];
        let filtered = filter_models(
            &models,
            Some(&oauth_credential(serde_json::json!(["a", "c"]))),
        );
        let ids: Vec<&str> = filtered.iter().map(|m| m.id()).collect();
        assert_eq!(ids, vec!["a", "c"]);
    }

    #[test]
    fn keeps_all_models_for_api_key_credentials() {
        let models = vec![test_model("a"), test_model("b")];
        let filtered = filter_models(&models, Some(&Credential::ApiKey(Default::default())));
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn keeps_all_models_for_malformed_available_ids() {
        let models = vec![test_model("a"), test_model("b")];
        for malformed in [
            serde_json::json!(null),
            serde_json::json!("a"),
            serde_json::json!(["a", 1]),
        ] {
            let filtered = filter_models(&models, Some(&oauth_credential(malformed)));
            assert_eq!(filtered.len(), 2);
        }
    }
}
