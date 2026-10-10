//! Port of `packages/ai/src/auth/oauth/load.ts` — accessors for the built-in
//! OAuth flow implementations.
//!
//! Upstream loads the flow modules through a runtime `import()` so bundlers
//! cannot follow them into Node-only code, with an optional statically
//! bundled loader registry for standalone Bun binaries
//! (`registerBundledOAuthFlowLoaders`). Rust links statically and the Bun
//! path does not apply, so both layers collapse into plain constructors
//! (deviation D34).

use crate::auth::types::OAuthAuth;

/// Upstream `loadAnthropicOAuth`.
pub fn load_anthropic_oauth() -> OAuthAuth {
    super::anthropic::anthropic_oauth()
}

/// Upstream `loadOpenAICodexOAuth`.
pub fn load_openai_codex_oauth() -> OAuthAuth {
    super::openai_codex::openai_codex_oauth()
}

/// Upstream `loadOpenAIChatGPTOAuth`.
pub fn load_openai_chatgpt_oauth() -> OAuthAuth {
    super::openai_chatgpt::openai_chatgpt_oauth()
}

/// Upstream `loadGitHubCopilotOAuth`.
pub fn load_github_copilot_oauth() -> OAuthAuth {
    super::github_copilot::github_copilot_oauth()
}

/// Upstream `loadOpenRouterOAuth`.
pub fn load_openrouter_oauth() -> OAuthAuth {
    super::openrouter::openrouter_oauth()
}

/// Upstream `loadKimiCodingOAuth`.
pub fn load_kimi_coding_oauth() -> OAuthAuth {
    super::kimi_coding::kimi_coding_oauth()
}

/// Upstream `loadMetaOAuth`.
pub fn load_meta_oauth() -> OAuthAuth {
    super::meta::meta_oauth()
}

/// Upstream `loadXaiOAuth`.
pub fn load_xai_oauth() -> OAuthAuth {
    super::xai::xai_oauth()
}

/// Upstream `loadRadiusOAuth(options)` — radius flows are per-gateway.
pub fn load_radius_oauth(options: super::radius::RadiusOAuthOptions) -> OAuthAuth {
    super::radius::create_radius_oauth(options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::oauth::OAuthTestExt;

    // Port of the load surface implied by test/oauth-auth.test.ts: every flow
    // constructs with its upstream display metadata.
    #[test]
    fn loads_every_flow_with_upstream_metadata() {
        assert_eq!(load_anthropic_oauth().name, "Anthropic (Claude Pro/Max)");
        assert_eq!(load_openai_codex_oauth().name, "OpenAI (ChatGPT Plus/Pro)");
        assert_eq!(
            load_openai_chatgpt_oauth().name,
            "OpenAI (ChatGPT subscription)"
        );
        assert_eq!(load_github_copilot_oauth().name, "GitHub Copilot");
        assert_eq!(load_openrouter_oauth().name, "OpenRouter OAuth");
        assert_eq!(load_kimi_coding_oauth().name, "Kimi Code (subscription)");
        assert_eq!(load_meta_oauth().name, "Meta (Muse subscription)");
        assert_eq!(load_xai_oauth().name, "xAI (Grok/X subscription)");
        assert_eq!(
            load_radius_oauth(super::super::radius::RadiusOAuthOptions {
                name: "Radius".to_string(),
                gateway: "https://radius.example".to_string(),
            })
            .name,
            "Radius"
        );
    }
}
