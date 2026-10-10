//! Port of `packages/ai/src/oauth.ts` — the type-only compatibility entry
//! point for coding-agent extension OAuth declarations.
//!
//! Upstream `bun-oauth.ts` (the Bun-specific re-export of the flow
//! implementations) is omitted: it exists to dodge bundler limitations in
//! Bun binaries and has no Rust equivalent (deviation D40). No flow
//! implementations are reachable from this module — only the extension
//! declaration types.

pub use crate::auth::types::OAuthCredentials;
pub use crate::compat::extension_oauth_types::{
    OAuthAuthInfo, OAuthDeviceCodeInfo, OAuthLoginCallbacks, OAuthPrompt, OAuthSelectOption,
    OAuthSelectPrompt,
};

#[cfg(test)]
mod tests {
    use super::*;
    // Port of test/oauth-auth.test.ts "keeps the extension OAuth barrel free of
    // built-in flow implementations": the barrel exposes no login functions.
    #[test]
    fn extension_oauth_barre_types_compile_without_flows() {
        let credentials = OAuthCredentials {
            refresh: "r".to_string(),
            access: "a".to_string(),
            expires: 0,
            extra: Default::default(),
        };
        assert_eq!(credentials.access, "a");
        let prompt = OAuthPrompt {
            message: "m".to_string(),
            placeholder: None,
            allow_empty: None,
        };
        assert_eq!(prompt.message, "m");
    }
}
