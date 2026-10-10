//! Port of `packages/ai/src/compat/extension-oauth-types.ts` — the legacy
//! extension OAuth declaration surface, retained only for coding-agent
//! extension compatibility (type-only upstream).
//!
//! Upstream `OAuthCredentials` re-exports the auth credential type; the Rust
//! port re-exports [`crate::auth::types::OAuthCredentials`] through
//! `crate::oauth`.

/// Legacy extension OAuth prompt (upstream `OAuthPrompt`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OAuthPrompt {
    pub message: String,
    pub placeholder: Option<String>,
    pub allow_empty: Option<bool>,
}

/// Legacy extension OAuth authorization link (upstream `OAuthAuthInfo`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OAuthAuthInfo {
    pub url: String,
    pub instructions: Option<String>,
}

/// Legacy extension OAuth device-code notification (upstream
/// `OAuthDeviceCodeInfo`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OAuthDeviceCodeInfo {
    pub user_code: String,
    pub verification_uri: String,
    pub interval_seconds: Option<u64>,
    pub expires_in_seconds: Option<u64>,
}

/// Legacy extension OAuth select option (upstream `OAuthSelectOption`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthSelectOption {
    pub id: String,
    pub label: String,
}

/// Legacy extension OAuth select prompt (upstream `OAuthSelectPrompt`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthSelectPrompt {
    pub message: String,
    pub options: Vec<OAuthSelectOption>,
}

/// Callback surface retained only for coding-agent extension compatibility
/// (upstream `OAuthLoginCallbacks`).
pub struct OAuthLoginCallbacks {
    pub signal: Option<crate::utils::abort::AbortSignal>,
    pub on_auth: Box<dyn Fn(OAuthAuthInfo) + Send + Sync>,
    pub on_device_code: Box<dyn Fn(OAuthDeviceCodeInfo) + Send + Sync>,
    pub on_prompt:
        Box<dyn Fn(OAuthPrompt) -> futures_util::future::BoxFuture<'static, String> + Send + Sync>,
    pub on_progress: Option<Box<dyn Fn(&str) + Send + Sync>>,
    pub on_manual_code_input:
        Option<Box<dyn Fn() -> futures_util::future::BoxFuture<'static, String> + Send + Sync>>,
    pub on_select: Box<
        dyn Fn(OAuthSelectPrompt) -> futures_util::future::BoxFuture<'static, Option<String>>
            + Send
            + Sync,
    >,
}
