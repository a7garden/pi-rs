//! Port of `packages/ai/src/auth/oauth/` — the built-in OAuth flows.
//!
//! Module layout mirrors upstream 1:1 (`pkce`, `device-code`,
//! `callback-server`, `load`, and one file per provider). The provider flows
//! are constructed eagerly as statics; `load.ts` re-exports them through the
//! upstream `loadXxxOAuth` names.
//!
//! # Ambient fetch seam
//!
//! Upstream flows call `globalThis.fetch`, which tests stub with
//! `vi.stubGlobal`. Rust has no ambient fetch (crate docs, D4), so this
//! module carries the oauth-layer equivalent: [`oauth_fetch`] returns the
//! ambient implementation and [`set_oauth_fetch`] replaces it process-wide
//! (deviation D29). Flows and their tests go through this seam; the default
//! is the crate's reqwest adapter.

pub mod anthropic;
pub mod callback_server;
pub mod device_code;
pub mod github_copilot;
pub mod kimi_coding;
pub mod load;
pub mod meta;
pub mod openai_chatgpt;
pub mod openai_codex;
pub mod openrouter;
pub mod pkce;
pub mod radius;
pub mod xai;

// Upstream re-export surface (auth/oauth files export their flow objects;
// the Rust port exposes the constructors + the shared flow types).
pub use crate::auth::types::{OAuthAuth, OAuthCredential, OAuthCredentials};
pub use anthropic::anthropic_oauth;
pub use callback_server::{
    start_oauth_callback_server, wait_for_callback_or_manual_input, CallbackOrManual,
    CallbackServerError, CompleteFn, OAuthCallbackServer, OAuthCallbackServerOptions,
};
pub use device_code::{
    abortable_sleep, poll_oauth_device_code_flow, OAuthDeviceCodePollOptions,
    OAuthDeviceCodePollResult,
};
pub use github_copilot::github_copilot_oauth;
pub use kimi_coding::kimi_coding_oauth;
pub use load::{
    load_anthropic_oauth, load_github_copilot_oauth, load_kimi_coding_oauth, load_meta_oauth,
    load_openai_chatgpt_oauth, load_openai_codex_oauth, load_openrouter_oauth, load_radius_oauth,
    load_xai_oauth,
};
pub use meta::meta_oauth;
pub use openai_chatgpt::openai_chatgpt_oauth;
pub use openai_codex::openai_codex_oauth;
pub use openrouter::openrouter_oauth;
pub use pkce::generate_pkce;
pub use radius::{create_radius_oauth, RadiusOAuthOptions};
pub use xai::xai_oauth;

use parking_lot::RwLock;

use crate::http::{default_fetch, FetchFn};
use crate::utils::abort::AbortSignal;
use crate::utils::thrown::ThrownValue;

/// The ambient oauth fetch (upstream `globalThis.fetch` inside the oauth
/// flow modules).
static OAUTH_FETCH: RwLock<Option<FetchFn>> = RwLock::new(None);

/// Replace the ambient oauth fetch (`vi.stubGlobal("fetch", …)` equivalent);
/// `None` restores the crate default adapter.
pub fn set_oauth_fetch(fetch: Option<FetchFn>) {
    *OAUTH_FETCH.write() = fetch;
}

/// Fetch used by every oauth flow request (upstream bare `fetch(...)` calls).
pub fn oauth_fetch() -> FetchFn {
    OAUTH_FETCH.read().clone().unwrap_or_else(default_fetch)
}

/// Error thrown by an oauth request wrapper (upstream `new Error(...)`) — the
/// crate's `ThrownValue::message`.
pub(crate) fn oauth_error(message: impl Into<String>) -> ThrownValue {
    ThrownValue::message(message)
}

/// Upstream `AbortSignal.any([signal, AbortSignal.timeout(ms)])`: a merged
/// signal that also aborts with the timeout reason after `timeout_ms`
/// (crate docs, D20 pattern).
pub(crate) fn timeout_signal(signal: &AbortSignal, timeout_ms: u64) -> AbortSignal {
    let controller = crate::utils::abort::AbortController::new();
    let watcher_signal = controller.signal();
    let watcher_for_spawn = watcher_signal.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(timeout_ms)).await;
        AbortSignal::abort_with(
            &watcher_for_spawn,
            crate::utils::abort::AbortReason::from("The operation was aborted due to timeout"),
        );
    });
    AbortSignal::any([signal.clone(), watcher_signal])
}

/// Node `Response.statusText` equivalents for statuses these flows render.
pub(crate) fn http_status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        408 => "Request Timeout",
        409 => "Conflict",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "",
    }
}

/// URL-encode a form body from ordered fields (upstream
/// `new URLSearchParams(fields).toString()`).
pub(crate) fn form_urlencode(fields: &[(&str, &str)]) -> String {
    fields
        .iter()
        .map(|(key, value)| {
            let mut encoded = url::form_urlencoded::Serializer::new(String::new());
            encoded.append_pair(key, value);
            encoded.finish()
        })
        .collect::<Vec<String>>()
        .join("&")
}

/// Serializes tests that stub the process-global oauth fetch (upstream
/// `describe.sequential` + `vi.stubGlobal`).
#[cfg(test)]
pub(crate) fn test_fetch_guard() -> &'static tokio::sync::Mutex<()> {
    static GUARD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    &GUARD
}

/// Method-call surface over the boxed callbacks of [`OAuthAuth`] for tests
/// (Rust cannot call `struct.field(args)`; upstream's object methods map to
/// these).
#[cfg(test)]
pub(crate) trait OAuthTestExt {
    fn login(
        self,
        interaction: crate::auth::types::ProviderAuthInteraction,
        options: Option<crate::auth::types::LoginOptions>,
    ) -> futures_util::future::BoxFuture<'static, Result<OAuthCredential, ThrownValue>>;
    fn refresh(
        self,
        credential: OAuthCredential,
        signal: crate::utils::abort::AbortSignal,
    ) -> futures_util::future::BoxFuture<'static, Result<OAuthCredential, ThrownValue>>;
    fn to_auth<'a>(
        self,
        credential: &'a OAuthCredential,
    ) -> futures_util::future::BoxFuture<'a, Result<crate::auth::types::ModelAuth, ThrownValue>>;
}

#[cfg(test)]
impl OAuthTestExt for OAuthAuth {
    fn login(
        self,
        interaction: crate::auth::types::ProviderAuthInteraction,
        options: Option<crate::auth::types::LoginOptions>,
    ) -> futures_util::future::BoxFuture<'static, Result<OAuthCredential, ThrownValue>> {
        (self.login)(interaction, options)
    }

    fn refresh(
        self,
        credential: OAuthCredential,
        signal: crate::utils::abort::AbortSignal,
    ) -> futures_util::future::BoxFuture<'static, Result<OAuthCredential, ThrownValue>> {
        (self.refresh)(credential, signal)
    }

    fn to_auth<'a>(
        self,
        credential: &'a OAuthCredential,
    ) -> futures_util::future::BoxFuture<'a, Result<crate::auth::types::ModelAuth, ThrownValue>>
    {
        (self.to_auth)(credential)
    }
}

/// Parse a `key=value&...` form body (upstream `new URLSearchParams(body)`
/// reads). Keys/values are percent-decoded.
pub(crate) fn parse_form(body: &str) -> std::collections::BTreeMap<String, String> {
    url::form_urlencoded::parse(body.as_bytes())
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_form_bodies_like_urlsearchparams() {
        let form = parse_form("grant_type=refresh_token&scope=read%3Auser&client_id=a+b");
        assert_eq!(
            form.get("grant_type").map(String::as_str),
            Some("refresh_token")
        );
        assert_eq!(form.get("scope").map(String::as_str), Some("read:user"));
        assert_eq!(form.get("client_id").map(String::as_str), Some("a b"));
    }

    #[tokio::test]
    async fn set_oauth_fetch_overrides_the_ambient_fetch() {
        let _guard = test_fetch_guard().lock().await;
        struct Probe;
        let probe = FetchFn::new(|request| {
            Box::pin(async move {
                Ok(crate::http::HttpResponse {
                    status: 201,
                    headers: Vec::new(),
                    body: crate::http::HttpBody::bytes(format!("probe {}", request.url)),
                })
            })
        });
        set_oauth_fetch(Some(probe));
        let response = oauth_fetch()
            .send(crate::http::HttpRequest::new(
                "GET",
                "https://example.invalid",
            ))
            .await
            .expect("response");
        assert_eq!(response.status, 201);
        set_oauth_fetch(None);
    }
}
