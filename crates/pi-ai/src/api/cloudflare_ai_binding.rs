//! Port of `packages/ai/src/api/cloudflare-ai-binding.ts`.
//!
//! AI Gateway transport over the Workers AI binding. pi's Cloudflare AI
//! Gateway support speaks HTTPS
//! (`gateway.ai.cloudflare.com/v1/{account}/{gateway}/{provider}/...`, see
//! [`crate::api::cloudflare`]), which needs a Cloudflare API token even when
//! the caller is a Worker in the gateway's own account. A Worker avoids that
//! token by talking to the gateway through the AI binding's `fetch`
//! passthrough (`env.AI.fetch()`), which serves the gateway's provider
//! passthrough at
//! `https://workers-binding.ai/ai-gateway/gateways/{gateway}/{provider}/...`
//! — the same shape as the HTTPS URL, minus the account id (the binding
//! channel carries identity). Binding calls are pre-authenticated in-account
//! and return the provider's native wire format as a regular (streaming)
//! `Response`, so API implementations behave identically over either
//! transport.
//!
//! A model whose `baseUrl` names that route therefore needs no translation:
//! point it there and pass [`create_ai_binding_fetch`] as the request `fetch`.
//! Nothing is rewritten, buffered or re-encoded — method, headers, query
//! string and the body stream go to the binding as they arrive, so every
//! method, non-JSON body and streaming request body works.
//!
//! # Deviations (per-file ledger)
//!
//! - **D15** — upstream `AiBinding` is a structural TypeScript interface over
//!   the `@cloudflare/workers-types` `Ai` class (its `fetch` is optional only
//!   because the published type does not declare it yet; `aiGatewayLogId`
//!   pins the type to the AI binding). Rust has no structural typing, so the
//!   binding is the plain [`AiBinding`] struct: `ai_gateway_log_id` plus the
//!   binding fetch as a [`FetchFn`] (the crate's fetch seam, crate deviation
//!   D4). The late runtime check for a binding without `fetch()` becomes
//!   `create_ai_binding_fetch` returning `Err` with the upstream `TypeError`
//!   message (Rust has no exceptions).
//! - **D16** — upstream binds `binding.fetch` eagerly because `fetch` is a
//!   mutable property; the Rust [`FetchFn`] is an `Arc`-cloned closure, so
//!   the returned value is the binding fetch itself and passes requests
//!   through untouched by construction.

use crate::http::FetchFn;

/// The Workers AI binding (`env.AI`), described structurally so this module
/// does not depend on the workers runtime (upstream `AiBinding`).
///
/// `fetch` is optional only because the published workers type doesn't
/// declare it yet — every real binding has it at runtime.
/// `ai_gateway_log_id` is here to pin the type to the AI binding: it is the
/// one member unique to `Ai`, so without it this struct would also accept an
/// `AiGateway` or any hand-rolled `{ fetch }` object, which is exactly the
/// mistake the runtime check reports late.
#[derive(Clone, Default)]
pub struct AiBinding {
    pub ai_gateway_log_id: Option<String>,
    pub fetch: Option<FetchFn>,
}

/// Placeholder value for auth headers on binding-routed requests
/// (`CLOUDFLARE_GATEWAY_BINDING_AUTH_SENTINEL`). API implementations require
/// an API key or a recognized auth header (`authorization`, `x-api-key`,
/// `cf-aig-authorization`) before dispatch; binding calls are
/// pre-authenticated, so pass
/// `cf-aig-authorization: Bearer {CLOUDFLARE_GATEWAY_BINDING_AUTH_SENTINEL}`
/// to satisfy the check. The gateway ignores (and strips)
/// `cf-aig-authorization` on binding-routed requests. Pair it with
/// `Authorization: null` / `x-api-key: null` so placeholder auth headers
/// never reach the gateway, which would treat a request-supplied auth header
/// as a BYOK provider key that overrides its stored keys — the same as it
/// would over HTTPS.
pub const CLOUDFLARE_GATEWAY_BINDING_AUTH_SENTINEL: &str = "cloudflare-gateway-binding";

/// Create a `fetch` backed by the AI binding, for models whose `baseUrl`
/// already names a route the binding serves — including the gateway's
/// provider passthrough,
/// `https://workers-binding.ai/ai-gateway/gateways/{gateway}/{provider}/...`.
/// Requests pass through untouched.
///
/// `fetch` is optional on the binding, so its presence is checked here —
/// early, rather than as a confusing failure on the first inference request.
/// The error carries the upstream `TypeError` message verbatim.
pub fn create_ai_binding_fetch(binding: &AiBinding) -> Result<FetchFn, String> {
    match &binding.fetch {
        Some(fetch) => Ok(fetch.clone()),
        None => Err("createAiBindingFetch: the AI binding does not expose fetch()".to_string()),
    }
}

#[cfg(test)]
mod tests {
    // Port of test/cloudflare-ai-binding.test.ts.
    use parking_lot::Mutex;
    use std::sync::Arc;

    use super::*;
    use crate::http::{HttpBody, HttpRequest, HttpResponse};
    use crate::types::{
        BaseModelFields, Context, ModelCost, ModelCostRates, ModelInput, ProviderHeaders,
        SimpleStreamOptions, UserContentField, UserMessage,
    };
    use crate::utils::transcript::normalize_context;

    const BINDING_PREFIX: &str = "https://workers-binding.ai/ai-gateway/gateways/my-gateway";

    /// Body shape of the prepared binding response (`HttpBody` streams are
    /// single-consumer and not `Sync`, so the fake carries a descriptor and
    /// rebuilds the body per call).
    enum PreparedBody {
        Bytes(Vec<u8>),
        Sse,
        Empty,
    }

    /// Recording binding fake (`fakeBinding`): captures every request and
    /// answers with the prepared response.
    fn fake_binding(
        status: u16,
        headers: Vec<(String, String)>,
        body: PreparedBody,
        requests: Arc<Mutex<Vec<HttpRequest>>>,
    ) -> AiBinding {
        AiBinding {
            ai_gateway_log_id: None,
            fetch: Some(FetchFn::new(move |request: HttpRequest| {
                requests.lock().push(request);
                let body = match &body {
                    PreparedBody::Bytes(bytes) => HttpBody::Bytes(bytes.clone()),
                    PreparedBody::Sse => HttpBody::stream(futures_util::stream::iter(vec![Ok(
                        b"data: {}\n\n".to_vec(),
                    )])),
                    PreparedBody::Empty => HttpBody::Empty,
                };
                let prepared = HttpResponse {
                    status,
                    headers: headers.clone(),
                    body,
                };
                Box::pin(async move { Ok(prepared) })
            })),
        }
    }

    fn user_hello() -> Context {
        Context {
            system_prompt: None,
            messages: vec![crate::types::Message::User(UserMessage {
                content: UserContentField::Text("hello".to_string()),
                timestamp: 1,
            })],
            tools: None,
        }
    }

    #[tokio::test]
    async fn passes_requests_to_the_binding_untouched() {
        let requests: Arc<Mutex<Vec<HttpRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let binding = fake_binding(
            200,
            vec![
                ("content-type".to_string(), "text/event-stream".to_string()),
                ("cf-aig-log-id".to_string(), "log-1".to_string()),
            ],
            PreparedBody::Sse,
            requests.clone(),
        );
        // No cast at the call site: an `Ai` binding is accepted as-is, the way `env.AI` is typed.
        let fetch_fn = create_ai_binding_fetch(&binding).expect("binding exposes fetch");
        let body = serde_json::json!({
            "model": "claude",
            "messages": [{ "role": "user", "content": "hi" }],
        })
        .to_string();

        let mut request = HttpRequest::new(
            "POST",
            format!("{BINDING_PREFIX}/anthropic/v1/messages?beta=true"),
        );
        request = request.header("content-type", "application/json");
        request = request.header(
            "cf-aig-authorization",
            format!("Bearer {CLOUDFLARE_GATEWAY_BINDING_AUTH_SENTINEL}"),
        );
        request = request.header("anthropic-version", "2023-06-01");
        request.body = HttpBody::Bytes(body.clone().into_bytes());

        let response = fetch_fn.send(request).await.expect("binding responds");

        let captured = requests.lock();
        assert_eq!(captured.len(), 1);
        assert_eq!(
            captured[0].url,
            format!("{BINDING_PREFIX}/anthropic/v1/messages?beta=true")
        );
        assert_eq!(captured[0].method, "POST");
        assert_eq!(
            captured[0].headers,
            vec![
                ("content-type".to_string(), "application/json".to_string()),
                (
                    "cf-aig-authorization".to_string(),
                    format!("Bearer {CLOUDFLARE_GATEWAY_BINDING_AUTH_SENTINEL}")
                ),
                ("anthropic-version".to_string(), "2023-06-01".to_string()),
            ]
        );
        let sent_body = match &captured[0].body {
            HttpBody::Bytes(bytes) => String::from_utf8_lossy(bytes).to_string(),
            _ => String::new(),
        };
        assert_eq!(sent_body, body);
        // Upstream asserts `response` is the exact prepared binding response;
        // the observable content (status, headers, streamed body) is checked instead.
        assert_eq!(response.status, 200);
        assert_eq!(response.header("cf-aig-log-id"), Some("log-1"));
        let collected = response.body.collect().await.expect("body collects");
        assert_eq!(String::from_utf8_lossy(&collected), "data: {}\n\n");
    }

    #[test]
    fn rejects_a_binding_with_no_fetch_at_construction_not_on_first_request() {
        // A well-typed binding on a runtime that predates `Ai#fetch`: `fetch` is optional on
        // the type, so the check has to happen at runtime — early, with a message that names the cause.
        let error = create_ai_binding_fetch(&AiBinding {
            ai_gateway_log_id: None,
            fetch: None,
        })
        .expect_err("binding without fetch rejects");
        assert!(
            error.contains("does not expose fetch()"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn keeps_sdk_placeholder_auth_off_the_wire_when_paired_with_null_auth_headers() {
        // The full header contract from the module docs, end to end through a real API impl: the
        // sentinel satisfies pi's request-auth check, and the explicit nulls make the SDK delete
        // its own `Authorization: Bearer unused` placeholder before dispatch.
        let requests: Arc<Mutex<Vec<HttpRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let binding = fake_binding(
            400,
            vec![("content-type".to_string(), "application/json".to_string())],
            PreparedBody::Bytes(
                serde_json::json!({ "error": { "type": "bad_request", "message": "stubbed" } })
                    .to_string()
                    .into_bytes(),
            ),
            requests.clone(),
        );
        let model = crate::types::Model {
            base: BaseModelFields {
                id: "test-model".to_string(),
                name: "Test Model".to_string(),
                api: "openai-completions".to_string(),
                provider: "openai".to_string(),
                base_url: format!("{BINDING_PREFIX}/openai"),
                input: vec![ModelInput::Text],
                input_limits: None,
                cost: ModelCost {
                    rates: ModelCostRates {
                        input: 0.0,
                        output: 0.0,
                        cache_read: 0.0,
                        cache_write: 0.0,
                    },
                    tiers: None,
                },
                headers: None,
            },
            model_type: None,
            reasoning: false,
            thinking_level_map: None,
            prompt_cache: None,
            context_window: 10_000,
            max_tokens: 1_000,
            sampling_params: None,
            sampling_params_by_thinking_level: None,
            compat: None,
        };

        let mut options = SimpleStreamOptions::default();
        let mut headers = ProviderHeaders::new();
        headers.insert(
            "cf-aig-authorization".to_string(),
            Some(format!("Bearer {CLOUDFLARE_GATEWAY_BINDING_AUTH_SENTINEL}")),
        );
        headers.insert("Authorization".to_string(), None);
        headers.insert("x-api-key".to_string(), None);
        options.base.base.headers = Some(headers);
        options.base.base.fetch =
            Some(create_ai_binding_fetch(&binding).expect("binding exposes fetch"));
        options.base.base.max_retries = Some(0);

        let stream = crate::api::openai_completions::stream_simple(
            &model,
            &normalize_context(&user_hello()),
            &options,
        );
        let result = stream.result().await;

        assert_eq!(result.stop_reason, crate::types::StopReason::Error);
        let captured = requests.lock();
        assert_eq!(captured.len(), 1);
        assert_eq!(
            captured[0].url,
            format!("{BINDING_PREFIX}/openai/chat/completions")
        );
        let header_names: Vec<String> = captured[0]
            .headers
            .iter()
            .map(|(name, _)| name.to_lowercase())
            .collect();
        assert!(!header_names.contains(&"authorization".to_string()));
        assert!(!header_names.contains(&"x-api-key".to_string()));
    }
}
