//! Port of `packages/agent/test/proxy.test.ts`.

mod common;

use futures_util::StreamExt;
use pi_agent::{stream_proxy, ProxyStreamOptions};
use pi_ai::http::{FetchFn, HttpBody, HttpResponse};
use pi_ai::types::{
    AssistantMessageEvent, Context, Model, ModelCost, ModelCostRates, ModelInput, Usage,
};
use serde_json::{json, Value};

fn proxy_model() -> Model {
    Model {
        base: pi_ai::types::BaseModelFields {
            id: "gpt-5.4".into(),
            name: "GPT-5.4".into(),
            api: "openai-responses".into(),
            provider: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
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
        reasoning: true,
        thinking_level_map: None,
        prompt_cache: None,
        context_window: 400_000,
        max_tokens: 128_000,
        sampling_params: None,
        sampling_params_by_thinking_level: None,
        compat: None,
    }
}

fn proxy_usage() -> Usage {
    Usage::default()
}

/// Serialize SSE lines the way the fake servers do upstream.
fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| format!("data: {}\n\n", event))
        .collect()
}

/// Fake fetch returning a 200 response with a fully buffered SSE body
/// (upstream `vi.stubGlobal("fetch", ...)`; crate docs D5).
fn fake_fetch(body: String) -> FetchFn {
    FetchFn::new(move |_request| {
        let body = body.clone();
        Box::pin(async move {
            Ok(HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: HttpBody::bytes(body.into_bytes()),
            })
        })
    })
}

fn proxy_options(fetch: FetchFn) -> ProxyStreamOptions {
    ProxyStreamOptions {
        auth_token: "test-token".into(),
        proxy_url: "https://proxy.example.com".into(),
        fetch: Some(fetch),
        ..Default::default()
    }
}

async fn run(
    stream: pi_ai::utils::event_stream::AssistantMessageEventStream,
) -> (Vec<AssistantMessageEvent>, pi_ai::types::AssistantMessage) {
    let mut rx = stream.clone();
    let mut events = Vec::new();
    while let Some(event) = rx.next().await {
        events.push(event);
    }
    let result = stream.result().await;
    (events, result)
}

#[tokio::test]
async fn preserves_tool_call_metadata_received_only_on_toolcall_end() {
    let proxy_events = vec![
        json!({ "type": "start" }),
        json!({ "type": "toolcall_start", "contentIndex": 0, "id": "call_test|fc_test", "toolName": "lookup" }),
        json!({ "type": "toolcall_delta", "contentIndex": 0, "delta": "{\"value\":\"hello\"}" }),
        json!({
            "type": "toolcall_end",
            "contentIndex": 0,
            "toolCall": {
                "type": "toolCall",
                "id": "call_test|fc_test",
                "name": "lookup",
                "arguments": { "value": "hello" },
                "namespace": "dynamic_tools"
            }
        }),
        json!({ "type": "done", "reason": "toolUse", "usage": serde_json::to_value(proxy_usage()).unwrap() }),
    ];
    let body = sse(&proxy_events);

    let stream = stream_proxy(
        &proxy_model(),
        &pi_ai::utils::transcript::normalize_context(&Context {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: None,
        }),
        proxy_options(fake_fetch(body)),
    );
    let (events, result) = run(stream).await;
    let end_event = events.iter().find_map(|event| match event {
        AssistantMessageEvent::ToolcallEnd { tool_call, .. } => Some(tool_call.clone()),
        _ => None,
    });

    let end_event = end_event.expect("toolcall_end event");
    assert_eq!(end_event.namespace.as_deref(), Some("dynamic_tools"));
    match &result.content[0] {
        pi_ai::types::AssistantContent::ToolCall(tool_call) => {
            assert_eq!(tool_call.arguments.get("value"), Some(&json!("hello")));
            assert_eq!(tool_call.namespace.as_deref(), Some("dynamic_tools"));
        }
        other => panic!("expected toolCall content, got {other:?}"),
    }
}

// Regression tests for https://github.com/earendil-works/pi/issues/8996
#[tokio::test]
async fn processes_terminal_metadata_when_the_event_is_not_newline_terminated() {
    let start = format!("data: {}\n\n", json!({ "type": "start" }));
    let done = json!({
        "type": "done",
        "reason": "stop",
        "usage": serde_json::to_value(proxy_usage()).unwrap(),
        "providerThinkingLevel": "high"
    })
    .to_string();

    let stream = stream_proxy(
        &proxy_model(),
        &pi_ai::utils::transcript::normalize_context(&Context {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: None,
        }),
        proxy_options(fake_fetch(format!("{start}data: {done}"))),
    );
    let (events, result) = run(stream).await;

    let types: Vec<&'static str> = events
        .iter()
        .map(|event| match event {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
            _ => "other",
        })
        .collect();
    assert_eq!(types, vec!["start", "done"]);
    assert_eq!(result.stop_reason, pi_ai::types::StopReason::Stop);
    assert_eq!(result.provider_thinking_level.as_deref(), Some("high"));
}

#[tokio::test]
async fn emits_an_error_instead_of_hanging_when_the_stream_ends_without_a_terminal_event() {
    let body = format!("data: {}\n\n", json!({ "type": "start" }));

    let stream = stream_proxy(
        &proxy_model(),
        &pi_ai::utils::transcript::normalize_context(&Context {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: None,
        }),
        proxy_options(fake_fetch(body)),
    );
    let (events, result) = run(stream).await;

    let types: Vec<&'static str> = events
        .iter()
        .map(|event| match event {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::Error { .. } => "error",
            _ => "other",
        })
        .collect();
    assert_eq!(types, vec!["start", "error"]);
    assert_eq!(result.stop_reason, pi_ai::types::StopReason::Error);
    let error_message = result.error_message.expect("error message");
    assert!(
        error_message.contains("Connection closed by proxy server"),
        "unexpected: {error_message}"
    );
}
