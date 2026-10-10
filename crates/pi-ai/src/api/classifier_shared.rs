//! Port of `packages/ai/src/api/classifier-shared.ts` — the shared HTTP client
//! for classifier APIs (System One services, OpenAI Decisions).
//!
//! Deviations (carried over from the system-one port):
//! - D24: upstream throws `ClassifierHttpError` objects carrying
//!   `status`/`headers`/`body`; Rust folds that shape into
//!   [`crate::http::ProviderHttpError`], the crate's single provider error
//!   carrier whose completed-response fields (`status`, `headers`, `body`)
//!   feed [`crate::utils::provider_retry`] and
//!   [`crate::utils::error_body`].
//! - D25: upstream builds per-attempt `AbortSignal.timeout` signals; Rust has
//!   no such primitive, so each attempt starts a timer that aborts a
//!   pure-Rust [`crate::utils::abort::AbortSignal`], combined with the caller
//!   signal via `AbortSignal::any`. When the timer fires and the caller signal
//!   has not, the failure is converted to the upstream `"Request timed out
//!   after Nms"` error.
//! - D26: the shared `onPayload`/`onResponse` hooks are typed with the chat
//!   [`Model`] in this crate (foundation `PayloadHook`/`ResponseHook`); the
//!   classifier model is passed as a chat-`Model` view (same id/provider/api/
//!   baseUrl/headers/cost) because [`crate::types::ClassifierModel`] cannot be
//!   converted losslessly. Hooks reading only model identity — the upstream
//!   contract for these hooks — observe correct values.

use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::http::{FetchFn, HttpBody, HttpRequest, HttpResponse, ProviderHttpError};
use crate::models::calculate_cost;
use crate::types::{
    AnyModel, ClassifierModel, ClassifierOptions, Model, ProviderHeaders, ProviderResponse, Usage,
    UsageCost,
};
use crate::utils::abort::{AbortController, AbortSignal};
use crate::utils::headers::{headers_to_record, provider_headers_to_record};
use crate::utils::provider_retry::{retry_provider_request, ProviderRetryOptions};

/// `httpError`: an HTTP failure in the shape `retryProviderRequest` and
/// `normalizeProviderError` understand.
pub fn http_error(
    label: &str,
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
) -> ProviderHttpError {
    ProviderHttpError {
        message: format!("{label} returned {status}"),
        status: Some(status),
        headers,
        body: Some(body),
        source: None,
    }
}

/// `timeoutError`: `name: "TimeoutError"`, no status — always retryable
/// upstream; here a plain message error.
pub fn timeout_error(timeout_ms: u64) -> ProviderHttpError {
    ProviderHttpError::new(format!("Request timed out after {timeout_ms}ms"))
}

/// `isRecord`: a JSON object (not array/null).
pub fn is_record(value: &Value) -> bool {
    value.is_object()
}

/// `requiredNumber`: a finite number, or an error naming the field.
pub fn required_number(
    label: &str,
    value: Option<&Value>,
    field: &str,
) -> Result<f64, ProviderHttpError> {
    match value
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
    {
        Some(number) => Ok(number),
        None => Err(ProviderHttpError::new(format!(
            "{label} returned an invalid {field}"
        ))),
    }
}

/// `requestHeaders`: bearer auth and JSON content type, merged with model and
/// caller headers (case-insensitive, `null` suppresses).
pub fn request_headers(
    model: &ClassifierModel,
    api_key: &str,
    options_headers: Option<&ProviderHeaders>,
) -> Vec<(String, String)> {
    let mut base = ProviderHeaders::new();
    base.insert(
        "authorization".to_string(),
        Some(format!("Bearer {api_key}")),
    );
    base.insert(
        "content-type".to_string(),
        Some("application/json".to_string()),
    );
    let model_headers: Option<ProviderHeaders> = model.base.headers.as_ref().map(|headers| {
        headers
            .iter()
            .map(|(name, value)| (name.clone(), Some(value.clone())))
            .collect()
    });
    provider_headers_to_record(&[Some(&base), model_headers.as_ref(), options_headers])
        .unwrap_or_default()
        .into_iter()
        .collect()
}

/// Chat-`Model` view of the classifier model for the shared request hooks
/// (deviation D26).
pub fn classifier_model_view(model: &ClassifierModel) -> Model {
    Model {
        base: model.base.clone(),
        model_type: None,
        reasoning: false,
        thinking_level_map: None,
        prompt_cache: None,
        context_window: 0,
        max_tokens: 0,
        sampling_params: None,
        sampling_params_by_thinking_level: None,
        compat: None,
    }
}

/// `tokenCount`: positive finite counts only; everything else is 0.
fn token_count(value: Option<&Value>) -> u64 {
    value
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite() && *number > 0.0)
        .map(|number| number as u64)
        .unwrap_or(0)
}

/// `parseClassifierUsage`: usage from a `{ input_tokens, output_tokens }`
/// object, priced from the model catalog like chat usage. A missing or
/// malformed usage object leaves the result without usage instead of failing
/// it.
pub fn parse_classifier_usage(value: Option<&Value>, model: &ClassifierModel) -> Option<Usage> {
    let object = value.and_then(Value::as_object)?;
    if !object.contains_key("input_tokens") && !object.contains_key("output_tokens") {
        return None;
    }
    let input = token_count(object.get("input_tokens"));
    let output = token_count(object.get("output_tokens"));
    let mut usage = Usage {
        input,
        output,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: input + output,
        cost: UsageCost::default(),
    };
    calculate_cost(&AnyModel::Classifier(model.clone()), &mut usage);
    Some(usage)
}

/// `postClassifierRequest`: posts one JSON classifier request with bearer
/// auth, `onPayload`/`onResponse` hooks, a fresh timeout per attempt, and
/// provider retries. Returns the parsed response body; fails with a
/// [`ProviderHttpError`]. `no_retry_statuses` lists HTTP statuses that fail at
/// once although they are normally retried.
pub async fn post_classifier_request(
    label: &str,
    url: String,
    model: &ClassifierModel,
    payload: Value,
    options: Option<&ClassifierOptions>,
    no_retry_statuses: Option<&[u16]>,
) -> Result<Value, ProviderHttpError> {
    // Upstream: `if (!options?.apiKey) throw new Error(...)`.
    let Some(options) = options else {
        return Err(ProviderHttpError::new(format!(
            "No API key for provider: {}",
            model.provider()
        )));
    };
    let Some(api_key) = options.base.api_key.clone() else {
        return Err(ProviderHttpError::new(format!(
            "No API key for provider: {}",
            model.provider()
        )));
    };
    let mut payload = payload;
    if let Some(hook) = &options.base.on_payload {
        // (deviation D26) hook receives a chat-Model view of the classifier.
        if let Some(transformed) = hook(payload.clone(), classifier_model_view(model)).await {
            payload = transformed;
        }
    }
    let request_fetch: FetchFn = options.base.fetch.clone().unwrap_or_default();
    let caller_signal = options.base.signal.clone();
    let attempt = || async {
        // Fresh timeout per attempt (deviation D25).
        let timeout_signal: Option<AbortSignal> = options.base.timeout_ms.map(|timeout_ms| {
            let controller = AbortController::new();
            let signal = controller.signal();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
                controller.abort();
            });
            signal
        });
        let request_signal = match (&caller_signal, &timeout_signal) {
            (Some(caller), Some(timeout)) => {
                Some(AbortSignal::any([caller.clone(), timeout.clone()]))
            }
            (Some(caller), None) => Some(caller.clone()),
            (None, Some(timeout)) => Some(timeout.clone()),
            (None, None) => None,
        };
        let timed_out = || -> Option<ProviderHttpError> {
            let fired = timeout_signal
                .as_ref()
                .map(|signal| signal.aborted())
                .unwrap_or(false);
            let caller_aborted = caller_signal
                .as_ref()
                .map(|signal| signal.aborted())
                .unwrap_or(false);
            if fired && !caller_aborted {
                Some(timeout_error(
                    options.base.timeout_ms.expect("timeout configured"),
                ))
            } else {
                None
            }
        };

        let mut request = HttpRequest::new("POST", url.clone());
        for (name, value) in request_headers(model, &api_key, options.base.headers.as_ref()) {
            request = request.header(name, value);
        }
        let body = serde_json::to_vec(&payload).expect("request body serializes");
        request.body = HttpBody::Bytes(body);
        request.signal = request_signal;

        let HttpResponse {
            status,
            headers,
            body,
        } = match request_fetch.send(request).await {
            Ok(response) => response,
            Err(error) => return Err(timed_out().unwrap_or(error)),
        };
        if !(200..300).contains(&status) {
            let body = match body.collect().await {
                Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                Err(error) => return Err(timed_out().unwrap_or(error)),
            };
            return Err(timed_out().unwrap_or_else(|| http_error(label, status, headers, body)));
        }
        let bytes = match body.collect().await {
            Ok(bytes) => bytes,
            Err(error) => return Err(timed_out().unwrap_or(error)),
        };
        let parsed: Value = match serde_json::from_slice(&bytes) {
            Ok(parsed) => parsed,
            Err(error) => {
                return Err(ProviderHttpError::with_source(
                    "invalid JSON response".to_string(),
                    Arc::new(error),
                ))
            }
        };
        Ok((status, headers, parsed))
    };
    let retry_options = ProviderRetryOptions {
        max_retries: Some(options.base.max_retries.unwrap_or(2)),
        max_retry_delay_ms: options.base.max_retry_delay_ms,
        signal: caller_signal.clone(),
        no_retry_statuses: no_retry_statuses.map(|statuses| statuses.to_vec()),
    };
    let (status, headers, parsed) = retry_provider_request(attempt, retry_options).await?;

    if let Some(hook) = &options.base.on_response {
        // (deviation D26) hook receives a chat-Model view of the classifier.
        hook(
            ProviderResponse {
                status,
                headers: headers_to_record(&headers),
            },
            classifier_model_view(model),
        )
        .await;
    }
    Ok(parsed)
}
