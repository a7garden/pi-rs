//! Port of `packages/ai/src/utils/provider-retry.ts`.
//!
//! Reproduces the retry behavior used by the OpenAI and Anthropic SDKs while
//! making their backoff sleep interruptible: callers invoke the request with
//! `maxRetries: 0` at the SDK layer and wrap it with
//! [`retry_provider_request`]. Provider-requested delays above
//! `max_retry_delay_ms` fail immediately (60 seconds by default); set it to
//! `Some(0)` to disable the limit.
//!
//! Deviations (crate docs ledger):
//! - **D17 — the runtime `isProviderError` shape probe becomes the
//!   [`ProviderRequestError`] protocol.** Upstream re-checks
//!   `error instanceof Error && "status" in error` at runtime; Rust errors are
//!   statically typed, so error types declare their `status`/headers/message.
//!   [`crate::http::ProviderHttpError`] implements the protocol directly, and
//!   API modules map richer SDK-shaped errors onto it (see
//!   `api/openai_responses.rs::ResponsesRequestError`). Errors without a
//!   status are retryable, matching upstream `error.status === undefined`.
//! - **D18 — sleeps use `tokio::time::sleep`** raced against the pure-Rust
//!   [`crate::utils::abort::AbortSignal`]; aborts surface through
//!   [`ProviderRequestError::from_abort`] with the upstream `AbortError`
//!   message `"Request aborted"`.
//! - **D18a — `Math.random` jitter uses `getrandom`.**
//! - **D17a — `retry-after` HTTP-date values are parsed natively** (upstream
//!   `Date.parse`); only IMF-fixdate is accepted, numeric seconds otherwise.
//! - **D17b — the thrown "Server requested Ns retry delay" error is built by
//!   [`ProviderRequestError::from_server_delay`]** (upstream throws a fresh
//!   `Error` from `validateServerRetryDelayMs`); the default falls back to
//!   [`ProviderRequestError::from_abort`] for error types that do not carry
//!   ad-hoc messages.

use std::future::Future;
use std::time::Duration;

use crate::http::ProviderHttpError;
use crate::utils::abort::AbortSignal;

/// Upstream `DEFAULT_MAX_RETRY_DELAY_MS`.
const DEFAULT_MAX_RETRY_DELAY_MS: u64 = 60_000;

/// The error protocol [`retry_provider_request`] classifies (upstream the
/// `ProviderError` shape: an `Error` carrying `status`/`headers`).
pub trait ProviderRequestError: Sized {
    /// `error.status`: `None` = unknown (upstream treats as retryable).
    fn status(&self) -> Option<u16>;
    /// Case-insensitive response-header lookup (`Headers.get`).
    fn header(&self, name: &str) -> Option<String>;
    /// `error.message`.
    fn message(&self) -> &str;
    /// The `AbortError` synthesized when the signal aborts (`Request aborted`).
    fn from_abort() -> Self;
    /// The error thrown when a server-requested retry delay exceeds the cap
    /// (upstream `validateServerRetryDelayMs` throwing
    /// `"Server requested Ns retry delay (max: Ms). <message>"`). Defaults to
    /// [`ProviderRequestError::from_abort`] for error types that cannot carry
    /// ad-hoc messages (deviation D17b).
    fn from_server_delay(_delay_ms: f64, _max_delay_ms: f64, _provider_message: &str) -> Self {
        Self::from_abort()
    }
}

impl ProviderRequestError for ProviderHttpError {
    fn status(&self) -> Option<u16> {
        self.status
    }

    fn header(&self, name: &str) -> Option<String> {
        ProviderHttpError::header(self, name).map(str::to_string)
    }

    fn message(&self) -> &str {
        &self.message
    }

    fn from_abort() -> Self {
        ProviderHttpError::new("Request aborted")
    }

    fn from_server_delay(delay_ms: f64, max_delay_ms: f64, provider_message: &str) -> Self {
        ProviderHttpError::new(format!(
            "Server requested {}s retry delay (max: {}s). {}",
            (delay_ms / 1000.0).ceil(),
            (max_delay_ms / 1000.0).ceil(),
            provider_message
        ))
    }
}

/// Upstream `ProviderRetryOptions`.
#[derive(Clone, Default)]
pub struct ProviderRetryOptions {
    /// `options.maxRetries` (`None`/absent = 0 retries).
    pub max_retries: Option<u32>,
    /// Cap for server-requested retry delays; `Some(0)` disables the cap.
    pub max_retry_delay_ms: Option<u64>,
    /// Cooperative cancellation; checked between attempts and during backoff.
    pub signal: Option<AbortSignal>,
    /// HTTP statuses that fail at once although the default policy would retry
    /// them (upstream `noRetryStatuses`).
    pub no_retry_statuses: Option<Vec<u16>>,
}

/// Upstream `isRetryableProviderError`: mirrors the pinned OpenAI/Anthropic SDK
/// retry policy; review when either SDK is upgraded.
fn is_retryable_provider_error<E: ProviderRequestError>(error: &E) -> bool {
    match error.header("x-should-retry").as_deref() {
        Some("true") => return true,
        Some("false") => return false,
        _ => {}
    }
    match error.status() {
        None => true,
        Some(status) => status == 408 || status == 409 || status == 429 || status >= 500,
    }
}

/// Upstream `validateServerRetryDelayMs`.
fn validate_server_retry_delay_ms<E: ProviderRequestError>(
    delay_ms: f64,
    max_retry_delay_ms: Option<u64>,
    error: &E,
) -> Result<f64, E> {
    let max_delay_ms = max_retry_delay_ms.unwrap_or(DEFAULT_MAX_RETRY_DELAY_MS) as f64;
    if max_delay_ms > 0.0 && delay_ms > max_delay_ms {
        return Err(E::from_server_delay(
            delay_ms,
            max_delay_ms,
            error.message(),
        ));
    }
    Ok(delay_ms)
}

/// Upstream `getRetryDelayMs`.
fn get_retry_delay_ms<E: ProviderRequestError>(
    error: &E,
    retry_index: u32,
    max_retry_delay_ms: Option<u64>,
) -> Result<f64, E> {
    if let Some(retry_after_ms) = error.header("retry-after-ms") {
        if let Ok(value) = retry_after_ms.parse::<f64>() {
            if value.is_finite() {
                return validate_server_retry_delay_ms(value, max_retry_delay_ms, error);
            }
        }
    }

    if let Some(retry_after) = error.header("retry-after") {
        let seconds = retry_after.parse::<f64>().ok();
        let delay_ms = match seconds {
            Some(seconds) => seconds * 1000.0,
            None => match parse_http_date_ms(&retry_after) {
                Some(date_ms) => date_ms - now_ms() as f64,
                None => f64::NAN,
            },
        };
        if delay_ms.is_finite() {
            return validate_server_retry_delay_ms(delay_ms, max_retry_delay_ms, error);
        }
    }

    let exponential_delay = (0.5 * 2f64.powi(retry_index as i32)).min(8.0) * 1000.0;
    Ok(exponential_delay * (1.0 - random_fraction() * 0.25))
}

/// Upstream `abortableSleep`: `Err` when the signal aborts during the wait.
async fn abortable_sleep<E: ProviderRequestError>(
    ms: f64,
    signal: Option<&AbortSignal>,
) -> Result<(), E> {
    let ms = ms.max(0.0) as u64;
    match signal {
        Some(signal) if signal.aborted() => Err(E::from_abort()),
        Some(signal) => {
            tokio::select! {
                biased;
                _ = tokio::time::sleep(Duration::from_millis(ms)) => Ok(()),
                _ = signal.aborted_future() => Err(E::from_abort()),
            }
        }
        None => {
            tokio::time::sleep(Duration::from_millis(ms)).await;
            Ok(())
        }
    }
}

/// Reproduce the SDK retry loop while keeping the backoff interruptible.
/// Each retry is a fresh request (upstream `X-Stainless-Retry-Count` stays zero).
///
/// Upstream `retryProviderRequest`.
pub async fn retry_provider_request<T, E, F, Fut>(
    mut request: F,
    options: ProviderRetryOptions,
) -> Result<T, E>
where
    E: ProviderRequestError,
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let max_retries = options.max_retries.unwrap_or(0);
    let mut retries_remaining = max_retries;

    loop {
        let value = match request().await {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };
        if options
            .signal
            .as_ref()
            .is_some_and(|signal| signal.aborted())
        {
            return Err(E::from_abort());
        }
        if retries_remaining == 0 || !is_retryable_provider_error(&value) {
            return Err(value);
        }
        if let Some(status) = value.status() {
            if options
                .no_retry_statuses
                .as_ref()
                .is_some_and(|statuses| statuses.contains(&status))
            {
                return Err(value);
            }
        }

        let retry_index = max_retries - retries_remaining;
        retries_remaining -= 1;
        let delay_ms = get_retry_delay_ms(&value, retry_index, options.max_retry_delay_ms)?;
        abortable_sleep(delay_ms, options.signal.as_ref()).await?;
    }
}

/// Uniform random fraction in `[0, 1)` (upstream `Math.random`).
fn random_fraction() -> f64 {
    let mut bytes = [0u8; 8];
    getrandom::getrandom(&mut bytes).expect("platform randomness is available");
    (u64::from_le_bytes(bytes) >> 11) as f64 / (1u64 << 53) as f64
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

/// Minimal IMF-fixdate parser (`Sun, 06 Nov 1994 08:49:37 GMT`) for
/// `retry-after` HTTP-date values; returns epoch milliseconds.
fn parse_http_date_ms(value: &str) -> Option<f64> {
    let value = value.trim();
    let rest = value
        .strip_suffix(" GMT")
        .or_else(|| value.strip_suffix(" UTC"))?;
    let mut parts = rest.split_whitespace();
    let weekday = parts.next()?;
    if !weekday.ends_with(',') {
        return None;
    }
    let day: u32 = parts.next()?.parse().ok()?;
    let month = match parts.next()? {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: i64 = parts.next()?.parse().ok()?;
    let mut time_parts = parts.next()?.split(':');
    let hour: i64 = time_parts.next()?.parse().ok()?;
    let minute: i64 = time_parts.next()?.parse().ok()?;
    let second: i64 = time_parts.next()?.parse().ok()?;

    // Days from civil (Howard Hinnant's algorithm).
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some((days * 86_400 + hour * 3_600 + minute * 60 + second) as f64 * 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn provider_error(status: Option<u16>, headers: &[(&str, &str)]) -> ProviderHttpError {
        let message = format!(
            "Provider error: {}",
            status
                .map(|status| status.to_string())
                .unwrap_or_else(|| "none".into())
        );
        let mut error = match status {
            Some(status) => ProviderHttpError::from_response(message, status, None),
            None => ProviderHttpError::new(message),
        };
        error.headers = headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        error
    }

    /// Counting request stub: fails with `error` on the first `failures` calls,
    /// then resolves to `"ok"`.
    fn counting_request(
        failures: usize,
        error: ProviderHttpError,
    ) -> (
        impl FnMut() -> futures_core::future::BoxFuture<'static, Result<String, ProviderHttpError>>
            + Send,
        Arc<AtomicUsize>,
    ) {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let request = move || {
            let counter = counter.clone();
            let error = ProviderHttpError {
                message: error.message.clone(),
                status: error.status,
                headers: error.headers.clone(),
                body: error.body.clone(),
                source: error.source.clone(),
            };
            Box::pin(async move {
                if counter.fetch_add(1, Ordering::SeqCst) < failures {
                    Err(error)
                } else {
                    Ok("ok".to_string())
                }
            })
                as futures_core::future::BoxFuture<'static, Result<String, ProviderHttpError>>
        };
        (request, calls)
    }

    #[tokio::test(start_paused = true)]
    async fn retries_retryable_provider_errors() {
        let (request, calls) =
            counting_request(1, provider_error(Some(429), &[("retry-after-ms", "1000")]));
        let options = ProviderRetryOptions {
            max_retries: Some(1),
            ..Default::default()
        };
        let handle = tokio::spawn(retry_provider_request(request, options));

        tokio::time::advance(Duration::from_millis(999)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        tokio::time::advance(Duration::from_millis(1)).await;

        let result = handle.await.expect("task joins").expect("request succeeds");
        assert_eq!(result, "ok");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn does_not_retry_errors_the_provider_marks_as_non_retryable() {
        let (request, calls) =
            counting_request(1, provider_error(Some(429), &[("x-should-retry", "false")]));
        let options = ProviderRetryOptions {
            max_retries: Some(2),
            ..Default::default()
        };

        let result = retry_provider_request(request, options).await;

        assert_eq!(
            result.expect_err("request fails").message,
            "Provider error: 429"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn does_not_retry_statuses_listed_in_no_retry_statuses() {
        let (request, calls) =
            counting_request(1, provider_error(Some(504), &[("retry-after-ms", "0")]));
        let options = ProviderRetryOptions {
            max_retries: Some(2),
            no_retry_statuses: Some(vec![504]),
            ..Default::default()
        };

        let result = retry_provider_request(request, options).await;

        assert_eq!(
            result.expect_err("request fails").message,
            "Provider error: 504"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn rejects_a_provider_requested_retry_delay_above_the_limit() {
        let (request, calls) =
            counting_request(1, provider_error(Some(429), &[("retry-after", "277403")]));
        let options = ProviderRetryOptions {
            max_retries: Some(1),
            max_retry_delay_ms: Some(1000),
            signal: None,
            no_retry_statuses: None,
        };
        let result = retry_provider_request(request, options).await;

        let error = result.expect_err("delay above the limit fails");
        assert!(
            error
                .message
                .starts_with("Server requested 277403s retry delay (max: 1s). "),
            "{}",
            error.message
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn allows_disabling_the_provider_requested_retry_delay_cap() {
        let (request, calls) =
            counting_request(1, provider_error(Some(429), &[("retry-after", "2")]));
        let options = ProviderRetryOptions {
            max_retries: Some(1),
            max_retry_delay_ms: Some(0),
            signal: None,
            no_retry_statuses: None,
        };
        let handle = tokio::spawn(retry_provider_request(request, options));

        tokio::time::advance(Duration::from_millis(1999)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        tokio::time::advance(Duration::from_millis(1)).await;

        let result = handle.await.expect("task joins").expect("request succeeds");
        assert_eq!(result, "ok");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn aborts_a_provider_requested_retry_delay() {
        use crate::utils::abort::AbortController;

        let controller = AbortController::new();
        let (request, calls) = counting_request(
            usize::MAX,
            provider_error(Some(429), &[("retry-after", "277403")]),
        );
        let options = ProviderRetryOptions {
            max_retries: Some(2),
            max_retry_delay_ms: Some(0),
            signal: Some(controller.signal()),
            no_retry_statuses: None,
        };
        let handle = tokio::spawn(retry_provider_request(request, options));

        tokio::time::advance(Duration::from_millis(0)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        controller.abort();

        let result = handle.await.expect("task joins").expect_err("aborted");
        assert_eq!(result.message, "Request aborted");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retries_status_free_transport_errors_without_delay_headers() {
        let (request, calls) = counting_request(1, provider_error(None, &[]));
        let options = ProviderRetryOptions {
            max_retries: Some(1),
            ..Default::default()
        };

        let result = retry_provider_request(request, options).await;

        assert_eq!(result.expect("status-free errors are retryable"), "ok");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn checks_the_abort_signal_between_attempts() {
        use crate::utils::abort::AbortController;

        let controller = AbortController::new();
        controller.abort();
        let (request, calls) = counting_request(usize::MAX, provider_error(Some(500), &[]));
        let options = ProviderRetryOptions {
            max_retries: Some(3),
            max_retry_delay_ms: None,
            signal: Some(controller.signal()),
            no_retry_statuses: None,
        };
        let result = retry_provider_request(request, options).await;

        assert_eq!(
            result.expect_err("already aborted").message,
            "Request aborted"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn parses_imf_fixdate_retry_after_values() {
        assert_eq!(
            parse_http_date_ms("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(784111777000.0)
        );
        assert_eq!(parse_http_date_ms("not a date"), None);
    }
}
