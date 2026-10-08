//! Port of `packages/coding-agent/src/utils/management-http.ts` — bounded
//! immediate retry for idempotent management requests (version checks,
//! catalogs, downloads).
//!
//! This is intentionally a transport-level helper for idempotent management
//! requests. It must not be used for agent/model operations: those can fail
//! after the HTTP request starts and are retried by their semantic caller
//! instead.
//!
//! Deviation (crate docs, **D12**): JS `fetch`/`AbortSignal` becomes a
//! [`Fetcher`] trait seam with a working default [`ReqwestFetcher`] (requires
//! the `reqwest` dependency — coordinator: add
//! `reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"] }`
//! to this crate). Caller cancellation is terminal via dropping the returned
//! future; `timeout_ms` is a terminal overall budget, `attempt_timeout_ms`
//! aborts only the current attempt so a hung connection can be retried.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

/// Status codes retried when `retry_on_status` is set (upstream
/// `RETRYABLE_STATUS_CODES`).
const RETRYABLE_STATUS_CODES: [u16; 7] = [408, 425, 429, 500, 502, 503, 504];

/// A minimal HTTP response (upstream `Response` subset used by callers).
#[derive(Debug, Clone)]
pub struct HttpResponse {
	pub status: u16,
	pub body: Vec<u8>,
}

impl HttpResponse {
	pub fn ok(&self) -> bool {
		self.status >= 200 && self.status < 300
	}
}

/// Transport seam standing in for JS `fetch` (D12).
pub trait Fetcher: Send + Sync {
	/// Perform a GET-ish request; `headers` are appended, `timeout_ms` bounds
	/// the single attempt.
	fn fetch(
		&self,
		url: &str,
		headers: &[(String, String)],
		timeout_ms: Option<u64>,
	) -> std::pin::Pin<Box<dyn Future<Output = std::io::Result<HttpResponse>> + Send + '_>>;
}

/// Default [`Fetcher`] backed by `reqwest` (rustls, no default features).
#[derive(Debug, Clone)]
pub struct ReqwestFetcher {
	client: Option<reqwest::Client>,
}

impl Default for ReqwestFetcher {
	fn default() -> Self {
		ReqwestFetcher { client: reqwest::Client::builder().build().ok() }
	}
}

impl ReqwestFetcher {
	fn client(&self) -> std::io::Result<&reqwest::Client> {
		if let Some(client) = &self.client {
			return Ok(client);
		}
		Err(std::io::Error::other("fetch failed"))
	}
}

impl Fetcher for ReqwestFetcher {
	fn fetch(
		&self,
		url: &str,
		headers: &[(String, String)],
		timeout_ms: Option<u64>,
	) -> std::pin::Pin<Box<dyn Future<Output = std::io::Result<HttpResponse>> + Send + '_>> {
		let client = self.client().ok().cloned();
		let url = url.to_string();
		let headers = headers.to_vec();
		Box::pin(async move {
			let client = client.as_ref().ok_or_else(|| std::io::Error::other("fetch failed"))?;
			let mut request = client.get(&url);
			for (name, value) in &headers {
				request = request.header(name, value);
			}
			if let Some(timeout_ms) = timeout_ms {
				request = request.timeout(Duration::from_millis(timeout_ms));
			}
			let response = request.send().await.map_err(|e| std::io::Error::other(format!("fetch failed ({e})")))?;
			let status = response.status().as_u16();
			let body = response.bytes().await.map_err(|e| std::io::Error::other(format!("fetch failed ({e})")))?;
			Ok(HttpResponse { status, body: body.to_vec() })
		})
	}
}

/// Retry options (upstream `FetchRetryOptions`).
#[derive(Debug, Clone, Default)]
pub struct FetchRetryOptions {
	/// Number of additional attempts after the initial request. Defaults to two.
	pub max_retries: Option<u32>,
	/// Retry transient HTTP responses as well as transport failures. Defaults to true.
	pub retry_on_status: Option<bool>,
	/// Overall time budget shared by all attempts.
	pub timeout_ms: Option<u64>,
	/// Per-attempt timeout. A new timeout is created for every attempt.
	pub attempt_timeout_ms: Option<u64>,
}

fn is_retryable_status(status: u16) -> bool {
	RETRYABLE_STATUS_CODES.contains(&status)
}

/// Fetch a management HTTP resource with a bounded immediate retry
/// (`fetchWithRetry`).
pub async fn fetch_with_retry(
	fetcher: Arc<dyn Fetcher>,
	input: &str,
	headers: &[(String, String)],
	options: &FetchRetryOptions,
) -> std::io::Result<HttpResponse> {
	let max_retries = match options.max_retries {
		None => 2u32,
		Some(max_retries) => max_retries,
	};
	let retry_on_status = options.retry_on_status.unwrap_or(true);
	let overall_deadline = options
		.timeout_ms
		.filter(|ms| *ms > 0)
		.map(|ms| tokio::time::Instant::now() + Duration::from_millis(ms));
	let attempt_timeout_ms = options.attempt_timeout_ms.filter(|ms| *ms > 0);

	let mut attempt: u32 = 0;
	loop {
		if let Some(deadline) = overall_deadline {
			if tokio::time::Instant::now() >= deadline {
				return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "The operation was aborted due to timeout"));
			}
		}

		let result = {
			let remaining = overall_deadline
				.map(|deadline| deadline.saturating_duration_since(tokio::time::Instant::now()).as_millis() as u64);
			let attempt_timeout = attempt_timeout_ms.map(|ms| ms.min(remaining.unwrap_or(u64::MAX)));
			let timeout = attempt_timeout.or(remaining);
			fetcher.fetch(input, headers, timeout).await
		};

		match result {
			Ok(response) => {
				let should_retry = retry_on_status && is_retryable_status(response.status) && attempt < max_retries;
				if !should_retry {
					return Ok(response);
				}
				// The response body is discarded before the retry; nothing
				// useful to do if cancelling it also fails (no-op here).
			}
			Err(error) => {
				let overall_timed_out = overall_deadline
					.map(|deadline| tokio::time::Instant::now() >= deadline)
					.unwrap_or(false);
				if overall_timed_out || attempt >= max_retries {
					return Err(error);
				}
			}
		}

		attempt += 1;
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::atomic::{AtomicU32, Ordering};

	/// Fetcher that fails `fail_times` times, then returns `response`.
	struct MockFetcher {
		calls: Arc<AtomicU32>,
		fail_times: u32,
		response: HttpResponse,
	}

	impl Fetcher for MockFetcher {
		fn fetch(
			&self,
			_url: &str,
			_headers: &[(String, String)],
			_timeout_ms: Option<u64>,
		) -> std::pin::Pin<Box<dyn Future<Output = std::io::Result<HttpResponse>> + Send + '_>> {
			let calls = Arc::clone(&self.calls);
			let fail_times = self.fail_times;
			let response = self.response.clone();
			Box::pin(async move {
				if calls.fetch_add(1, Ordering::SeqCst) < fail_times {
					return Err(std::io::Error::other("fetch failed"));
				}
				Ok(response)
			})
		}
	}

	fn mock(fail_times: u32, status: u16) -> (Arc<dyn Fetcher>, Arc<AtomicU32>) {
		let calls = Arc::new(AtomicU32::new(0));
		(
			Arc::new(MockFetcher {
				calls: Arc::clone(&calls),
				fail_times,
				response: HttpResponse { status, body: b"{}".to_vec() },
			}),
			calls,
		)
	}

	#[tokio::test]
	async fn retries_transport_failures_up_to_max_retries() {
		let (fetcher, calls) = mock(2, 200);
		let response = fetch_with_retry(fetcher.clone(), "https://example.test/", &[], &FetchRetryOptions::default()).await.unwrap();
		assert_eq!(response.status, 200);
		assert_eq!(calls.load(Ordering::SeqCst), 3);
	}

	#[tokio::test]
	async fn gives_up_after_max_retries() {
		let (fetcher, calls) = mock(u32::MAX, 200);
		assert!(fetch_with_retry(fetcher.clone(), "https://example.test/", &[], &FetchRetryOptions::default()).await.is_err());
		assert_eq!(calls.load(Ordering::SeqCst), 3); // initial + 2 retries
	}

	#[tokio::test]
	async fn retries_transient_statuses_when_enabled() {
		let (fetcher, calls) = mock(0, 503);
		// A 503 after exhausting retries is still returned, not an error.
		let response = fetch_with_retry(fetcher.clone(), "https://example.test/", &[], &FetchRetryOptions::default()).await.unwrap();
		assert_eq!(response.status, 503);
		assert_eq!(calls.load(Ordering::SeqCst), 3);
	}

	#[tokio::test]
	async fn does_not_retry_statuses_when_disabled() {
		let (fetcher, calls) = mock(0, 503);
		let response = fetch_with_retry(
			fetcher.clone(),
			"https://example.test/",
			&[],
			&FetchRetryOptions { retry_on_status: Some(false), ..Default::default() },
		)
		.await
		.unwrap();
		assert_eq!(response.status, 503);
		assert_eq!(calls.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn overall_timeout_is_terminal() {
		let (fetcher, calls) = mock(u32::MAX, 200);
		let started = tokio::time::Instant::now();
		let result = fetch_with_retry(
			fetcher,
			"https://example.test/",
			&[],
			&FetchRetryOptions { timeout_ms: Some(100), ..Default::default() },
		)
		.await;
		assert!(result.is_err());
		assert!(started.elapsed() < Duration::from_secs(5));
		// Instant transport failures still respect max_retries; the deadline
		// only bounds attempts that would run past it.
		assert_eq!(calls.load(Ordering::SeqCst), 3);
	}
}
