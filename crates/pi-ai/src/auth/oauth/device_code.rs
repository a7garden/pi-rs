//! Port of `packages/ai/src/auth/oauth/device-code.ts` — the shared RFC 8628
//! polling loop. Timing behavior matches upstream exactly (minimum 1s
//! interval, 5s default, +5s `slow_down` increment, optional wait before the
//! first poll, deadline from `expiresInSeconds`).

use std::future::Future;
use std::sync::Arc;

use crate::utils::abort::AbortSignal;
use crate::utils::thrown::ThrownValue;

const CANCEL_MESSAGE: &str = "Login cancelled";
const TIMEOUT_MESSAGE: &str = "Device flow timed out";
const SLOW_DOWN_TIMEOUT_MESSAGE: &str = "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again.";
const MINIMUM_INTERVAL_MS: u64 = 1000;
/// RFC 8628 section 3.2: if the authorization server omits `interval`, the
/// client must use 5 seconds.
const DEFAULT_POLL_INTERVAL_SECONDS: f64 = 5.0;
/// RFC 8628 section 3.5: `slow_down` means the polling interval must increase
/// by 5 seconds.
const SLOW_DOWN_INTERVAL_INCREMENT_MS: u64 = 5000;

/// Result of one poll attempt (upstream `OAuthDeviceCodePollResult`).
pub enum OAuthDeviceCodePollResult<T> {
	Pending,
	SlowDown { interval_seconds: Option<f64> },
	Failed { message: String },
	Complete { value: T },
}

/// Options bag for [`poll_oauth_device_code_flow`] (upstream
/// `OAuthDeviceCodePollOptions` minus the `poll` callback, which is a
/// separate parameter like upstream's object field).
pub struct OAuthDeviceCodePollOptions {
	pub interval_seconds: Option<f64>,
	pub expires_in_seconds: Option<f64>,
	pub wait_before_first_poll: bool,
	pub signal: AbortSignal,
}

/// Upstream `now()` — wall clock, in milliseconds. Test hooks record poll
/// times with it; the flow deadline itself uses the tokio clock (D43) so
/// paused-clock tests advance it the way vitest fake timers advance Date.now.
fn unix_now_ms() -> i64 {
	std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|duration| duration.as_millis() as i64)
		.unwrap_or(0)
}

/// Sleep that rejects with `cancelMessage` when the signal aborts (upstream
/// `abortableSleep`). Cancellation is error `Login cancelled`.
pub async fn abortable_sleep(ms: u64, signal: &AbortSignal, cancel_message: &str) -> Result<(), ThrownValue> {
	if signal.aborted() {
		return Err(ThrownValue::message(cancel_message));
	}
	let sleep = tokio::time::sleep(std::time::Duration::from_millis(ms));
	tokio::select! {
		biased;
		reason = signal.aborted_future() => {
			let _ = reason;
			Err(ThrownValue::message(cancel_message))
		}
		() = sleep => Ok(()),
	}
}

/// The generic RFC 8628 polling loop (upstream `pollOAuthDeviceCodeFlow`).
/// The `poll` callback may fail; a failure aborts the whole flow with the
/// same error (upstream a throw inside the callback).
pub async fn poll_oauth_device_code_flow<T, F, Fut>(
	options: OAuthDeviceCodePollOptions,
	mut poll: F,
) -> Result<T, ThrownValue>
where
	F: FnMut() -> Fut,
	Fut: Future<Output = Result<OAuthDeviceCodePollResult<T>, ThrownValue>>,
{
	let deadline: Option<tokio::time::Instant> = options
		.expires_in_seconds
		.map(|seconds| tokio::time::Instant::now() + std::time::Duration::from_millis((seconds * 1000.0) as u64));
	let mut interval_ms = std::cmp::max(
		MINIMUM_INTERVAL_MS,
		(options.interval_seconds.unwrap_or(DEFAULT_POLL_INTERVAL_SECONDS) * 1000.0) as u64,
	);

	let mut slow_down_responses: u32 = 0;
	if options.wait_before_first_poll {
		let remaining_ms = deadline
			.map(|deadline| deadline.saturating_duration_since(tokio::time::Instant::now()).as_millis() as u64)
			.unwrap_or(i64::MAX as u64);
		if remaining_ms > 0 {
			abortable_sleep(
				std::cmp::min(interval_ms, remaining_ms as u64),
				&options.signal,
				CANCEL_MESSAGE,
			)
			.await?;
		}
	}

	while deadline.is_none_or(|deadline| tokio::time::Instant::now() < deadline) {
		if options.signal.aborted() {
			return Err(ThrownValue::message(CANCEL_MESSAGE));
		}

		let result = poll().await?;
		match result {
			OAuthDeviceCodePollResult::Complete { value } => return Ok(value),
			OAuthDeviceCodePollResult::Failed { message } => return Err(ThrownValue::message(message)),
			OAuthDeviceCodePollResult::SlowDown { interval_seconds } => {
				slow_down_responses += 1;
				// Use the server-provided interval when given (GitHub reports the new
				// required minimum in `interval`); trusting only a client-tracked value
				// risks polling early forever under WSL/VM clock drift. Otherwise apply
				// RFC 8628 section 3.5: increase by 5 seconds.
				interval_ms = match interval_seconds {
					Some(seconds) if seconds.is_finite() && seconds > 0.0 => {
						std::cmp::max(MINIMUM_INTERVAL_MS, (seconds * 1000.0) as u64)
					}
					_ => std::cmp::max(MINIMUM_INTERVAL_MS, interval_ms + SLOW_DOWN_INTERVAL_INCREMENT_MS),
				};
			}
			OAuthDeviceCodePollResult::Pending => {}
		}

		let remaining_ms = deadline
			.map(|deadline| deadline.saturating_duration_since(tokio::time::Instant::now()).as_millis() as u64)
			.unwrap_or(i64::MAX as u64);
		if remaining_ms == 0 {
			break;
		}

		abortable_sleep(
			std::cmp::min(interval_ms, remaining_ms as u64),
			&options.signal,
			CANCEL_MESSAGE,
		)
		.await?;
	}

	Err(ThrownValue::message(if slow_down_responses > 0 {
		SLOW_DOWN_TIMEOUT_MESSAGE
	} else {
		TIMEOUT_MESSAGE
	}))
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::{Arc};
	use {parking_lot::Mutex};

	fn never_aborted_signal() -> AbortSignal {
		crate::utils::abort::AbortController::new().signal()
	}

	/// Records `unix_now_ms()` at each poll (upstream `pollTimes`).
	#[derive(Clone)]
	struct PollTimes(Arc<Mutex<Vec<i64>>>);

	impl PollTimes {
		fn new() -> PollTimes {
			PollTimes(Arc::new(Mutex::new(Vec::new())))
		}

		fn push_now(&self) {
			self.0.lock().push(unix_now_ms());
		}

		fn len(&self) -> usize {
			self.0.lock().len()
		}
	}

	// Port of test/oauth-device-code.test.ts "polls immediately and returns the
	// completed value".
	#[tokio::test(start_paused = true)]
	async fn polls_immediately_and_returns_the_completed_value() {
		let times = PollTimes::new();
		let record = times.clone();
		let first = Arc::new(Mutex::new(true));
		let poll_first = first.clone();
		let result = tokio::spawn(poll_oauth_device_code_flow(
			OAuthDeviceCodePollOptions {
				interval_seconds: Some(2.0),
				expires_in_seconds: Some(30.0),
				wait_before_first_poll: false,
				signal: never_aborted_signal(),
			},
			move || {
				times.push_now();
				let is_first = {
					let mut flag = poll_first.lock();
					let was = *flag;
					*flag = false;
					was
				};
				let fut = if is_first {
					std::future::ready(Ok(OAuthDeviceCodePollResult::<String>::Pending))
				} else {
					std::future::ready(Ok(OAuthDeviceCodePollResult::Complete { value: "token".to_string() }))
				};
				async move { fut.await }
			},
		));

		tokio::time::advance(std::time::Duration::from_millis(0)).await;
		assert_eq!(record.len(), 1);

		tokio::time::advance(std::time::Duration::from_millis(1999)).await;
		assert_eq!(record.len(), 1);

		tokio::time::advance(std::time::Duration::from_millis(1)).await;
		assert_eq!(result.await.expect("task").expect("result"), "token");
		assert_eq!(record.len(), 2);
	}

	// Port of "can wait before the first poll".
	#[tokio::test(start_paused = true)]
	async fn can_wait_before_the_first_poll() {
		let times = PollTimes::new();
		let record = times.clone();
		let result = tokio::spawn(poll_oauth_device_code_flow(
			OAuthDeviceCodePollOptions {
				interval_seconds: Some(2.0),
				expires_in_seconds: Some(30.0),
				wait_before_first_poll: true,
				signal: never_aborted_signal(),
			},
			move || {
				times.push_now();
				async { Ok(OAuthDeviceCodePollResult::<String>::Complete { value: "token".to_string() }) }
			},
		));

		tokio::time::advance(std::time::Duration::from_millis(1999)).await;
		assert_eq!(record.len(), 0);

		tokio::time::advance(std::time::Duration::from_millis(1)).await;
		assert_eq!(result.await.expect("task").expect("result"), "token");
		assert_eq!(record.len(), 1);
	}

	// Port of "increases the interval by 5 seconds after slow_down without a
	// server interval".
	#[tokio::test(start_paused = true)]
	async fn increases_interval_by_five_seconds_after_slow_down() {
		let times = PollTimes::new();
		let record = times.clone();
		let results: Arc<Mutex<Vec<OAuthDeviceCodePollResult<String>>>> = Arc::new(Mutex::new(vec![
			OAuthDeviceCodePollResult::SlowDown { interval_seconds: None },
			OAuthDeviceCodePollResult::Complete { value: "token".to_string() },
		]));
		let results_for_poll = results.clone();
		let result = tokio::spawn(poll_oauth_device_code_flow(
			OAuthDeviceCodePollOptions {
				interval_seconds: Some(2.0),
				expires_in_seconds: Some(900.0),
				wait_before_first_poll: false,
				signal: never_aborted_signal(),
			},
			move || {
				times.push_now();
				let next = results_for_poll.lock().remove(0);
				async move { Ok(next) }
			},
		));

		tokio::time::advance(std::time::Duration::from_millis(0)).await;
		assert_eq!(record.len(), 1);

		tokio::time::advance(std::time::Duration::from_millis(6999)).await;
		assert_eq!(record.len(), 1);

		tokio::time::advance(std::time::Duration::from_millis(1)).await;
		assert_eq!(result.await.expect("task").expect("result"), "token");
		assert_eq!(record.len(), 2);
	}

	// Port of "honors a server-provided slow_down interval".
	#[tokio::test(start_paused = true)]
	async fn honors_server_provided_slow_down_interval() {
		let times = PollTimes::new();
		let record = times.clone();
		let results: Arc<Mutex<Vec<OAuthDeviceCodePollResult<String>>>> = Arc::new(Mutex::new(vec![
			OAuthDeviceCodePollResult::SlowDown { interval_seconds: Some(30.0) },
			OAuthDeviceCodePollResult::Complete { value: "token".to_string() },
		]));
		let results_for_poll = results.clone();
		let result = tokio::spawn(poll_oauth_device_code_flow(
			OAuthDeviceCodePollOptions {
				interval_seconds: Some(2.0),
				expires_in_seconds: Some(900.0),
				wait_before_first_poll: false,
				signal: never_aborted_signal(),
			},
			move || {
				times.push_now();
				let next = results_for_poll.lock().remove(0);
				async move { Ok(next) }
			},
		));

		tokio::time::advance(std::time::Duration::from_millis(0)).await;
		assert_eq!(record.len(), 1);

		tokio::time::advance(std::time::Duration::from_millis(29999)).await;
		assert_eq!(record.len(), 1);

		tokio::time::advance(std::time::Duration::from_millis(1)).await;
		assert_eq!(result.await.expect("task").expect("result"), "token");
		assert_eq!(record.len(), 2);
	}

	// Port of "cancels an in-flight wait".
	#[tokio::test(start_paused = true)]
	async fn cancels_an_in_flight_wait() {
		let controller = crate::utils::abort::AbortController::new();
		let signal = controller.signal();
		let result = tokio::spawn(poll_oauth_device_code_flow(
			OAuthDeviceCodePollOptions {
				interval_seconds: Some(5.0),
				expires_in_seconds: Some(30.0),
				wait_before_first_poll: false,
				signal: signal.clone(),
			},
			|| async { Ok(OAuthDeviceCodePollResult::<String>::Pending) },
		));

		// Let the flow reach its first sleep, then abort.
		tokio::time::advance(std::time::Duration::from_millis(1)).await;
		tokio::task::yield_now().await;
		controller.abort();
		tokio::time::advance(std::time::Duration::from_millis(1)).await;

		let error = result.await.expect("task").expect_err("cancelled");
		assert_eq!(crate::models::thrown_message(&error), "Login cancelled");
	}

	// Port of the timeout invariants: the slow_down timeout message.
	#[tokio::test(start_paused = true)]
	async fn times_out_with_the_slow_down_message_after_repeated_slow_downs() {
		let result = tokio::spawn(poll_oauth_device_code_flow(
			OAuthDeviceCodePollOptions {
				interval_seconds: Some(1.0),
				expires_in_seconds: Some(1.0),
				wait_before_first_poll: false,
				signal: never_aborted_signal(),
			},
			|| async { Ok(OAuthDeviceCodePollResult::<String>::SlowDown { interval_seconds: None }) },
		));

		tokio::time::advance(std::time::Duration::from_secs(3)).await;
		let error = result.await.expect("task").expect_err("timeout");
		assert_eq!(
			crate::models::thrown_message(&error),
			"Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again."
		);
	}

	#[tokio::test(start_paused = true)]
	async fn times_out_with_the_plain_message_without_slow_downs() {
		let result = tokio::spawn(poll_oauth_device_code_flow(
			OAuthDeviceCodePollOptions {
				interval_seconds: Some(1.0),
				expires_in_seconds: Some(1.0),
				wait_before_first_poll: false,
				signal: never_aborted_signal(),
			},
			|| async { Ok(OAuthDeviceCodePollResult::<String>::Pending) },
		));

		tokio::time::advance(std::time::Duration::from_secs(3)).await;
		let error = result.await.expect("task").expect_err("timeout");
		assert_eq!(crate::models::thrown_message(&error), "Device flow timed out");
	}
}

