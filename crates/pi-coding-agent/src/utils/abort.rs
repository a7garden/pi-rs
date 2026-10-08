//! Port of `packages/coding-agent/src/utils/abort.ts` — signal composition
//! modeled on tokio sync primitives instead of the DOM `AbortSignal`.
//!
//! Deviation D9 — JS `AbortSignal`/`AbortController` become a small local
//! [`AbortSignal`]/[`AbortController`] pair backed by `tokio::sync::Notify`.
//! An aborted operation surfaces its reason as [`AbortReason`] instead of a
//! thrown `AbortError`; `raceWithAbortSignal` cancels (drops) the abandoned
//! future on abort rather than letting it run to settlement like an
//! unobserved JS promise.

use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use tokio::sync::Notify;

/// Why an operation was aborted. Mirrors upstream `abortReason`: a custom
/// reason when one was given, otherwise the default `AbortError`.
#[derive(Debug, Clone, PartialEq)]
pub enum AbortReason {
	/// Default `AbortError` — "The operation was aborted".
	AbortError,
	/// A custom reason supplied to `AbortController::abort_with_reason`.
	Custom(serde_json::Value),
}

impl fmt::Display for AbortReason {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			AbortReason::AbortError => f.write_str("The operation was aborted"),
			AbortReason::Custom(value) => write!(f, "{value}"),
		}
	}
}

impl std::error::Error for AbortReason {}

struct SignalInner {
	aborted: AtomicBool,
	notify: Notify,
	reason: Mutex<Option<AbortReason>>,
}

/// A cancellation signal — the local counterpart of the DOM `AbortSignal`.
#[derive(Clone)]
pub struct AbortSignal {
	inner: Arc<SignalInner>,
}

impl AbortSignal {
	/// A fresh signal that is never aborted on its own (the counterpart of
	/// `new AbortController().signal`).
	pub fn new() -> Self {
		Self {
			inner: Arc::new(SignalInner {
				aborted: AtomicBool::new(false),
				notify: Notify::new(),
				reason: parking_lot::Mutex::new(None),
			}),
		}
	}

	/// Whether the signal has been aborted (`signal.aborted`).
	pub fn aborted(&self) -> bool {
		self.inner.aborted.load(Ordering::SeqCst)
	}

	/// The abort reason, if one was recorded.
	pub fn reason(&self) -> Option<AbortReason> {
		self.inner.reason.lock().clone()
	}

	/// Resolves once the signal is aborted (a no-op future if already
	/// aborted). Cancellation-safe: each call registers a fresh waiter.
	pub async fn aborted_future(&self) {
		if self.aborted() {
			return;
		}
		let notified = self.inner.notify.notified();
		if self.aborted() {
			return;
		}
		notified.await;
	}

	fn abort_with(&self, reason: Option<AbortReason>) {
		{
			let mut guard = self.inner.reason.lock();
			if guard.is_none() {
				*guard = Some(reason.unwrap_or(AbortReason::AbortError));
			}
		}
		if !self.inner.aborted.swap(true, Ordering::SeqCst) {
			self.inner.notify.notify_waiters();
		}
	}
}

impl Default for AbortSignal {
	fn default() -> Self {
		Self::new()
	}
}

impl fmt::Debug for AbortSignal {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("AbortSignal").field("aborted", &self.aborted()).finish()
	}
}

/// The counterpart of the DOM `AbortController`.
#[derive(Clone)]
pub struct AbortController {
	signal: AbortSignal,
}

impl AbortController {
	pub fn new() -> Self {
		Self { signal: AbortSignal::new() }
	}

	/// The controller's signal.
	pub fn signal(&self) -> &AbortSignal {
		&self.signal
	}

	/// `controller.abort()` — abort with the default `AbortError` reason.
	pub fn abort(&self) {
		self.signal.abort_with(None);
	}

	/// `controller.abort(reason)` — abort with a custom reason.
	pub fn abort_with_reason(&self, reason: serde_json::Value) {
		self.signal.abort_with(Some(AbortReason::Custom(reason)));
	}
}

impl Default for AbortController {
	fn default() -> Self {
		Self::new()
	}
}

/// Upstream `abortReason(signal)`: the recorded reason, or the default
/// `AbortError` when none was supplied.
pub fn abort_reason(signal: &AbortSignal) -> AbortReason {
	signal.reason().unwrap_or(AbortReason::AbortError)
}

/// `operationSignal(signal)`: normalize an optional public signal without
/// imposing a deadline.
pub fn operation_signal(signal: Option<&AbortSignal>) -> AbortSignal {
	match signal {
		Some(signal) => signal.clone(),
		None => AbortSignal::new(),
	}
}

/// `raceWithAbortSignal(operation, signal)`: resolve with the operation's
/// output, or fail with the abort reason if the signal fires first. The
/// abandoned operation is cancelled (dropped) — see D9.
pub async fn race_with_abort_signal<T, F>(operation: F, signal: Option<&AbortSignal>) -> Result<T, AbortReason>
where
	F: Future<Output = T>,
{
	let Some(signal) = signal else {
		return Ok(operation.await);
	};
	if signal.aborted() {
		return Err(abort_reason(signal));
	}
	tokio::select! {
		value = operation => Ok(value),
		() = signal.aborted_future() => Err(abort_reason(signal)),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[tokio::test]
	async fn signal_starts_unaborted_and_reports_reason_on_abort() {
		let controller = AbortController::new();
		let signal = controller.signal().clone();
		assert!(!signal.aborted());
		assert_eq!(abort_reason(&signal), AbortReason::AbortError);
		controller.abort();
		assert!(signal.aborted());
		assert_eq!(abort_reason(&signal), AbortReason::AbortError);
	}

	#[tokio::test]
	async fn custom_reason_is_reported() {
		let controller = AbortController::new();
		controller.abort_with_reason(serde_json::json!({"code": "shutting down"}));
		assert_eq!(
			abort_reason(controller.signal()),
			AbortReason::Custom(serde_json::json!({"code": "shutting down"}))
		);
	}

	#[tokio::test]
	async fn aborted_future_resolves_when_abort_fires() {
		let controller = AbortController::new();
		let signal = controller.signal().clone();
		let waiter = tokio::spawn(async move {
			signal.aborted_future().await;
		});
		tokio::task::yield_now().await;
		controller.abort();
		waiter.await.unwrap();
	}

	#[tokio::test]
	async fn aborted_future_resolves_immediately_when_already_aborted() {
		let controller = AbortController::new();
		controller.abort();
		tokio::time::timeout(std::time::Duration::from_millis(10), controller.signal().aborted_future())
			.await
			.expect("already-aborted future should resolve");
	}

	#[tokio::test]
	async fn operation_signal_normalizes_none() {
		let signal = operation_signal(None);
		assert!(!signal.aborted());
		let existing = AbortSignal::new();
		let normalized = operation_signal(Some(&existing));
		existing
			.inner
			.aborted
			.store(true, Ordering::SeqCst);
		assert!(normalized.aborted());
	}

	#[tokio::test]
	async fn race_resolves_with_operation_value() {
		let signal = AbortSignal::new();
		let value = race_with_abort_signal(async { 42 }, Some(&signal)).await;
		assert_eq!(value, Ok(42));
	}

	#[tokio::test]
	async fn race_fails_immediately_when_already_aborted() {
		let controller = AbortController::new();
		controller.abort();
		let result: Result<(), AbortReason> =
			race_with_abort_signal(async { unreachable!() }, Some(controller.signal())).await;
		assert_eq!(result, Err(AbortReason::AbortError));
	}

	#[tokio::test]
	async fn race_aborts_pending_operation() {
		let controller = Arc::new(AbortController::new());
		let signal = controller.signal().clone();
		let handle = tokio::spawn(async move {
			let result: Result<(), AbortReason> =
				race_with_abort_signal(tokio::time::sleep(std::time::Duration::from_secs(60)), Some(&signal)).await;
			result
		});
		tokio::task::yield_now().await;
		controller.abort();
		let result = handle.await.unwrap();
		assert_eq!(result, Err(AbortReason::AbortError));
	}
}
