//! Port of `packages/ai/src/utils/abort-signals.ts`.

use crate::utils::abort::{AbortController, AbortSignal};

/// Combined signal plus a cleanup that detaches listeners
/// (`CombinedAbortSignal`). The signal is `None` when no input signal was
/// supplied.
pub struct CombinedAbortSignal {
	pub signal: Option<AbortSignal>,
	pub cleanup: Box<dyn FnOnce() + Send>,
}

/// Combine optional signals into one that aborts when any active input aborts.
/// With zero active inputs there is no signal and cleanup is a no-op; with one,
/// it is returned as-is.
pub fn combine_abort_signals(signals: Vec<Option<AbortSignal>>) -> CombinedAbortSignal {
	let active_signals: Vec<AbortSignal> = signals.into_iter().flatten().collect();
	if active_signals.is_empty() {
		return CombinedAbortSignal { signal: None, cleanup: Box::new(|| {}) };
	}
	if active_signals.len() == 1 {
		let signal = active_signals.into_iter().next().expect("one signal");
		return CombinedAbortSignal { signal: Some(signal), cleanup: Box::new(|| {}) };
	}

	let controller = AbortController::new();
	let combined = controller.signal();
	let mut disposers: Vec<Box<dyn FnOnce() + Send>> = Vec::new();
	for signal in active_signals {
		if signal.aborted() {
			let reason = signal.reason();
			AbortSignal::abort_with(&combined, reason.unwrap_or(crate::utils::abort::AbortReason::Aborted));
			break;
		}
		let target = combined.clone();
		let signal_for_listener = signal.clone();
		let listener = move || {
			let reason = signal_for_listener.reason();
			AbortSignal::abort_with(&target, reason.unwrap_or(crate::utils::abort::AbortReason::Aborted));
		};
		disposers.push(signal.add_listener(listener));
	}

	CombinedAbortSignal {
		signal: Some(combined),
		cleanup: Box::new(move || {
			for disposer in disposers {
				disposer();
			}
		}),
	}
}
