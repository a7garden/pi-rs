//! Port of `packages/ai/src/utils/sleep.ts`.

use crate::utils::abort::{AbortReason, AbortSignal};

/// Resolve after `ms`, or reject with the signal's abort reason
/// (upstream `sleep(ms, signal)` rejects on abort).
pub async fn sleep(ms: u64, signal: &AbortSignal) -> Result<(), AbortReason> {
    signal.check()?;
    tokio::select! {
        biased;
        reason = signal.aborted_future() => Err(reason),
        _ = tokio::time::sleep(std::time::Duration::from_millis(ms)) => Ok(()),
    }
}
