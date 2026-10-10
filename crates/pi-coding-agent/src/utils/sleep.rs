//! Port of `packages/coding-agent/src/utils/sleep.ts` — a sleep helper that
//! respects an abort signal (see [`super::abort`], deviation D9, for the
//! `AbortSignal` counterpart).

use std::time::Duration;

use super::abort::AbortSignal;

/// Rejection of [`sleep`] — upstream rejects with `new Error("Aborted")`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("Aborted")]
pub struct SleepAborted;

/// Sleep for `ms` milliseconds, or fail early when the signal aborts.
pub async fn sleep(ms: u64, signal: Option<&AbortSignal>) -> Result<(), SleepAborted> {
    if signal.is_some_and(|signal| signal.aborted()) {
        return Err(SleepAborted);
    }
    let Some(signal) = signal else {
        tokio::time::sleep(Duration::from_millis(ms)).await;
        return Ok(());
    };
    tokio::select! {
        () = tokio::time::sleep(Duration::from_millis(ms)) => Ok(()),
        () = signal.aborted_future() => Err(SleepAborted),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::abort::AbortController;
    use super::*;

    #[tokio::test]
    async fn resolves_after_the_delay() {
        sleep(5, None).await.unwrap();
    }

    #[tokio::test]
    async fn fails_immediately_when_already_aborted() {
        let controller = AbortController::new();
        controller.abort();
        let result = sleep(50, Some(controller.signal())).await;
        assert_eq!(result, Err(SleepAborted));
    }

    #[tokio::test]
    async fn fails_early_when_aborted_mid_sleep() {
        let controller = Arc::new(AbortController::new());
        let signal = controller.signal().clone();
        let aborter = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            controller.abort();
        });
        let started = std::time::Instant::now();
        let result = sleep(60_000, Some(&signal)).await;
        assert_eq!(result, Err(SleepAborted));
        assert!(started.elapsed() < Duration::from_secs(60));
        aborter.await.unwrap();
    }

    #[tokio::test]
    async fn completes_without_a_signal() {
        let started = std::time::Instant::now();
        sleep(10, None).await.unwrap();
        assert!(started.elapsed() >= Duration::from_millis(10));
    }
}
