//! Port of `packages/coding-agent/src/core/output-guard.ts`.
//!
//! Raw-stdout channel for the JSON/RPC/print modes: protocol output must
//! reach stdout verbatim while library chatter stays on stderr.
//!
//! Deviation **D91** — Node overrides `process.stdout.write` and retries
//! `ENOBUFS`/`EAGAIN` writes through a promise tail; Rust stdout is a
//! synchronous blocking handle with no replaceable `write` hook, so
//! [`take_over_stdout`] only records the takeover state (library code in this
//! crate never `println!`s) and [`write_raw_stdout`] performs a locked
//! blocking write. [`wait_for_raw_stdout_backpressure`] is therefore a no-op;
//! the signature stays so mode call sites keep the upstream shape.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

static STDOUT_TAKEN_OVER: AtomicBool = AtomicBool::new(false);
static STDOUT_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// `takeOverStdout`: redirect `process.stdout.write` to stderr so only raw
/// writes reach stdout. Rust has no replaceable stdout hook (D91); this
/// records the takeover flag for [`is_stdout_taken_over`]/[`restore_stdout`].
pub fn take_over_stdout() {
    STDOUT_TAKEN_OVER.store(true, Ordering::SeqCst);
}

/// `restoreStdout`.
pub fn restore_stdout() {
    STDOUT_TAKEN_OVER.store(false, Ordering::SeqCst);
}

/// `isStdoutTakenOver`.
pub fn is_stdout_taken_over() -> bool {
    STDOUT_TAKEN_OVER.load(Ordering::SeqCst)
}

/// `writeRawStdout`: queue `text` on the serialized raw-stdout writer.
pub fn write_raw_stdout(text: &str) {
    if text.is_empty() {
        return;
    }
    let _guard = STDOUT_LOCK.lock();
    let mut stdout = std::io::stdout().lock();
    // Upstream retries ENOBUFS/EAGAIN; a blocking pipe write either succeeds
    // or fails fatally (D91).
    let _ = stdout.write_all(text.as_bytes());
    let _ = stdout.flush();
}

/// `waitForRawStdoutBackpressure`: resolve once the queued writes drained.
/// Writes are synchronous (D91), so this returns immediately.
pub async fn wait_for_raw_stdout_backpressure() {}

/// `flushRawStdout`: drain queued writes and flush the underlying handle.
pub async fn flush_raw_stdout() {
    let _guard = STDOUT_LOCK.lock();
    let _ = std::io::stdout().lock().flush();
}
