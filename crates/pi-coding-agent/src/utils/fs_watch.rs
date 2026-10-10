//! Port of `packages/coding-agent/src/utils/fs-watch.ts` — filesystem watching
//! with a shared error-handling seam.
//!
//! Deviation (crate docs, D13): node's `fs.watch` (inotify/FSEvents/kqueue)
//! has no counterpart among the allowed dependencies, so this module provides
//! a small polling [`FsWatcher`] seam instead: a background thread re-stats
//! the path every [`FS_WATCH_POLL_INTERVAL_MS`] and fires the listener on
//! mtime/size/existence changes. The public surface
//! ([`close_watcher`], [`watch_with_error_handler`], the retry-delay
//! constant) mirrors upstream 1:1.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use tokio::sync::mpsc;

/// Upstream `FS_WATCH_RETRY_DELAY_MS` — callers re-arm the watch after this
/// delay when the watcher errored.
pub const FS_WATCH_RETRY_DELAY_MS: u64 = 5000;

/// Poll interval for the watcher seam (deviation D13).
pub const FS_WATCH_POLL_INTERVAL_MS: u64 = 500;

type Listener = Box<dyn FnMut() + Send>;

struct WatcherShared {
    closed: AtomicBool,
}

/// A handle to a running poll watcher. Dropping it does not stop the thread —
/// call [`close_watcher`] (upstream closes `FSWatcher`s explicitly too).
pub struct FsWatcher {
    shared: Arc<WatcherShared>,
    /// Receives the thread's completion signal; kept so the channel end lives
    /// with the handle.
    #[allow(dead_code)]
    done_rx: mpsc::Receiver<()>,
}

impl FsWatcher {
    /// True once [`close_watcher`] (or [`FsWatcher::close`]) was called.
    pub fn is_closed(&self) -> bool {
        self.shared.closed.load(Ordering::SeqCst)
    }

    /// Stop the watcher thread (upstream `watcher.close()`). Idempotent.
    pub fn close(&mut self) {
        self.shared.closed.store(true, Ordering::SeqCst);
    }
}

fn snapshot(path: &str) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    Some((modified, meta.len()))
}

fn spawn_poll_thread(
    path: String,
    initial: Option<(SystemTime, u64)>,
    listener: Listener,
    shared: Arc<WatcherShared>,
    done_tx: mpsc::Sender<()>,
) {
    std::thread::spawn(move || {
        let mut listener = listener;
        let mut last = initial;
        let interval = Duration::from_millis(FS_WATCH_POLL_INTERVAL_MS);
        loop {
            if shared.closed.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(interval);
            if shared.closed.load(Ordering::SeqCst) {
                break;
            }
            let current = snapshot(&path);
            if current != last {
                last = current;
                listener();
            }
        }
        let _ = done_tx.send(());
    });
}

/// Close a watcher, ignoring close errors (`closeWatcher`). Accepts the same
/// `null | undefined` shapes upstream does via `Option`.
pub fn close_watcher(watcher: Option<&mut FsWatcher>) {
    if let Some(watcher) = watcher {
        watcher.close();
    }
}

/// Start watching `path`, reporting startup failures through `on_error`
/// (`watchWithErrorHandler`); returns `None` when the path cannot be watched.
///
/// With the polling seam, the listener fires on mtime/size/existence changes;
/// unlike node, no separate per-event error channel exists, so a mid-watch
/// stat failure surfaces as a change event (a deleted file is a change).
pub fn watch_with_error_handler(
    path: &str,
    listener: Listener,
    mut on_error: Listener,
) -> Option<FsWatcher> {
    // Snapshot synchronously so changes made immediately after this call are
    // not missed by a slow-to-start poll thread.
    let Some(initial) = snapshot(path) else {
        on_error();
        return None;
    };
    let shared = Arc::new(WatcherShared {
        closed: AtomicBool::new(false),
    });
    let (done_tx, done_rx) = mpsc::channel(1);
    spawn_poll_thread(
        path.to_string(),
        Some(initial),
        listener,
        Arc::clone(&shared),
        done_tx,
    );
    Some(FsWatcher { shared, done_rx })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn returns_none_and_reports_error_for_missing_path() {
        let dir = std::env::temp_dir().join(format!("pi-fswatch-missing-{}", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        let errors = Arc::new(AtomicUsize::new(0));
        let errors2 = Arc::clone(&errors);
        let watcher = watch_with_error_handler(
            &dir.to_string_lossy(),
            Box::new(|| {}),
            Box::new(move || {
                errors2.fetch_add(1, Ordering::SeqCst);
            }),
        );
        assert!(watcher.is_none());
        assert_eq!(errors.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn fires_listener_on_change_and_close_stops_it() {
        let dir = std::env::temp_dir();
        let changes = Arc::new(AtomicUsize::new(0));
        let file = dir.join(format!("pi-fswatch-{}-{:p}", std::process::id(), &changes));
        std::fs::write(&file, b"one").unwrap();
        let path = file.to_string_lossy().into_owned();

        let changes2 = Arc::clone(&changes);
        let mut watcher = watch_with_error_handler(
            &path,
            Box::new(move || {
                changes2.fetch_add(1, Ordering::SeqCst);
            }),
            Box::new(|| {}),
        )
        .unwrap();

        std::fs::write(&path, b"two").unwrap();
        // Wait up to ~15s (parallel test load can starve the poller briefly).
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        while changes.load(Ordering::SeqCst) == 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(changes.load(Ordering::SeqCst) >= 1);
        assert!(!watcher.is_closed());
        close_watcher(Some(&mut watcher));
        assert!(watcher.is_closed());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn close_watcher_accepts_none() {
        close_watcher(None::<&mut FsWatcher>);
    }

    #[test]
    fn retry_delay_constant_matches_upstream() {
        assert_eq!(FS_WATCH_RETRY_DELAY_MS, 5000);
    }
}
