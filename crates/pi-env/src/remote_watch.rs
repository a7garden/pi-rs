//! A watcher running in the daemon (upstream `packages/env/src/watch.ts`): Durable's `NodeFileWatcher`
//! next to the files. When the connection is lost, the watcher is opened again once the connection can
//! be started again, and reports `overflow`: changes in between were not seen.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use parking_lot::Mutex;
use pi_chord::context::{AbortController, Context};
use pi_durable::env::{
    FileError, FileErrorCode, FileWatcher, WatchCallback, WatchChange, WatchMode, WatchTarget,
};
use serde_json::{Value, json};

use crate::{
    connection::{Connection, Json, RemoteError, RequestOptions},
    errors::to_file_error,
};

/// How the daemon watches, like `NodeExecutionEnv`'s `NodeWatchOptions`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RemoteWatchOptions {
    /// Force a mode; by default Windows and file systems that do not report remote changes poll.
    pub mode: Option<WatchModeChoice>,
    /// Interval between snapshots in `polling` mode; default 2000 ms.
    pub poll_interval_ms: Option<u64>,
    /// Most directories one watcher covers; default 10,000.
    pub max_directories: Option<u64>,
}

/// Which mechanism the daemon uses, as the watch request's `mode` and its `ready` event report it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchModeChoice {
    Native,
    Polling,
}

impl WatchModeChoice {
    fn as_str(self) -> &'static str {
        match self {
            WatchModeChoice::Native => "native",
            WatchModeChoice::Polling => "polling",
        }
    }

    fn from_str(value: &str) -> WatchModeChoice {
        if value == "polling" {
            WatchModeChoice::Polling
        } else {
            WatchModeChoice::Native
        }
    }
}

impl From<WatchModeChoice> for WatchMode {
    fn from(choice: WatchModeChoice) -> WatchMode {
        match choice {
            WatchModeChoice::Native => WatchMode::Native,
            WatchModeChoice::Polling => WatchMode::Polling,
        }
    }
}

const RECONNECT_FIRST: Duration = Duration::from_secs(1);
const RECONNECT_MAX: Duration = Duration::from_secs(30);

/// State shared by the watcher and its current request task.
struct Shared {
    connection: Connection,
    /// The watch request's fields: resolved targets plus the options.
    fields: Json,
    on_change: WatchCallback,
    mode: Mutex<WatchMode>,
    closed: AtomicBool,
    controller: Mutex<Option<AbortController>>,
    request: Mutex<Option<tokio::task::JoinHandle<()>>>,
    wake: Arc<tokio::sync::Notify>,
}

/// A watcher running in the daemon. Open with [`RemoteWatcher::open`]; it fails like
/// `NodeFileWatcher.open` if coverage cannot be established.
#[derive(Clone)]
pub struct RemoteWatcher {
    shared: Arc<Shared>,
}

impl RemoteWatcher {
    /// Watch `targets` (resolved paths).
    pub async fn open(
        connection: Connection,
        targets: &[WatchTarget],
        on_change: WatchCallback,
        options: RemoteWatchOptions,
    ) -> Result<RemoteWatcher, RemoteError> {
        let targets: Vec<Value> = targets
			.iter()
			.map(|target| {
				json!({
					"path": target.path,
					"recursive": target.recursive,
					"exclude": {
						"hidden": target.exclude.as_ref().is_some_and(|exclude| exclude.hidden),
						"names": target.exclude.as_ref().map(|exclude| exclude.names.clone()).unwrap_or_default(),
					},
				})
			})
			.collect();
        let mut fields = Json::new();
        fields.insert("targets".into(), json!(targets));
        if let Some(mode) = options.mode {
            fields.insert("mode".into(), json!(mode.as_str()));
        }
        if let Some(interval) = options.poll_interval_ms {
            fields.insert("pollIntervalMs".into(), json!(interval));
        }
        if let Some(max) = options.max_directories {
            fields.insert("maxDirectories".into(), json!(max));
        }
        let watcher = RemoteWatcher {
            shared: Arc::new(Shared {
                connection,
                fields,
                on_change,
                mode: Mutex::new(WatchMode::Native),
                closed: AtomicBool::new(false),
                controller: Mutex::new(None),
                request: Mutex::new(None),
                wake: Arc::new(tokio::sync::Notify::new()),
            }),
        };
        let ready = spawn_attempt(watcher.shared.clone());
        match ready.await {
            Ok(result) => result?,
            // The task only ends with the request; without its answer the watcher was closed.
            Err(_) => {
                if !watcher.shared.closed.load(Ordering::SeqCst) {
                    return Err(RemoteError::new(
                        "unknown",
                        "watch ended before it was ready",
                    ));
                }
            }
        }
        Ok(watcher)
    }

    /// How the daemon currently watches.
    pub fn mode(&self) -> WatchMode {
        *self.shared.mode.lock()
    }

    /// Stop watching; no `onChange` call starts after this resolves. Idempotent.
    pub async fn close(&self) {
        if self.shared.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Some(controller) = self.shared.controller.lock().take() {
            controller.abort_default();
        }
        self.shared.wake.notify_one();
        let request = self.shared.request.lock().take();
        if let Some(request) = request {
            let _ = request.await;
        }
    }
}

#[async_trait::async_trait]
impl FileWatcher for RemoteWatcher {
    fn mode(&self) -> WatchMode {
        RemoteWatcher::mode(self)
    }

    async fn close(&self, _context: &Context) {
        RemoteWatcher::close(self).await;
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // Losing the last watcher handle stops it, like closing it would.
        self.closed.store(true, Ordering::SeqCst);
        if let Some(controller) = self.controller.lock().take() {
            controller.abort_default();
        }
        self.wake.notify_one();
    }
}

fn deliver(shared: &Shared, change: WatchChange) {
    if shared.closed.load(Ordering::SeqCst) {
        return;
    }
    // A panicking callback must not stop watching.
    let callback = shared.on_change.clone();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || callback(change)));
}

/// Open the daemon's watcher once; the returned channel resolves when it reports coverage, or with the
/// failure that ended the attempt.
fn spawn_attempt(shared: Arc<Shared>) -> tokio::sync::oneshot::Receiver<Result<(), RemoteError>> {
    let (ready_sender, ready) = tokio::sync::oneshot::channel();
    let controller = AbortController::new();
    *shared.controller.lock() = Some(controller.clone());
    let ready_sent = Arc::new(AtomicBool::new(false));
    let request = {
        let event_shared = shared.clone();
        let ready_sender = Arc::new(Mutex::new(Some(ready_sender)));
        let event_ready_sender = ready_sender.clone();
        let event_ready_sent = ready_sent.clone();
        let on_event = Arc::new(move |event: Json, _payload: Vec<u8>| {
            let kind = event
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match kind {
                "ready" => {
                    let choice = WatchModeChoice::from_str(
                        event
                            .get("mode")
                            .and_then(Value::as_str)
                            .unwrap_or("native"),
                    );
                    *event_shared.mode.lock() = WatchMode::from(choice);
                    if !event_ready_sent.swap(true, Ordering::SeqCst)
                        && let Some(sender) = event_ready_sender.lock().take()
                    {
                        let _ = sender.send(Ok(()));
                    }
                }
                "change" => {
                    if event.get("mode").and_then(Value::as_str) == Some("polling") {
                        *event_shared.mode.lock() = WatchMode::Polling;
                    }
                    if event.get("overflow").and_then(Value::as_bool) == Some(true) {
                        deliver(&event_shared, WatchChange::Overflow);
                    } else {
                        let paths = event
                            .get("paths")
                            .and_then(Value::as_array)
                            .map(|paths| {
                                paths
                                    .iter()
                                    .filter_map(Value::as_str)
                                    .map(str::to_owned)
                                    .collect()
                            })
                            .unwrap_or_default();
                        deliver(&event_shared, WatchChange::Paths(paths));
                    }
                }
                "error" => {
                    let code =
                        if event.get("code").and_then(Value::as_str) == Some("permission_denied") {
                            FileErrorCode::PermissionDenied
                        } else {
                            FileErrorCode::Invalid
                        };
                    let message = event
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("watch failed")
                        .to_owned();
                    deliver(
                        &event_shared,
                        WatchChange::Error(FileError::new(code, message, None)),
                    );
                    event_shared.closed.store(true, Ordering::SeqCst);
                }
                _ => {}
            }
        });
        let task_shared = shared.clone();
        tokio::spawn(async move {
            let controller = task_shared.controller.lock().clone();
            let mut options = RequestOptions::default();
            options.signal = controller.as_ref().map(AbortController::signal);
            options.on_event = Some(on_event);
            let result = task_shared
                .connection
                .request("watch", task_shared.fields.clone(), options)
                .await;
            // Settled: the daemon answered, or the connection failed under the open watcher.
            let was_ready = ready_sent.load(Ordering::SeqCst);
            if !was_ready {
                if task_shared.closed.load(Ordering::SeqCst) {
                    return;
                }
                if let Some(sender) = ready_sender.lock().take() {
                    let _ = sender.send(Err(match result {
                        Err(error) => error,
                        Ok(_) => RemoteError::new("unknown", "watch ended before it was ready"),
                    }));
                }
                return;
            }
            let Err(error) = result else { return };
            if task_shared.closed.load(Ordering::SeqCst) {
                return;
            }
            if error.lost() {
                reconnect(task_shared).await;
                return;
            }
            deliver(
                &task_shared,
                WatchChange::Error(to_file_error(&error, None)),
            );
            task_shared.closed.store(true, Ordering::SeqCst);
        })
    };
    *shared.request.lock() = Some(request);
    ready
}

/// Reopen the daemon's watcher with exponential backoff; changes while disconnected were not seen.
async fn reconnect(shared: Arc<Shared>) {
    let mut delay = RECONNECT_FIRST;
    while !shared.closed.load(Ordering::SeqCst) {
        let _ = tokio::time::timeout(delay, shared.wake.notified()).await;
        if shared.closed.load(Ordering::SeqCst) {
            return;
        }
        let ready = spawn_attempt(shared.clone());
        match ready.await {
            Ok(Ok(())) => {
                // Changes while disconnected were not seen.
                deliver(&shared, WatchChange::Overflow);
                return;
            }
            Ok(Err(error)) if error.lost() => delay = (delay * 2).min(RECONNECT_MAX),
            Ok(Err(error)) => {
                deliver(&shared, WatchChange::Error(to_file_error(&error, None)));
                shared.closed.store(true, Ordering::SeqCst);
                return;
            }
            // The attempt task ended without an answer; it saw a close.
            Err(_) => return,
        }
    }
}
