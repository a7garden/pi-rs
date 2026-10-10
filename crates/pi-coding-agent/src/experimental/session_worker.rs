//! Port of `experimental/session-worker.ts` — the Session worker process
//! body: wire protocol, worker-local demand/Harness reconciliation
//! ([`WorkerLifecycle`]), and the process `run` loop.
//!
//! Deviations (D151):
//! - typebox schemas become serde types with explicit validation helpers
//!   ([`validate_session_worker_options`]); `WorkerOperationScope` is a
//!   serde mirror of [`pi_chord`]services' [`WorkerServiceScope`](crate::experimental::services::worker::WorkerServiceScope)
//!   with conversions at the seam.
//! - Node socket + JSON-line async iterators become a tokio Unix stream
//!   with a reader task; `socket.destroy` maps to dropping the stream.
//! - `process.exit` on terminal failure becomes [`std::process::exit`] at
//!   the entry seam; `run` returns errors instead.
//! - The default `createCodingAgentHarness` factory has landed with the
//!   `durable/` slice ([`create_coding_agent_harness`], over
//!   `durable::harness_setup` + `durable::prompt`; D151 release). The full
//!   worker machinery still runs against the [`CreateSessionWorkerHarness`]
//!   seam (`run_session_worker_with_harness`); `run_session_worker_process`
//!   binds the default factory (`run_session_worker_entry`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pi_chord::services::errors::{error, AnyError};
use pi_chord::types::{BoxFuture, ServiceCall};
use pi_durable::harness::harness::Harness;
use pi_durable::harness::task_graph::TaskGraph;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::unix::OwnedWriteHalf;
use tokio::sync::mpsc;

use crate::experimental::coordinator::COORDINATOR_PROTOCOL_VERSION;
use crate::experimental::process::{
    consume_internal_process_role, encode_control_line, InternalProcessRole, MAX_CONTROL_LINE_BYTES,
};
use crate::experimental::services::worker::{
    create_session_worker_services, SessionWorkerRuntime, SessionWorkerServices, WorkerServiceScope,
};
use crate::experimental::session_catalog::session_storage_path;
#[cfg(test)]
use crate::experimental::TEST_GUARD;

pub use crate::experimental::services::worker::SessionWorkerRuntime as SessionWorkerRuntimeExport;

/// `PI_SESSION_WORKER_CONTROL_ADDRESS`.
pub const SESSION_WORKER_CONTROL_ADDRESS_ENV: &str = "PI_SESSION_WORKER_CONTROL_ADDRESS";
/// `PI_SESSION_WORKER_CONTROL_TOKEN`.
pub const SESSION_WORKER_CONTROL_TOKEN_ENV: &str = "PI_SESSION_WORKER_CONTROL_TOKEN";
/// `PI_SESSION_WORKER_SESSION_KEY_BASE64`.
pub const SESSION_WORKER_SESSION_KEY_ENV: &str = "PI_SESSION_WORKER_SESSION_KEY_BASE64";
/// `PI_SESSION_WORKER_PEER_ID`.
pub const SESSION_WORKER_PEER_ID_ENV: &str = "PI_SESSION_WORKER_PEER_ID";

const DEFAULT_INITIAL_DEMAND_GRACE_MS: u64 = 10_000;
const DEFAULT_ORPHAN_DEMAND_GRACE_MS: u64 = 30_000;
/// `__PI_SESSION_WORKER_INITIAL_DEMAND_GRACE_MS`.
pub const SESSION_WORKER_INITIAL_DEMAND_GRACE_ENV: &str =
    "__PI_SESSION_WORKER_INITIAL_DEMAND_GRACE_MS";
/// `__PI_SESSION_WORKER_ORPHAN_DEMAND_GRACE_MS`.
pub const SESSION_WORKER_ORPHAN_DEMAND_GRACE_ENV: &str =
    "__PI_SESSION_WORKER_ORPHAN_DEMAND_GRACE_MS";

// ---------------------------------------------------------------------------
// Wire types (upstream typebox schemas).
// ---------------------------------------------------------------------------

/// Upstream `SessionWorkerMetadataSchema`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct SessionWorkerMetadata {
    pub id: String,
    pub created_at: i64,
    pub cwd: String,
    pub path: String,
}

impl SessionWorkerMetadata {
    fn is_valid(&self) -> bool {
        !self.id.is_empty()
            && Path::new(&self.cwd).is_absolute()
            && Path::new(&self.path).is_absolute()
    }
}

/// Upstream `SessionWorkerOptionsSchema`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct SessionWorkerOptions {
    pub session_dir: String,
    pub metadata: SessionWorkerMetadata,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub plugin_manifest_paths: Vec<String>,
}

/// Upstream `WorkerOperationScopeSchema` (serde mirror of
/// [`WorkerServiceScope`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct WorkerOperationScope {
    pub server_connection_id: String,
    pub attachment_id: String,
}

impl From<&WorkerServiceScope> for WorkerOperationScope {
    fn from(scope: &WorkerServiceScope) -> Self {
        Self {
            server_connection_id: scope.server_connection_id.clone(),
            attachment_id: scope.attachment_id.clone(),
        }
    }
}

impl From<WorkerOperationScope> for WorkerServiceScope {
    fn from(scope: WorkerOperationScope) -> Self {
        Self {
            server_connection_id: scope.server_connection_id,
            attachment_id: scope.attachment_id,
        }
    }
}

/// Upstream `WorkerOperationRequestSchema`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct WorkerOperationRequest {
    #[serde(rename = "type")]
    pub kind: OperationRequestKind,
    pub request_id: String,
    pub scope: WorkerOperationScope,
    pub call: ServiceCall,
}

/// The `"operation"` type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationRequestKind {
    Operation,
}

/// Upstream `WorkerOperationResponseSchema`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    tag = "type"
)]
pub enum WorkerOperationResponse {
    OperationResult {
        request_id: String,
        scope: WorkerOperationScope,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
    },
    OperationError {
        request_id: String,
        scope: WorkerOperationScope,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<String>,
        message: String,
    },
}

impl WorkerOperationResponse {
    pub fn request_id(&self) -> &str {
        match self {
            Self::OperationResult { request_id, .. } | Self::OperationError { request_id, .. } => {
                request_id
            }
        }
    }

    pub fn scope(&self) -> &WorkerOperationScope {
        match self {
            Self::OperationResult { scope, .. } | Self::OperationError { scope, .. } => scope,
        }
    }
}

/// Upstream `SessionWorkerCommandSchema`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    tag = "type"
)]
pub enum SessionWorkerCommand {
    Shutdown,
    DiscoverWorkers,
    SessionDemand {
        #[serde(rename = "serverConnectionId")]
        server_connection_id: String,
        request_id: String,
        attachment_id: String,
        attached: bool,
    },
    #[serde(rename = "operation")]
    Operation(WorkerOperationRequest),
    OperationCancel {
        request_id: String,
        scope: WorkerOperationScope,
    },
}

/// Upstream `SessionWorkerEventSchema`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    tag = "type"
)]
pub enum SessionWorkerEvent {
    WorkerReady {
        token: String,
        session_key: String,
        session_id: String,
        pid: i32,
        metadata: SessionWorkerMetadata,
        plugin_manifest_paths: Vec<String>,
    },
    WorkerFailed {
        token: String,
        session_key: String,
        message: String,
    },
    DemandApplied {
        token: String,
        session_key: String,
        request_id: String,
        attachment_id: String,
        attached: bool,
    },
    DemandRejected {
        token: String,
        session_key: String,
        request_id: String,
        message: String,
    },
    OperationResponse {
        token: String,
        session_key: String,
        response: WorkerOperationResponse,
    },
    ServiceUpdate {
        token: String,
        session_key: String,
        scope: WorkerOperationScope,
        subscription_id: String,
        update: Value,
    },
}

/// Upstream `CoordinatorInputSchema` (worker side).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum CoordinatorInput {
    PeerRegistered {
        peer_id: String,
        #[serde(
            default,
            rename = "serverConnectionId",
            skip_serializing_if = "Option::is_none"
        )]
        server_connection_id: Option<String>,
    },
    ServerConnected {
        #[serde(rename = "serverConnectionId")]
        server_connection_id: String,
    },
    ServerDisconnected {
        #[serde(rename = "serverConnectionId")]
        server_connection_id: String,
    },
    Message {
        from: String,
        payload: Value,
    },
}

/// Upstream `validate_session_worker_options` inline checks
/// (`isAbsolute` + provider/model coherence + schema strictness).
pub fn validate_session_worker_options(options: &SessionWorkerOptions) -> Result<(), AnyError> {
    let invalid = || error("Session worker received invalid options");
    if options.session_dir.is_empty()
        || !Path::new(&options.session_dir).is_absolute()
        || !options.metadata.is_valid()
        || (options.provider.is_some() && options.model.is_none())
        || options
            .provider
            .as_ref()
            .is_some_and(|provider| provider.is_empty())
        || options.model.as_ref().is_some_and(|model| model.is_empty())
        || options.plugin_manifest_paths.is_empty()
        || options
            .plugin_manifest_paths
            .iter()
            .any(|path| path.is_empty())
    {
        return Err(invalid());
    }
    Ok(())
}

fn parse_remote_service_error_code(value: &str) -> Option<String> {
    pi_chord::REMOTE_SERVICE_ERROR_CODES
        .contains(&value)
        .then(|| value.to_string())
}

// ---------------------------------------------------------------------------
// Worker lifecycle (upstream `WorkerLifecycle`).
// ---------------------------------------------------------------------------

fn demand_key(server_connection_id: &str, attachment_id: &str) -> String {
    format!("{server_connection_id}\0{attachment_id}")
}

struct DemandEntry {
    server_connection_id: String,
    attachment_id: String,
    timer: Option<tokio::task::JoinHandle<()>>,
}

/// Worker-local reconciliation of server-generation demand and Harness
/// activity (upstream `WorkerLifecycle`).
pub struct WorkerLifecycle {
    initial_demand_grace_ms: u64,
    orphan_demand_grace_ms: u64,
    on_retire: Box<dyn Fn() + Send + Sync>,
    demands: parking_lot::Mutex<HashMap<String, DemandEntry>>,
    harness_active: AtomicBool,
    current_server_connection_id: parking_lot::Mutex<Option<String>>,
    initial_timer: parking_lot::Mutex<Option<tokio::task::JoinHandle<()>>>,
    demand_initialized: AtomicBool,
    retirement_holds: AtomicUsize,
    retiring: AtomicBool,
}

impl WorkerLifecycle {
    pub fn new(
        initial_server_connection_id: Option<String>,
        initial_demand_grace_ms: u64,
        orphan_demand_grace_ms: u64,
        on_retire: Box<dyn Fn() + Send + Sync>,
    ) -> Arc<Self> {
        let lifecycle = Arc::new(Self {
            initial_demand_grace_ms,
            orphan_demand_grace_ms,
            on_retire,
            demands: parking_lot::Mutex::new(HashMap::new()),
            harness_active: AtomicBool::new(false),
            current_server_connection_id: parking_lot::Mutex::new(initial_server_connection_id),
            initial_timer: parking_lot::Mutex::new(None),
            demand_initialized: AtomicBool::new(false),
            retirement_holds: AtomicUsize::new(0),
            retiring: AtomicBool::new(false),
        });
        let initial = Arc::clone(&lifecycle);
        lifecycle
            .initial_timer
            .lock()
            .replace(tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(
                    initial.initial_demand_grace_ms,
                ))
                .await;
                initial.initial_timer.lock().take();
                initial.demand_initialized.store(true, Ordering::SeqCst);
                initial.reconcile();
            }));
        lifecycle
    }

    /// `serverConnected(serverConnectionId)`.
    pub fn server_connected(&self, server_connection_id: &str) {
        *self.current_server_connection_id.lock() = Some(server_connection_id.to_string());
        for demand in self.demands.lock().values_mut() {
            if demand.server_connection_id == server_connection_id {
                demand.timer.take();
            }
        }
    }

    /// `serverDisconnected(serverConnectionId)`.
    pub fn server_disconnected(self: &Arc<Self>, server_connection_id: &str) {
        {
            let mut current = self.current_server_connection_id.lock();
            if current.as_deref() == Some(server_connection_id) {
                *current = None;
            }
        }
        let lifecycle = Arc::clone(self);
        let orphan_ms = self.orphan_demand_grace_ms;
        for (key, demand) in self.demands.lock().iter_mut() {
            if demand.server_connection_id != server_connection_id || demand.timer.is_some() {
                continue;
            }
            let key = key.clone();
            let lifecycle = Arc::clone(&lifecycle);
            let disconnected = server_connection_id.to_string();
            demand.timer = Some(tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(orphan_ms)).await;
                let mut demands = lifecycle.demands.lock();
                // `this.#demands.get(key) !== demand` — a replaced entry survives.
                if demands.get(&key).is_some_and(|entry| {
                    entry.server_connection_id == disconnected && entry.timer.is_none()
                }) {
                    demands.remove(&key);
                    drop(demands);
                    lifecycle.reconcile();
                }
            }));
        }
    }

    /// `beginRequest(...)`: validate the request against the active
    /// attachment and hold retirement while it runs.
    pub fn begin_request(
        self: &Arc<Self>,
        server_connection_id: &str,
        attachment_id: &str,
    ) -> Result<Box<dyn FnOnce() + Send>, AnyError> {
        if self.retiring.load(Ordering::SeqCst) {
            return Err(error("Session worker is retiring"));
        }
        if self.current_server_connection_id.lock().as_deref() != Some(server_connection_id) {
            return Err(error(
                "Session worker received a request from a stale server generation",
            ));
        }
        let demand_ok = self
            .demands
            .lock()
            .get(&demand_key(server_connection_id, attachment_id))
            .is_some_and(|demand| demand.timer.is_none());
        if !demand_ok {
            return Err(error(
                "Session worker request does not match the active attachment",
            ));
        }
        Ok(self.hold_retirement())
    }

    /// `holdRetirement()`: one reference keeping the worker alive.
    pub fn hold_retirement(self: &Arc<Self>) -> Box<dyn FnOnce() + Send> {
        self.retirement_holds.fetch_add(1, Ordering::SeqCst);
        let lifecycle = Arc::clone(self);
        let released = Arc::new(AtomicBool::new(false));
        let release_flag = Arc::clone(&released);
        Box::new(move || {
            if release_flag.swap(true, Ordering::SeqCst) {
                return;
            }
            lifecycle.retirement_holds.fetch_sub(1, Ordering::SeqCst);
            lifecycle.reconcile();
        })
    }

    /// `setDemand(serverConnectionId, attachmentId, attached)`.
    pub fn set_demand(
        self: &Arc<Self>,
        server_connection_id: &str,
        attachment_id: &str,
        attached: bool,
    ) -> Result<(), AnyError> {
        if self.retiring.load(Ordering::SeqCst) {
            return Err(error("Session worker is retiring"));
        }
        if self.current_server_connection_id.lock().as_deref() != Some(server_connection_id) {
            return Err(error(
                "Session worker received demand from a stale server generation",
            ));
        }
        self.demand_initialized.store(true, Ordering::SeqCst);
        self.initial_timer.lock().take();
        let key = demand_key(server_connection_id, attachment_id);
        {
            let mut demands = self.demands.lock();
            if let Some(previous) = demands.get_mut(&key) {
                previous.timer.take();
            }
            if attached {
                demands.insert(
                    key,
                    DemandEntry {
                        server_connection_id: server_connection_id.to_string(),
                        attachment_id: attachment_id.to_string(),
                        timer: None,
                    },
                );
            } else {
                demands.remove(&key);
            }
        }
        self.reconcile();
        Ok(())
    }

    /// `setHarnessActive(active)`: live Harness tasks hold the worker.
    pub fn set_harness_active(&self, active: bool) {
        self.harness_active.store(active, Ordering::SeqCst);
        if !active {
            self.reconcile();
        }
    }

    /// `close()`.
    pub fn close(&self) {
        self.initial_timer.lock().take();
        for demand in self.demands.lock().values_mut() {
            demand.timer.take();
        }
        self.demands.lock().clear();
    }

    fn reconcile(&self) {
        if self.retiring.load(Ordering::SeqCst)
            || !self.demand_initialized.load(Ordering::SeqCst)
            || self.retirement_holds.load(Ordering::SeqCst) != 0
            || self.harness_active.load(Ordering::SeqCst)
            || !self.demands.lock().is_empty()
        {
            return;
        }
        self.retiring.store(true, Ordering::SeqCst);
        (self.on_retire)();
    }
}

fn lifecycle_delay(name: &str, fallback: u64) -> Result<u64, AnyError> {
    match std::env::var(name) {
        Err(_) => Ok(fallback),
        Ok(value) => value
            .parse::<u64>()
            .map_err(|_| error(format!("{name} must be a non-negative safe integer"))),
    }
}

// ---------------------------------------------------------------------------
// Control connection (upstream `connectControl`).
// ---------------------------------------------------------------------------

struct WorkerControl {
    initial_server_connection_id: Option<String>,
    messages: tokio::sync::Mutex<mpsc::Receiver<Value>>,
    writer: Arc<tokio::sync::Mutex<Option<OwnedWriteHalf>>>,
}

impl WorkerControl {
    async fn send(&self, event: &SessionWorkerEvent) -> Result<(), AnyError> {
        let payload = serde_json::to_value(event).expect("worker events serialize");
        let line = encode_control_line(
            &serde_json::json!({ "type": "send", "to": "server", "payload": payload }),
        )
        .map_err(error)?;
        let mut writer = self.writer.lock().await;
        match writer.as_mut() {
            Some(writer) => writer.write_all(line.as_bytes()).await.map_err(|io_error| {
                error(format!("Session worker control send failed: {io_error}"))
            }),
            None => Err(error("Session worker control connection is closed")),
        }
    }
}

async fn connect_control() -> Result<WorkerControl, AnyError> {
    let invalid = || error("Session worker requires a control address");
    let address = std::env::var(SESSION_WORKER_CONTROL_ADDRESS_ENV)
        .ok()
        .filter(|value| !value.is_empty());
    let token = std::env::var(SESSION_WORKER_CONTROL_TOKEN_ENV)
        .ok()
        .filter(|value| !value.is_empty());
    let encoded_session_key = std::env::var(SESSION_WORKER_SESSION_KEY_ENV)
        .ok()
        .filter(|value| !value.is_empty());
    let (address, _token, _encoded_session_key) = match (address, token, encoded_session_key) {
        (Some(address), Some(token), Some(encoded)) => (address, token, encoded),
        _ => return Err(invalid()),
    };
    let peer_id = std::env::var(SESSION_WORKER_PEER_ID_ENV)
        .ok()
        .filter(|value| !value.is_empty());
    let Some(peer_id) = peer_id else {
        return Err(error("Session worker requires a peer ID"));
    };
    let socket = tokio::net::UnixStream::connect(&address)
        .await
        .map_err(|io_error| error(format!("Could not connect to the coordinator: {io_error}")))?;
    let (mut reader, writer) = socket.into_split();
    // Register as a peer before anything else.
    let register = encode_control_line(&serde_json::json!({
        "type": "register_peer",
        "protocol": COORDINATOR_PROTOCOL_VERSION,
        "peerId": peer_id,
    }))
    .map_err(error)?;
    let mut register_writer = writer;
    register_writer
        .write_all(register.as_bytes())
        .await
        .map_err(|io_error| {
            error(format!(
                "Could not register with the coordinator: {io_error}"
            ))
        })?;
    let writer = Arc::new(tokio::sync::Mutex::new(Some(register_writer)));

    let (messages_tx, messages_rx) = mpsc::channel::<Value>(64);
    tokio::spawn(async move {
        let mut line_reader = crate::experimental::process::LineReader::new();
        let mut buffer = vec![0u8; 64 * 1024];
        loop {
            match reader.read(&mut buffer).await {
                Ok(0) | Err(_) => return,
                Ok(read) => {
                    let lines = match line_reader.push(&buffer[..read]) {
                        Ok(lines) => lines,
                        Err(_) => return,
                    };
                    for line in lines {
                        if messages_tx.send(line).await.is_err() {
                            return;
                        }
                    }
                }
            }
        }
    });

    // Await `peer_registered`.
    let messages = tokio::sync::Mutex::new(messages_rx);
    let initial_server_connection_id =
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                let value =
                    messages.lock().await.recv().await.ok_or_else(|| {
                        error("Coordinator rejected the session worker registration")
                    })?;
                if let Ok(input) = serde_json::from_value::<CoordinatorInput>(value.clone()) {
                    if let CoordinatorInput::PeerRegistered {
                        server_connection_id,
                        ..
                    } = input
                    {
                        return Ok::<_, AnyError>(server_connection_id);
                    }
                }
            }
        })
        .await
        .map_err(|_| error("Coordinator rejected the session worker registration"))??;
    Ok(WorkerControl {
        initial_server_connection_id,
        messages,
        writer,
    })
}

/// Parse a control value into a coordinator input, mirroring the
/// `CoordinatorInputSchema` check (the routed `from` must be `"server"`).
fn parse_coordinator_input(value: &Value) -> Option<CoordinatorInput> {
    let input = serde_json::from_value::<CoordinatorInput>(value.clone()).ok()?;
    if let CoordinatorInput::Message { from, .. } = &input {
        if from != "server" {
            return None;
        }
    }
    Some(input)
}

/// `use serde_json::from_value` shape check for worker commands.
fn parse_worker_command(payload: &Value) -> Option<SessionWorkerCommand> {
    serde_json::from_value::<SessionWorkerCommand>(payload.clone()).ok()
}

// ---------------------------------------------------------------------------
// Run loop (upstream `run`).
// ---------------------------------------------------------------------------

/// Upstream `CreateSessionWorkerHarness`.
pub type CreateSessionWorkerHarness = Arc<
    dyn Fn(PathBuf, SessionWorkerOptions) -> BoxFuture<Result<SessionWorkerRuntime, AnyError>>
        + Send
        + Sync,
>;
/// Shared run state handed to every task (upstream closure variables).
struct RunContext {
    control: Arc<WorkerControl>,
    token: String,
    session_key: String,
    metadata: SessionWorkerMetadata,
    plugin_manifest_paths: Vec<String>,
    ready: AtomicBool,
    active_requests: Arc<parking_lot::Mutex<HashMap<String, ActiveRequest>>>,
    services: parking_lot::Mutex<Option<Arc<SessionWorkerServices>>>,
    lifecycle: parking_lot::Mutex<Option<Arc<WorkerLifecycle>>>,
    /// Everything the close sequence disposes; set once startup completes.
    close: parking_lot::Mutex<Option<Arc<CloseState>>>,
}

struct ActiveRequest {
    scope: WorkerOperationScope,
    cancel: Arc<pi_chord::context::AbortController>,
}

/// The assembled disposables behind the close sequence (upstream the
/// `closing` memoized promise over `closeResources`).
struct CloseState {
    lifecycle: Arc<WorkerLifecycle>,
    activity: pi_durable::harness::task_graph::TaskGraphWatch,
    services: Arc<SessionWorkerServices>,
    /// Shared with the run context; the close sequence cancels the same
    /// active requests.
    requests: Arc<parking_lot::Mutex<HashMap<String, ActiveRequest>>>,
    runtime: parking_lot::Mutex<Option<SessionWorkerRuntime>>,
    release_ownership: parking_lot::Mutex<Option<crate::experimental::lockfile::FileLock>>,
    started: tokio::sync::OnceCell<()>,
}

impl CloseState {
    async fn run_close(&self) {
        self.lifecycle.close();
        self.activity.stop_detached();
        self.services.remove_subscriptions(|_scope| true);
        for (_, request) in self.requests.lock().drain() {
            request.cancel.abort("Session worker is closing");
        }
        let runtime = self.runtime.lock().take();
        let ownership = self.release_ownership.lock().take();
        let _ = close_resources(runtime, Some(Arc::clone(&self.services)), None, ownership).await;
    }
}

impl RunContext {
    fn lifecycle(&self) -> Option<Arc<WorkerLifecycle>> {
        self.lifecycle.lock().clone()
    }

    fn services(&self) -> Option<Arc<SessionWorkerServices>> {
        self.services.lock().clone()
    }

    fn announce(self: &Arc<Self>) {
        if !self.ready.load(Ordering::SeqCst) {
            return;
        }
        let context = Arc::clone(self);
        tokio::spawn(async move {
            if context
                .send_event(worker_ready_event(&context))
                .await
                .is_err()
            {
                context.spawn_close_and_exit(None);
            }
        });
    }

    async fn send_event(&self, event: SessionWorkerEvent) -> Result<(), AnyError> {
        self.control.send(&event).await
    }

    fn spawn_close_and_exit(self: &Arc<Self>, failure: Option<AnyError>) {
        if let Some(failure) = &failure {
            eprintln!("{failure}");
        }
        let context = Arc::clone(self);
        let close = self.close.lock().clone();
        tokio::spawn(async move {
            if let Some(close) = close {
                close
                    .started
                    .get_or_init(|| async { close.run_close().await })
                    .await;
            }
            std::process::exit(u8::from(failure.is_some()) as i32);
        });
    }
}

fn worker_ready_event(context: &RunContext) -> SessionWorkerEvent {
    SessionWorkerEvent::WorkerReady {
        token: context.token.clone(),
        session_key: context.session_key.clone(),
        session_id: context.metadata.id.clone(),
        pid: std::process::id() as i32,
        metadata: context.metadata.clone(),
        plugin_manifest_paths: context.plugin_manifest_paths.clone(),
    }
}

/// `run(options, createHarness)` — the full worker process body.
async fn run(
    options: SessionWorkerOptions,
    create_harness: CreateSessionWorkerHarness,
) -> Result<(), AnyError> {
    let metadata = options.metadata.clone();
    let control = Arc::new(connect_control().await?);
    let token = std::env::var(SESSION_WORKER_CONTROL_TOKEN_ENV).unwrap_or_default();
    let session_key = decode_session_key()?;
    let plugin_manifest_paths = options.plugin_manifest_paths.clone();

    let context = Arc::new(RunContext {
        control: Arc::clone(&control),
        token,
        session_key,
        metadata: metadata.clone(),
        plugin_manifest_paths: plugin_manifest_paths.clone(),
        ready: AtomicBool::new(false),
        active_requests: Arc::new(parking_lot::Mutex::new(HashMap::new())),
        services: parking_lot::Mutex::new(None),
        lifecycle: parking_lot::Mutex::new(None),
        close: parking_lot::Mutex::new(None),
    });

    // The worker owns the Session for its lifetime (upstream
    // `lockfile.lock(metadata.path, ...)`); proper-lockfile locks the
    // directory path, the port locks `<path>.lock` next to it (D149).
    let ownership_path = lock_path_for_session(&metadata.path);
    let release_ownership = crate::experimental::lockfile::FileLock::acquire(
        &ownership_path,
        crate::experimental::lockfile::LockOptions {
            retry_ms: 25,
            wait_ms: 8_000,
        },
    )
    .await
    .map_err(error)?;

    // Open the Session's Harness while the worker holds the Session lock;
    // on failure close whatever it opened.
    let created = async {
        let catalog_metadata = crate::experimental::session_catalog::SessionCatalogMetadata {
            id: metadata.id.clone(),
            created_at: metadata.created_at,
            cwd: metadata.cwd.clone(),
            path: metadata.path.clone(),
        };
        let runtime =
            create_harness(session_storage_path(&catalog_metadata), options.clone()).await?;
        let watch = runtime
            .harness
            .task_graph
            .watch(&pi_chord::context::TODO_CONTEXT)
            .await
            .map_err(error)?;
        Ok::<_, AnyError>((runtime, watch))
    }
    .await;
    let (runtime, watch) = match created {
        Ok(created) => created,
        Err(creation_error) => {
            let cleanup_error = close_resources(None, None, None, Some(release_ownership))
                .await
                .err();
            return match (creation_error, cleanup_error) {
                (creation_error, Some(cleanup_error)) => {
                    Err(pi_chord::services::errors::AggregateError::new(
                        "Session worker startup and cleanup failed",
                        vec![creation_error, cleanup_error],
                    ))
                }
                (creation_error, None) => Err(creation_error),
            };
        }
    };

    // `createSessionWorkerServices` with a placeholder publisher (service
    // updates publish through the per-invocation chord publisher below).
    let services =
        match create_session_worker_services(CreateSessionWorkerServicesShim::options(&runtime))
            .await
        {
            Ok(services) => services,
            Err(services_error) => {
                watch.stop_detached();
                let cleanup_error =
                    close_resources(None, None, Some(watch), Some(release_ownership))
                        .await
                        .err();
                return match (services_error, cleanup_error) {
                    (services_error, Some(cleanup_error)) => {
                        Err(pi_chord::services::errors::AggregateError::new(
                            "Session worker startup and cleanup failed",
                            vec![services_error, cleanup_error],
                        ))
                    }
                    (services_error, None) => Err(services_error),
                };
            }
        };
    *context.services.lock() = Some(Arc::clone(&services));

    // Worker lifecycle with the real retire hook (upstream constructs it
    // before the services; the hook needs the assembled state).
    let lifecycle_context = Arc::downgrade(&context);
    let lifecycle = WorkerLifecycle::new(
        control.initial_server_connection_id.clone(),
        lifecycle_delay(
            SESSION_WORKER_INITIAL_DEMAND_GRACE_ENV,
            DEFAULT_INITIAL_DEMAND_GRACE_MS,
        )?,
        lifecycle_delay(
            SESSION_WORKER_ORPHAN_DEMAND_GRACE_ENV,
            DEFAULT_ORPHAN_DEMAND_GRACE_MS,
        )?,
        Box::new(move || {
            let Some(context) = lifecycle_context.upgrade() else {
                std::process::exit(0);
            };
            context.spawn_close_and_exit(None);
        }),
    );
    *context.lifecycle.lock() = Some(Arc::clone(&lifecycle));

    let close_state = Arc::new(CloseState {
        lifecycle,
        activity: watch,
        services,
        requests: Arc::clone(&context.active_requests),
        runtime: parking_lot::Mutex::new(Some(runtime)),
        release_ownership: parking_lot::Mutex::new(Some(release_ownership)),
        started: tokio::sync::OnceCell::new(),
    });
    *context.close.lock() = Some(close_state);

    // Every live task, background work included, keeps the worker open.
    {
        let close = context.close.lock().clone().expect("close state");
        let lifecycle = Arc::clone(&close.lifecycle);
        let _ = close.activity.start(Arc::new(
            move |graph: TaskGraph, _ops: Vec<_>, _context: pi_chord::Context| {
                lifecycle.set_harness_active(!graph.tasks.is_empty());
                Box::pin(async {}) as pi_chord::types::BoxFuture<()>
            },
        ));
    }
    // Recovered work from an interrupted turn continues now.
    context
        .close
        .lock()
        .clone()
        .expect("close state")
        .runtime
        .lock()
        .as_ref()
        .expect("runtime")
        .harness
        .resume();

    // `readCommands` — the control reader loop.
    let reader_context = Arc::clone(&context);
    let reader = tokio::spawn(async move { read_commands(reader_context).await });
    let _ = reader;

    // `process.once("SIGTERM"/"SIGINT", closeAndExit)`.
    let signal_context = Arc::clone(&context);
    tokio::spawn(async move {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("sigterm handler");
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("sigint handler");
        tokio::select! {
            _ = sigterm.recv() => signal_context.spawn_close_and_exit(None),
            _ = sigint.recv() => signal_context.spawn_close_and_exit(None),
        }
    });

    // Readiness announcement; a failed send closes and reports (upstream
    // `closeAndExit` through AggregateError on the awaited close).
    context.ready.store(true, Ordering::SeqCst);
    if let Err(ready_error) = context.send_event(worker_ready_event(&context)).await {
        let close = context.close.lock().clone();
        if let Some(close) = close {
            close
                .started
                .get_or_init(|| async { close.run_close().await })
                .await;
        }
        return Err(ready_error);
    }

    // The process body lives until `closeAndExit` terminates it (upstream
    // stays alive on sockets and timers).
    std::future::pending::<()>().await;
    Ok(())
}

/// `<session path>.lock` (upstream `lockfile.lock(metadata.path)`, D149).
fn lock_path_for_session(session_path: &str) -> PathBuf {
    PathBuf::from(format!("{session_path}.lock"))
}

/// `readCommands(control, handlers)` over the shared run context.
async fn read_commands(context: Arc<RunContext>) {
    loop {
        let Some(value) = context.control.messages.lock().await.recv().await else {
            // The control socket closed or errored.
            context.spawn_close_and_exit(None);
            return;
        };
        let Some(input) = parse_coordinator_input(&value) else {
            // `socket.destroy(new Error("Coordinator sent an invalid worker message"))`
            context.spawn_close_and_exit(Some(error("Coordinator sent an invalid worker message")));
            return;
        };
        match input {
            CoordinatorInput::ServerConnected {
                server_connection_id,
            } => {
                if let Some(lifecycle) = context.lifecycle() {
                    lifecycle.server_connected(&server_connection_id);
                }
            }
            CoordinatorInput::ServerDisconnected {
                server_connection_id,
            } => {
                if let Some(services) = context.services() {
                    let server = server_connection_id.clone();
                    services.remove_subscriptions(move |scope: &WorkerServiceScope| {
                        scope.server_connection_id == server
                    });
                }
                {
                    let mut active = context.active_requests.lock();
                    let stale: Vec<String> = active
                        .iter()
                        .filter(|(_, request)| {
                            request.scope.server_connection_id == server_connection_id
                        })
                        .map(|(id, _)| id.clone())
                        .collect();
                    for id in stale {
                        if let Some(request) = active.remove(&id) {
                            request.cancel.abort("Server disconnected");
                        }
                    }
                }
                if let Some(lifecycle) = context.lifecycle() {
                    lifecycle.server_disconnected(&server_connection_id);
                }
            }
            CoordinatorInput::PeerRegistered { .. } => {}
            CoordinatorInput::Message { payload, .. } => match parse_worker_command(&payload) {
                None => continue,
                Some(SessionWorkerCommand::Shutdown) => {
                    context.spawn_close_and_exit(None);
                    return;
                }
                Some(SessionWorkerCommand::DiscoverWorkers) => context.announce(),
                Some(command @ SessionWorkerCommand::SessionDemand { .. }) => {
                    // Upstream awaits demand handling before the next message.
                    handle_demand(&context, command).await;
                }
                Some(SessionWorkerCommand::OperationCancel { request_id, scope }) => {
                    let active = context.active_requests.lock();
                    if let Some(request) = active.get(&request_id) {
                        if request.scope == scope {
                            request.cancel.abort("Service operation cancelled");
                        }
                    }
                }
                Some(SessionWorkerCommand::Operation(request)) => {
                    let operation_context = Arc::clone(&context);
                    tokio::spawn(async move {
                        // `handleOperation(request).catch(() => closeAndExit())`.
                        if let Err(failure) = handle_operation(&operation_context, request).await {
                            operation_context.spawn_close_and_exit(Some(failure));
                        }
                    });
                }
            },
        }
    }
}

/// `onDemand` — apply or reject the demand update while holding retirement.
async fn handle_demand(context: &Arc<RunContext>, command: SessionWorkerCommand) {
    let SessionWorkerCommand::SessionDemand {
        server_connection_id,
        request_id,
        attachment_id,
        attached,
    } = command
    else {
        return;
    };
    let Some(lifecycle) = context.lifecycle() else {
        return;
    };
    let Some(services) = context.services() else {
        return;
    };
    let release = lifecycle.hold_retirement();
    let outcome = async {
        if !attached {
            let server = server_connection_id.clone();
            let attachment = attachment_id.clone();
            services.remove_subscriptions(move |scope: &WorkerServiceScope| {
                scope.server_connection_id == server && scope.attachment_id == attachment
            });
        }
        lifecycle.set_demand(&server_connection_id, &attachment_id, attached)
    }
    .await;
    let send_result = match outcome {
        Err(demand_error) => {
            context
                .send_event(SessionWorkerEvent::DemandRejected {
                    token: context.token.clone(),
                    session_key: context.session_key.clone(),
                    request_id: request_id.clone(),
                    message: demand_error.to_string(),
                })
                .await
        }
        Ok(()) => {
            context
                .send_event(SessionWorkerEvent::DemandApplied {
                    token: context.token.clone(),
                    session_key: context.session_key.clone(),
                    request_id: request_id.clone(),
                    attachment_id: attachment_id.clone(),
                    attached,
                })
                .await
        }
    };
    if send_result.is_err() {
        context.spawn_close_and_exit(None);
    }
    release();
}

/// `handleOperation` — run one service call against the assembled services.
/// Begin failures answer with an `operation_error`; only a failed response
/// send is terminal (upstream's outer `catch(() => closeAndExit())`).
async fn handle_operation(
    context: &Arc<RunContext>,
    request: WorkerOperationRequest,
) -> Result<(), AnyError> {
    let Some(lifecycle) = context.lifecycle() else {
        return Err(error("Session worker is closing"));
    };
    let Some(services) = context.services() else {
        return Err(error("Session worker is closing"));
    };
    let cancellable = pi_chord::context::AbortController::new();
    let cancellable_context = pi_chord::context::with_abort_signal(
        cancellable.signal(),
        pi_chord::context::BACKGROUND_CONTEXT.clone(),
    );
    let response = match lifecycle.begin_request(
        &request.scope.server_connection_id,
        &request.scope.attachment_id,
    ) {
        Err(begin_error) => WorkerOperationResponse::OperationError {
            request_id: request.request_id.clone(),
            scope: request.scope.clone(),
            code: None,
            message: begin_error.to_string(),
        },
        Ok(release) => {
            let scope = request.scope.clone();
            let invocation_scope = scope.clone();
            context.active_requests.lock().insert(
                request.request_id.clone(),
                ActiveRequest {
                    scope,
                    cancel: Arc::new(cancellable),
                },
            );
            // Service updates for this scope's subscriptions publish to the
            // server (upstream's captured `publish`).
            let publisher_context = Arc::clone(context);
            let publisher_scope = invocation_scope.clone();
            let publish: pi_chord::services::provider::ServiceUpdatePublisher = Arc::new(
                move |subscription_id: String,
                      update: &pi_chord::ServiceProviderUpdate,
                      _context: &pi_chord::Context| {
                    let context = Arc::clone(&publisher_context);
                    let scope = publisher_scope.clone();
                    let update = serde_json::to_value(update).unwrap_or(serde_json::Value::Null);
                    Box::pin(async move {
                        let _ = context
                            .send_event(SessionWorkerEvent::ServiceUpdate {
                                token: context.token.clone(),
                                session_key: context.session_key.clone(),
                                scope,
                                subscription_id,
                                update,
                            })
                            .await;
                    })
                },
            );
            let result = services
                .invoke(
                    &request.call,
                    &invocation_scope.into(),
                    publish,
                    &cancellable_context,
                )
                .await;
            // Only remove the entry this invocation installed.
            {
                let mut active = context.active_requests.lock();
                if active
                    .get(&request.request_id)
                    .is_some_and(|active| active.scope == request.scope)
                {
                    active.remove(&request.request_id);
                }
            }
            drop(release);
            match result {
                Ok(result) => WorkerOperationResponse::OperationResult {
                    request_id: request.request_id.clone(),
                    scope: request.scope.clone(),
                    result,
                },
                Err(failure) => WorkerOperationResponse::OperationError {
                    request_id: request.request_id.clone(),
                    scope: request.scope.clone(),
                    code: remote_error_code(&failure),
                    message: failure.to_string(),
                },
            }
        }
    };
    context
        .send_event(SessionWorkerEvent::OperationResponse {
            token: context.token.clone(),
            session_key: context.session_key.clone(),
            response,
        })
        .await
}

/// The `RemoteServiceError` code extraction (upstream also inspects ad-hoc
/// `error.code` fields; Rust errors carry codes only through the typed
/// error).
fn remote_error_code(failure: &AnyError) -> Option<String> {
    failure
        .downcast_ref::<pi_chord::services::errors::RemoteServiceError>()
        .map(|remote| remote.code.as_str().to_string())
}

/// `CreateSessionWorkerServicesOptions` assembly (the shim keeps the
/// construction next to the run loop).
struct CreateSessionWorkerServicesShim;

impl CreateSessionWorkerServicesShim {
    fn options(
        runtime: &SessionWorkerRuntime,
    ) -> crate::experimental::services::worker::CreateSessionWorkerServicesOptions {
        crate::experimental::services::worker::CreateSessionWorkerServicesOptions {
            harness: Arc::clone(&runtime.harness),
            conversation: Arc::clone(&runtime.conversation),
            model_runtime: runtime.model_runtime.clone(),
            settings_manager: runtime.settings_manager.clone(),
            facet_loader: runtime.facet_loader.clone(),
            // Upstream threads the control publisher here; updates flow
            // through the per-invocation chord publisher (see
            // `handle_operation`), so the unused slot stays empty.
            publish: Arc::new(
                |_scope: WorkerServiceScope,
                 _subscription_id: String,
                 _update: pi_chord::ServiceProviderUpdate| { Box::pin(async {}) },
            ),
        }
    }
}
struct ControlWriter {
    control: Arc<()>,
}

/// `closeResources` — dispose services, close the Harness, run cleanup, and
/// release ownership; errors aggregate like upstream.
async fn close_resources(
    runtime: Option<SessionWorkerRuntime>,
    services: Option<Arc<SessionWorkerServices>>,
    watch: Option<pi_durable::harness::task_graph::TaskGraphWatch>,
    release_ownership: Option<crate::experimental::lockfile::FileLock>,
) -> Result<(), AnyError> {
    let mut errors: Vec<AnyError> = Vec::new();
    if let Some(services) = services {
        if let Err(dispose_error) = services.dispose().await {
            errors.push(dispose_error);
        }
    }
    if let Some(watch) = watch {
        watch.stop_detached();
    }
    if let Some(runtime) = runtime {
        if let Err(close_error) = runtime
            .harness
            .close(&pi_chord::context::TODO_CONTEXT)
            .await
        {
            errors.push(error(close_error));
        }
        if let Some(cleanup) = &runtime.cleanup {
            if let Err(cleanup_error) = cleanup().await {
                errors.push(cleanup_error);
            }
        }
    }
    drop(release_ownership);
    match errors.len() {
        0 => Ok(()),
        1 => Err(errors.pop().expect("one error")),
        _ => Err(pi_chord::services::errors::AggregateError::new(
            "Session worker cleanup failed",
            errors,
        )),
    }
}

fn decode_session_key() -> Result<String, AnyError> {
    use base64::Engine as _;
    let encoded = std::env::var(SESSION_WORKER_SESSION_KEY_ENV).unwrap_or_default();
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded.as_bytes())
        .map_err(|decode_error| {
            error(format!(
                "Session worker received an invalid session key: {decode_error}"
            ))
        })
        .and_then(|bytes| {
            String::from_utf8(bytes).map_err(|utf8_error| error(utf8_error.to_string()))
        })
}

/// `runSessionWorkerWithHarness(args, createHarness)`.
pub async fn run_session_worker_with_harness(
    args: &[String],
    create_harness: CreateSessionWorkerHarness,
) -> Result<(), AnyError> {
    if args.len() != 1 {
        return Err(error("Session worker requires one options argument"));
    }
    let options: SessionWorkerOptions = serde_json::from_str(&args[0]).map_err(|serde_error| {
        error(format!(
            "Session worker received invalid options: {serde_error}"
        ))
    })?;
    validate_session_worker_options(&options)?;
    run(options, create_harness).await
}

/// Consume the internal role and run the Session worker process with `args`
/// (upstream `runSessionWorkerProcess`: `runSessionWorkerWithHarness(args,
/// createCodingAgentHarness)`).
pub async fn run_session_worker_entry(args: &[String]) -> Result<(), String> {
    let role = consume_internal_process_role().map_err(|message| message)?;
    if role != Some(InternalProcessRole::SessionWorker) {
        return Err(
            "Session worker entrypoint requires an internal session-worker invocation".into(),
        );
    }
    run_session_worker_with_harness(
        args,
        Arc::new(|path: PathBuf, options: SessionWorkerOptions| {
            Box::pin(create_coding_agent_harness(path, options)) as pi_chord::types::BoxFuture<_>
        }),
    )
    .await
    .map_err(|failure| failure.to_string())
}

/// The default coding-agent harness factory (upstream
/// `createCodingAgentHarness`): pi's model runtime, settings, HTTP setup,
/// coding registry, and per-directory execution environments over the
/// Session's database; the initial model applies only when the root
/// conversation is created, so later starts keep its durable model.
pub async fn create_coding_agent_harness(
    database_path: PathBuf,
    options: SessionWorkerOptions,
) -> Result<SessionWorkerRuntime, AnyError> {
    use crate::experimental::durable::harness_setup::{
        configure_harness_http, create_coding_registry, create_harness_settings,
        create_model_runtime, create_settings_manager, find_initial_agent_model, CliModel,
        ExecutionEnvs,
    };
    use pi_durable::harness::harness::open_harness;
    use pi_durable::harness::types::{AgentChange, RootConversationOptions, SetOrClear};
    use pi_durable::storage::{NodeSqliteStorageOptions, SqliteStorage};

    let cwd = options.metadata.cwd.clone();
    let model_runtime = create_model_runtime()
        .await
        .map_err(|message| error(message))?;
    let settings_manager = create_settings_manager(&cwd);
    configure_harness_http(&settings_manager.lock()).map_err(|message| error(message))?;
    let envs = Arc::new(ExecutionEnvs::new(cwd.clone()));
    let registry = create_coding_registry(Arc::clone(&settings_manager), &cwd);
    let opened = open_harness(
        Arc::new(
            SqliteStorage::open_path(&database_path, &NodeSqliteStorageOptions::default())
                .await
                .map_err(|storage_error| error(storage_error.to_string()))?,
        ),
        pi_durable::harness::types::HarnessOptions {
            models: Arc::clone(&model_runtime) as Arc<dyn pi_ai::Models>,
            registry: Arc::new(registry) as Arc<dyn pi_durable::harness::types::RegistryReader>,
            settings: None,
            settings_provider: Some(create_harness_settings(Arc::clone(&settings_manager))),
            env: Some(envs.builder()),
            conversation_created: None,
            now: None,
            on_report: Some(Arc::new(|report: String| eprintln!("{report}"))),
        },
        &pi_chord::context::TODO_CONTEXT,
    )
    .await;
    let harness = match opened {
        Ok(harness) => harness,
        // Failed opens close what they opened and aggregate (upstream the
        // catch around the body).
        Err(open_error) => {
            envs.cleanup(&pi_chord::context::TODO_CONTEXT).await;
            return Err(error(open_error));
        }
    };
    // The initial model applies only when the root conversation is created;
    // later starts keep its durable model.
    let created = harness
        .conversation(
            pi_durable::types::ROOT_CONVERSATION_ID,
            &pi_chord::context::TODO_CONTEXT,
        )
        .await
        .map_err(|message| error(message))?
        .is_none();
    let cli = match (&options.provider, &options.model) {
        (Some(provider), Some(model)) => Some(CliModel {
            provider: Some(provider.clone()),
            model: model.clone(),
        }),
        (Some(_), None) | (None, Some(_)) => {
            return Err(error("Session worker received invalid options"));
        }
        (None, None) => None,
    };
    let initial = if created {
        Some(
            find_initial_agent_model(&settings_manager.lock(), &model_runtime, cli)
                .map_err(|message| error(message))?,
        )
    } else {
        None
    };
    let initial = initial.unwrap_or_default();
    let conversation = harness
        .root(
            &pi_chord::context::TODO_CONTEXT,
            Some(RootConversationOptions {
                agent: Some(AgentChange {
                    cwd: Some(SetOrClear::Set(cwd)),
                    model: initial.model.map(SetOrClear::Set),
                    thinking_level: initial.thinking_level.map(SetOrClear::Set),
                    ..AgentChange::default()
                }),
                init: None,
            }),
        )
        .await
        .map_err(|message| error(message))?;
    let facet_loader = crate::experimental::plugins::bundled::create_session_plugin_facet_loader(
        &options.plugin_manifest_paths,
    );
    Ok(SessionWorkerRuntime {
        harness,
        conversation: Arc::new(conversation),
        model_runtime: Some(model_runtime),
        settings_manager: Some(settings_manager),
        facet_loader,
        cleanup: {
            let envs = Arc::clone(&envs);
            Some(Arc::new(move || {
                let envs = Arc::clone(&envs);
                Box::pin(async move {
                    envs.cleanup(&pi_chord::context::TODO_CONTEXT).await;
                    Ok(())
                }) as futures_util::future::BoxFuture<'static, Result<(), AnyError>>
            })
                as Arc<
                    dyn Fn() -> futures_util::future::BoxFuture<'static, Result<(), AnyError>>
                        + Send
                        + Sync,
                >)
        },
    })
}

fn _assert_send<T: Send>(value: T) -> T {
    value
}

#[allow(unused)]
fn _task_graph_tasks(graph: &TaskGraph) -> usize {
    graph.tasks.len()
}

#[allow(unused)]
fn _max_control_line() -> usize {
    MAX_CONTROL_LINE_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> SessionWorkerOptions {
        SessionWorkerOptions {
            session_dir: "/sessions".into(),
            metadata: SessionWorkerMetadata {
                id: "s".into(),
                created_at: 1,
                cwd: "/cwd".into(),
                path: "/sessions/s".into(),
            },
            provider: None,
            model: None,
            plugin_manifest_paths: vec!["/m.json".into()],
        }
    }

    #[test]
    fn options_wire_format_round_trips() {
        let options = options();
        let value = serde_json::to_value(&options).unwrap();
        assert_eq!(value["sessionDir"], "/sessions");
        assert_eq!(value["metadata"]["createdAt"], 1);
        assert_eq!(value["pluginManifestPaths"][0], "/m.json");
        let parsed: SessionWorkerOptions = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, options);
    }

    #[test]
    fn options_validation_rejects_relative_paths_and_dangling_provider() {
        assert!(validate_session_worker_options(&options()).is_ok());
        let mut relative = options();
        relative.metadata.cwd = "cwd".into();
        assert!(validate_session_worker_options(&relative).is_err());
        let mut dangling = options();
        dangling.provider = Some("anthropic".into());
        assert!(validate_session_worker_options(&dangling).is_err());
        let mut empty_paths = options();
        empty_paths.plugin_manifest_paths = Vec::new();
        assert!(validate_session_worker_options(&empty_paths).is_err());
    }

    #[test]
    fn operation_response_wire_format() {
        let response = WorkerOperationResponse::OperationResult {
            request_id: "r".into(),
            scope: WorkerOperationScope {
                server_connection_id: "srv".into(),
                attachment_id: "att".into(),
            },
            result: Some(serde_json::json!({ "ok": true })),
        };
        let value = serde_json::to_value(&response).unwrap();
        assert_eq!(value["type"], "operation_result");
        assert_eq!(value["requestId"], "r");
        assert_eq!(value["scope"]["serverConnectionId"], "srv");
        let parsed: WorkerOperationResponse = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, response);

        let failure = WorkerOperationResponse::OperationError {
            request_id: "r".into(),
            scope: WorkerOperationScope {
                server_connection_id: "srv".into(),
                attachment_id: "att".into(),
            },
            code: Some("service_not_found".into()),
            message: "boom".into(),
        };
        let value = serde_json::to_value(&failure).unwrap();
        assert_eq!(value["code"], "service_not_found");
    }

    #[test]
    fn commands_and_events_use_snake_case_tags() {
        let command: SessionWorkerCommand =
            serde_json::from_value(serde_json::json!({ "type": "shutdown" })).unwrap();
        assert!(matches!(command, SessionWorkerCommand::Shutdown));
        let command: SessionWorkerCommand = serde_json::from_value(serde_json::json!({
            "type": "session_demand",
            "serverConnectionId": "srv",
            "requestId": "r",
            "attachmentId": "att",
            "attached": true,
        }))
        .unwrap();
        assert!(matches!(
            command,
            SessionWorkerCommand::SessionDemand { attached: true, .. }
        ));
        let event: SessionWorkerEvent = serde_json::from_value(serde_json::json!({
            "type": "worker_ready",
            "token": "t",
            "sessionKey": "/sessions/s",
            "sessionId": "s",
            "pid": 42,
            "metadata": { "id": "s", "createdAt": 1, "cwd": "/cwd", "path": "/sessions/s" },
            "pluginManifestPaths": ["/m.json"],
        }))
        .unwrap();
        assert!(matches!(
            event,
            SessionWorkerEvent::WorkerReady { pid: 42, .. }
        ));
    }

    #[test]
    fn coordinator_input_rejects_non_server_messages() {
        assert!(parse_coordinator_input(&serde_json::json!({
            "type": "message", "from": "server", "payload": {}
        }))
        .is_some());
        assert!(parse_coordinator_input(&serde_json::json!({
            "type": "message", "from": "worker-1", "payload": {}
        }))
        .is_none());
        assert!(parse_coordinator_input(&serde_json::json!({
            "type": "server_connected", "serverConnectionId": "srv"
        }))
        .is_some());
    }

    #[test]
    fn scope_conversion_preserves_fields() {
        let scope = WorkerServiceScope {
            server_connection_id: "srv".into(),
            attachment_id: "att".into(),
        };
        let wire = WorkerOperationScope::from(&scope);
        assert_eq!(wire.server_connection_id, "srv");
        let back: WorkerServiceScope = wire.into();
        assert_eq!(back, scope);
    }

    #[tokio::test]
    async fn lifecycle_retires_after_initial_grace() {
        let _guard = TEST_GUARD.lock();
        let retired = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&retired);
        let lifecycle = WorkerLifecycle::new(
            None,
            10,
            10_000,
            Box::new(move || {
                flag.store(true, Ordering::SeqCst);
            }),
        );
        // A live demand defers retirement.
        lifecycle.server_connected("srv");
        lifecycle.set_demand("srv", "att", true).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!retired.load(Ordering::SeqCst));
        lifecycle.set_demand("srv", "att", false).unwrap();
        assert!(retired.load(Ordering::SeqCst));
        assert!(lifecycle.retiring.load(Ordering::SeqCst));
        lifecycle.close();
    }

    #[tokio::test]
    async fn lifecycle_demands_from_stale_generations_reject() {
        let _guard = TEST_GUARD.lock();
        let lifecycle = WorkerLifecycle::new(Some("srv-1".into()), 10_000, 10_000, Box::new(|| {}));
        lifecycle.server_connected("srv-2");
        assert_eq!(
            lifecycle
                .set_demand("srv-1", "att", true)
                .unwrap_err()
                .to_string(),
            "Session worker received demand from a stale server generation"
        );
        lifecycle.set_demand("srv-2", "att", true).unwrap();
        assert!(lifecycle.begin_request("srv-2", "att").is_ok());
        assert_eq!(
            lifecycle
                .begin_request("srv-2", "other")
                .err()
                .expect("mismatched attachment rejects")
                .to_string(),
            "Session worker request does not match the active attachment"
        );
        lifecycle.close();
    }

    #[tokio::test]
    async fn lifecycle_holds_and_harness_block_retirement() {
        let _guard = TEST_GUARD.lock();
        let retired = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&retired);
        let lifecycle = WorkerLifecycle::new(
            None,
            5,
            5,
            Box::new(move || {
                flag.store(true, Ordering::SeqCst);
            }),
        );
        lifecycle.demand_initialized.store(true, Ordering::SeqCst);
        lifecycle.initial_timer.lock().take();
        let release = lifecycle.hold_retirement();
        lifecycle.set_harness_active(true);
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert!(!retired.load(Ordering::SeqCst));
        lifecycle.set_harness_active(false);
        assert!(!retired.load(Ordering::SeqCst));
        release();
        assert!(retired.load(Ordering::SeqCst));
    }

    #[test]
    fn lifecycle_delay_reads_env() {
        let _guard = TEST_GUARD.lock();
        assert_eq!(
            lifecycle_delay(SESSION_WORKER_INITIAL_DEMAND_GRACE_ENV, 7).unwrap(),
            7
        );
        std::env::set_var(SESSION_WORKER_INITIAL_DEMAND_GRACE_ENV, "42");
        assert_eq!(
            lifecycle_delay(SESSION_WORKER_INITIAL_DEMAND_GRACE_ENV, 7).unwrap(),
            42
        );
        std::env::set_var(SESSION_WORKER_INITIAL_DEMAND_GRACE_ENV, "-1");
        assert!(lifecycle_delay(SESSION_WORKER_INITIAL_DEMAND_GRACE_ENV, 7).is_err());
        std::env::remove_var(SESSION_WORKER_INITIAL_DEMAND_GRACE_ENV);
    }

    #[tokio::test]
    async fn entry_requires_session_worker_role() {
        let _guard = TEST_GUARD.lock();
        std::env::set_var(super::super::process::INTERNAL_PROCESS_ENV, "server");
        let error = run_session_worker_entry(&[]).await.unwrap_err();
        assert_eq!(
            error,
            "Session worker entrypoint requires an internal session-worker invocation"
        );
        std::env::remove_var(super::super::process::INTERNAL_PROCESS_ENV);
    }

    #[tokio::test]
    async fn with_harness_validates_arguments() {
        let _guard = TEST_GUARD.lock();
        let create: CreateSessionWorkerHarness =
            Arc::new(|_path, _options| Box::pin(async { unreachable!("never invoked") }));
        let error = run_session_worker_with_harness(&[], Arc::clone(&create))
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Session worker requires one options argument"
        );
        let error = run_session_worker_with_harness(&["not json".to_string()], create)
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .starts_with("Session worker received invalid options"));
    }
}
