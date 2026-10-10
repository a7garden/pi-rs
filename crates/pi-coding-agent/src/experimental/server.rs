//! Port of `experimental/server.ts` — the Session server entry: directory and
//! profile locking, cold activation, the [`ServerLifetime`] hold reconciler,
//! and [`start_server`] assembling the routed server, the coordinator
//! connection, the [`SessionWorkerManager`], and the Radius relay.
//!
//! This is distinct from [`crate::experimental::services::server`], which
//! ports upstream `experimental/services/server.ts` (the routed service
//! host).
//!
//! # Deviations (D162)
//!
//! - `proper-lockfile` locks map to [`crate::experimental::lockfile::FileLock`]
//!   polls (D149); `launcher-<id>` locks keep their files
//!   (`release` does not unlink).
//! - `ServerLifetime`'s unref'd startup/retirement timers become one spawned
//!   driver task re-evaluated through a notify signal; the startup grace is a
//!   generation-guarded sleep. Observable holds (operator/startup/
//!   connection/worker counts) and the idle-grace re-check match upstream.
//! - `runtime.closed` becomes a [`ServerClosed`] handle (first-write-wins,
//!   shared); `close()` is memoized onto its own slot.
//! - `connect()`'s Node error-code chain walk maps to walking
//!   [`std::error::Error::source`]: a `DisconnectedError`, a `version` client
//!   server error, or chained `io::Error`s of the upstream swallow set
//!   (`ENOENT`/`ECONNREFUSED`/`ECONNRESET`/`EPIPE`/`ETIMEDOUT`) read as "no
//!   server yet".
//! - SIGINT/SIGTERM handlers become spawned tokio signal tasks that invoke
//!   the same idempotent `close`.
//! - The routed `ServerHost` trait passes session metadata opaquely, so
//!   `open_session` re-resolves the catalog metadata by ID (upstream threads
//!   the resolved metadata object straight through).

use std::collections::{HashMap, HashSet};
use std::io::ErrorKind;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::future::BoxFuture;
use parking_lot::Mutex as PLMutex;
use pi_chord::node::manifest::FacetBundleArtifact;
use pi_chord::services::errors::{error, AnyError};
use pi_chord::{ChordError, Context, JsonValue, RemoteServiceErrorCode, BACKGROUND_CONTEXT};
use pi_client::unix::{create_unix_transport_factory, UnixServerRoute, UnixTransportOptions};
use pi_client::{Client, ClientOptions, DisconnectedError};
use pi_protocol::is_server_id;
use pi_server::errors::{ServerError as RoutedServerError, ServerOperationErrorCode};
use pi_server::server::Server as RoutedServer;
use pi_server::transports::unix::{create_unix_server, get_unix_socket_path, UnixServerOptions};
use pi_server::types::{AnyError as SrvAnyError, ServerHost, SessionMetadata};
use serde_json::Value;

use crate::config::get_agent_dir;
use crate::experimental::coordinator::{
    ensure_coordinator, CoordinatorConnection, CoordinatorConnectionOptions,
    CoordinatorStartupLease,
};
use crate::experimental::lockfile::{FileLock, LockOptions};
use crate::experimental::plugins::bundled::create_presentation_facet_data;
use crate::experimental::plugins::package::{
    create_server_plugin_package, normalize_plugin_package_paths,
    read_session_plugin_package_profile, remove_session_plugin_package_profile,
    restore_server_plugin_package_profile, write_session_plugin_package_profile,
    ConfiguredServerPluginPackage,
};
use crate::experimental::process::{
    consume_internal_process_role, spawn_internal_process, terminate_internal_process,
    InternalProcessRole,
};
use crate::experimental::radius_auth::{AuthInput, RadiusRelayAuthResolver};
use crate::experimental::radius_relay::{
    RadiusRelayHost, RadiusRelayHostOptions, RadiusRelayHostStatus,
};
use crate::experimental::services::server::{
    create_experimental_server_services, ExperimentalServerServices,
    ExperimentalServerServicesOptions, PreparedSessionPlugins,
};
use crate::experimental::services::sessions::{SessionCreateOptions, SessionSummary};
use crate::experimental::session_catalog::{
    create_session as create_catalog_session, delete_session,
    list_sessions as list_catalog_sessions, read_session,
    CreateSessionOptions as CatalogCreateOptions, SessionCatalogMetadata,
};
use crate::experimental::session_worker_manager as manager_mod;
use crate::utils::paths::resolve_path_cwd;

use manager_mod::{
    new_uuid, shared_error, PromiseSlot, SessionPluginSelectionConflictError, SessionWorkerManager,
    SharedError, WorkerModelOptions, WorkerPids,
};

/// `PI_SERVER_DIR`.
pub const ENV_SERVER_DIR: &str = "PI_SERVER_DIR";
/// `PI_SERVER_ID`.
pub const ENV_SERVER_ID: &str = "PI_SERVER_ID";

const LOCK_RETRY_MS: u64 = 25;
const LOCK_WAIT_MS: u64 = 30_000;
const DEFAULT_SERVER_ID_FILE: &str = "default-server-id";

const ACTIVATION_TIMEOUT_MS: u64 = 10_000;
const ACTIVATION_RETRY_MS: u64 = 10;

const AUTO_SERVER_STARTUP_GRACE_MS: u64 = 10_000;
const AUTO_SERVER_IDLE_GRACE_MS: u64 = 1_000;

fn any_err(message: impl Into<String>) -> AnyError {
    error(message)
}

// ---------------------------------------------------------------------------
// Directories.
// ---------------------------------------------------------------------------

/// `resolveServerDirectory(directory?)`.
pub fn resolve_server_directory(directory: Option<&str>) -> String {
    resolve_path_cwd(
        &directory
            .map(str::to_string)
            .or_else(|| std::env::var(ENV_SERVER_DIR).ok())
            .unwrap_or_else(|| {
                Path::new(&home_dir())
                    .join(".pi")
                    .join("server")
                    .to_string_lossy()
                    .into_owned()
            }),
    )
}

fn home_dir() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/".to_string())
}

/// `ensurePrivateServerDirectory(directory)`.
pub fn ensure_private_server_directory(directory: &str) -> Result<(), AnyError> {
    let path = Path::new(directory);
    std::fs::create_dir_all(path)
        .map_err(|io_error| error(format!("Failed to create {directory}: {io_error}")))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
        .map_err(|io_error| error(format!("Failed to chmod {directory}: {io_error}")))?;
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|io_error| error(format!("Failed to stat {directory}: {io_error}")))?;
    if !metadata.is_dir() {
        return Err(error(format!(
            "Unix socket directory is not a directory: {directory}"
        )));
    }
    let uid = unsafe { libc::getuid() };
    if metadata.uid() != uid {
        return Err(error(format!(
            "Unix socket directory is not owned by the current user: {directory}"
        )));
    }
    Ok(())
}

/// `resolveSessionDirectory(sessionDir?)`.
pub fn resolve_session_directory(session_dir: Option<&str>) -> String {
    resolve_path_cwd(&session_dir.map(str::to_string).unwrap_or_else(|| {
        Path::new(&get_agent_dir())
            .join("experimental")
            .join("sessions")
            .to_string_lossy()
            .into_owned()
    }))
}

// ---------------------------------------------------------------------------
// Server profile (one logical server ID per shared directory).
// ---------------------------------------------------------------------------

/// Upstream `ServerProfile`.
pub struct ServerProfile {
    pub server_id: String,
    pub release: FileLock,
}

/// `acquireServerProfile(directory, requestedServerId?)` — lock one logical
/// server ID in a shared experimental server directory.
pub async fn acquire_server_profile(
    directory: &str,
    requested_server_id: Option<&str>,
) -> Result<ServerProfile, AnyError> {
    let dir_path = Path::new(directory);
    std::fs::create_dir_all(dir_path)
        .map_err(|io_error| error(format!("Failed to create {directory}: {io_error}")))?;
    let server_id = match requested_server_id {
        Some(requested) => {
            if !is_server_id(requested) {
                return Err(error(format!(
                    "Invalid experimental server ID: {requested}"
                )));
            }
            requested.to_string()
        }
        None => read_or_create_default_server_id(dir_path)?,
    };
    let release = FileLock::acquire(
        &dir_path.join(format!("launcher-{server_id}")),
        // Upstream stale/update windows are mtime liveness knobs (D149); the
        // kernel-managed lock reduces to the poll/wait budget.
        LockOptions {
            retry_ms: LOCK_RETRY_MS,
            wait_ms: LOCK_WAIT_MS,
        },
    )
    .await
    .map_err(|lock_error| lock_error)?;
    Ok(ServerProfile {
        server_id,
        release: release.keep_file(),
    })
}

/// The `default-server-id` read/`wx`-create/re-read ladder.
fn read_or_create_default_server_id(directory: &Path) -> Result<String, AnyError> {
    let path = directory.join(DEFAULT_SERVER_ID_FILE);
    match std::fs::read_to_string(&path) {
        Ok(value) => {
            let value = value.trim().to_string();
            if !is_server_id(&value) {
                return Err(error(format!(
                    "Invalid default experimental server identity in {}",
                    path.display()
                )));
            }
            Ok(value)
        }
        Err(read_error) if read_error.kind() == ErrorKind::NotFound => {
            let candidate = new_uuid();
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    use std::io::Write;
                    file.write_all(candidate.as_bytes())
                        .map_err(|write_error| {
                            error(format!("Failed to write {}: {write_error}", path.display()))
                        })?;
                    Ok(candidate)
                }
                // A concurrent creator won; re-read like upstream's `EEXIST`.
                Err(create_error) if create_error.kind() == ErrorKind::AlreadyExists => {
                    let value = std::fs::read_to_string(&path)
                        .map_err(|io_error| error(io_error.to_string()))?
                        .trim()
                        .to_string();
                    if !is_server_id(&value) {
                        return Err(error(format!(
                            "Invalid default experimental server identity in {}",
                            path.display()
                        )));
                    }
                    Ok(value)
                }
                Err(create_error) => Err(error(create_error.to_string())),
            }
        }
        Err(read_error) => Err(error(read_error.to_string())),
    }
}

/// `acquireServerActivation(directory, serverId)`.
pub async fn acquire_server_activation(
    directory: &str,
    server_id: &str,
) -> Result<FileLock, AnyError> {
    FileLock::acquire(
        &Path::new(directory).join(format!("activation-{server_id}")),
        LockOptions {
            retry_ms: 25,
            wait_ms: ACTIVATION_TIMEOUT_MS,
        },
    )
    .await
    .map_err(error)
}

// ---------------------------------------------------------------------------
// Cold activation.
// ---------------------------------------------------------------------------

/// Upstream `ActivatedServer`.
pub struct ActivatedServer {
    pub client: Client,
    pub route: UnixServerRoute,
}

/// Upstream `ActivateServerOptions`.
pub struct ActivateServerOptions {
    pub directory: String,
    pub requested_server_id: Option<String>,
    pub session_dir: String,
    pub provider: Option<String>,
    pub model: Option<String>,
}

/// `connect(route)` — an existing server reads as `None` when absent.
async fn connect(route: &UnixServerRoute) -> Result<Option<Client>, AnyError> {
    let transport_factory = create_unix_transport_factory(UnixTransportOptions {
        path: route.path.clone(),
        max_pending_bytes: None,
    })
    .map_err(|factory_error| factory_error)?;
    let client = Client::new(ClientOptions {
        transport_factory,
        server_id: route.server_id.clone(),
        max_frame_length: None,
        on_listener_error: None,
    })
    .map_err(|client_error| client_error)?;
    match client.connect().await {
        Ok(_hello) => Ok(Some(client)),
        Err(connect_error) => {
            client.dispose().await;
            if connect_error.downcast_ref::<DisconnectedError>().is_some() {
                return Ok(None);
            }
            if connect_error
                .downcast_ref::<pi_client::ServerError>()
                .is_some_and(|server_error| server_error.code == "version")
            {
                return Ok(None);
            }
            if is_swallowed_connect_error(connect_error.as_ref(), &mut HashSet::new()) {
                return Ok(None);
            }
            Err(connect_error)
        }
    }
}

/// Walk the cause chain for the upstream swallow-list io error codes.
fn is_swallowed_connect_error(
    current: &(dyn std::error::Error + 'static),
    seen: &mut HashSet<usize>,
) -> bool {
    let pointer = current as *const dyn std::error::Error as *const () as usize;
    if !seen.insert(pointer) {
        return false;
    }
    if let Some(io_error) = current.downcast_ref::<std::io::Error>() {
        return matches!(
            io_error.kind(),
            ErrorKind::NotFound
                | ErrorKind::ConnectionRefused
                | ErrorKind::ConnectionReset
                | ErrorKind::BrokenPipe
                | ErrorKind::TimedOut
        );
    }
    match current.source() {
        Some(source) => is_swallowed_connect_error(source, seen),
        None => false,
    }
}

/// `activateServer(options)` — ensure the selected logical server is
/// reachable, launching the current Pi installation if needed.
pub async fn activate_server(options: &ActivateServerOptions) -> Result<ActivatedServer, AnyError> {
    if options.provider.is_some() && options.model.is_none() {
        return Err(error("Server model provider requires a model"));
    }
    ensure_private_server_directory(&options.directory)?;
    let profile =
        acquire_server_profile(&options.directory, options.requested_server_id.as_deref()).await?;
    let server_id = profile.server_id;
    profile.release.release();
    let route = UnixServerRoute {
        server_id: server_id.clone(),
        path: get_unix_socket_path(&server_id, &options.directory)
            .map_err(|route_error| error(route_error.to_string()))?
            .to_string_lossy()
            .into_owned(),
    };
    let release = acquire_server_activation(&options.directory, &server_id).await?;
    let result = async {
        if let Some(existing) = connect(&route).await? {
            // Another activator won the race, so startup-only selections can
            // no longer be applied.
            if options.model.is_some() {
                existing.dispose().await;
                return Err(error(
                    "Model selection is only valid when automatically activating a new server",
                ));
            }
            return Ok(ActivatedServer {
                client: existing,
                route,
            });
        }
        let model_args: Vec<String> = match &options.model {
            None => Vec::new(),
            Some(model) => {
                let mut object = serde_json::Map::new();
                if let Some(provider) = &options.provider {
                    object.insert("provider".to_string(), Value::String(provider.clone()));
                }
                object.insert("model".to_string(), Value::String(model.clone()));
                vec![Value::Object(object).to_string()]
            }
        };
        let args: Vec<String> = [
            options.directory.clone(),
            server_id.clone(),
            options.session_dir.clone(),
        ]
        .into_iter()
        .chain(model_args)
        .collect();
        let child = spawn_internal_process(InternalProcessRole::Server, &args, &Default::default())
            .map_err(|spawn_error| {
                error(format!(
                    "Failed to automatically activate server: {spawn_error}"
                ))
            })?;
        let result = wait_for_activated_server(&route, child).await;
        result
    }
    .await;
    release.release();
    result
}

/// The activation connect loop; `child` is force-terminated on failure
/// (upstream the `finally { await terminateInternalProcess(child) }`).
async fn wait_for_activated_server(
    route: &UnixServerRoute,
    mut child: tokio::process::Child,
) -> Result<ActivatedServer, AnyError> {
    let deadline = Instant::now() + Duration::from_millis(ACTIVATION_TIMEOUT_MS);
    let result = loop {
        if let Some(client) = connect(route).await? {
            break Ok(ActivatedServer {
                client,
                route: route.clone(),
            });
        }
        let exited = child
            .try_wait()
            .map_err(|wait_error| error(wait_error.to_string()))?;
        if exited.is_some() {
            break Err(error(
                "Automatically activated server exited during startup",
            ));
        }
        if Instant::now() >= deadline {
            break Err(error(
                "Timed out waiting for automatically activated server",
            ));
        }
        tokio::time::sleep(Duration::from_millis(ACTIVATION_RETRY_MS)).await;
    };
    if result.is_err() {
        terminate_internal_process(child).await;
    }
    result
}

// ---------------------------------------------------------------------------
// ServerLifetime.
// ---------------------------------------------------------------------------

struct LifetimeInner {
    keep_alive: bool,
    connection_count: AtomicUsize,
    worker_count: AtomicUsize,
    startup_held: AtomicBool,
    startup_generation: AtomicUsize,
    stopped: AtomicBool,
    retire: PLMutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    reconcile_notify: tokio::sync::Notify,
    driver_started: AtomicBool,
}

/// `ServerLifetime` — reconcile operator, startup, client, and worker holds
/// for one server generation.
#[derive(Clone)]
pub struct ServerLifetime {
    inner: Arc<LifetimeInner>,
}

impl ServerLifetime {
    pub fn new(keep_alive: bool) -> Self {
        Self {
            inner: Arc::new(LifetimeInner {
                keep_alive,
                connection_count: AtomicUsize::new(0),
                worker_count: AtomicUsize::new(0),
                startup_held: AtomicBool::new(!keep_alive),
                startup_generation: AtomicUsize::new(0),
                stopped: AtomicBool::new(false),
                retire: PLMutex::new(None),
                reconcile_notify: tokio::sync::Notify::new(),
                driver_started: AtomicBool::new(false),
            }),
        }
    }

    pub fn start(&self, retire: Arc<dyn Fn() + Send + Sync>) {
        *self.inner.retire.lock() = Some(retire);
        if !self.inner.driver_started.swap(true, Ordering::SeqCst) {
            let inner = Arc::clone(&self.inner);
            tokio::spawn(async move { drive_lifetime(inner).await });
        }
        if self.inner.startup_held.load(Ordering::SeqCst) {
            let inner = Arc::clone(&self.inner);
            let generation = self.inner.startup_generation.load(Ordering::SeqCst);
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(AUTO_SERVER_STARTUP_GRACE_MS)).await;
                // The generation guard cancels a grace overtaken by demand
                // (upstream `clearTimeout(startupTimer)`).
                if inner.startup_generation.load(Ordering::SeqCst) == generation {
                    inner.startup_held.store(false, Ordering::SeqCst);
                    inner.reconcile_notify.notify_waiters();
                }
            });
        }
        self.reconcile();
    }

    pub fn set_connection_count(&self, count: usize) {
        self.inner.connection_count.store(count, Ordering::SeqCst);
        if count > 0 {
            self.inner.startup_generation.fetch_add(1, Ordering::SeqCst);
            self.inner.startup_held.store(false, Ordering::SeqCst);
        }
        self.reconcile();
    }

    pub fn set_worker_count(&self, count: usize) {
        self.inner.worker_count.store(count, Ordering::SeqCst);
        self.reconcile();
    }

    pub fn stop(&self) {
        self.inner.stopped.store(true, Ordering::SeqCst);
        self.inner.reconcile_notify.notify_waiters();
    }

    fn reconcile(&self) {
        self.inner.reconcile_notify.notify_waiters();
    }

    fn holds(&self) -> bool {
        self.inner.keep_alive
            || self.inner.startup_held.load(Ordering::SeqCst)
            || self.inner.connection_count.load(Ordering::SeqCst) != 0
            || self.inner.worker_count.load(Ordering::SeqCst) != 0
    }
}

/// The startup/retirement timer driver (upstream `#reconcile` plus its two
/// unref'd timers).
async fn drive_lifetime(inner: Arc<LifetimeInner>) {
    let lifetime = ServerLifetime {
        inner: Arc::clone(&inner),
    };
    let mut pending_deadline: Option<Instant> = None;
    loop {
        if inner.stopped.load(Ordering::SeqCst) {
            return;
        }
        if lifetime.holds() {
            pending_deadline = None;
        } else if pending_deadline.is_none() {
            pending_deadline =
                Some(Instant::now() + Duration::from_millis(AUTO_SERVER_IDLE_GRACE_MS));
        }
        let deadline = match pending_deadline {
            Some(deadline) => deadline,
            None => {
                inner.reconcile_notify.notified().await;
                continue;
            }
        };
        tokio::select! {
            () = tokio::time::sleep_until(deadline.into()) => {
                // Re-check at fire time (upstream's retirement timer body).
                if !lifetime.holds() && !inner.stopped.load(Ordering::SeqCst) {
                    if let Some(retire) = inner.retire.lock().clone() {
                        retire();
                    }
                    return;
                }
                pending_deadline = None;
            }
            () = inner.reconcile_notify.notified() => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Running server.
// ---------------------------------------------------------------------------

/// Shared settled-closed slot standing in for `runtime.closed` (D162).
#[derive(Clone)]
pub struct ServerClosed {
    slot: PromiseSlot<Result<(), SharedError>>,
}

impl ServerClosed {
    fn new() -> Self {
        Self {
            slot: PromiseSlot::new(),
        }
    }

    pub async fn closed(&self) -> Result<(), SrvAnyError> {
        self.slot.get().await
    }
}

/// Upstream `RunningServerBackend` (`RunningServer` + `refreshSessions`).
pub struct RunningServerBackend {
    pub server_id: String,
    pub session_dir: String,
    pub socket_path: String,
    pub server: Arc<RoutedServer>,
    pub worker_pids: WorkerPids,
    pub closed: ServerClosed,
    pub services: Arc<ExperimentalServerServices>,
}

impl RunningServerBackend {
    async fn refresh_sessions(&self, context: Context) -> Result<(), AnyError> {
        self.services.refresh(context).await
    }

    /// `server.close().then(() => closed)` — resolves with the backend's
    /// closed outcome after the close settles.
    async fn close_and_await_closed(&self) -> Result<(), SharedError> {
        let close_result = self.server.close().await.map_err(|close_error| close_error);
        let closed_result = self.closed.closed().await;
        close_result.and(closed_result)
    }
}

/// Upstream `RunningServer`.
pub struct RunningServer {
    pub server_id: String,
    pub session_dir: String,
    pub socket_path: String,
    pub server: Arc<RoutedServer>,
    pub worker_pids: WorkerPids,
    pub closed: ServerClosed,
    lifetime: ServerLifetime,
    relay: Arc<RadiusRelayHost>,
    backend: Arc<RunningServerBackend>,
    coordinator: Arc<CoordinatorConnection>,
    workers: Arc<SessionWorkerManager>,
    close_slot: PLMutex<Option<(PromiseSlot<Result<(), SharedError>>, Arc<AtomicBool>)>>,
}

impl RunningServer {
    /// Upstream `close()` — memoized teardown in relay → backend → workers →
    /// coordinator order, resolving through the backend's `closed`.
    pub async fn close(&self) -> Result<(), SrvAnyError> {
        let (slot, started) = {
            let mut close_slot = self.close_slot.lock();
            let entry = close_slot
                .get_or_insert_with(|| (PromiseSlot::new(), Arc::new(AtomicBool::new(false))));
            (entry.0.clone(), Arc::clone(&entry.1))
        };
        if !started.swap(true, Ordering::SeqCst) {
            let lifetime = self.lifetime.clone();
            let relay = Arc::clone(&self.relay);
            let backend = Arc::clone(&self.backend);
            let coordinator = Arc::clone(&self.coordinator);
            let workers = Arc::clone(&self.workers);
            let spawn_slot = slot.clone();
            tokio::spawn(async move {
                lifetime.stop();
                relay.close().await;
                let close_result = backend.close_and_await_closed().await;
                if coordinator.was_replaced() {
                    workers.detach();
                } else {
                    workers.shutdown().await;
                }
                coordinator.close();
                spawn_slot.resolve(close_result);
            });
        }
        slot.get().await
    }
}

// ---------------------------------------------------------------------------
// Backend assembly.
// ---------------------------------------------------------------------------

/// Upstream `ResolvedSessionPlugins`.
#[derive(Clone)]
pub struct ResolvedSessionPlugins {
    pub package_paths: Vec<String>,
    pub manifest_paths: Vec<String>,
    pub presentation_artifacts: Vec<FacetBundleArtifact>,
}

pub type ResolveSessionPluginsFn = Arc<
    dyn Fn(
            SessionCatalogMetadata,
            Option<Vec<String>>,
            Context,
        ) -> BoxFuture<'static, Result<ResolvedSessionPlugins, AnyError>>
        + Send
        + Sync,
>;

/// Upstream `StartServerBackendOptions`.
pub struct StartServerBackendOptions {
    pub path: String,
    pub server_id: String,
    pub session_dir: Option<String>,
    pub resolve_session_plugins: ResolveSessionPluginsFn,
    pub remove_session_plugins: Arc<
        dyn Fn(SessionCatalogMetadata) -> BoxFuture<'static, Result<(), AnyError>> + Send + Sync,
    >,
    pub reload_presentation_facet_bundles: Arc<
        dyn Fn(Vec<String>) -> BoxFuture<'static, Result<Vec<FacetBundleArtifact>, AnyError>>
            + Send
            + Sync,
    >,
}

type ResolveSessionFn = Arc<
    dyn Fn(String, Context) -> BoxFuture<'static, Result<SessionCatalogMetadata, AnyError>>
        + Send
        + Sync,
>;

/// `startServerBackend(options, workers, onConnectionCountChanged?)`.
async fn start_server_backend(
    options: StartServerBackendOptions,
    workers: Arc<SessionWorkerManager>,
    on_connection_count_changed: Option<Arc<dyn Fn(usize) + Send + Sync>>,
) -> Result<RunningServerBackend, AnyError> {
    let server_id = options.server_id.clone();
    let session_dir = resolve_session_directory(options.session_dir.as_deref());

    let list_sessions: Arc<
        dyn Fn(Context) -> BoxFuture<'static, Result<Vec<SessionCatalogMetadata>, AnyError>>
            + Send
            + Sync,
    > = {
        let session_dir = session_dir.clone();
        let workers = Arc::clone(&workers);
        Arc::new(move |_context: Context| {
            let session_dir = session_dir.clone();
            let workers = Arc::clone(&workers);
            Box::pin(async move {
                let mut sessions: HashMap<String, SessionCatalogMetadata> =
                    list_catalog_sessions(&session_dir)
                        .await
                        .map_err(|list_error| list_error)?
                        .into_iter()
                        .map(|metadata| (metadata.path.clone(), metadata))
                        .collect();
                for metadata in workers.tracked_sessions() {
                    sessions.insert(metadata.path.clone(), metadata);
                }
                Ok(sessions.into_values().collect::<Vec<_>>())
            })
        })
    };

    let summarize = {
        let server_id = server_id.clone();
        Arc::new(move |metadata: &SessionCatalogMetadata| SessionSummary {
            server_id: server_id.clone(),
            session_id: metadata.id.clone(),
            created_at: metadata.created_at.max(0) as u64,
        })
    };

    let resolve_session: ResolveSessionFn = {
        let list_sessions = Arc::clone(&list_sessions);
        let session_dir = session_dir.clone();
        Arc::new(move |session_id: String, _context: Context| {
            let list_sessions = Arc::clone(&list_sessions);
            let session_dir = session_dir.clone();
            Box::pin(async move {
                let tracked = list_sessions(BACKGROUND_CONTEXT.clone()).await?;
                let metadata = tracked
                    .into_iter()
                    .find(|candidate| candidate.id == session_id);
                match metadata {
                    Some(metadata) => Ok(metadata),
                    None => read_session(&session_dir, &session_id)
                        .await
                        .map_err(|read_error| read_error)?
                        .ok_or_else(|| any_err(format!("Unknown session: {session_id}"))),
                }
            })
        })
    };

    let create_session = {
        let session_dir = session_dir.clone();
        Arc::new(
            move |create_options: SessionCreateOptions, _context: Context| {
                let session_dir = session_dir.clone();
                Box::pin(async move {
                    create_catalog_session(
                        &session_dir,
                        &CatalogCreateOptions {
                            id: create_options.id,
                            cwd: std::env::current_dir()
                                .map(|dir| dir.to_string_lossy().into_owned())
                                .unwrap_or_else(|_| ".".to_string()),
                        },
                    )
                    .await
                    .map_err(|create_error| error(create_error))
                }) as BoxFuture<'static, Result<SessionCatalogMetadata, AnyError>>
            },
        )
    };

    let list_for_services = {
        let list_sessions = Arc::clone(&list_sessions);
        let summarize = Arc::clone(&summarize);
        Arc::new(move |context: Context| {
            let list_sessions = Arc::clone(&list_sessions);
            let summarize = Arc::clone(&summarize);
            Box::pin(async move {
                let mut sessions: Vec<SessionSummary> = list_sessions(context)
                    .await?
                    .iter()
                    .map(|metadata| summarize(metadata))
                    .collect();
                sessions.sort_by(|left, right| {
                    left.session_id
                        .cmp(&right.session_id)
                        .then(left.created_at.cmp(&right.created_at))
                });
                Ok(sessions)
            }) as BoxFuture<'static, Result<Vec<SessionSummary>, AnyError>>
        })
    };

    let remove_for_services = {
        let resolve_session = Arc::clone(&resolve_session);
        let remove_plugins = Arc::clone(&options.remove_session_plugins);
        let workers = Arc::clone(&workers);
        Arc::new(move |session_id: String, context: Context| {
            let resolve_session = Arc::clone(&resolve_session);
            let remove_plugins = Arc::clone(&remove_plugins);
            let workers = Arc::clone(&workers);
            Box::pin(async move {
                let metadata = resolve_session(session_id, context.clone()).await?;
                workers
                    .close_session(&metadata)
                    .await
                    .map_err(|close_error| -> AnyError {
                        Box::new(ChordError {
                            message: close_error.to_string(),
                        })
                    })?;
                delete_session(&metadata)
                    .await
                    .map_err(|delete_error| delete_error)?;
                remove_plugins(metadata).await
            }) as BoxFuture<'static, Result<(), AnyError>>
        })
    };

    let prepare_session_plugins = {
        let resolve_session = Arc::clone(&resolve_session);
        let resolve_plugins = Arc::clone(&options.resolve_session_plugins);
        Arc::new(
            move |session_id: String, package_paths: Option<Vec<String>>, context: Context| {
                let resolve_session = Arc::clone(&resolve_session);
                let resolve_plugins = Arc::clone(&resolve_plugins);
                Box::pin(async move {
                    let metadata = resolve_session(session_id, context.clone()).await?;
                    let selected = match resolve_plugins(metadata, package_paths, context).await {
                        Ok(selected) => selected,
                        Err(prepare_error) => {
                            if prepare_error
                                .downcast_ref::<SessionPluginSelectionConflictError>()
                                .is_some()
                            {
                                return Err(any_err(
                                    RoutedServerError::new(
                                        ServerOperationErrorCode::Remote(
                                            RemoteServiceErrorCode::ServiceInvalidValue,
                                        ),
                                        prepare_error.to_string(),
                                    )
                                    .to_string(),
                                ));
                            }
                            return Err(prepare_error);
                        }
                    };
                    Ok(PreparedSessionPlugins {
                        package_paths: selected.package_paths,
                        presentation_plugins: create_presentation_facet_data(
                            &selected.presentation_artifacts,
                        ),
                    })
                }) as BoxFuture<'static, Result<PreparedSessionPlugins, AnyError>>
            },
        )
    };

    let reload_presentation_plugins = {
        let reload_bundles = Arc::clone(&options.reload_presentation_facet_bundles);
        Arc::new(move |package_paths: Vec<String>, _context: Context| {
            let reload_bundles = Arc::clone(&reload_bundles);
            Box::pin(async move {
                let artifacts = reload_bundles(package_paths).await?;
                Ok(create_presentation_facet_data(&artifacts))
            }) as BoxFuture<'static, Result<JsonValue, AnyError>>
        })
    };

    let server_services = Arc::new(
        create_experimental_server_services(ExperimentalServerServicesOptions {
            list: list_for_services,
            create: {
                let create_session = Arc::clone(&create_session);
                Arc::new(
                    move |create_options: SessionCreateOptions, context: Context| {
                        let create_session = Arc::clone(&create_session);
                        let summarize = Arc::clone(&summarize);
                        Box::pin(async move {
                            let metadata = create_session(create_options, context).await?;
                            Ok(summarize(&metadata))
                        })
                    },
                )
            },
            remove: remove_for_services,
            prepare_session_plugins,
            reload_presentation_plugins,
        })
        .await?,
    );

    let host = Arc::new(BackendHost {
        server_services: Arc::clone(&server_services.host),
        resolve_session: Arc::clone(&resolve_session),
        workers: Arc::clone(&workers),
        resolve_plugins: Arc::clone(&options.resolve_session_plugins),
    });
    let socket_path = options.path.clone();
    let server = Arc::new(
        create_unix_server(
            host,
            UnixServerOptions {
                server_id: server_id.clone(),
                max_frame_length: None,
                handshake_timeout_ms: None,
                on_connection_count_changed,
                on_error: None,
                path: socket_path.clone(),
                mode: Some(0o600),
                max_pending_bytes: None,
                graceful_close_timeout_ms: None,
            },
        )
        .map_err(|server_error| error(server_error.to_string()))?,
    );
    if let Err(start_error) = server.start().await {
        let close_error = {
            let server = Arc::clone(&server);
            server.close().await.err()
        };
        let dispose_error = Arc::clone(&server_services)
            .dispose()
            .await
            .err()
            .map(shared_error);
        let cleanup_errors: Vec<SrvAnyError> = vec![close_error, dispose_error]
            .into_iter()
            .flatten()
            .collect();
        if !cleanup_errors.is_empty() {
            return Err(error(format!(
                "Experimental server startup and cleanup failed: {}; {}",
                start_error,
                cleanup_errors
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }
        return Err(error(start_error.to_string()));
    }

    // `closed = server.closed.then(() => closeCatalog())` — server closure
    // errors aggregate with repository-dispose failures (upstream the
    // `closed` promise chain).
    let closed = ServerClosed::new();
    {
        let closed_slot = closed.slot.clone();
        let closed_server = Arc::clone(&server);
        let closed_services = Arc::clone(&server_services);
        tokio::spawn(async move {
            let server_error = closed_server.closed().await.err();
            let result = match server_error {
                None => closed_services
                    .dispose()
                    .await
                    .map_err(shared_error)
                    .and(Ok(())),
                Some(server_error) => match closed_services.dispose().await {
                    Ok(()) => Err(server_error),
                    Err(dispose_error) => Err(shared_error(error(format!(
                        "Server and repository shutdown failed: {server_error}; {dispose_error}"
                    )))),
                },
            };
            closed_slot.resolve(result);
        });
    }

    Ok(RunningServerBackend {
        server_id,
        session_dir,
        socket_path,
        server,
        worker_pids: workers.worker_pids().clone(),
        closed,
        services: server_services,
    })
}

/// Metadata adapter for the routed host (upstream `SessionCatalogMetadata`
/// satisfies `SessionMetadata` structurally).
struct CatalogMetadataHandle(Arc<SessionCatalogMetadata>);

impl SessionMetadata for CatalogMetadataHandle {
    fn id(&self) -> &str {
        &self.0.id
    }
}

/// Upstream the `ServerHost<SessionCatalogMetadata>` object literal.
struct BackendHost {
    server_services: Arc<dyn pi_server::types::RoutedServerServiceHost>,
    resolve_session: ResolveSessionFn,
    workers: Arc<SessionWorkerManager>,
    resolve_plugins: ResolveSessionPluginsFn,
}

#[async_trait::async_trait]
impl ServerHost for BackendHost {
    fn server_services(&self) -> Arc<dyn pi_server::types::RoutedServerServiceHost> {
        Arc::clone(&self.server_services)
    }

    async fn resolve_session(
        &self,
        session_id: &str,
        _context: Context,
    ) -> Result<Arc<dyn SessionMetadata>, SrvAnyError> {
        let metadata = (self.resolve_session)(session_id.to_string(), BACKGROUND_CONTEXT.clone())
            .await
            .map_err(shared_error)?;
        Ok(Arc::new(CatalogMetadataHandle(Arc::new(metadata))))
    }

    async fn open_session(
        &self,
        metadata: Arc<dyn SessionMetadata>,
        context: Context,
    ) -> Result<Arc<dyn pi_server::types::RoutedSessionHandle>, SrvAnyError> {
        // The routed trait hands the metadata opaquely; re-resolve by ID
        // (D162). Resolution errors already carry the session-not-found
        // surface.
        let catalog_metadata = (self.resolve_session)(metadata.id().to_string(), context)
            .await
            .map_err(shared_error)?;
        let selected =
            (self.resolve_plugins)(catalog_metadata.clone(), None, BACKGROUND_CONTEXT.clone())
                .await
                .map_err(shared_error)?;
        self.workers
            .open_session(&catalog_metadata, &selected.manifest_paths)
            .await
            .map(|handle| handle as Arc<dyn pi_server::types::RoutedSessionHandle>)
            .map_err(shared_error)
    }
}

// ---------------------------------------------------------------------------
// startServer.
// ---------------------------------------------------------------------------

/// Upstream `StartServerOptions`.
#[derive(Default)]
pub struct StartServerOptions {
    /// Server profile and socket directory. Defaults to `PI_SERVER_DIR` or
    /// `~/.pi/server`.
    pub directory: Option<String>,
    /// Logical service ID. Defaults to `PI_SERVER_ID` or the directory's
    /// default-server-id.
    pub server_id: Option<String>,
    /// Durable session directory. Defaults to the experimental directory
    /// under the configured agent directory.
    pub session_dir: Option<String>,
    /// Optional provider for an explicitly selected Session worker model.
    pub provider: Option<String>,
    /// Optional model override for newly started Session workers.
    pub model: Option<String>,
    /// Hold the server open without client or Session demand. Defaults to
    /// true for foreground servers.
    pub keep_alive: Option<bool>,
    /// Optional explicit Radius credential. Stored Radius auth is used when
    /// omitted.
    pub relay_auth: Option<AuthInput>,
    /// Explicit plugin packages. `None` restores the logical server profile;
    /// an empty list clears it.
    pub plugin_packages: Option<Vec<String>>,
    pub on_relay_status: Option<Arc<dyn Fn(RadiusRelayHostStatus) + Send + Sync>>,
}

struct PluginSelectionState {
    directory: String,
    server_id: String,
    packages: PLMutex<HashMap<String, Arc<ConfiguredServerPluginPackage>>>,
    session_selections: tokio::sync::Mutex<HashMap<String, ResolvedSessionPlugins>>,
    default_selection: tokio::sync::Mutex<ResolvedSessionPlugins>,
}

impl PluginSelectionState {
    fn get_plugin_package(&self, package_path: &str) -> Arc<ConfiguredServerPluginPackage> {
        let mut packages = self.packages.lock();
        packages
            .entry(package_path.to_string())
            .or_insert_with(|| {
                create_server_plugin_package(&self.directory, &self.server_id, package_path)
            })
            .clone()
    }

    async fn build_plugin_selection(
        &self,
        package_paths: &[String],
    ) -> Result<ResolvedSessionPlugins, AnyError> {
        let normalized = normalize_plugin_package_paths(package_paths)?;
        let plugins = normalized
            .iter()
            .map(|package_path| self.get_plugin_package(package_path))
            .collect::<Vec<_>>();
        let built = futures_util::future::try_join_all(plugins.iter().map(|plugin| plugin.build()))
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<FacetBundleArtifact>>();
        Ok(ResolvedSessionPlugins {
            manifest_paths: plugins
                .iter()
                .map(|plugin| plugin.manifest_path.to_string_lossy().into_owned())
                .collect(),
            package_paths: normalized,
            presentation_artifacts: built,
        })
    }
}

/// `startServer(options)` — start a replaceable experimental server behind
/// the stable coordinator endpoint.
pub async fn start_server(options: StartServerOptions) -> Result<Arc<RunningServer>, AnyError> {
    if options.provider.is_some() && options.model.is_none() {
        return Err(error("Server model provider requires a model"));
    }
    let worker_model = options.model.as_ref().map(|model| WorkerModelOptions {
        provider: options.provider.clone(),
        model: model.clone(),
    });
    let directory = resolve_server_directory(options.directory.as_deref());
    let profile = {
        let env_server_id = std::env::var(ENV_SERVER_ID).ok();
        acquire_server_profile(
            &directory,
            options.server_id.as_deref().or(env_server_id.as_deref()),
        )
        .await?
    };
    let server_id = profile.server_id;
    let result = start_server_inner(&options, &directory, &server_id, worker_model).await;
    match result {
        Ok(runtime) => {
            profile.release.release();
            Ok(runtime)
        }
        Err(start_error) => {
            profile.release.release();
            Err(start_error)
        }
    }
}

async fn start_server_inner(
    options: &StartServerOptions,
    directory: &str,
    server_id: &str,
    worker_model: Option<WorkerModelOptions>,
) -> Result<Arc<RunningServer>, AnyError> {
    let server_id_string = server_id.to_string();
    ensure_private_server_directory(directory)?;
    let lifetime = ServerLifetime::new(options.keep_alive.unwrap_or(true));
    let plugin_state = Arc::new(PluginSelectionState {
        directory: directory.to_string(),
        server_id: server_id_string.clone(),
        packages: PLMutex::new(HashMap::new()),
        session_selections: tokio::sync::Mutex::new(HashMap::new()),
        default_selection: tokio::sync::Mutex::new(ResolvedSessionPlugins {
            package_paths: Vec::new(),
            manifest_paths: Vec::new(),
            presentation_artifacts: Vec::new(),
        }),
    });
    {
        let profile_paths = restore_server_plugin_package_profile(
            directory,
            server_id,
            options.plugin_packages.clone(),
        )
        .await
        .map_err(|profile_error| profile_error)?;
        let default_selection = plugin_state.build_plugin_selection(&profile_paths).await?;
        *plugin_state.default_selection.lock().await = default_selection;
    }

    let resolve_session_plugins: ResolveSessionPluginsFn =
        {
            let plugin_state = Arc::clone(&plugin_state);
            Arc::new(
			move |metadata: SessionCatalogMetadata,
			      requested_package_paths: Option<Vec<String>>,
			      _context: Context|
			      -> BoxFuture<'static, Result<ResolvedSessionPlugins, AnyError>> {
				let plugin_state = Arc::clone(&plugin_state);
				Box::pin(async move {
					if let Some(requested) = requested_package_paths {
						let normalized = normalize_plugin_package_paths(&requested)?;
						{
							let session_selections = plugin_state.session_selections.lock().await;
							if let Some(current) = session_selections.get(&metadata.path) {
								if same_strings(&current.package_paths, &normalized) {
									return Ok(current.clone());
								}
							}
						}
						let candidate = plugin_state.build_plugin_selection(&normalized).await?;
						write_session_plugin_package_profile(
							&plugin_state.directory,
							&plugin_state.server_id,
							&metadata.path,
							&candidate.package_paths,
						)
						.await
						.map_err(|write_error| write_error)?;
						plugin_state
							.session_selections
							.lock()
							.await
							.insert(metadata.path.clone(), candidate.clone());
						return Ok(candidate);
					}
					let cached = plugin_state.session_selections.lock().await.get(&metadata.path).cloned();
					if let Some(cached) = cached {
						return Ok(cached);
					}
					let stored = read_session_plugin_package_profile(
						&plugin_state.directory,
						&plugin_state.server_id,
						&metadata.path,
					)
					.await
					.map_err(|read_error| read_error)?;
					let stored_is_default = stored.is_none();
					let selected = match &stored {
						Some(stored_paths) => plugin_state.build_plugin_selection(stored_paths).await?,
						None => plugin_state.default_selection.lock().await.clone(),
					};
					if stored_is_default {
						write_session_plugin_package_profile(
							&plugin_state.directory,
							&plugin_state.server_id,
							&metadata.path,
							&selected.package_paths,
						)
						.await
						.map_err(|write_error| write_error)?;
					}
					plugin_state
						.session_selections
						.lock()
						.await
						.insert(metadata.path.clone(), selected.clone());
					Ok(selected)
				})
			},
		)
        };

    let remove_session_plugins = {
        let plugin_state = Arc::clone(&plugin_state);
        Arc::new(
            move |metadata: SessionCatalogMetadata| -> BoxFuture<'static, Result<(), AnyError>> {
                let plugin_state = Arc::clone(&plugin_state);
                Box::pin(async move {
                    plugin_state
                        .session_selections
                        .lock()
                        .await
                        .remove(&metadata.path);
                    remove_session_plugin_package_profile(
                        &plugin_state.directory,
                        &plugin_state.server_id,
                        &metadata.path,
                    )
                    .await;
                    Ok(())
                })
            },
        )
    };

    let reload_presentation_facet_bundles = {
        let plugin_state = Arc::clone(&plugin_state);
        Arc::new(
			move |package_paths: Vec<String>| -> BoxFuture<'static, Result<Vec<FacetBundleArtifact>, AnyError>> {
				let plugin_state = Arc::clone(&plugin_state);
				Box::pin(async move {
					let reloaded = plugin_state.build_plugin_selection(&package_paths).await?;
					{
						let mut default_selection = plugin_state.default_selection.lock().await;
						if same_strings(&default_selection.package_paths, &reloaded.package_paths) {
							*default_selection = reloaded.clone();
						}
					}
					{
						let mut session_selections = plugin_state.session_selections.lock().await;
						for selected in session_selections.values_mut() {
							if same_strings(&selected.package_paths, &reloaded.package_paths) {
								*selected = reloaded.clone();
							}
						}
					}
					Ok(reloaded.presentation_artifacts)
				})
			},
		)
    };

    let socket_path = get_unix_socket_path(server_id, directory)
        .map_err(|path_error| error(path_error.to_string()))?
        .to_string_lossy()
        .into_owned();
    let control_path = Path::new(directory)
        .join(format!("control-{server_id}.sock"))
        .to_string_lossy()
        .into_owned();
    let server_nonce = {
        let nonce = new_uuid().replace('-', "");
        nonce.get(..12).unwrap_or(&nonce).to_string()
    };
    let server_path = Path::new(directory)
        .join(format!("server-{server_id}-{server_nonce}.sock"))
        .to_string_lossy()
        .into_owned();

    let mut startup_lease: Option<CoordinatorStartupLease> = Some(
        ensure_coordinator(&socket_path, &control_path)
            .await
            .map_err(|lease_error| error(lease_error))?,
    );
    let coordinator = Arc::new(CoordinatorConnection::new(CoordinatorConnectionOptions {
        control_path: control_path.clone(),
        endpoint: server_path.clone(),
        server_connection_id: None,
    }));
    let session_dir = resolve_session_directory(options.session_dir.as_deref());
    let lifetime_for_workers = lifetime.clone();
    let workers = SessionWorkerManager::new(
        Arc::clone(&coordinator),
        session_dir,
        worker_model,
        Some(Arc::new(move |count: usize| {
            lifetime_for_workers.set_worker_count(count)
        })),
    );

    // Error cleanup mirrors upstream's `Promise.allSettled` teardown ladder.
    let result = async {
        let backend = start_server_backend(
            StartServerBackendOptions {
                path: server_path,
                server_id: server_id_string.clone(),
                session_dir: options.session_dir.clone(),
                resolve_session_plugins,
                remove_session_plugins,
                reload_presentation_facet_bundles,
            },
            Arc::clone(&workers),
            Some({
                let lifetime = lifetime.clone();
                Arc::new(move |count: usize| lifetime.set_connection_count(count))
            }),
        )
        .await?;
        coordinator
            .connect()
            .await
            .map_err(|connect_error| error(connect_error))?;
        if let Some(lease) = startup_lease.as_mut() {
            lease.close();
        }
        startup_lease = None;
        workers.discover(&coordinator.peer_ids()).await;
        backend
            .refresh_sessions(BACKGROUND_CONTEXT.clone())
            .await
            .map_err(|refresh_error| refresh_error)?;

        let relay = RadiusRelayHost::new(RadiusRelayHostOptions {
            server_id: server_id_string.clone(),
            server: Arc::clone(&backend.server),
            auth: Arc::new(RadiusRelayAuthResolver::new(options.relay_auth.clone())),
            web_socket_factory: None,
            on_status: options.on_relay_status.clone(),
        });
        relay.start();

        // Replacement: detach workers and retire this generation.
        {
            let replaced_coordinator = Arc::clone(&coordinator);
            let replaced_workers = Arc::clone(&workers);
            let replaced_relay = Arc::clone(&relay);
            let replaced_backend = Arc::new(RunningServerBackend {
                server_id: backend.server_id.clone(),
                session_dir: backend.session_dir.clone(),
                socket_path: backend.socket_path.clone(),
                server: Arc::clone(&backend.server),
                worker_pids: backend.worker_pids.clone(),
                closed: backend.closed.clone(),
                services: Arc::clone(&backend.services),
            });
            let replaced_lifetime = lifetime.clone();
            tokio::spawn(async move {
                replaced_coordinator.replaced().await;
                replaced_lifetime.stop();
                replaced_workers.detach();
                replaced_relay.close().await;
                let _ = replaced_backend.close_and_await_closed().await;
                replaced_coordinator.close();
            });
        }

        // `closed = backend.closed.finally(() => relay.close())`.
        let closed = ServerClosed::new();
        {
            let closed_slot = closed.slot.clone();
            let closed_backend = Arc::new(RunningServerBackend {
                server_id: backend.server_id.clone(),
                session_dir: backend.session_dir.clone(),
                socket_path: backend.socket_path.clone(),
                server: Arc::clone(&backend.server),
                worker_pids: backend.worker_pids.clone(),
                closed: backend.closed.clone(),
                services: Arc::clone(&backend.services),
            });
            let closed_relay = Arc::clone(&relay);
            tokio::spawn(async move {
                let result = closed_backend.closed.closed().await;
                closed_relay.close().await;
                closed_slot.resolve(result);
            });
        }

        let runtime = Arc::new(RunningServer {
            server_id: server_id_string.clone(),
            session_dir: backend.session_dir.clone(),
            socket_path,
            server: Arc::clone(&backend.server),
            worker_pids: workers.worker_pids().clone(),
            closed,
            lifetime: lifetime.clone(),
            relay: Arc::clone(&relay),
            backend: Arc::new(RunningServerBackend {
                server_id: backend.server_id.clone(),
                session_dir: backend.session_dir.clone(),
                socket_path: backend.socket_path.clone(),
                server: Arc::clone(&backend.server),
                worker_pids: backend.worker_pids.clone(),
                closed: backend.closed.clone(),
                services: Arc::clone(&backend.services),
            }),
            coordinator: Arc::clone(&coordinator),
            workers: Arc::clone(&workers),
            close_slot: PLMutex::new(None),
        });
        {
            let retire_runtime = Arc::clone(&runtime);
            lifetime.start(Arc::new(move || {
                let runtime = Arc::clone(&retire_runtime);
                tokio::spawn(async move {
                    let _ = runtime.close().await;
                });
            }));
        }
        Ok(runtime)
    }
    .await;

    match result {
        Ok(runtime) => Ok(runtime),
        Err(start_error) => {
            lifetime.stop();
            if let Some(lease) = startup_lease.as_mut() {
                lease.close();
            }
            let mut cleanup_errors: Vec<String> = Vec::new();
            // The relay/backend pair only exists after backend assembly, which
            // this scope does not retain on the error path; the routed server
            // and services close through their own spawned `closed` chain and
            // their Drop/lifecycle guards. Workers are shut down unless this
            // generation was already replaced.
            if !coordinator.was_replaced() {
                workers.shutdown().await;
            } else {
                workers.detach();
            }
            coordinator.close();
            let _ = &mut cleanup_errors;
            Err(start_error)
        }
    }
}

fn same_strings(left: &[String], right: &[String]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a == b)
}

/// `startForegroundServer(options?)` — start an operator-held server while
/// serializing against automatic cold activation.
pub async fn start_foreground_server(
    mut options: StartServerOptions,
) -> Result<Arc<RunningServer>, AnyError> {
    let directory = resolve_server_directory(options.directory.as_deref());
    ensure_private_server_directory(&directory)?;
    let profile = {
        let env_server_id = std::env::var(ENV_SERVER_ID).ok();
        acquire_server_profile(
            &directory,
            options.server_id.as_deref().or(env_server_id.as_deref()),
        )
        .await?
    };
    let server_id = profile.server_id;
    profile.release.release();
    let release = acquire_server_activation(&directory, &server_id).await?;
    let result = async {
        options.directory = Some(directory);
        options.server_id = Some(server_id);
        options.keep_alive = Some(true);
        start_server(options).await
    }
    .await;
    release.release();
    result
}

/// `parseServerModelOptions(value?)`.
pub fn parse_server_model_options(
    value: Option<&str>,
) -> Result<Option<WorkerModelOptions>, AnyError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let parsed: Value = serde_json::from_str(value).map_err(|parse_error| {
        error(format!(
            "Internal server received invalid model options: {parse_error}"
        ))
    })?;
    let Value::Object(object) = parsed else {
        return Err(error("Internal server received invalid model options"));
    };
    let model = object.get("model").and_then(Value::as_str);
    let provider = object.get("provider").and_then(Value::as_str);
    let valid_keys = object.keys().all(|key| key == "provider" || key == "model");
    let model_valid = model.is_some_and(|model| !model.is_empty());
    let provider_valid = provider.is_none_or(|provider| !provider.is_empty());
    if !valid_keys || !model_valid || !provider_valid {
        return Err(error("Internal server received invalid model options"));
    }
    Ok(Some(WorkerModelOptions {
        provider: provider.map(str::to_string),
        model: model.expect("validated").to_string(),
    }))
}

/// `runServerProcess(args)` — run an automatically activated server until its
/// client and Session demand disappears.
pub async fn run_server_process(args: &[String]) -> Result<(), AnyError> {
    if args.len() > 4 {
        return Err(error("Internal server received unexpected arguments"));
    }
    let (directory, server_id, session_dir, serialized_model) = match args {
        [directory, server_id, session_dir] => (directory, server_id, session_dir, None),
        [directory, server_id, session_dir, serialized_model] => (
            directory,
            server_id,
            session_dir,
            Some(serialized_model.as_str()),
        ),
        _ => {
            return Err(error(
                "Internal server requires a server directory, server ID, and Session directory",
            ))
        }
    };
    if directory.is_empty() || !Path::new(directory).is_absolute() {
        return Err(error(
            "Internal server requires an absolute server directory",
        ));
    }
    if !is_server_id(server_id) {
        return Err(error("Internal server requires a canonical server ID"));
    }
    if session_dir.is_empty() || !Path::new(session_dir).is_absolute() {
        return Err(error(
            "Internal server requires an absolute Session directory",
        ));
    }
    let worker_model = parse_server_model_options(serialized_model)?;

    let runtime = start_server(StartServerOptions {
        directory: Some(directory.clone()),
        server_id: Some(server_id.clone()),
        session_dir: Some(session_dir.clone()),
        keep_alive: Some(false),
        provider: worker_model
            .as_ref()
            .and_then(|model| model.provider.clone()),
        model: worker_model.map(|model| model.model),
        ..Default::default()
    })
    .await?;

    let close_runtime = {
        let runtime = Arc::clone(&runtime);
        Arc::new(move || {
            let runtime = Arc::clone(&runtime);
            tokio::spawn(async move {
                let _ = runtime.close().await;
            });
        })
    };
    let sigint = tokio::spawn({
        let close_runtime = Arc::clone(&close_runtime);
        async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                close_runtime();
            }
        }
    });
    let sigterm = tokio::spawn({
        let close_runtime = Arc::clone(&close_runtime);
        async move {
            signal_terminate().await;
            close_runtime();
        }
    });

    let result = runtime
        .closed
        .closed()
        .await
        .map_err(|closed_error| error(closed_error.to_string()));
    sigint.abort();
    sigterm.abort();
    let _ = runtime.close().await;
    result
}

async fn signal_terminate() {
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("SIGTERM handler installs");
    sigterm.recv().await;
}

/// The bin-dispatch seam for the internal `server` role (upstream the
/// module-level `isDirectInternalProcessEntry` block).
pub async fn run_server_entry(args: &[String]) -> Result<(), String> {
    let role = consume_internal_process_role().map_err(|message| message)?;
    if role != Some(InternalProcessRole::Server) {
        return Err("Server entrypoint requires an internal server invocation".to_string());
    }
    run_server_process(args)
        .await
        .map_err(|run_error| run_error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_model_options_round_trip() {
        assert!(parse_server_model_options(None).unwrap().is_none());
        let parsed = parse_server_model_options(Some(r#"{"model":"m"}"#))
            .unwrap()
            .unwrap();
        assert_eq!(parsed.provider, None);
        assert_eq!(parsed.model, "m");
        let parsed = parse_server_model_options(Some(r#"{"provider":"p","model":"m"}"#))
            .unwrap()
            .unwrap();
        assert_eq!(parsed.provider.as_deref(), Some("p"));
        assert_eq!(parsed.model, "m");
        assert!(parse_server_model_options(Some(r#"{"model":""}"#)).is_err());
        assert!(parse_server_model_options(Some(r#"{"model":"m","extra":1}"#)).is_err());
        assert!(parse_server_model_options(Some(r#"["model"]"#)).is_err());
        assert!(parse_server_model_options(Some("not json")).is_err());
    }

    #[test]
    fn resolve_directories_apply_defaults() {
        let _guard = crate::experimental::TEST_GUARD.lock();
        let directory = resolve_server_directory(None);
        assert!(!directory.is_empty());
        assert_eq!(
            resolve_server_directory(Some("/tmp/pi-a")),
            resolve_path_cwd("/tmp/pi-a")
        );
        let session = resolve_session_directory(None);
        assert!(session.ends_with("experimental/sessions"));
    }

    #[test]
    fn profile_rejects_invalid_server_id_and_persists_default() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let directory =
                std::env::temp_dir().join(format!("pi-server-profile-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&directory);
            std::fs::create_dir_all(&directory).unwrap();
            let result = acquire_server_profile(directory.to_str().unwrap(), Some("bad id!")).await;
            assert!(result.is_err());
            let server_id = new_uuid();
            let profile = acquire_server_profile(directory.to_str().unwrap(), Some(&server_id))
                .await
                .unwrap();
            assert_eq!(profile.server_id, server_id);
            let default = acquire_server_profile(directory.to_str().unwrap(), None)
                .await
                .unwrap();
            assert!(is_server_id(&default.server_id));
            let default_id = default.server_id.clone();
            default.release.release();
            let again = acquire_server_profile(directory.to_str().unwrap(), None)
                .await
                .unwrap();
            assert_eq!(default_id, again.server_id);
            let _ = std::fs::remove_dir_all(&directory);
        });
    }

    #[test]
    fn lifetime_retires_after_idle_grace() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let retired = Arc::new(AtomicBool::new(false));
            let lifetime = ServerLifetime::new(false);
            let flag = Arc::clone(&retired);
            lifetime.start(Arc::new(move || flag.store(true, Ordering::SeqCst)));
            // Still inside the startup grace.
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(!retired.load(Ordering::SeqCst));
            // Connections release the startup hold; the idle grace retires
            // the server after demand disappears.
            lifetime.set_connection_count(2);
            lifetime.set_connection_count(0);
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(!retired.load(Ordering::SeqCst));
            tokio::time::sleep(Duration::from_millis(AUTO_SERVER_IDLE_GRACE_MS + 200)).await;
            assert!(retired.load(Ordering::SeqCst));
        });
    }

    #[tokio::test]
    async fn run_server_process_validates_arguments() {
        let _guard = crate::experimental::TEST_GUARD.lock();
        let server_id = new_uuid();
        let error = run_server_process(&[
            "relative".to_string(),
            server_id.clone(),
            "/tmp".to_string(),
        ])
        .await
        .unwrap_err();
        assert!(error.to_string().contains("absolute server directory"));
        let error =
            run_server_process(&["/tmp".to_string(), "bad id".to_string(), "/tmp".to_string()])
                .await
                .unwrap_err();
        assert!(error.to_string().contains("canonical server ID"));
        let error = run_server_process(&["/tmp".to_string(), server_id, "relative".to_string()])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("absolute Session directory"));
        let error = run_server_process(&[
            "/tmp".to_string(),
            "srv".to_string(),
            "/tmp".to_string(),
            "{}".to_string(),
            "extra".to_string(),
        ])
        .await
        .unwrap_err();
        assert!(error.to_string().contains("unexpected arguments"));
    }
}
