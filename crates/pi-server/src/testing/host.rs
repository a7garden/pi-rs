//! Port of `packages/server/src/testing/host.ts`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use pi_chord::{Context, JsonValue, ServiceCall};
use std::sync::OnceLock;

use crate::errors::ServerError;
use crate::types::{
    AnyError, RoutedServerPresentation, RoutedServerServiceAttachment, RoutedServerServiceHost,
    RoutedSessionAttachment, RoutedSessionHandle, ServerHost, ServicePublish, SessionMetadata,
};

/// Upstream `Deferred<T>` (shared one-shot resolution).
pub struct Deferred<T> {
    value: parking_lot::Mutex<Option<T>>,
    notify: tokio::sync::Notify,
}

impl<T: Clone + Send + Sync> Deferred<T> {
    pub fn new() -> Deferred<T> {
        Deferred {
            value: parking_lot::Mutex::new(None),
            notify: tokio::sync::Notify::new(),
        }
    }

    pub fn resolve(&self, value: T) {
        let mut guard = self.value.lock();
        if guard.is_none() {
            *guard = Some(value);
            drop(guard);
            self.notify.notify_waiters();
        }
    }

    pub async fn promise(&self) -> T {
        loop {
            let notified = self.notify.notified();
            if let Some(value) = self.value.lock().clone() {
                return value;
            }
            notified.await;
        }
    }
}

impl<T: Clone + Send + Sync> Default for Deferred<T> {
    fn default() -> Self {
        Deferred::new()
    }
}

#[derive(Clone)]
pub struct OpenGate {
    pub entered: Arc<Deferred<()>>,
    pub release: Arc<Deferred<()>>,
}

fn open_gate() -> OpenGate {
    OpenGate {
        entered: Arc::new(Deferred::new()),
        release: Arc::new(Deferred::new()),
    }
}

/// Upstream `TestHarness`: records attachment traffic for one open Session and
/// implements both the routed handle and its attachment.
pub struct TestHarness {
    pub metadata: Arc<dyn SessionMetadata>,
    pub closed: Arc<Deferred<()>>,
    termination: Arc<Deferred<Option<AnyError>>>,
    pub attached_clients: AtomicUsize,
    pub attachment_release_count: AtomicUsize,
    pub close_count: AtomicUsize,
    released: AtomicBool,
    pub service_calls: parking_lot::Mutex<Vec<ServiceCall>>,
    pub fail_attachment_release: parking_lot::Mutex<Option<AnyError>>,
    pub fail_close: parking_lot::Mutex<Option<AnyError>>,
    pub next_service_error: parking_lot::Mutex<Option<AnyError>>,
    pub next_service_result: parking_lot::Mutex<Option<JsonValue>>,
    next_close_gate: parking_lot::Mutex<Option<OpenGate>>,
    next_service_gate: parking_lot::Mutex<Option<OpenGate>>,
    self_arc: OnceLock<std::sync::Weak<TestHarness>>,
}

impl TestHarness {
    fn new(metadata: Arc<dyn SessionMetadata>) -> Arc<TestHarness> {
        Arc::new_cyclic(|weak| TestHarness {
            metadata,
            closed: Arc::new(Deferred::new()),
            termination: Arc::new(Deferred::new()),
            attached_clients: AtomicUsize::new(0),
            attachment_release_count: AtomicUsize::new(0),
            close_count: AtomicUsize::new(0),
            released: AtomicBool::new(false),
            service_calls: parking_lot::Mutex::new(Vec::new()),
            fail_attachment_release: parking_lot::Mutex::new(None),
            fail_close: parking_lot::Mutex::new(None),
            next_service_error: parking_lot::Mutex::new(None),
            next_service_result: parking_lot::Mutex::new(Some(serde_json::json!({ "ok": true }))),
            next_close_gate: parking_lot::Mutex::new(None),
            next_service_gate: parking_lot::Mutex::new(None),
            self_arc: OnceLock::from(weak.clone()),
        })
    }

    fn handle(&self) -> Arc<TestHarness> {
        self.self_arc
            .get()
            .expect("harness arc")
            .upgrade()
            .expect("harness alive")
    }

    pub fn attached_clients_count(&self) -> usize {
        self.attached_clients.load(Ordering::SeqCst)
    }

    pub fn attachment_release_count_value(&self) -> usize {
        self.attachment_release_count.load(Ordering::SeqCst)
    }

    pub fn close_count_value(&self) -> usize {
        self.close_count.load(Ordering::SeqCst)
    }

    pub fn terminate(&self, error: Option<AnyError>) {
        self.termination.resolve(error);
    }

    pub fn gate_next_close(&self) -> OpenGate {
        let gate = open_gate();
        *self.next_close_gate.lock() = Some(gate.clone());
        gate
    }

    pub fn gate_next_service_call(&self) -> OpenGate {
        let gate = open_gate();
        *self.next_service_gate.lock() = Some(gate.clone());
        gate
    }

    /// Exposes the terminated future for tests (upstream `harness.terminated`).
    pub async fn terminated(&self) -> Option<AnyError> {
        self.termination.promise().await
    }

    pub async fn closed_promise(&self) {
        self.closed.promise().await;
    }
}

#[async_trait]
impl RoutedSessionAttachment for TestHarness {
    fn invoke_service(
        &self,
        call: ServiceCall,
        _publish: ServicePublish,
        _context: Context,
    ) -> crate::types::BoxFuture<Result<Option<JsonValue>, AnyError>> {
        let future_self = self.handle();
        Box::pin(async move {
            let this = future_self;
            this.service_calls.lock().push(call);
            if let Some(error) = this.next_service_error.lock().take() {
                return Err(error);
            }
            let gate = this.next_service_gate.lock().take();
            if let Some(gate) = gate {
                gate.entered.resolve(());
                gate.release.promise().await;
            }
            let result = this.next_service_result.lock().take();
            *this.next_service_result.lock() = Some(serde_json::json!({ "ok": true }));
            Ok(result)
        })
    }

    fn release(&self, _context: Context) -> crate::types::MaybeFuture<Result<(), AnyError>> {
        let this = self.handle();
        Box::pin(async move {
            if this.released.swap(true, Ordering::SeqCst) {
                return Ok(());
            }
            this.attachment_release_count.fetch_add(1, Ordering::SeqCst);
            if let Some(error) = this.fail_attachment_release.lock().clone() {
                return Err(error);
            }
            this.attached_clients.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

#[async_trait]
impl RoutedSessionHandle for TestHarness {
    fn attach_client(
        &self,
        _context: Context,
    ) -> crate::types::MaybeFuture<Result<Arc<dyn RoutedSessionAttachment>, AnyError>> {
        self.attached_clients.fetch_add(1, Ordering::SeqCst);
        let handle: Arc<dyn RoutedSessionAttachment> = self.handle();
        Box::pin(async move { Ok(handle) })
    }

    fn terminated(&self) -> Option<crate::types::BoxFuture<Option<AnyError>>> {
        let termination = Arc::clone(&self.termination);
        Some(Box::pin(async move { termination.promise().await }))
    }

    async fn close(&self, _context: Context) -> Result<(), AnyError> {
        self.close_count.fetch_add(1, Ordering::SeqCst);
        let gate = self.next_close_gate.lock().take();
        if let Some(gate) = gate {
            gate.entered.resolve(());
            gate.release.promise().await;
        }
        if let Some(error) = self.fail_close.lock().take() {
            return Err(error);
        }
        self.closed.resolve(());
        self.termination.resolve(None);
        Ok(())
    }
}

/// Upstream `createTestServerServices` — `pi.session-management`
/// attach/detach over the presentation surface.
pub fn create_test_server_services() -> Arc<dyn RoutedServerServiceHost> {
    Arc::new(TestServerServices)
}

struct TestServerServices;

struct TestServerAttachment {
    presentation: Arc<dyn RoutedServerPresentation>,
}

#[async_trait]
impl RoutedServerServiceHost for TestServerServices {
    async fn attach_client(
        &self,
        presentation: Arc<dyn RoutedServerPresentation>,
        _context: Context,
    ) -> Result<Arc<dyn RoutedServerServiceAttachment>, AnyError> {
        Ok(Arc::new(TestServerAttachment { presentation }))
    }
}

fn is_session_management_attach(call: &ServiceCall) -> bool {
    call.instance.is_none()
        && call.service_id == "pi.session-management"
        && call.member == "attach"
        && call.args.len() == 1
        && call.args[0].is_string()
}

fn is_session_management_detach(call: &ServiceCall) -> bool {
    call.instance.is_none()
        && call.service_id == "pi.session-management"
        && call.member == "detach"
        && call.args.is_empty()
}

impl RoutedServerServiceAttachment for TestServerAttachment {
    fn invoke_service(
        &self,
        call: ServiceCall,
        _publish: ServicePublish,
        context: Context,
    ) -> crate::types::BoxFuture<Result<Option<JsonValue>, AnyError>> {
        let presentation = Arc::clone(&self.presentation);
        Box::pin(async move {
            if is_session_management_attach(&call) {
                let session_id = call.args[0].as_str().expect("checked").to_string();
                presentation.attach_session(&session_id, context).await?;
                return Ok(Some(JsonValue::Null));
            }
            if is_session_management_detach(&call) {
                presentation.detach_session(context).await?;
                return Ok(Some(JsonValue::Null));
            }
            Err(crate::types::any(pi_chord::ChordError {
                message: format!(
                    "Unsupported test server service {}.{}",
                    call.service_id, call.member
                ),
            }))
        })
    }

    fn release(&self, _context: Context) -> crate::types::MaybeFuture<Result<(), AnyError>> {
        Box::pin(async { Ok(()) })
    }
}

/// Upstream `TestServerHost`.
pub struct TestServerHost {
    server_services: Arc<dyn RoutedServerServiceHost>,
    pub sessions: parking_lot::Mutex<HashMap<String, Arc<dyn SessionMetadata>>>,
    pub harnesses: parking_lot::Mutex<HashMap<String, Vec<Arc<TestHarness>>>>,
    open_session_count: AtomicUsize,
    pub next_open_session_error: parking_lot::Mutex<Option<AnyError>>,
    pub next_harness_close_error: parking_lot::Mutex<Option<AnyError>>,
    next_open_session_gate: parking_lot::Mutex<Option<OpenGate>>,
}

impl TestServerHost {
    pub fn new() -> Arc<TestServerHost> {
        Arc::new(TestServerHost {
            server_services: create_test_server_services(),
            sessions: parking_lot::Mutex::new(HashMap::new()),
            harnesses: parking_lot::Mutex::new(HashMap::new()),
            open_session_count: AtomicUsize::new(0),
            next_open_session_error: parking_lot::Mutex::new(None),
            next_harness_close_error: parking_lot::Mutex::new(None),
            next_open_session_gate: parking_lot::Mutex::new(None),
        })
    }

    pub fn open_session_count_value(&self) -> usize {
        self.open_session_count.load(Ordering::SeqCst)
    }

    pub async fn seed(&self) -> Arc<dyn SessionMetadata> {
        self.seed_with_id("session-1").await
    }

    pub async fn seed_with_id(&self, id: &str) -> Arc<dyn SessionMetadata> {
        let metadata: Arc<dyn SessionMetadata> =
            Arc::new(crate::types::BasicSessionMetadata { id: id.to_string() });
        self.sessions
            .lock()
            .insert(id.to_string(), Arc::clone(&metadata));
        metadata
    }

    pub fn gate_next_open_session(&self) -> OpenGate {
        let gate = open_gate();
        *self.next_open_session_gate.lock() = Some(gate.clone());
        gate
    }

    pub fn latest_harness(&self, id: &str) -> Result<Arc<TestHarness>, AnyError> {
        self.harnesses
            .lock()
            .get(id)
            .and_then(|harnesses| harnesses.last().cloned())
            .ok_or_else(|| -> AnyError {
                Arc::new(pi_chord::ChordError {
                    message: format!("No harness for {id}"),
                })
            })
    }

    pub fn harness_count(&self, id: &str) -> usize {
        self.harnesses.lock().get(id).map(Vec::len).unwrap_or(0)
    }

    pub fn harness_total(&self) -> usize {
        self.harnesses.lock().values().map(Vec::len).sum()
    }
}

#[async_trait]
impl ServerHost for TestServerHost {
    fn server_services(&self) -> Arc<dyn RoutedServerServiceHost> {
        Arc::clone(&self.server_services)
    }

    async fn resolve_session(
        &self,
        session_id: &str,
        _context: Context,
    ) -> Result<Arc<dyn SessionMetadata>, AnyError> {
        self.sessions
            .lock()
            .get(session_id)
            .cloned()
            .ok_or_else(|| -> AnyError {
                Arc::new(ServerError::session_not_found_with(format!(
                    "Unknown session: {session_id}"
                )))
            })
    }

    async fn open_session(
        &self,
        metadata: Arc<dyn SessionMetadata>,
        _context: Context,
    ) -> Result<Arc<dyn RoutedSessionHandle>, AnyError> {
        self.open_session_count.fetch_add(1, Ordering::SeqCst);
        let gate = self.next_open_session_gate.lock().take();
        if let Some(gate) = gate {
            gate.entered.resolve(());
            gate.release.promise().await;
        }
        if !self.sessions.lock().contains_key(metadata.id()) {
            return Err(Arc::new(ServerError::session_not_found_with(format!(
                "Unknown session: {}",
                metadata.id()
            ))));
        }
        if let Some(error) = self.next_open_session_error.lock().take() {
            return Err(error);
        }
        let harness = TestHarness::new(Arc::clone(&metadata));
        if let Some(error) = self.next_harness_close_error.lock().take() {
            *harness.fail_close.lock() = Some(error);
        }
        self.harnesses
            .lock()
            .entry(metadata.id().to_string())
            .or_default()
            .push(Arc::clone(&harness));
        Ok(harness)
    }
}
