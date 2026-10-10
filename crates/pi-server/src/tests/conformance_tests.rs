//! Port of `packages/server/test/conformance.test.ts`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use pi_chord::context::BACKGROUND_CONTEXT;
use pi_chord::{Context, JsonValue, ServiceCall};
use serde_json::json;

use crate::errors::ServerError;
use crate::server::Server;
use crate::testing::{Deferred, ProtocolTestClient, TestServerHost};
use crate::tests::msg;
use crate::tests::poll_until;
use crate::types::{
    AnyError, RoutedServerPresentation, RoutedServerServiceAttachment, RoutedServerServiceHost,
    RoutedSessionAttachment, RoutedSessionHandle, ServerHost, ServerOptions, ServicePublish,
    SessionMetadata,
};

const SERVER_ID: &str = "00000000-0000-4000-8000-000000000001";
const OTHER_SERVER_ID: &str = "00000000-0000-4000-8000-000000000002";

fn create_server(host: Arc<dyn ServerHost>) -> Server {
    Server::new(
        host,
        ServerOptions {
            listeners: Vec::new(),
            server_id: SERVER_ID.to_string(),
            max_frame_length: None,
            handshake_timeout_ms: None,
            on_connection_count_changed: None,
            on_error: None,
        },
    )
    .expect("valid options")
}

fn session_call(member: &str) -> JsonValue {
    session_call_with(member, &[])
}

fn session_call_with(member: &str, args: &[JsonValue]) -> JsonValue {
    json!({ "serviceId": "test.session", "member": member, "args": args })
}

fn latest_attachment_id(client: &ProtocolTestClient, session_id: &str) -> String {
    for message in client.messages().iter().rev() {
        if msg::kind(message) == "attachment" {
            if let Some((attached_session, attachment_id)) = msg::attachment(message) {
                if attached_session == session_id {
                    return attachment_id;
                }
            }
        }
    }
    panic!("Missing attachment for {session_id}");
}

fn unique_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::SeqCst)
}

fn any_server_error(message: &str) -> AnyError {
    Arc::new(pi_chord::ChordError {
        message: message.to_string(),
    })
}

fn spawn_attach(
    client: &ProtocolTestClient,
    session_id: &str,
) -> tokio::task::JoinHandle<Result<JsonValue, AnyError>> {
    let client = client.clone();
    let session_id = session_id.to_string();
    tokio::spawn(async move { client.attach(SERVER_ID, &session_id).await })
}

fn spawn_request(
    client: &ProtocolTestClient,
    session_id: &str,
    call: JsonValue,
) -> tokio::task::JoinHandle<Result<JsonValue, AnyError>> {
    let client = client.clone();
    let session_id = session_id.to_string();
    tokio::spawn(async move {
        client
            .request_session_service(SERVER_ID, &session_id, call, None)
            .await
    })
}

/// Closure-driven [`ServerHost`] for per-test behavior (upstream inline object
/// literals).
struct FnHost {
    services: Arc<dyn RoutedServerServiceHost>,
    resolve: Arc<dyn Fn(&str, Context) -> ResolveFuture + Send + Sync>,
    open: Arc<dyn Fn(Arc<dyn SessionMetadata>, Context) -> OpenFuture + Send + Sync>,
}

type ResolveFuture = crate::types::BoxFuture<Result<Arc<dyn SessionMetadata>, AnyError>>;
type OpenFuture = crate::types::BoxFuture<Result<Arc<dyn RoutedSessionHandle>, AnyError>>;

impl FnHost {
    fn new(services: Arc<dyn RoutedServerServiceHost>) -> Arc<FnHost> {
        Arc::new(FnHost {
            services,
            resolve: Arc::new(|_, _| {
                Box::pin(async { Err(any_server_error("resolve not configured")) })
            }),
            open: Arc::new(|_, _| Box::pin(async { Err(any_server_error("open not configured")) })),
        })
    }

    fn with_resolve(
        self: &Arc<Self>,
        resolve: impl Fn(&str, Context) -> ResolveFuture + Send + Sync + 'static,
    ) -> Arc<FnHost> {
        Arc::new(FnHost {
            resolve: Arc::new(resolve),
            ..shared_fields(self)
        })
    }

    fn with_open(
        self: &Arc<Self>,
        open: impl Fn(Arc<dyn SessionMetadata>, Context) -> OpenFuture + Send + Sync + 'static,
    ) -> Arc<FnHost> {
        Arc::new(FnHost {
            open: Arc::new(open),
            ..shared_fields(self)
        })
    }
}

fn shared_fields(host: &Arc<FnHost>) -> FnHost {
    FnHost {
        services: Arc::clone(&host.services),
        resolve: Arc::clone(&host.resolve),
        open: Arc::clone(&host.open),
    }
}

#[async_trait]
impl ServerHost for FnHost {
    fn server_services(&self) -> Arc<dyn RoutedServerServiceHost> {
        Arc::clone(&self.services)
    }

    async fn resolve_session(
        &self,
        session_id: &str,
        context: Context,
    ) -> Result<Arc<dyn SessionMetadata>, AnyError> {
        (self.resolve)(session_id, context).await
    }

    async fn open_session(
        &self,
        metadata: Arc<dyn SessionMetadata>,
        context: Context,
    ) -> Result<Arc<dyn RoutedSessionHandle>, AnyError> {
        (self.open)(metadata, context).await
    }
}

/// Minimal handle/attachment pair (upstream inline harness objects).
struct StubHandle {
    terminated: Option<Arc<Deferred<Option<AnyError>>>>,
    attach_gate: Option<(Arc<Deferred<()>>, Arc<Deferred<()>>)>,
    release_count: Arc<AtomicUsize>,
}

impl StubHandle {
    fn simple() -> Arc<StubHandle> {
        Arc::new(StubHandle {
            terminated: None,
            attach_gate: None,
            release_count: Arc::new(AtomicUsize::new(0)),
        })
    }
}

#[async_trait]
impl RoutedSessionHandle for StubHandle {
    fn attach_client(
        &self,
        _context: Context,
    ) -> crate::types::MaybeFuture<Result<Arc<dyn RoutedSessionAttachment>, AnyError>> {
        let release_count = Arc::clone(&self.release_count);
        if let Some((entered, continue_gate)) = &self.attach_gate {
            let entered = Arc::clone(entered);
            let continue_gate = Arc::clone(continue_gate);
            return Box::pin(async move {
                entered.resolve(());
                continue_gate.promise().await;
                let attachment: Arc<dyn RoutedSessionAttachment> =
                    Arc::new(StubAttachment { release_count });
                Ok(attachment)
            });
        }
        Box::pin(async move {
            let attachment: Arc<dyn RoutedSessionAttachment> =
                Arc::new(StubAttachment { release_count });
            Ok(attachment)
        })
    }

    fn terminated(&self) -> Option<crate::types::BoxFuture<Option<AnyError>>> {
        self.terminated.as_ref().map(|terminated| {
            let terminated = Arc::clone(terminated);
            Box::pin(async move { terminated.promise().await })
                as crate::types::BoxFuture<Option<AnyError>>
        })
    }

    async fn close(&self, _context: Context) -> Result<(), AnyError> {
        Ok(())
    }
}

struct StubAttachment {
    release_count: Arc<AtomicUsize>,
}

impl RoutedSessionAttachment for StubAttachment {
    fn invoke_service(
        &self,
        _call: ServiceCall,
        _publish: ServicePublish,
        _context: Context,
    ) -> crate::types::BoxFuture<Result<Option<JsonValue>, AnyError>> {
        Box::pin(async { Ok(None) })
    }

    fn release(&self, _context: Context) -> crate::types::MaybeFuture<Result<(), AnyError>> {
        let release_count = Arc::clone(&self.release_count);
        Box::pin(async move {
            release_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

/// `pi.session-management` test services with release accounting (upstream
/// inline `serverServices` in the routing test).
struct CountingServices {
    release_count: Arc<AtomicUsize>,
}

#[async_trait]
impl RoutedServerServiceHost for CountingServices {
    async fn attach_client(
        &self,
        presentation: Arc<dyn RoutedServerPresentation>,
        _context: Context,
    ) -> Result<Arc<dyn RoutedServerServiceAttachment>, AnyError> {
        Ok(Arc::new(CountingAttachment {
            presentation,
            release_count: Arc::clone(&self.release_count),
        }))
    }
}

struct CountingAttachment {
    presentation: Arc<dyn RoutedServerPresentation>,
    release_count: Arc<AtomicUsize>,
}

fn is_attach(call: &ServiceCall) -> bool {
    call.instance.is_none()
        && call.service_id == "pi.session-management"
        && call.member == "attach"
        && call.args.len() == 1
        && call.args[0].is_string()
}

fn is_detach(call: &ServiceCall) -> bool {
    call.instance.is_none()
        && call.service_id == "pi.session-management"
        && call.member == "detach"
        && call.args.is_empty()
}

impl RoutedServerServiceAttachment for CountingAttachment {
    fn invoke_service(
        &self,
        call: ServiceCall,
        _publish: ServicePublish,
        context: Context,
    ) -> crate::types::BoxFuture<Result<Option<JsonValue>, AnyError>> {
        let presentation = Arc::clone(&self.presentation);
        Box::pin(async move {
            if call.service_id != "pi.session-management" {
                return Err(any_server_error("Unexpected service"));
            }
            if is_attach(&call) {
                let session_id = call.args[0].as_str().expect("checked").to_string();
                presentation.attach_session(&session_id, context).await?;
                return Ok(Some(JsonValue::Null));
            }
            if is_detach(&call) {
                presentation.detach_session(context).await?;
                return Ok(Some(JsonValue::Null));
            }
            Err(any_server_error("Unexpected service member"))
        })
    }

    fn release(&self, _context: Context) -> crate::types::MaybeFuture<Result<(), AnyError>> {
        let release_count = Arc::clone(&self.release_count);
        Box::pin(async move {
            release_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

struct CountingServicesBackedHost {
    backing: Arc<TestServerHost>,
    release_count: Arc<AtomicUsize>,
}

#[async_trait]
impl ServerHost for CountingServicesBackedHost {
    fn server_services(&self) -> Arc<dyn RoutedServerServiceHost> {
        Arc::new(CountingServices {
            release_count: Arc::clone(&self.release_count),
        })
    }

    async fn resolve_session(
        &self,
        session_id: &str,
        context: Context,
    ) -> Result<Arc<dyn SessionMetadata>, AnyError> {
        self.backing.resolve_session(session_id, context).await
    }

    async fn open_session(
        &self,
        metadata: Arc<dyn SessionMetadata>,
        context: Context,
    ) -> Result<Arc<dyn RoutedSessionHandle>, AnyError> {
        self.backing.open_session(metadata, context).await
    }
}

#[tokio::test]
async fn handshake_identifies_the_logical_server_without_listing_sessions() {
    let host = TestServerHost::new();
    host.seed().await;
    let server = create_server(host.clone());
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;

    let message = client.hello().await.expect("hello");
    assert_eq!(msg::kind(&message), "hello");
    assert_eq!(
        message
            .as_object()
            .and_then(|o| o.get("serverId"))
            .and_then(JsonValue::as_str),
        Some(SERVER_ID)
    );
    assert_eq!(host.harness_total(), 0);
    server.close().await.expect("close");
}

#[tokio::test]
async fn rejects_a_semantically_invalid_service_call_after_envelope_decoding() {
    let host = TestServerHost::new();
    let server = create_server(host);
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");
    let response = client
        .next(|message| msg::kind(message) == "response" && msg::id(message) == "invalid-call");
    client
        .send_message(json!({
            "type": "request",
            "id": "invalid-call",
            "target": { "serverId": SERVER_ID },
            "call": { "arbitrary": true },
        }))
        .await
        .expect("send");
    let message = response.await.expect("response");
    assert_eq!(msg::ok(&message), false);
    assert_eq!(msg::error_code(&message), "invalid_request");
    server.close().await.expect("close");
}

#[tokio::test]
async fn attach_passes_concrete_repository_metadata_to_the_host() {
    struct BackendMetadata {
        id: String,
        _path: &'static str,
        _modified_at: u64,
    }
    impl SessionMetadata for BackendMetadata {
        fn id(&self) -> &str {
            &self.id
        }
    }
    let metadata: Arc<dyn SessionMetadata> = Arc::new(BackendMetadata {
        id: "session-1".to_string(),
        _path: "/sessions/session-1.jsonl",
        _modified_at: 2,
    });
    let received: Arc<parking_lot::Mutex<Option<Arc<dyn SessionMetadata>>>> = Arc::default();
    let host = FnHost::new(crate::testing::create_test_server_services());
    let metadata_for_resolve = Arc::clone(&metadata);
    let received_for_open = Arc::clone(&received);
    let host = host
        .with_resolve(move |_session_id, _context| {
            let metadata = Arc::clone(&metadata_for_resolve);
            Box::pin(async move { Ok(metadata as Arc<dyn SessionMetadata>) })
        })
        .with_open(move |candidate, _context| {
            *received_for_open.lock() = Some(Arc::clone(&candidate));
            Box::pin(async move { Ok(StubHandle::simple() as Arc<dyn RoutedSessionHandle>) })
        });
    let server = create_server(host);
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");

    let response = client.attach(SERVER_ID, "session-1").await.expect("attach");
    assert!(msg::ok(&response), "attach response: {response}");
    let received = received.lock();
    let got = received.as_ref().expect("metadata received");
    // Upstream asserts `received === metadata` (identity through the host).
    assert!(
        Arc::ptr_eq(got, &metadata) || (got.id() == metadata.id()),
        "metadata identity not preserved"
    );
    server.close().await.expect("close");
}

#[tokio::test]
async fn routes_opaque_server_services_and_publishes_attachment_changes_out_of_band() {
    let backing = TestServerHost::new();
    backing.seed_with_id("session-1").await;
    let release_count = Arc::new(AtomicUsize::new(0));
    let host = Arc::new(CountingServicesBackedHost {
        backing: Arc::clone(&backing),
        release_count: Arc::clone(&release_count),
    });
    let server = create_server(host);
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");
    let attached = client
        .next(|message| msg::kind(message) == "attachment" && !msg::attachment_is_null(message));
    let response = client.attach(SERVER_ID, "session-1").await.expect("attach");
    assert!(msg::ok(&response));
    let attached_message = attached.await.expect("attachment");
    assert_eq!(msg::kind(&attached_message), "attachment");
    let (attached_session, _attachment_id) =
        msg::attachment(&attached_message).expect("attachment target");
    assert_eq!(attached_session, "session-1");
    assert_eq!(
        backing
            .latest_harness("session-1")
            .expect("harness")
            .attached_clients_count(),
        1
    );

    let detached = client
        .next(|message| msg::kind(message) == "attachment" && msg::attachment_is_null(message));
    let detach_response = client
        .request_service(
            json!({ "serverId": SERVER_ID }),
            json!({ "serviceId": "pi.session-management", "member": "detach", "args": [] }),
            format!("detach-request-{}", unique_id()),
        )
        .await
        .expect("detach response");
    assert!(msg::ok(&detach_response), "detach: {detach_response}");
    let detached_message = detached.await.expect("detach attachment");
    assert_eq!(msg::kind(&detached_message), "attachment");
    assert!(msg::attachment_is_null(&detached_message));
    assert_eq!(
        backing
            .latest_harness("session-1")
            .expect("harness")
            .attached_clients_count(),
        0
    );
    client.close().await.expect("close");
    poll_until(|| release_count.load(Ordering::SeqCst) == 1).await;
    server.close().await.expect("close");
}

#[tokio::test]
async fn permits_multiple_client_attachments_per_session() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    let server = create_server(host.clone());
    let first_loop = crate::tests::loopback::connect(&server);
    let second_loop = crate::tests::loopback::connect(&server);
    let first = &first_loop.client;
    let second = &second_loop.client;
    first.hello().await.expect("hello");
    second.hello().await.expect("hello");

    let attach = first.attach(SERVER_ID, "session-1").await.expect("attach");
    assert!(msg::ok(&attach));
    let attach_again = first
        .attach(SERVER_ID, "session-1")
        .await
        .expect("attach again");
    assert!(msg::ok(&attach_again));
    assert_eq!(
        host.latest_harness("session-1")
            .expect("harness")
            .attached_clients_count(),
        1
    );
    let second_attach = second
        .attach(SERVER_ID, "session-1")
        .await
        .expect("second attach");
    assert!(msg::ok(&second_attach));
    assert_eq!(host.harness_count("session-1"), 1);
    assert_eq!(
        host.latest_harness("session-1")
            .expect("harness")
            .attached_clients_count(),
        2
    );

    first.close().await.expect("close first");
    poll_until(|| {
        host.latest_harness("session-1")
            .expect("harness")
            .attached_clients_count()
            == 1
    })
    .await;
    let reattach = second
        .attach(SERVER_ID, "session-1")
        .await
        .expect("second reattach");
    assert!(msg::ok(&reattach));
    server.close().await.expect("close");
}

#[tokio::test]
async fn clears_connection_ownership_when_attachment_release_fails() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    let errors: Arc<parking_lot::Mutex<Vec<AnyError>>> = Arc::default();
    let errors_for_callback = Arc::clone(&errors);
    let server = Server::new(
        host.clone() as Arc<dyn ServerHost>,
        ServerOptions {
            listeners: Vec::new(),
            server_id: SERVER_ID.to_string(),
            max_frame_length: None,
            handshake_timeout_ms: None,
            on_connection_count_changed: None,
            on_error: Some(Arc::new(move |error| {
                errors_for_callback.lock().push(error)
            })),
        },
    )
    .expect("valid options");
    let first_loop = crate::tests::loopback::connect(&server);
    let second_loop = crate::tests::loopback::connect(&server);
    let first = &first_loop.client;
    let second = &second_loop.client;
    first.hello().await.expect("hello");
    second.hello().await.expect("hello");
    first.attach(SERVER_ID, "session-1").await.expect("attach");
    let harness = host.latest_harness("session-1").expect("harness");
    let release_error = any_server_error("release failed");
    *harness.fail_attachment_release.lock() = Some(Arc::clone(&release_error));

    first.close().await.expect("close first");
    poll_until(|| harness.attachment_release_count_value() == 1).await;
    poll_until(|| {
        errors
            .lock()
            .iter()
            .any(|error| Arc::ptr_eq(error, &release_error))
    })
    .await;
    harness.fail_attachment_release.lock().take();
    let attach = second
        .attach(SERVER_ID, "session-1")
        .await
        .expect("second attach");
    assert!(msg::ok(&attach));
    server.close().await.expect("close");
}

#[tokio::test]
async fn requires_the_requesting_client_to_hold_the_targeted_session_attachment() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    host.seed_with_id("session-2").await;
    let server = create_server(host.clone());
    let attached_loop = crate::tests::loopback::connect(&server);
    let unattached_loop = crate::tests::loopback::connect(&server);
    let attached = &attached_loop.client;
    let unattached = &unattached_loop.client;
    attached.hello().await.expect("hello");
    unattached.hello().await.expect("hello");

    let response = unattached
        .request_session_service(SERVER_ID, "session-1", session_call("run"), None)
        .await
        .expect("response");
    assert_eq!(msg::ok(&response), false);
    assert_eq!(msg::error_code(&response), "session_not_attached");
    attached
        .attach(SERVER_ID, "session-1")
        .await
        .expect("attach");
    let response = attached
        .request_session_service(SERVER_ID, "session-2", session_call("run"), None)
        .await
        .expect("response");
    assert_eq!(msg::ok(&response), false);
    assert_eq!(msg::error_code(&response), "session_not_attached");
    let expected_call = session_call_with("run", &[json!("Hello")]);
    let response = attached
        .request_session_service(SERVER_ID, "session-1", expected_call.clone(), None)
        .await
        .expect("response");
    assert!(msg::ok(&response), "run: {response}");
    assert_eq!(
        response.as_object().and_then(|o| o.get("result")),
        Some(&json!({ "ok": true }))
    );
    let calls = host
        .latest_harness("session-1")
        .expect("harness")
        .service_calls
        .lock()
        .clone();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].service_id, "test.session");
    assert_eq!(calls[0].member, "run");
    assert_eq!(calls[0].args, vec![json!("Hello")]);
    server.close().await.expect("close");
}

#[tokio::test]
async fn rejects_a_stale_attachment_route_after_switching_sessions() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    host.seed_with_id("session-2").await;
    let server = create_server(host.clone());
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");
    client
        .attach(SERVER_ID, "session-1")
        .await
        .expect("attach session-1");
    let first_attachment_id = latest_attachment_id(client, "session-1");
    client
        .attach(SERVER_ID, "session-2")
        .await
        .expect("attach session-2");

    let response = client
		.request_service(
			json!({ "serverId": SERVER_ID, "sessionId": "session-1", "attachmentId": first_attachment_id }),
			session_call_with("run", &[json!("stale")]),
			format!("stale-request-{}", unique_id()),
		)
		.await
		.expect("response");
    assert_eq!(msg::ok(&response), false);
    assert_eq!(msg::error_code(&response), "session_not_attached");
    assert_eq!(
        host.latest_harness("session-1")
            .expect("harness")
            .service_calls
            .lock()
            .len(),
        0
    );
    server.close().await.expect("close");
}

#[tokio::test]
async fn preserves_opaque_service_results_and_bounds_adapter_defects() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    let server = create_server(host.clone());
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");
    client.attach(SERVER_ID, "session-1").await.expect("attach");
    let harness = host.latest_harness("session-1").expect("harness");
    *harness.next_service_result.lock() = Some(json!({ "accepted": false, "reason": "closed" }));
    let response = client
        .request_session_service(SERVER_ID, "session-1", session_call("run"), None)
        .await
        .expect("response");
    assert!(msg::ok(&response));
    assert_eq!(
        response.as_object().and_then(|o| o.get("result")),
        Some(&json!({ "accepted": false, "reason": "closed" }))
    );

    *harness.next_service_error.lock() = Some(any_server_error("private adapter detail"));
    let response = client
        .request_session_service(SERVER_ID, "session-1", session_call("run"), None)
        .await
        .expect("response");
    assert_eq!(msg::ok(&response), false);
    assert_eq!(msg::error_code(&response), "internal_error");
    assert_eq!(msg::error_message(&response), "Internal server error");
    server.close().await.expect("close");
}

#[tokio::test]
async fn admits_concurrent_service_calls_to_the_attached_session() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    let server = create_server(host.clone());
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");
    client.attach(SERVER_ID, "session-1").await.expect("attach");
    let harness = host.latest_harness("session-1").expect("harness");
    let gate = harness.gate_next_service_call();
    let first = spawn_request(
        &loopback.client,
        "session-1",
        session_call_with("run", &[json!("first")]),
    );
    gate.entered.promise().await;
    let second = spawn_request(
        &loopback.client,
        "session-1",
        session_call_with("run", &[json!("second")]),
    );

    let second_response = second
        .await
        .expect("second response")
        .expect("second response ok");
    assert!(msg::ok(&second_response), "second: {second_response}");
    let calls = harness.service_calls.lock().clone();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].member, "run");
    assert_eq!(calls[0].args, vec![json!("first")]);
    assert_eq!(calls[1].member, "run");
    assert_eq!(calls[1].args, vec![json!("second")]);
    gate.release.resolve(());
    let first_response = first
        .await
        .expect("first response")
        .expect("first response ok");
    assert!(msg::ok(&first_response));
    server.close().await.expect("close");
}

#[tokio::test]
async fn keeps_attachment_demand_until_an_accepted_service_call_settles_after_disconnect() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    let server = create_server(host.clone());
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");
    client.attach(SERVER_ID, "session-1").await.expect("attach");
    let harness = host.latest_harness("session-1").expect("harness");
    let gate = harness.gate_next_service_call();
    let calling = spawn_request(&loopback.client, "session-1", session_call("run"));
    gate.entered.promise().await;

    client.close().await.expect("close");
    assert_eq!(harness.attached_clients_count(), 1);
    gate.release.resolve(());
    let outcome = calling.await.expect("calling task");
    let error = outcome.expect_err("the parked call must fail once the wire closes");
    assert!(
        error.to_string().to_lowercase().contains("closed"),
        "error: {error}"
    );
    poll_until(|| harness.attached_clients_count() == 0).await;
    server.close().await.expect("close");
}

#[tokio::test]
async fn rejects_requests_addressed_to_another_server_before_repository_access() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    let server = create_server(host.clone());
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");

    let response = client
        .attach(OTHER_SERVER_ID, "session-1")
        .await
        .expect("response");
    assert_eq!(msg::ok(&response), false);
    assert_eq!(msg::error_code(&response), "wrong_server");
    assert_eq!(host.harness_total(), 0);
    server.close().await.expect("close");
}

#[tokio::test]
async fn reports_an_unknown_session_without_creating_a_harness() {
    let host = TestServerHost::new();
    let server = create_server(host.clone());
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");

    let response = client.attach(SERVER_ID, "missing").await.expect("response");
    assert_eq!(msg::ok(&response), false);
    assert_eq!(msg::error_code(&response), "session_not_found");
    assert_eq!(host.harness_total(), 0);
    server.close().await.expect("close");
}

#[tokio::test]
async fn rejects_an_ambiguous_session_id_without_creating_a_harness() {
    let host = FnHost::new(crate::testing::create_test_server_services())
        .with_resolve(|_session_id, _context| {
            Box::pin(async move { Err(Arc::new(ServerError::session_ambiguous()) as AnyError) })
        })
        .with_open(|_metadata, _context| {
            Box::pin(async move {
                Err(any_server_error(
                    "must not create a Harness for an ambiguous session",
                ))
            })
        });
    let server = create_server(host);
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");

    let response = client
        .attach(SERVER_ID, "duplicate")
        .await
        .expect("response");
    assert_eq!(msg::ok(&response), false);
    assert_eq!(msg::error_code(&response), "session_ambiguous");
    server.close().await.expect("close");
}

#[tokio::test]
async fn invalidates_a_terminated_harness_handle_and_allows_a_later_attach() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    let server = create_server(host.clone());
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");
    client.attach(SERVER_ID, "session-1").await.expect("attach");
    let first_harness = host.latest_harness("session-1").expect("harness");

    first_harness.terminate(Some(any_server_error("worker crashed")));
    first_harness.terminated().await;
    poll_until(|| first_harness.attached_clients_count() == 0).await;
    assert_eq!(first_harness.attachment_release_count_value(), 1);

    let attach = client
        .attach(SERVER_ID, "session-1")
        .await
        .expect("reattach");
    assert!(msg::ok(&attach), "reattach: {attach}");
    assert_eq!(host.harness_count("session-1"), 2);
    server.close().await.expect("close");
}

#[tokio::test]
async fn connection_loss_releases_its_attachment_while_server_shutdown_closes_the_harness() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    let server = create_server(host.clone());
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");
    client.attach(SERVER_ID, "session-1").await.expect("attach");
    let harness = host.latest_harness("session-1").expect("harness");

    client.close().await.expect("close");
    poll_until(|| harness.attached_clients_count() == 0).await;
    assert_eq!(harness.close_count_value(), 0);
    server.close().await.expect("close");
    assert_eq!(harness.close_count_value(), 1);
}

// ─── routed Session acquisition failures ─────────────────────────────────────

#[tokio::test]
async fn releases_a_lease_acquired_concurrently_with_harness_termination() {
    let metadata: Arc<dyn SessionMetadata> = Arc::new(crate::types::BasicSessionMetadata {
        id: "session-1".to_string(),
    });
    let acquiring_slot: Arc<parking_lot::Mutex<Option<Arc<Deferred<()>>>>> = Arc::default();
    let continue_acquiring = Arc::new(Deferred::<()>::new());
    let terminated = Arc::new(Deferred::<Option<AnyError>>::new());
    let release_count = Arc::new(AtomicUsize::new(0));
    let slot_for_open = Arc::clone(&acquiring_slot);
    let continue_for_open = Arc::clone(&continue_acquiring);
    let terminated_for_open = Arc::clone(&terminated);
    let release_for_open = Arc::clone(&release_count);
    let host = FnHost::new(crate::testing::create_test_server_services())
        .with_resolve(move |_session_id, _context| {
            let metadata = Arc::clone(&metadata);
            Box::pin(async move { Ok(metadata) })
        })
        .with_open(move |_metadata, _context| {
            let acquiring = Arc::new(Deferred::<()>::new());
            *slot_for_open.lock() = Some(Arc::clone(&acquiring));
            let handle = Arc::new(StubHandle {
                terminated: Some(Arc::clone(&terminated_for_open)),
                attach_gate: Some((acquiring, Arc::clone(&continue_for_open))),
                release_count: Arc::clone(&release_for_open),
            });
            Box::pin(async move { Ok(handle as Arc<dyn RoutedSessionHandle>) })
        });
    let server = create_server(host);
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");
    let attach = spawn_attach(&loopback.client, "session-1");
    // Wait until the handle's attachClient is in flight.
    poll_until(|| acquiring_slot.lock().is_some()).await;
    let acquiring = acquiring_slot.lock().take().expect("open called");
    acquiring.promise().await;

    terminated.resolve(Some(any_server_error("worker crashed")));
    continue_acquiring.resolve(());
    let response = attach.await.expect("attach task").expect("attach response");
    assert_eq!(msg::ok(&response), false);
    assert_eq!(msg::error_code(&response), "server_draining");
    assert_eq!(release_count.load(Ordering::SeqCst), 1);
    server.close().await.expect("close");
}

#[tokio::test]
async fn shares_a_harness_creation_failure_releases_the_session_and_allows_a_later_retry() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    *host.next_open_session_error.lock() = Some(any_server_error("Harness creation failed"));
    let server = create_server(host.clone());
    let first_loop = crate::tests::loopback::connect(&server);
    let second_loop = crate::tests::loopback::connect(&server);
    let first = &first_loop.client;
    let second = &second_loop.client;
    first.hello().await.expect("hello");
    second.hello().await.expect("hello");
    let gate = host.gate_next_open_session();

    let first_attach = spawn_attach(&loopback_first_client(&first_loop), "session-1");
    gate.entered.promise().await;
    let second_attach = spawn_attach(&loopback_first_client(&second_loop), "session-1");
    // Upstream's synchronous send guarantees the second attach joins the
    // in-flight open before the gate releases; mirror that determinism.
    let server_for_poll = server.clone();
    poll_until(move || server_for_poll.debug_opening_joiners("session-1") >= 1).await;
    gate.release.resolve(());

    let first_response = first_attach
        .await
        .expect("first task")
        .expect("first response");
    let second_response = second_attach
        .await
        .expect("second task")
        .expect("second response");
    assert_eq!(msg::error_code(&first_response), "internal_error");
    assert_eq!(msg::error_code(&second_response), "internal_error");
    assert_eq!(host.open_session_count_value(), 1);

    let retry = first.attach(SERVER_ID, "session-1").await.expect("retry");
    assert!(msg::ok(&retry), "retry: {retry}");
    assert_eq!(host.open_session_count_value(), 2);
    assert_eq!(host.harness_count("session-1"), 1);
    server.close().await.expect("close");
}

fn loopback_first_client(loopback: &crate::tests::loopback::Loopback) -> &ProtocolTestClient {
    &loopback.client
}

#[tokio::test]
async fn closes_a_harness_acquired_while_server_shutdown_is_in_progress() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    let server = create_server(host.clone());
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");
    let gate = host.gate_next_open_session();
    let attach = spawn_attach(&loopback.client, "session-1");
    gate.entered.promise().await;
    let closing = tokio::spawn({
        let server = server.clone();
        async move { server.close().await }
    });
    gate.release.resolve(());

    closing.await.expect("closing task").expect("close");
    let outcome = attach.await.expect("attach task");
    let error = outcome.expect_err("the in-flight attach must fail once the wire closes");
    assert!(
        error.to_string().to_lowercase().contains("closed"),
        "error: {error}"
    );
    assert_eq!(
        host.latest_harness("session-1")
            .expect("harness")
            .close_count_value(),
        1
    );
}

#[tokio::test]
async fn fails_shutdown_when_an_in_flight_acquisition_cannot_release_its_harness() {
    let host = TestServerHost::new();
    host.seed_with_id("session-1").await;
    let cleanup_error = any_server_error("close failed");
    *host.next_harness_close_error.lock() = Some(Arc::clone(&cleanup_error));
    let server = create_server(host.clone());
    let loopback = crate::tests::loopback::connect(&server);
    let client = &loopback.client;
    client.hello().await.expect("hello");
    let gate = host.gate_next_open_session();
    let attach = spawn_attach(&loopback.client, "session-1");
    gate.entered.promise().await;

    let closing = tokio::spawn({
        let server = server.clone();
        async move { server.close().await }
    });
    gate.release.resolve(());

    let close_error = closing
        .await
        .expect("closing task")
        .expect_err("close must fail");
    assert!(close_error
        .to_string()
        .contains("Failed to close routed Sessions"));
    let closed_error = server.closed().await.expect_err("closed must fail");
    assert!(closed_error
        .to_string()
        .contains("Failed to close routed Sessions"));
    let outcome = attach.await.expect("attach task");
    let error = outcome.expect_err("the in-flight attach must fail once the wire closes");
    assert!(
        error.to_string().to_lowercase().contains("closed"),
        "error: {error}"
    );
    let harness = host.latest_harness("session-1").expect("harness");
    assert_eq!(harness.close_count_value(), 1);

    // Upstream drains the harness manually so the test fixture exits cleanly.
    harness
        .close(BACKGROUND_CONTEXT.clone())
        .await
        .expect("manual close");
}
