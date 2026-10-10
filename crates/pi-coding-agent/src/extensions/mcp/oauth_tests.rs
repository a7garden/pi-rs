//! Ports of `test/mcp-oauth-store.test.ts` and
//! `test/mcp-oauth-refresh.test.ts`, with the `suite/mcp-oauth-server.ts`
//! fixture rebuilt on a tokio TCP listener (upstream: `node:http` — D88: the
//! fixture is an in-module test server; the browser fetch is `reqwest`).

use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use parking_lot::Mutex;
use serde_json::{json, Value};
use url::Url;

use pi_mcp::auth_provider::{default_fetch, UnauthorizedContext, UnauthorizedResponse};
use pi_mcp::client::AbortController;
use pi_mcp::oauth::{
    McpOAuthStateStore, OAuthChallenge, OAuthClientInformationMixed, OAuthIssuerMismatchError,
};
use pi_mcp::protocol::jsonrpc::McpError;
use pi_mcp::AuthProvider;

use super::*;
use crate::core::auth_storage::{InMemoryAuthStorageBackend, LockResult};

// ---------------------------------------------------------------------------
// Credential store (mcp-oauth-store.test.ts)
// ---------------------------------------------------------------------------

const SERVER_URL: &str = "https://mcp.example.com/mcp";

fn stored_state(access_token: &str) -> Value {
    json!({
        "serverUrl": SERVER_URL,
        "tokens": { "access_token": access_token, "token_type": "Bearer" }
    })
}

fn stored_keys(backend: &InMemoryAuthStorageBackend) -> Vec<String> {
    backend
        .with_lock(|current| {
            let keys: Vec<String> = parse_states(current.as_deref()).keys().cloned().collect();
            LockResult {
                result: keys,
                next: None,
            }
        })
        .expect("in-memory lock")
}

/// Port of "keeps separate credentials for servers sharing a SERVER_URL"
/// (earendil-works/pi#10252).
#[tokio::test]
async fn keeps_separate_credentials_for_servers_sharing_a_server_url() {
    let store = McpOAuthCredentialStore::new(Arc::new(InMemoryAuthStorageBackend::new()), None);
    store
        .for_server_borrowed("work", SERVER_URL)
        .save(&mcp_state("work-token"))
        .await
        .expect("save");
    store
        .for_server_borrowed("personal", SERVER_URL)
        .save(&mcp_state("personal-token"))
        .await
        .expect("save");

    assert_eq!(
        store
            .for_server_borrowed("work", SERVER_URL)
            .load()
            .await
            .expect("load")
            .expect("state")
            .tokens
            .expect("tokens")
            .access_token,
        "work-token"
    );
    assert_eq!(
        store
            .for_server_borrowed("personal", SERVER_URL)
            .load()
            .await
            .expect("load")
            .expect("state")
            .tokens
            .expect("tokens")
            .access_token,
        "personal-token"
    );

    assert!(store.remove("work", SERVER_URL));
    assert!(store
        .for_server_borrowed("work", SERVER_URL)
        .load()
        .await
        .expect("load")
        .is_none());
    assert_eq!(
        store.tokens("personal", SERVER_URL).expect("tokens")["access_token"],
        json!("personal-token")
    );
}

fn mcp_state(access_token: &str) -> pi_mcp::oauth::McpOAuthState {
    pi_mcp::oauth::McpOAuthState {
        server_url: SERVER_URL.to_string(),
        tokens: Some(pi_mcp::oauth::OAuthTokens {
            access_token: access_token.to_string(),
            token_type: "Bearer".to_string(),
            expires_in: None,
            scope: None,
            refresh_token: None,
            id_token: None,
        }),
        ..Default::default()
    }
}

fn write_raw_state(backend: &InMemoryAuthStorageBackend, states: Value) {
    backend
        .with_lock(|_current| LockResult {
            result: (),
            next: Some(serde_json::to_string(&states).expect("serialize")),
        })
        .expect("in-memory lock");
}

/// Port of "moves credentials stored by SERVER_URL to the first server that
/// loads them".
#[tokio::test]
async fn moves_credentials_stored_by_server_url_to_the_first_server_that_loads_them() {
    let backend = Arc::new(InMemoryAuthStorageBackend::new());
    write_raw_state(
        &backend,
        json!({ (SERVER_URL): stored_state("legacy-token") }),
    );
    let store = McpOAuthCredentialStore::new(backend.clone(), None);

    // Reading tokens does not take the legacy state over.
    assert_eq!(
        store.tokens("work", SERVER_URL).expect("tokens")["access_token"],
        json!("legacy-token")
    );
    assert_eq!(stored_keys(&backend), vec![SERVER_URL.to_string()]);

    assert_eq!(
        store
            .for_server_borrowed("my_work", SERVER_URL)
            .load()
            .await
            .expect("load")
            .expect("state")
            .tokens
            .expect("tokens")
            .access_token,
        "legacy-token"
    );
    // Names differing only in `-` and `_` are the same server.
    assert_eq!(
        store.tokens("my-work", SERVER_URL).expect("tokens")["access_token"],
        json!("legacy-token")
    );
    assert!(store
        .for_server_borrowed("personal", SERVER_URL)
        .load()
        .await
        .expect("load")
        .is_none());
    assert_eq!(
        stored_keys(&backend),
        vec![format!("mcp__my_work|{SERVER_URL}")]
    );
}

/// Port of "signs out of credentials stored by SERVER_URL".
#[tokio::test]
async fn signs_out_of_credentials_stored_by_server_url() {
    let backend = Arc::new(InMemoryAuthStorageBackend::new());
    write_raw_state(
        &backend,
        json!({ (SERVER_URL): stored_state("legacy-token") }),
    );
    let store = McpOAuthCredentialStore::new(backend.clone(), None);

    assert!(store.remove("work", SERVER_URL));
    assert!(stored_keys(&backend).is_empty());
    assert!(!store.remove("work", SERVER_URL));
}

// ---------------------------------------------------------------------------
// OAuth MCP server fixture (suite/mcp-oauth-server.ts)
// ---------------------------------------------------------------------------

struct Fixture {
    url: String,
    origin: String,
    log: Mutex<Vec<String>>,
    registrations: Mutex<Vec<Value>>,
    authorizations: Mutex<Vec<Vec<(String, String)>>>,
    token_requests: Mutex<Vec<String>>,
    valid_tokens: Mutex<std::collections::HashSet<String>>,
    refresh_tokens: Mutex<std::collections::HashSet<String>>,
    challenges: Mutex<std::collections::HashMap<String, String>>,
    issued: std::sync::atomic::AtomicU64,
    iss: Option<String>,
    iss_parameter: bool,
    cimd: bool,
    redirect_path: Option<String>,
    listener: Arc<tokio::net::TcpListener>,
}

struct FixtureHandle {
    url: String,
    inner: Arc<Fixture>,
}

impl FixtureHandle {
    fn expire_access_tokens(&self) {
        self.inner.valid_tokens.lock().clear();
    }

    async fn close(self) {
        let _ = self.inner.listener;
    }
}

async fn start_oauth_mcp_server(options: impl FnOnce(&mut FixtureOptions)) -> FixtureHandle {
    let mut fixture_options = FixtureOptions::default();
    options(&mut fixture_options);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let origin = format!("http://127.0.0.1:{port}");
    let fixture = Arc::new(Fixture {
        url: format!("{origin}/mcp"),
        origin: origin.clone(),
        log: Mutex::new(Vec::new()),
        registrations: Mutex::new(Vec::new()),
        authorizations: Mutex::new(Vec::new()),
        token_requests: Mutex::new(Vec::new()),
        valid_tokens: Mutex::new(Default::default()),
        refresh_tokens: Mutex::new(Default::default()),
        challenges: Mutex::new(Default::default()),
        issued: std::sync::atomic::AtomicU64::new(0),
        iss: fixture_options.iss,
        iss_parameter: fixture_options.iss_parameter,
        cimd: fixture_options.cimd,
        redirect_path: fixture_options.redirect_path,
        listener: Arc::new(listener),
    });
    let server = Arc::clone(&fixture);
    tokio::spawn(async move { serve(server).await });
    FixtureHandle {
        url: fixture.url.clone(),
        inner: fixture,
    }
}

#[derive(Default)]
struct FixtureOptions {
    iss: Option<String>,
    iss_parameter: bool,
    cimd: bool,
    redirect_path: Option<String>,
}

async fn serve(fixture: Arc<Fixture>) {
    loop {
        let Ok((mut stream, _)) = fixture.listener.accept().await else {
            break;
        };
        let fixture = Arc::clone(&fixture);
        tokio::spawn(async move {
            let _ = handle_connection(&fixture, &mut stream).await;
            let _ = stream.shutdown().await;
        });
    }
}

async fn handle_connection(
    fixture: &Arc<Fixture>,
    stream: &mut tokio::net::TcpStream,
) -> std::io::Result<()> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    // Read the header.
    let header_end;
    loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Ok(());
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(position) = find_subsequence(&buffer, b"\r\n\r\n") {
            header_end = position + 4;
            break;
        }
        if buffer.len() > 1 << 20 {
            return Ok(());
        }
    }
    let header_text = String::from_utf8_lossy(&buffer[..header_end]).into_owned();
    let mut lines = header_text.lines();
    let request_line = lines.next().unwrap_or_default().to_string();
    let mut content_length = 0usize;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    while buffer.len() < header_end + content_length {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    let body = String::from_utf8_lossy(&buffer[header_end..]).into_owned();
    let mut parts = request_line.split(' ');
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path.to_string(), Some(query.to_string())),
        None => (target, None),
    };
    route(fixture, &method, &path, query.as_deref(), &body, stream).await
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

async fn route(
    fixture: &Arc<Fixture>,
    method: &str,
    path: &str,
    query: Option<&str>,
    body: &str,
    stream: &mut tokio::net::TcpStream,
) -> std::io::Result<()> {
    let origin = fixture.origin.clone();
    match path {
        "/.well-known/oauth-protected-resource/mcp" => {
            json_response(
                stream,
                200,
                json!({ "resource": format!("{origin}/mcp"), "authorization_servers": [origin] }),
            )
            .await
        }
        "/.well-known/oauth-authorization-server" => {
            let mut metadata = json!({
                "issuer": origin,
                "authorization_endpoint": format!("{origin}/authorize"),
                "token_endpoint": format!("{origin}/token"),
                "registration_endpoint": format!("{origin}/register"),
                "response_types_supported": ["code"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": ["none"],
            });
            if fixture.cimd {
                metadata["client_id_metadata_document_supported"] = json!(true);
            }
            if fixture.iss_parameter {
                metadata["authorization_response_iss_parameter_supported"] = json!(true);
            }
            json_response(stream, 200, metadata).await
        }
        "/register" => {
            let metadata: Value = serde_json::from_str(body).unwrap_or(json!({}));
            fixture.log.lock().push("register".to_string());
            fixture.registrations.lock().push(metadata.clone());
            let mut registered = metadata;
            registered["client_id"] = json!("client-1");
            json_response(stream, 201, registered).await
        }
        "/authorize" => {
            let params = parse_query(query.unwrap_or_default());
            fixture.authorizations.lock().push(params.clone());
            let index = fixture.challenges.lock().len() + 1;
            let code = format!("code-{index}");
            let challenge = query_get(&params, "code_challenge")
                .unwrap_or_default()
                .to_string();
            fixture.challenges.lock().insert(code.clone(), challenge);
            let mut redirect =
                Url::parse(query_get(&params, "redirect_uri").unwrap_or("http://127.0.0.1/"))
                    .expect("redirect");
            if let Some(redirect_path) = &fixture.redirect_path {
                redirect.set_path(redirect_path);
            }
            redirect.query_pairs_mut().append_pair("code", &code);
            redirect
                .query_pairs_mut()
                .append_pair("state", query_get(&params, "state").unwrap_or(""));
            let iss = fixture.iss.clone().or_else(|| {
                if fixture.iss_parameter {
                    Some(origin.clone())
                } else {
                    None
                }
            });
            if let Some(iss) = iss {
                redirect.query_pairs_mut().append_pair("iss", &iss);
            }
            let location = redirect.to_string();
            let response = format!("HTTP/1.1 302 Found\r\nlocation: {location}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
            stream.write_all(response.as_bytes()).await
        }
        "/token" => {
            let params: Vec<(String, String)> = form_urlencoded(body);
            let get = |key: &str| {
                params
                    .iter()
                    .find(|(name, _)| name == key)
                    .map(|(_, value)| value.clone())
            };
            fixture
                .token_requests
                .lock()
                .push(serde_json::to_string(&params).expect("serialize"));
            if get("grant_type").as_deref() == Some("authorization_code") {
                let code = get("code").unwrap_or_default();
                let challenge = fixture.challenges.lock().get(&code).cloned();
                let verifier = get("code_verifier").unwrap_or_default();
                let expected = base64_url_no_pad(&Sha256::digest(verifier.as_bytes()));
                let valid = challenge.as_deref() == Some(expected.as_str());
                fixture.challenges.lock().remove(&code);
                if !valid {
                    return json_response(stream, 400, json!({ "error": "invalid_grant" })).await;
                }
                fixture.log.lock().push("token code".to_string());
                let tokens = issue_tokens(fixture);
                return json_response(stream, 200, tokens).await;
            }
            let refresh = get("refresh_token").unwrap_or_default();
            if !fixture.refresh_tokens.lock().remove(&refresh) {
                return json_response(stream, 400, json!({ "error": "invalid_grant" })).await;
            }
            fixture.log.lock().push("token refresh".to_string());
            let tokens = issue_tokens(fixture);
            json_response(stream, 200, tokens).await
        }
        _ => {
            let response =
                "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n";
            stream.write_all(response.as_bytes()).await
        }
    }
}

fn issue_tokens(fixture: &Fixture) -> Value {
    let issued = fixture
        .issued
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        + 1;
    let access = format!("access-{issued}");
    let refresh = format!("refresh-{issued}");
    fixture.valid_tokens.lock().insert(access.clone());
    fixture.refresh_tokens.lock().insert(refresh.clone());
    json!({ "access_token": access, "refresh_token": refresh, "token_type": "Bearer", "expires_in": 3600 })
}

fn parse_query(query: &str) -> Vec<(String, String)> {
    form_urlencoded(query)
}

fn form_urlencoded(input: &str) -> Vec<(String, String)> {
    input
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| match pair.split_once('=') {
            Some((name, value)) => (urldecode(name), urldecode(value)),
            None => (urldecode(pair), String::new()),
        })
        .collect()
}

fn urldecode(input: &str) -> String {
    let mut output = Vec::new();
    let bytes = input.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                output.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("00");
                output.push(u8::from_str_radix(hex, 16).unwrap_or(b'%'));
                index += 3;
            }
            byte => {
                output.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&output).into_owned()
}

async fn json_response(
    stream: &mut tokio::net::TcpStream,
    status: u16,
    body: Value,
) -> std::io::Result<()> {
    let body = serde_json::to_string(&body).expect("serialize");
    let response = format!(
		"HTTP/1.1 {status} OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
		body.len()
	);
    stream.write_all(response.as_bytes()).await
}

/// A browser that GETs every authorization URL (following redirects to the
/// loopback callback), upstream `showAuthorizationUrl: (url) => void fetch(url)`.
fn browser_prompt() -> McpSignInPrompt {
    McpSignInPrompt {
        show_authorization_url: Arc::new(|url| {
            let url = url.to_string();
            tokio::spawn(async move {
                let _ = reqwest::Client::builder()
                    .redirect(reqwest::redirect::Policy::limited(5))
                    .build()
                    .expect("client")
                    .get(&url)
                    .send()
                    .await;
            });
        }),
        prompt_for_redirect_url: Arc::new(|signal| {
            Box::pin(async move {
                signal.cancelled().await;
                None
            })
        }),
    }
}

fn cancelled_prompt() -> McpSignInPrompt {
    McpSignInPrompt {
        show_authorization_url: Arc::new(|_| {}),
        prompt_for_redirect_url: Arc::new(|signal| {
            Box::pin(async move {
                signal.cancelled().await;
                None
            })
        }),
    }
}

fn paste_prompt(paste: impl Fn() -> Option<String> + Send + Sync + 'static) -> McpSignInPrompt {
    let paste = Arc::new(paste);
    McpSignInPrompt {
        show_authorization_url: Arc::new(|_| {}),
        prompt_for_redirect_url: Arc::new(move |_signal| {
            let paste = Arc::clone(&paste);
            Box::pin(async move { paste() })
        }),
    }
}

async fn in_memory_server_store(server_url: &str) -> Arc<dyn McpOAuthStateStore> {
    let store = McpOAuthCredentialStore::new(Arc::new(InMemoryAuthStorageBackend::new()), None)
        .for_server_borrowed("test", server_url);
    store as Arc<dyn McpOAuthStateStore>
}

// ---------------------------------------------------------------------------
// Refresh (mcp-oauth-refresh.test.ts)
// ---------------------------------------------------------------------------

async fn signed_in() -> (FixtureHandle, String, Process) {
    let server = start_oauth_mcp_server(|_| {}).await;
    let lock_dir = {
        let dir = std::env::temp_dir().join(format!(
            "pi-mcp-refresh-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("lock dir");
        dir.to_string_lossy().into_owned()
    };
    let url = server.url.clone();
    // Stores sharing the credential file and lock directory stand in for
    // separate pi processes.
    let backend = Arc::new(InMemoryAuthStorageBackend::new());
    let process: Process = {
        let backend = Arc::clone(&backend);
        let lock_dir = lock_dir.clone();
        let url = url.clone();
        Arc::new(move || {
            let store = McpOAuthCredentialStore::new(Arc::clone(&backend), Some(lock_dir.clone()))
                .for_server_borrowed("test", &url);
            McpAuthProvider::new(McpAuthProviderOptions {
                server_url: url.clone(),
                store,
                settings: Arc::new(|| McpOAuthSettings::default()),
                on_challenge: Arc::new(|_| {}),
            })
        })
    };
    sign_in_mcp_server(McpSignInOptions {
        server_url: url.clone(),
        store: McpOAuthCredentialStore::new(Arc::clone(&backend), Some(lock_dir.clone()))
            .for_server_borrowed("test", &url),
        settings: McpOAuthSettings::default(),
        challenge: None,
        prompt: browser_prompt(),
        signal: None,
    })
    .await
    .expect("sign in");
    (server, lock_dir, process)
}

/// One simulated pi "process": its provider over the shared backend.
type Process = Arc<dyn Fn() -> Arc<McpAuthProvider> + Send>;

/// Port of "refreshes once when several processes find the same token
/// rejected".
#[tokio::test(flavor = "multi_thread")]
async fn refreshes_once_when_several_processes_find_the_same_token_rejected() {
    let (server, lock_dir, process) = signed_in().await;
    server.expire_access_tokens();
    let providers: Vec<Arc<McpAuthProvider>> = (0..3).map(|_| process()).collect();

    // The server rotates refresh tokens: a second refresh with refresh-1 would
    // fail with invalid_grant.
    for provider in &providers {
        let _ = provider.refresh(Some("access-1".to_string()), None).await;
    }

    assert_eq!(
        server
            .inner
            .log
            .lock()
            .iter()
            .filter(|entry| entry.as_str() == "token refresh")
            .count(),
        1
    );
    for provider in &providers {
        assert_eq!(
            SharedMcpAuthProvider(Arc::clone(provider))
                .token()
                .await
                .expect("token"),
            Some("access-2".to_string())
        );
    }
    // The lock is released.
    let entries: Vec<_> = std::fs::read_dir(&lock_dir)
        .expect("lock dir")
        .filter_map(|entry| entry.ok())
        .collect();
    assert!(entries.is_empty(), "lock files remain: {entries:?}");
    let _ = std::fs::remove_dir_all(&lock_dir);
    server.close().await;
}

/// Port of "waits for a running refresh to save the new tokens".
#[tokio::test(flavor = "multi_thread")]
async fn waits_for_a_running_refresh_to_save_the_new_tokens() {
    let (server, lock_dir, process) = signed_in().await;
    server.expire_access_tokens();
    let provider = process();

    let refresh = provider.refresh(Some("access-1".to_string()), None);
    tokio::pin!(refresh);
    // Upstream's `onUnauthorized` promise starts eagerly; the lazy Rust future
    // registers itself in `refreshing` on its first poll, so poll it once.
    let _ = futures_util::poll!(refresh.as_mut());
    provider.settled().await;
    assert_eq!(
        SharedMcpAuthProvider(Arc::clone(&provider))
            .token()
            .await
            .expect("token"),
        Some("access-2".to_string())
    );
    refresh.await.expect("refresh");
    let _ = std::fs::remove_dir_all(&lock_dir);
    server.close().await;
}

// ---------------------------------------------------------------------------
// Sign-in (mcp-oauth-refresh.test.ts "MCP OAuth sign-in")
// ---------------------------------------------------------------------------

/// Port of "rejects an authorization response from another issuer".
#[tokio::test(flavor = "multi_thread")]
async fn rejects_an_authorization_response_from_another_issuer() {
    let server = start_oauth_mcp_server(|options| {
        options.iss = Some("https://attacker.example".to_string());
    })
    .await;
    let url = server.url.clone();
    let error = sign_in_mcp_server(McpSignInOptions {
        server_url: url,
        store: in_memory_server_store(&server.url).await,
        settings: McpOAuthSettings::default(),
        challenge: None,
        prompt: browser_prompt(),
        signal: None,
    })
    .await
    .expect_err("must reject");
    assert!(
        matches!(
            error,
            McpError::IssuerMismatch(OAuthIssuerMismatchError { .. })
        ),
        "unexpected error: {error}"
    );
    server.close().await;
}

/// Port of "keeps the granted scope when the server asks for more".
#[tokio::test(flavor = "multi_thread")]
async fn keeps_the_granted_scope_when_the_server_asks_for_more() {
    let server = start_oauth_mcp_server(|_| {}).await;
    let url = server.url.clone();
    let store = McpOAuthCredentialStore::new(Arc::new(InMemoryAuthStorageBackend::new()), None)
        .for_server_borrowed("test", &url);
    let opened: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let sign_in_with = |challenge: OAuthChallenge| {
        let store = Arc::clone(&store);
        let opened = Arc::clone(&opened);
        let url = url.clone();
        async move {
            sign_in_mcp_server(McpSignInOptions {
                server_url: url,
                store,
                settings: McpOAuthSettings::default(),
                challenge: Some(challenge),
                prompt: recording_prompt(opened),
                signal: None,
            })
            .await
        }
    };

    sign_in_with(OAuthChallenge {
        scope: Some("issues:read".to_string()),
        ..Default::default()
    })
    .await
    .expect("first sign-in");
    // The token response names no scope, so the grant has the requested one.
    assert_eq!(
        store
            .load()
            .await
            .expect("load")
            .expect("state")
            .tokens
            .expect("tokens")
            .scope
            .as_deref(),
        Some("issues:read")
    );
    // The step-up challenge lists only the missing scope. Requesting just that
    // would lose issues:read, so the next request would ask for sign-in again.
    sign_in_with(OAuthChallenge {
        error: Some("insufficient_scope".to_string()),
        scope: Some("issues:write".to_string()),
        ..Default::default()
    })
    .await
    .expect("step-up sign-in");
    let scopes: Vec<Option<String>> = opened
        .lock()
        .iter()
        .map(|url| {
            Url::parse(url).ok().and_then(|url| {
                url.query_pairs()
                    .find(|(name, _)| name == "scope")
                    .map(|(_, value)| value.into_owned())
            })
        })
        .collect();
    assert_eq!(
        scopes,
        vec![
            Some("issues:read".to_string()),
            Some("issues:read issues:write".to_string()),
        ]
    );
    assert_eq!(
        store
            .load()
            .await
            .expect("load")
            .expect("state")
            .tokens
            .expect("tokens")
            .scope
            .as_deref(),
        Some("issues:read issues:write")
    );
    server.close().await;
}

fn recording_prompt(opened: Arc<Mutex<Vec<String>>>) -> McpSignInPrompt {
    McpSignInPrompt {
        show_authorization_url: Arc::new(move |url| {
            let url = url.to_string();
            opened.lock().push(url.clone());
            let client = reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::limited(5))
                .build()
                .expect("client");
            tokio::spawn(async move {
                let _ = client.get(&url).send().await;
            });
        }),
        prompt_for_redirect_url: Arc::new(|signal| {
            Box::pin(async move {
                signal.cancelled().await;
                None
            })
        }),
    }
}

/// Port of #10172 "uses the configured authorization server metadata URL".
#[tokio::test(flavor = "multi_thread")]
async fn uses_the_configured_authorization_server_metadata_url() {
    let server = start_oauth_mcp_server(|_| {}).await;
    let url = server.url.clone();
    let settings = McpOAuthSettings {
        auth_server_metadata_url: Some(
            Url::parse(&format!("{}/missing", server.inner.origin)).expect("url"),
        ),
        ..Default::default()
    };
    let error = sign_in_mcp_server(McpSignInOptions {
        server_url: url,
        store: in_memory_server_store(&server.url).await,
        settings,
        challenge: None,
        prompt: browser_prompt(),
        signal: None,
    })
    .await
    .expect_err("must fail");
    assert!(
        error.to_string().contains("authorization server metadata"),
        "unexpected error: {error}"
    );
    server.close().await;
}

// ---------------------------------------------------------------------------
// Client ID Metadata Documents (#10302)
// ---------------------------------------------------------------------------

struct CimdServer {
    handle: FixtureHandle,
    store: Arc<dyn McpOAuthStateStore>,
    url: String,
}

async fn start_cimd_server(options: impl FnOnce(&mut FixtureOptions)) -> CimdServer {
    let handle = start_oauth_mcp_server(options).await;
    let store = McpOAuthCredentialStore::new(Arc::new(InMemoryAuthStorageBackend::new()), None)
        .for_server_borrowed("test", &handle.url);
    CimdServer {
        store: store as Arc<dyn McpOAuthStateStore>,
        url: handle.url.clone(),
        handle,
    }
}

impl CimdServer {
    async fn sign_in(&self, settings: McpOAuthSettings) -> Result<(), McpError> {
        self.sign_in_with(settings, browser_prompt()).await
    }

    async fn sign_in_with(
        &self,
        settings: McpOAuthSettings,
        prompt: McpSignInPrompt,
    ) -> Result<(), McpError> {
        sign_in_mcp_server(McpSignInOptions {
            server_url: self.url.clone(),
            store: Arc::clone(&self.store),
            settings,
            challenge: None,
            prompt,
            signal: None,
        })
        .await
    }

    fn registrations(&self) -> Vec<Value> {
        self.handle.inner.registrations.lock().clone()
    }

    fn authorizations(&self) -> Vec<Vec<(String, String)>> {
        self.handle.inner.authorizations.lock().clone()
    }

    fn token_requests(&self) -> Vec<Vec<(String, String)>> {
        // The fixture serializes the parsed form params as JSON.
        self.handle
            .inner
            .token_requests
            .lock()
            .iter()
            .map(|encoded| {
                serde_json::from_str::<Vec<(String, String)>>(encoded).expect("token params json")
            })
            .collect()
    }

    async fn stored(&self) -> Option<pi_mcp::oauth::McpOAuthState> {
        self.store.load().await.expect("load")
    }
}

fn cimd_settings() -> McpOAuthSettings {
    McpOAuthSettings {
        client_registration: Some("cimd".to_string()),
        ..Default::default()
    }
}

fn query_get<'a>(params: &'a [(String, String)], key: &str) -> Option<&'a str> {
    params
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
}

/// Port of "registers dynamically by default, even when the server supports
/// documents".
#[tokio::test(flavor = "multi_thread")]
async fn registers_dynamically_by_default_even_when_the_server_supports_documents() {
    let server = start_cimd_server(|options| {
        options.cimd = true;
        options.iss_parameter = true;
    })
    .await;
    server
        .sign_in(McpOAuthSettings::default())
        .await
        .expect("sign in");
    assert_eq!(server.registrations().len(), 1);
    let authorizations = server.authorizations();
    assert_eq!(query_get(&authorizations[0], "client_id"), Some("client-1"));
    server.handle.close().await;
}

/// Port of "uses pi's document when authorization responses name their
/// issuer".
#[tokio::test(flavor = "multi_thread")]
async fn uses_pis_document_when_authorization_responses_name_their_issuer() {
    let server = start_cimd_server(|options| {
        options.cimd = true;
        options.iss_parameter = true;
    })
    .await;
    server.sign_in(cimd_settings()).await.expect("sign in");
    let authorizations = server.authorizations();
    println!("DEBUG authorizations: {authorizations:?}");
    assert_eq!(
        query_get(&authorizations[0], "client_id"),
        Some("https://pi.dev/oauth/client.json")
    );
    let redirect =
        Url::parse(query_get(&authorizations[0], "redirect_uri").expect("redirect")).expect("url");
    assert_eq!(
        format!("{}{}", redirect.host_str().expect("host"), redirect.path()),
        "127.0.0.1/callback"
    );
    assert!(redirect.port().is_some());
    assert!(server.registrations().is_empty());
    let tokens = server.token_requests();
    assert_eq!(
        query_get(&tokens[0], "client_id"),
        Some("https://pi.dev/oauth/client.json")
    );
    assert_eq!(
        query_get(&tokens[0], "redirect_uri"),
        Some(redirect.as_str())
    );
    // The document is not stored, so signing in again refreshes the tokens
    // instead of discarding them.
    assert!(server
        .stored()
        .await
        .expect("load")
        .client_information
        .is_none());
    server
        .sign_in(cimd_settings())
        .await
        .expect("second sign-in");
    assert_eq!(server.authorizations().len(), 1);
    let tokens = server.token_requests();
    assert_eq!(query_get(&tokens[1], "grant_type"), Some("refresh_token"));
    assert_eq!(
        query_get(&tokens[1], "client_id"),
        Some("https://pi.dev/oauth/client.json")
    );
    server.handle.close().await;
}

/// Port of "uses a document and callback path specific to the MCP server
/// without the iss parameter".
#[tokio::test(flavor = "multi_thread")]
async fn uses_a_document_and_callback_path_specific_to_the_mcp_server_without_the_iss_parameter() {
    let server = start_cimd_server(|options| {
        options.cimd = true;
    })
    .await;
    server.sign_in(cimd_settings()).await.expect("sign in");
    // Computed like Codex: the first 9 bytes of the SHA-256 of the MCP server URL.
    let id = base64_url_no_pad(&Sha256::digest(server.url.as_bytes())[..9]);
    let authorizations = server.authorizations();
    assert_eq!(
        query_get(&authorizations[0], "client_id"),
        Some(format!("https://pi.dev/oauth/{id}/client.json").as_str())
    );
    let redirect =
        Url::parse(query_get(&authorizations[0], "redirect_uri").expect("redirect")).expect("url");
    assert_eq!(redirect.path(), format!("/callback/{id}"));
    assert!(server.registrations().is_empty());
    let tokens = server.token_requests();
    assert_eq!(
        query_get(&tokens[0], "redirect_uri"),
        Some(redirect.as_str())
    );
    server.handle.close().await;
}

/// Port of "replaces a registered client when switching to the document".
#[tokio::test(flavor = "multi_thread")]
async fn replaces_a_registered_client_when_switching_to_the_document() {
    let server = start_cimd_server(|options| {
        options.cimd = true;
        options.iss_parameter = true;
    })
    .await;
    server
        .sign_in(McpOAuthSettings::default())
        .await
        .expect("sign in");
    assert!(server
        .stored()
        .await
        .expect("load")
        .client_information
        .is_some());
    server
        .sign_in(cimd_settings())
        .await
        .expect("second sign-in");
    // The registered client's tokens are not refreshed with another client.
    let authorizations = server.authorizations();
    let client_ids: Vec<Option<&str>> = authorizations
        .iter()
        .map(|authorization| query_get(authorization, "client_id"))
        .collect();
    assert_eq!(
        client_ids,
        vec![Some("client-1"), Some("https://pi.dev/oauth/client.json")]
    );
    assert!(server
        .stored()
        .await
        .expect("load")
        .client_information
        .is_none());
    server.handle.close().await;
}

/// Port of "accepts the authorization response only on the server-specific
/// redirect URI".
#[tokio::test(flavor = "multi_thread")]
async fn accepts_the_authorization_response_only_on_the_server_specific_redirect_uri() {
    // A mixed-up authorization server redirects to the shared callback path.
    let server = start_cimd_server(|options| {
        options.cimd = true;
        options.redirect_path = Some("/callback".to_string());
    })
    .await;
    let error = server
        .sign_in(cimd_settings())
        .await
        .expect_err("must reject");
    assert!(
        error
            .to_string()
            .contains("arrived on another redirect URI"),
        "unexpected: {error}"
    );
    server.handle.close().await;

    // The same for a redirect URL pasted from the browser.
    let server = start_cimd_server(|options| {
        options.cimd = true;
    })
    .await;
    let shown: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let shown_sink = Arc::clone(&shown);
    let paste = paste_prompt(move || {
        let shown = shown_sink.lock().clone();
        let Some(shown) = shown else {
            return None;
        };
        let mut redirect = Url::parse(&shown)
            .ok()
            .and_then(|url| {
                url.query_pairs()
                    .find(|(name, _)| name == "redirect_uri")
                    .map(|(_, value)| value.into_owned())
            })
            .and_then(|value| Url::parse(&value).ok())?;
        redirect.set_path("/callback");
        let state = Url::parse(&shown).ok().and_then(|url| {
            url.query_pairs()
                .find(|(name, _)| name == "state")
                .map(|(_, value)| value.into_owned())
        })?;
        redirect.query_pairs_mut().append_pair("code", "code-1");
        redirect.query_pairs_mut().append_pair("state", &state);
        Some(redirect.to_string())
    });
    let shown_for_show = Arc::clone(&shown);
    let prompt = McpSignInPrompt {
        show_authorization_url: Arc::new(move |url| {
            *shown_for_show.lock() = Some(url.to_string());
        }),
        prompt_for_redirect_url: paste.prompt_for_redirect_url,
    };
    let error = server
        .sign_in_with(cimd_settings(), prompt)
        .await
        .expect_err("must reject");
    assert!(
        error
            .to_string()
            .contains("does not match this sign-in's redirect URI"),
        "unexpected: {error}"
    );
    server.handle.close().await;
}

/// Port of "fails instead of registering when the server does not support
/// documents".
#[tokio::test(flavor = "multi_thread")]
async fn fails_instead_of_registering_when_the_server_does_not_support_documents() {
    let server = start_cimd_server(|options| {
        options.iss_parameter = true;
    })
    .await;
    let error = server
        .sign_in(cimd_settings())
        .await
        .expect_err("must reject");
    assert!(
        error
            .to_string()
            .contains("does not support Client ID Metadata Documents"),
        "unexpected: {error}"
    );
    assert!(server.registrations().is_empty());
    server.handle.close().await;
}

// Keep unused-but-in-scope imports referenced (the challenge type appears in
// the runtime layer's challenge plumbing tested there).
#[allow(unused)]
fn _challenge_witness(challenge: &OAuthChallenge) {
    let _ = challenge.resource_metadata_url.clone();
    let _ = challenge.error_description.clone();
}

#[allow(unused)]
fn _mixed_client_witness(client: &OAuthClientInformationMixed) {
    let _ = client.client_id();
}

#[allow(unused)]
async fn _unauthorized_context_witness(server_url: Url) -> UnauthorizedContext {
    UnauthorizedContext {
        response: UnauthorizedResponse {
            status: 401,
            headers: Vec::new(),
        },
        server_url,
        fetch: default_fetch(),
        token: None,
    }
}

#[allow(unused)]
fn _abort_witness() -> AbortController {
    AbortController::new()
}

#[allow(unused)]
async fn _timeout_witness() {
    tokio::time::sleep(Duration::from_millis(0)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn probe_cimd_authorization() {
    let server = start_cimd_server(|options| {
        options.cimd = true;
        options.iss_parameter = true;
    })
    .await;
    let result = server.sign_in(cimd_settings()).await;
    println!(
        "PROBE sign_in: {:?}",
        result.as_ref().map(|_| "ok").map_err(|e| e.to_string())
    );
    for (index, params) in server.authorizations().iter().enumerate() {
        println!("PROBE authorization[{index}]: {:?}", params);
    }
    println!("PROBE registrations: {:?}", server.registrations());
    for (index, tokens) in server.token_requests().iter().enumerate() {
        println!("PROBE token[{index}]: {tokens:?}");
    }
    println!(
        "PROBE stored client: {:?}",
        server
            .stored()
            .await
            .map(|s| s.client_information.is_none())
    );
    server.handle.close().await;
}

// ============================================================================
// v1.1.0: request timeout + sign-in cancellation
// ============================================================================

/// `with_request_timeout` bounds a request even when the fetch never returns
/// (upstream `timedFetch`, formerly refresh-only, now sign-in wide).
#[tokio::test(flavor = "multi_thread")]
async fn request_timeout_bounds_a_slow_fetch() {
    let slow: pi_mcp::McpFetch = Arc::new(|_request| {
        Box::pin(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            Err(McpError::other("unreachable"))
        })
    });
    let bounded = with_request_timeout(slow, 10);
    let started = std::time::Instant::now();
    let request = pi_mcp::auth_provider::HttpRequest {
        method: "GET".to_string(),
        url: "https://authorization.example.com/authorize"
            .parse()
            .expect("url"),
        headers: Default::default(),
        body: None,
        signal: None,
    };
    let result = bounded(request).await;
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
    assert!(result.is_err());
}

/// An already-aborted signal cancels the sign-in before any network work.
#[tokio::test(flavor = "multi_thread")]
async fn sign_in_with_an_aborted_signal_fails_cancelled() {
    let controller = pi_mcp::client::AbortController::new();
    controller.abort(Some("cancelled".to_string()));
    let error = sign_in_mcp_server(McpSignInOptions {
        server_url: "https://example.com/mcp".to_string(),
        store: Arc::new(McpOAuthCredentialStore::new(
            Arc::new(InMemoryAuthStorageBackend::new()),
            None,
        ))
        .for_server_borrowed("test", "https://example.com/mcp"),
        settings: McpOAuthSettings::default(),
        challenge: None,
        prompt: browser_prompt(),
        signal: Some(controller.signal()),
    })
    .await
    .expect_err("aborted sign-in fails");
    assert!(is_sign_in_cancelled(&error), "{error}");
}

/// Aborting mid-flow — while the prompt waits for the redirect URL — reports
/// the cancellation instead of the abort reason.
#[tokio::test(flavor = "multi_thread")]
async fn sign_in_aborts_while_waiting_for_the_redirect_url() {
    struct NeverServer;
    impl NeverServer {
        fn url() -> String {
            "https://authorization.example.com".to_string()
        }
    }
    let _ = NeverServer::url();
    let controller = pi_mcp::client::AbortController::new();
    let prompt = McpSignInPrompt {
        show_authorization_url: Arc::new(|_url| {}),
        prompt_for_redirect_url: Arc::new(|signal| {
            Box::pin(async move {
                signal.cancelled().await;
                None
            })
        }),
    };
    // The flow cannot reach the network from this test (no server), so drive
    // only the abort-sensitive wait: the prompt observes the signal.
    let waiter = {
        let signal = controller.signal();
        tokio::spawn(async move { (prompt.prompt_for_redirect_url)(signal).await })
    };
    controller.abort(Some("test".to_string()));
    assert_eq!(waiter.await.unwrap(), None);
    assert!(is_sign_in_cancelled(&McpError::other(
        SIGN_IN_CANCELLED_MESSAGE
    )));
}
