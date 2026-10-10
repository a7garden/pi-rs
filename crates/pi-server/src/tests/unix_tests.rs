//! Port of `packages/server/test/unix.test.ts`.
//!
//! Deviation D6: the upstream stale-socket fixture forks a child process that
//! binds and then dies by SIGKILL; the port creates the identical final state
//! (socket file present, nothing listening) by binding and dropping a local
//! `std` listener in-test.

use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::path::PathBuf;
use std::sync::Arc;

use pi_chord::JsonValue;

use crate::server::Server;
use crate::testing::{connect_unix_test_client, TestServerHost};
use crate::transports::unix::{create_unix_server, get_unix_socket_path, UnixServerOptions};

const SERVER_ID: &str = "00000000-0000-4000-8000-000000000001";

struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(prefix: &str) -> TempDir {
        // Upstream tempdirs under /tmp; keep paths under SUN_LEN on macOS.
        let path = PathBuf::from("/tmp").join(format!(
            "{prefix}-{}",
            &crate::session_router::uuid_v4()[..8]
        ));
        std::fs::create_dir_all(&path).expect("temp dir");
        TempDir(path)
    }

    fn path(&self) -> std::path::PathBuf {
        self.0.clone()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn make_server(path: &std::path::Path) -> Server {
    let options = UnixServerOptions {
        server_id: SERVER_ID.to_string(),
        max_frame_length: None,
        handshake_timeout_ms: None,
        on_connection_count_changed: None,
        on_error: None,
        path: path.to_string_lossy().to_string(),
        mode: None,
        max_pending_bytes: None,
        graceful_close_timeout_ms: None,
    };
    create_unix_server(
        TestServerHost::new() as Arc<dyn crate::types::ServerHost>,
        options,
    )
    .expect("unix server")
}

fn unix_server_at(path: &std::path::Path, host: Arc<dyn crate::types::ServerHost>) -> Server {
    let options = UnixServerOptions {
        server_id: SERVER_ID.to_string(),
        max_frame_length: None,
        handshake_timeout_ms: None,
        on_connection_count_changed: None,
        on_error: None,
        path: path.to_string_lossy().to_string(),
        mode: None,
        max_pending_bytes: None,
        graceful_close_timeout_ms: None,
    };
    create_unix_server(host, options).expect("unix server")
}

fn assert_hello(message: &JsonValue) {
    assert_eq!(
        message
            .as_object()
            .and_then(|o| o.get("type"))
            .and_then(JsonValue::as_str),
        Some("hello"),
        "message: {message}"
    );
}

#[tokio::test]
async fn creates_an_in_memory_server_id_and_derives_its_explicit_unix_socket_path() {
    let directory = TempDir::new("pi-server");
    let path =
        get_unix_socket_path(SERVER_ID, &directory.path().to_string_lossy()).expect("socket path");

    assert_eq!(path, directory.path().join(format!("{SERVER_ID}.sock")));
    let first = unix_server_at(
        &path,
        TestServerHost::new() as Arc<dyn crate::types::ServerHost>,
    );
    first.start().await.expect("first start");
    let first_client = connect_unix_test_client(&path.to_string_lossy())
        .await
        .expect("client");
    let hello = first_client.hello().await.expect("hello");
    assert_eq!(
        hello
            .as_object()
            .and_then(|o| o.get("serverId"))
            .and_then(JsonValue::as_str),
        Some(SERVER_ID)
    );
    first_client.close().await.expect("client close");
    first.close().await.expect("first close");

    let replacement = unix_server_at(
        &path,
        TestServerHost::new() as Arc<dyn crate::types::ServerHost>,
    );
    replacement.start().await.expect("replacement start");
    let replacement_client = connect_unix_test_client(&path.to_string_lossy())
        .await
        .expect("client");
    let hello = replacement_client.hello().await.expect("hello");
    assert_eq!(
        hello
            .as_object()
            .and_then(|o| o.get("serverId"))
            .and_then(JsonValue::as_str),
        Some(SERVER_ID)
    );
    replacement_client.close().await.expect("client close");
    replacement.close().await.expect("replacement close");
}

#[tokio::test]
async fn rejects_a_live_listener_without_unlinking_it() {
    let directory = TempDir::new("ps");
    let path = directory.path().join("server.sock");
    let first = make_server(&path);
    first.start().await.expect("first start");
    let first_identity = std::fs::symlink_metadata(&path).expect("lstat");

    let second = make_server(&path);
    let error = second.start().await.expect_err("second start");
    assert!(
        error.to_string().contains("already running"),
        "error: {error}"
    );
    let current_identity = std::fs::symlink_metadata(&path).expect("lstat");
    assert!(current_identity.file_type().is_socket());
    assert_eq!(
        (current_identity.dev(), current_identity.ino()),
        (first_identity.dev(), first_identity.ino())
    );

    let client = connect_unix_test_client(&path.to_string_lossy())
        .await
        .expect("client");
    let hello = client.hello().await.expect("hello");
    assert_hello(&hello);
    client.close().await.expect("client close");
    first.close().await.expect("first close");
}

#[tokio::test]
async fn never_unlinks_a_regular_file_at_the_configured_path() {
    let directory = TempDir::new("ps");
    let path = directory.path().join("server.sock");
    std::fs::write(&path, b"do not remove").expect("write");
    let server = make_server(&path);
    let error = server.start().await.expect_err("start must refuse");
    assert!(error.to_string().contains("non-socket"), "error: {error}");
    let contents = std::fs::read(&path).expect("read");
    assert_eq!(contents, b"do not remove");
}

#[tokio::test]
async fn creates_nested_temp_parents_restricts_permissions_and_removes_its_own_socket() {
    let directory = TempDir::new("ps");
    let path = directory.path().join("p").join("n").join("server.sock");
    let server = make_server(&path);
    server.start().await.expect("start");
    let stats = std::fs::symlink_metadata(&path).expect("lstat");
    assert!(stats.file_type().is_socket());
    assert_eq!(stats.mode() & 0o777, 0o600, "socket mode");
    let mut siblings: Vec<String> = std::fs::read_dir(path.parent().expect("parent"))
        .expect("readdir")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .to_string()
        })
        .collect();
    siblings.sort();
    assert_eq!(siblings, vec!["server.sock".to_string()]);

    server.close().await.expect("close");
    let stat = std::fs::symlink_metadata(&path);
    assert!(stat.is_err(), "socket must be removed");
    assert_eq!(stat.unwrap_err().kind(), std::io::ErrorKind::NotFound);
}

#[tokio::test]
async fn does_not_remove_a_replacement_inode_during_shutdown() {
    let directory = TempDir::new("ps");
    let path = directory.path().join("server.sock");
    let server = make_server(&path);
    server.start().await.expect("start");
    std::fs::remove_file(&path).expect("unlink socket");
    std::fs::write(&path, b"replacement").expect("write replacement");

    server.close().await.expect("close");
    let contents = std::fs::read(&path).expect("read");
    assert_eq!(contents, b"replacement");
}

#[tokio::test]
async fn removes_a_genuinely_stale_socket_before_binding() {
    let directory = TempDir::new("ps");
    let path = directory.path().join("server.sock");
    // Deviation D6: recreate the fixture's final state — a socket file with
    // nothing listening behind it.
    {
        let stale = std::os::unix::net::UnixListener::bind(&path).expect("stale bind");
        let stale_identity = std::fs::symlink_metadata(&path).expect("lstat");
        assert!(stale_identity.file_type().is_socket());
        drop(stale);
    }

    let server = make_server(&path);
    server.start().await.expect("start over stale socket");
    let live_identity = std::fs::symlink_metadata(&path).expect("lstat");
    assert!(live_identity.file_type().is_socket());
    let client = connect_unix_test_client(&path.to_string_lossy())
        .await
        .expect("client");
    let hello = client.hello().await.expect("hello");
    assert_hello(&hello);
    client.close().await.expect("client close");
    server.close().await.expect("close");
}
