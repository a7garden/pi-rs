//! Client behavior against the real daemon (upstream `remote.test.ts`'s connection-level concerns).

use parking_lot::Mutex;
use pi_chord::context::AbortController;
use pi_durable::env::Shell;
use pi_env::{Connection, ConnectionOptions, Json, RequestOptions};
use serde_json::json;
use std::{sync::Arc, time::Duration};

fn connection() -> Connection {
    Connection::new(ConnectionOptions::new([
        env!("CARGO_BIN_EXE_pi-env").to_owned()
    ]))
}

fn exec_fields(cwd: &str) -> Json {
    let mut fields = Json::new();
    fields.insert("command".into(), json!("sleep 30"));
    fields.insert("cwd".into(), json!(cwd));
    fields.insert("env".into(), json!({}));
    fields.insert("inheritEnv".into(), json!(true));
    fields
}

#[tokio::test]
async fn abort_sends_cancel_and_waits_for_daemon_settlement() {
    let connection = connection();
    let cwd = std::env::current_dir()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let controller = AbortController::new();
    let signal = controller.signal();
    let started: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));
    let started_task = started.clone();
    let mut options = RequestOptions::default();
    options.on_start = Some(Arc::new(move |id, _session| {
        *started_task.lock() = Some(id);
    }));
    let request_connection = connection.clone();
    let task = tokio::spawn(async move {
        request_connection
            .request_with_signal("exec", exec_fields(&cwd), options, Some(signal))
            .await
    });
    // Abort only once the exec is in flight (a cold daemon start cannot receive the cancel first),
    // so the cancel always reaches the daemon's registered request.
    tokio::time::timeout(Duration::from_secs(10), async {
        while started.lock().is_none() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    controller.abort_default();
    let error = task.await.unwrap().unwrap_err();
    assert_eq!(error.code, "aborted");
    connection.close().await;
}

#[cfg(unix)]
#[tokio::test]
async fn restarts_daemon_after_child_exits() {
    let connection = connection();
    let first = connection.info().await.unwrap();
    let status = std::process::Command::new("kill")
        .args(["-9", &first.pid.to_string()])
        .status()
        .unwrap();
    assert!(status.success());
    tokio::time::sleep(Duration::from_millis(300)).await;
    let second = connection.info().await.unwrap();
    assert_ne!(first.pid, second.pid);
    connection.close().await;
}

#[tokio::test]
async fn close_fails_in_flight_requests() {
    let connection = connection();
    let cwd = std::env::current_dir()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let task_connection = connection.clone();
    let task = tokio::spawn(async move {
        task_connection
            .request("exec", exec_fields(&cwd), RequestOptions::default())
            .await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    connection.close().await;
    let error = task.await.unwrap().unwrap_err();
    assert_eq!(error.code, "unknown");
}

/// A stale handle (opened in a daemon that died) fails instead of reaching the new daemon, whose
/// handles start numbering from 1 again (upstream `remote.test.ts`).
#[cfg(unix)]
#[tokio::test]
async fn session_scoped_requests_fail_after_the_daemon_is_lost() {
    use pi_chord::context::BACKGROUND_CONTEXT;
    use pi_durable::env::FileSystem;
    use pi_env::RemoteExecutionEnv;

    let connection = connection();
    let cwd = std::env::temp_dir()
        .join(format!("pi-env-session-{}", std::process::id()))
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(&cwd).unwrap();
    let env = RemoteExecutionEnv::new(connection.clone(), "pi-env:test", cwd.clone());
    env.write_file(
        "a.txt",
        pi_durable::env::FileContent::from("AAA"),
        &BACKGROUND_CONTEXT,
    )
    .await
    .unwrap();
    env.write_file(
        "b.txt",
        pi_durable::env::FileContent::from("BBB"),
        &BACKGROUND_CONTEXT,
    )
    .await
    .unwrap();
    let a = env
        .open_binary_reader("a.txt", None, &BACKGROUND_CONTEXT)
        .await
        .unwrap();
    let pid = connection.info().await.unwrap().pid;
    std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()
        .unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    // The new daemon numbers its handles from 1 again.
    let b = env
        .open_binary_reader("b.txt", None, &BACKGROUND_CONTEXT)
        .await
        .unwrap();
    assert_ne!(connection.info().await.unwrap().pid, pid);
    let stale = a.read(0, 3, &BACKGROUND_CONTEXT).await;
    match stale {
        Err(error) => assert_eq!(error.code, pi_durable::env::FileErrorCode::Unknown),
        Ok(bytes) => panic!("stale handle answered: {bytes:?}"),
    }
    assert_eq!(b.read(0, 3, &BACKGROUND_CONTEXT).await.unwrap(), b"BBB");
    a.close(&BACKGROUND_CONTEXT).await;
    b.close(&BACKGROUND_CONTEXT).await;
    let _ = std::fs::remove_dir_all(&cwd);
    connection.close().await;
}

/// A computed command that fails once fails that start with its message, and the next request tries
/// again (upstream `remote.test.ts`).
#[tokio::test]
async fn computed_command_retries_a_failed_start() {
    use pi_chord::context::BACKGROUND_CONTEXT;
    use pi_durable::env::FileSystem;
    use pi_env::{CommandSource, RemoteExecutionEnv};

    let cwd = std::env::temp_dir()
        .join(format!("pi-env-retry-{}", std::process::id()))
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(&cwd).unwrap();
    let starts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let daemon = env!("CARGO_BIN_EXE_pi-env").to_string();
    let starts_provider = starts.clone();
    let mut options = ConnectionOptions::new([daemon.clone()]);
    options.command = CommandSource::Computed(Arc::new(move || {
        let daemon = daemon.clone();
        let starts = starts_provider.clone();
        Box::pin(async move {
            if starts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                return Err("ssh gpu-box failed: Network is unreachable".to_string());
            }
            Ok(vec![daemon])
        })
    }));
    let connection = Connection::new(options);
    let env = RemoteExecutionEnv::new(connection.clone(), "pi-env:test", cwd.clone());
    let offline = env.read_text_file("missing.txt", &BACKGROUND_CONTEXT).await;
    match offline {
        Err(error) => {
            assert_eq!(error.code, pi_durable::env::FileErrorCode::Unknown);
            assert_eq!(error.message, "ssh gpu-box failed: Network is unreachable");
        }
        Ok(text) => panic!("offline start answered: {text}"),
    }
    let command = env
        .exec(
            pi_durable::env::ExecCommand::Argv(vec!["sh".into(), "-c".into(), "exit 0".into()]),
            None,
            &BACKGROUND_CONTEXT,
        )
        .await;
    assert_eq!(command.unwrap().exit_code, 0);
    assert_eq!(starts.load(std::sync::atomic::Ordering::SeqCst), 2);
    let _ = std::fs::remove_dir_all(&cwd);
    connection.close().await;
}
