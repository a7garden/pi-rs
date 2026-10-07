use std::{sync::{mpsc, Arc, atomic::{AtomicUsize, Ordering}}, time::{Duration, SystemTime, UNIX_EPOCH}};
use parking_lot::Mutex;
use pi_chord::context::BACKGROUND_CONTEXT;
use pi_durable::env::{ExecCommand, FileContent, FileSystem, Shell, ShellExecOptions, WatchCallback, WatchChange, WatchTarget};
use pi_env::{Connection, ConnectionOptions, RemoteExecutionEnv};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn env() -> RemoteExecutionEnv {
    let connection = Connection::new(ConnectionOptions::new([env!("CARGO_BIN_EXE_pi-env").to_string()]));
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    RemoteExecutionEnv::new(connection, "pi-env:test", format!("{}/pi-env-{}-{nonce}", std::env::temp_dir().display(), NEXT.fetch_add(1, Ordering::SeqCst)))
}

#[tokio::test]
async fn remote_files_and_shell_match_execution_env_contract() {
    let env = env();
    let context = &BACKGROUND_CONTEXT;
    env.create_dir(".", None, context).await.unwrap();
    env.write_file("data.txt", FileContent::from("one\ntwo\n"), context).await.unwrap();
    assert_eq!(env.read_text_file("data.txt", context).await.unwrap(), "one\ntwo\n");
    assert_eq!(env.read_text_lines("data.txt", None, context).await.unwrap(), ["one", "two"]);
    assert_eq!(env.file_info("data.txt", context).await.unwrap().size, 8);
    env.append_file("data.txt", FileContent::from("three"), context).await.unwrap();
    assert_eq!(env.read_text_file("data.txt", context).await.unwrap(), "one\ntwo\nthree");
    let output = Arc::new(Mutex::new(String::new()));
    let received = output.clone();
    let mut options = ShellExecOptions::default();
    options.on_output = Some(Arc::new(move |chunk, _| received.lock().push_str(chunk)));
    let result = env.exec(ExecCommand::Shell("printf pi-env".into()), Some(options), context).await.unwrap();
    assert_eq!(result.exit_code, 0);
    assert_eq!(&*output.lock(), "pi-env");
    env.remove("data.txt", None, context).await.unwrap();
    env.remove(".", Some(pi_durable::env::RemoveOptions { recursive: Some(true), force: Some(true) }), context).await.unwrap();
    env.connection.close().await;
}

#[tokio::test]
async fn remote_watcher_reports_changes_and_closes() {
    let env = env();
    let context = &BACKGROUND_CONTEXT;
    env.create_dir(".", None, context).await.unwrap();
    let (tx, rx) = mpsc::channel();
    let callback: WatchCallback = Arc::new(move |change| { let _ = tx.send(change); });
    let watcher = env.watch(&[WatchTarget { path: env.cwd().to_owned(), recursive: true, exclude: None }], callback, context).await.unwrap();
    env.write_file("watched", FileContent::from("change"), context).await.unwrap();
    let change = tokio::task::spawn_blocking(move || rx.recv_timeout(Duration::from_secs(5)).unwrap()).await.unwrap();
    assert!(matches!(change, WatchChange::Paths(paths) if paths.iter().any(|p| p.ends_with("watched"))));
    watcher.close(context).await;
    env.remove("watched", None, context).await.unwrap();
    env.connection.close().await;
}
