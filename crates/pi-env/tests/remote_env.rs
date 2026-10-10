use parking_lot::Mutex;
use pi_chord::context::BACKGROUND_CONTEXT;
use pi_durable::env::{
    ExecCommand, FileContent, FileSystem, Shell, ShellExecOptions, WatchCallback, WatchChange,
    WatchTarget,
};
use pi_env::{Connection, ConnectionOptions, RemoteExecutionEnv};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn env() -> RemoteExecutionEnv {
    let connection = Connection::new(ConnectionOptions::new([
        env!("CARGO_BIN_EXE_pi-env").to_string()
    ]));
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    RemoteExecutionEnv::new(
        connection,
        "pi-env:test",
        format!(
            "{}/pi-env-{}-{nonce}",
            std::env::temp_dir().display(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ),
    )
}

#[tokio::test]
async fn remote_files_and_shell_match_execution_env_contract() {
    let env = env();
    let context = &BACKGROUND_CONTEXT;
    env.create_dir(".", None, context).await.unwrap();
    env.write_file("data.txt", FileContent::from("one\ntwo\n"), context)
        .await
        .unwrap();
    assert_eq!(
        env.read_text_file("data.txt", context).await.unwrap(),
        "one\ntwo\n"
    );
    assert_eq!(
        env.read_text_lines("data.txt", None, context)
            .await
            .unwrap(),
        ["one", "two"]
    );
    assert_eq!(env.file_info("data.txt", context).await.unwrap().size, 8);
    env.append_file("data.txt", FileContent::from("three"), context)
        .await
        .unwrap();
    assert_eq!(
        env.read_text_file("data.txt", context).await.unwrap(),
        "one\ntwo\nthree"
    );
    let output = Arc::new(Mutex::new(String::new()));
    let received = output.clone();
    let mut options = ShellExecOptions::default();
    options.on_output = Some(Arc::new(move |chunk, _| received.lock().push_str(chunk)));
    let result = env
        .exec(
            ExecCommand::Shell("printf pi-env".into()),
            Some(options),
            context,
        )
        .await
        .unwrap();
    assert_eq!(result.exit_code, 0);
    assert_eq!(&*output.lock(), "pi-env");
    env.remove("data.txt", None, context).await.unwrap();
    env.remove(
        ".",
        Some(pi_durable::env::RemoveOptions {
            recursive: Some(true),
            force: Some(true),
        }),
        context,
    )
    .await
    .unwrap();
    env.connection.close().await;
}

#[tokio::test]
async fn remote_watcher_reports_changes_and_closes() {
    let env = env();
    let context = &BACKGROUND_CONTEXT;
    env.create_dir(".", None, context).await.unwrap();
    let (tx, rx) = mpsc::channel();
    let callback: WatchCallback = Arc::new(move |change| {
        let _ = tx.send(change);
    });
    let watcher = env
        .watch(
            &[WatchTarget {
                path: env.cwd().to_owned(),
                recursive: true,
                exclude: None,
            }],
            callback,
            context,
        )
        .await
        .unwrap();
    env.write_file("watched", FileContent::from("change"), context)
        .await
        .unwrap();
    let change =
        tokio::task::spawn_blocking(move || rx.recv_timeout(Duration::from_secs(5)).unwrap())
            .await
            .unwrap();
    assert!(
        matches!(change, WatchChange::Paths(paths) if paths.iter().any(|p| p.ends_with("watched")))
    );
    watcher.close(context).await;
    env.remove("watched", None, context).await.unwrap();
    env.connection.close().await;
}

/// Upstream "transfers only about a window of output a caller keeps, with exact counts".
#[tokio::test(flavor = "multi_thread")]
async fn transfers_about_a_window_of_output_with_exact_counts() {
    use pi_durable::env::ShellExecOptions;

    let env = env();
    let context = &BACKGROUND_CONTEXT;
    let total = 5_000_000usize;
    let delivered = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let skipped = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tail = Arc::new(Mutex::new(String::new()));
    let received = tail.clone();
    let counted_delivered = delivered.clone();
    let counted_skipped = skipped.clone();
    let mut options = ShellExecOptions::default();
    options.window = Some(pi_durable::env::ShellOutputWindow {
        max_bytes: 1000,
        max_lines: 20,
        min_interval_ms: 20,
        bytes_per_second: 1_000_000.0,
    });
    options.on_output = Some(Arc::new(
        move |text: &str, info: &pi_durable::env::ShellOutputInfo| {
            counted_delivered.fetch_add(text.len(), Ordering::SeqCst);
            if let Some(skipped_info) = info.skipped {
                counted_skipped.fetch_add(skipped_info.bytes as usize, Ordering::SeqCst);
                received.lock().clear();
            }
            received.lock().push_str(text);
        },
    ));
    let result = env
        .exec(
            ExecCommand::Argv(vec![
                "sh".into(),
                "-c".into(),
                format!("yes 0123456789abcdef | head -c {total}"),
            ]),
            Some(options),
            context,
        )
        .await
        .unwrap();
    assert_eq!(result.exit_code, 0);
    let delivered = delivered.load(Ordering::SeqCst);
    let skipped = skipped.load(Ordering::SeqCst);
    assert_eq!(delivered + skipped, total);
    assert!(skipped > 0);
    assert!(delivered < total / 10);
    // After the last skip comes everything to the end, more than the window.
    let tail = tail.lock().clone();
    let mut full = "0123456789abcdef\n".repeat(total / 17 + 1);
    full.truncate(total);
    assert!(full.ends_with(&tail));
    assert!(tail.len() > 1000 || tail.matches('\n').count() > 20);
    env.connection.close().await;
}

/// Upstream "aborts commands cancelled right after they are sent".
#[tokio::test(flavor = "multi_thread")]
async fn aborts_commands_cancelled_right_after_they_are_sent() {
    use std::time::Instant;

    use pi_chord::context::AbortController;
    use pi_durable::env::ShellExecOptions;

    let env = env();
    let context = (*BACKGROUND_CONTEXT).clone();
    let started = Instant::now();
    for attempt in 0..20 {
        let controller = AbortController::new();
        let signal = controller.signal();
        // Upstream `withAbortSignal(context, signal)`: the same context with the test's signal.
        let context_with_signal = pi_chord::context::with_abort_signal(signal, context.clone());
        let running = env.exec(
            ExecCommand::Argv(vec!["sleep".into(), "5".into()]),
            None,
            &context_with_signal,
        );
        if attempt % 2 == 0 {
            controller.abort_default();
        } else {
            tokio::time::sleep(Duration::from_millis((attempt % 5) as u64)).await;
            controller.abort_default();
        }
        let result = running.await;
        assert!(
            matches!(&result, Err(error) if error.code == pi_durable::env::ExecutionErrorCode::Aborted),
            "attempt {attempt}: {result:?}"
        );
    }
    assert!(started.elapsed() < Duration::from_secs(10));
    env.connection.close().await;
}

/// Upstream "times out a command that produces no output".
#[tokio::test(flavor = "multi_thread")]
async fn times_out_a_command_that_produces_no_output() {
    use std::time::Instant;

    use pi_durable::env::ShellExecOptions;

    let env = env();
    let context = &BACKGROUND_CONTEXT;
    let started = Instant::now();
    let result = env
        .exec(
            ExecCommand::Argv(vec!["sleep".into(), "5".into()]),
            Some(ShellExecOptions {
                timeout: Some(0.3),
                ..Default::default()
            }),
            context,
        )
        .await;
    assert!(
        matches!(&result, Err(error) if error.code == pi_durable::env::ExecutionErrorCode::Timeout)
    );
    assert!(started.elapsed() < Duration::from_secs(3));
    env.connection.close().await;
}

/// Upstream "writes and reads files larger than one transfer chunk in order".
#[tokio::test(flavor = "multi_thread")]
async fn writes_and_reads_files_larger_than_one_chunk_in_order() {
    use pi_durable::env::FileSystem;

    let env = env();
    let context = &BACKGROUND_CONTEXT;
    let content: Vec<u8> = (0..3_500_017u32)
        .map(|index| ((index * 7919) % 251) as u8)
        .collect();
    env.write_file("big.bin", FileContent::Binary(content.clone()), context)
        .await
        .unwrap();
    env.append_file("big.bin", FileContent::Binary(content.clone()), context)
        .await
        .unwrap();
    let read = env.read_binary_file("big.bin", context).await.unwrap();
    assert_eq!(read.len(), content.len() * 2);
    assert_eq!(&read[..content.len()], &content[..]);
    assert_eq!(&read[content.len()..], &content[..]);
    let reader = env
        .open_binary_reader("big.bin", None, context)
        .await
        .unwrap();
    let range = reader.read(1_000_003, 2_000_000, context).await.unwrap();
    reader.close(context).await;
    assert_eq!(range, read[1_000_003..3_000_003]);
    env.remove("big.bin", None, context).await.unwrap();
    env.connection.close().await;
}
