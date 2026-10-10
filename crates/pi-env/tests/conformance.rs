use parking_lot::Mutex;
use pi_chord::context::BACKGROUND_CONTEXT;
use pi_durable::{
    env::{CreateDirOptions, FileSystem, RemoveOptions},
    testing::{EnvConformanceProvider, StorageConformanceRunner, register_env_conformance},
};
use pi_env::{
    Connection, ConnectionOptions, RemoteExecutionEnv, RemoteExecutionEnvOptions,
    RemoteWatchOptions,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[tokio::test(flavor = "multi_thread")]
async fn remote_execution_env_passes_durable_conformance_cases() {
    let connection = Connection::new(ConnectionOptions::new([
        env!("CARGO_BIN_EXE_pi-env").to_string()
    ]));
    let sequence = Arc::new(AtomicUsize::new(0));
    let provider_connection = connection.clone();
    let provider_sequence = sequence.clone();
    let with_env: EnvConformanceProvider = Arc::new(move |use_env| {
        let connection = provider_connection.clone();
        let index = provider_sequence.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let cwd = std::env::temp_dir()
                .join(format!("pi-env-conformance-{}-{index}", std::process::id()))
                .to_string_lossy()
                .into_owned();
            // Upstream runs the suite with `watch: { pollIntervalMs: 100 }` so watch cases settle quickly.
            let env = Arc::new(
                RemoteExecutionEnv::with_options(RemoteExecutionEnvOptions {
                    connection: Some(connection.clone()),
                    id: format!("pi-env:conformance:{index}"),
                    cwd: cwd.clone(),
                    watch: RemoteWatchOptions {
                        poll_interval_ms: Some(100),
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .expect("env builds"),
            );
            env.create_dir(
                ".",
                Some(CreateDirOptions {
                    recursive: Some(true),
                }),
                &BACKGROUND_CONTEXT,
            )
            .await
            .expect("fresh conformance cwd");
            use_env(env.clone()).await;
            env.cleanup(&BACKGROUND_CONTEXT).await;
            env.remove(
                ".",
                Some(RemoveOptions {
                    recursive: Some(true),
                    force: Some(true),
                }),
                &BACKGROUND_CONTEXT,
            )
            .await
            .expect("remove conformance cwd");
        })
    });
    let registrations = Arc::new(Mutex::new(Vec::new()));
    let registered = registrations.clone();
    let runner = StorageConformanceRunner {
        describe: Arc::new(|_, register| register()),
        expect: Arc::new(|condition, message| assert!(condition, "{message}")),
        it: Arc::new(move |name, test, _timeout_ms| {
            // `ItCase` is not `Send`, so the case runs as a spawned task and the test
            // collects its `JoinHandle` for the ordered await below.
            let handle = tokio::spawn(test());
            registered.lock().push((name.to_owned(), handle));
        }),
    };
    register_env_conformance(&runner, "RemoteExecutionEnv", with_env, None, None);
    let tests = std::mem::take(&mut *registrations.lock());
    assert!(!tests.is_empty());
    for (name, handle) in tests {
        let result = tokio::time::timeout(std::time::Duration::from_secs(40), handle)
            .await
            .unwrap_or_else(|_| panic!("conformance case timed out: {name}"));
        match result {
            Ok(()) => {}
            Err(error) if error.is_panic() => {
                let payload = error.into_panic();
                let message = payload
                    .downcast_ref::<&str>()
                    .map(|s| *s)
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("?");
                panic!("case '{name}' panicked: {message}");
            }
            Err(error) => panic!("conformance case {name} did not complete: {error}"),
        }
    }
    connection.close().await;
}
