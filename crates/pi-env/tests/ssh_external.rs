//! The CI suite against a real SSH server (upstream `packages/env/test/ssh-external.test.ts`).
//!
//! Everything is gated on `PI_ENV_SSH_HOST`: with it unset (the default) both tests return without any
//! network activity. When set, the target is the host's own server, configured by:
//!
//! - `PI_ENV_SSH_HOST` (required to opt in), `PI_ENV_SSH_USER`, `PI_ENV_SSH_PORT`,
//!   `PI_ENV_SSH_KEY` (identity file), `PI_ENV_SSH_PROGRAM` (the `ssh` binary),
//!   `PI_ENV_SSH_SHELL` (expected Windows default shell, default `cmd`).
//!
//! The daemon deployed is the one this test built (`CARGO_BIN_EXE_pi-env`); no packaged cross-target
//! artifacts are involved, so this is also the example of explicit-binary deployment — the packaged
//! path (`packaged_daemon`, driven by `PI_ENV_PACKAGE_ROOT`) is only exercised when no `binary` is
//! passed. On Windows this covers deployment and the daemon's launch through the server's default
//! shell. Each stage is logged to stderr, so a hang in CI shows where it is.

use std::{
    fs,
    path::PathBuf,
    process,
    sync::{Arc, OnceLock},
    time::SystemTime,
};

use parking_lot::Mutex;
use pi_chord::context::BACKGROUND_CONTEXT;
use pi_durable::{
    env::{FileSystem, RemoveOptions},
    testing::{EnvConformanceProvider, StorageConformanceRunner, register_env_conformance},
};
use pi_env::{
    Connection, RemoteExecutionEnv, RemoteExecutionEnvOptions, RemoteWatchOptions, SshCommandError,
    SshConnectOptions, SshTarget, accept_host_key, connect_ssh, deploy_daemon, detect_platform,
    scan_host_key,
};

fn host() -> Option<String> {
    std::env::var("PI_ENV_SSH_HOST")
        .ok()
        .filter(|host| !host.is_empty())
}

fn target_for(known_hosts_file: String) -> SshTarget {
    SshTarget {
        host: host().unwrap(),
        user: std::env::var("PI_ENV_SSH_USER")
            .ok()
            .filter(|user| !user.is_empty()),
        port: std::env::var("PI_ENV_SSH_PORT")
            .ok()
            .and_then(|port| port.parse().ok()),
        identity_file: std::env::var("PI_ENV_SSH_KEY")
            .ok()
            .filter(|key| !key.is_empty()),
        ssh: std::env::var("PI_ENV_SSH_PROGRAM")
            .ok()
            .filter(|program| !program.is_empty()),
        known_hosts_file,
        host_key_alias: "pi-env-external".into(),
        config_file: None,
    }
}

fn temp_known_hosts() -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "pi-env-ssh-external-{}-{}",
        process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    root.join("known_hosts")
}

fn log(stage: &str) {
    eprintln!("[ssh-external {}] {stage}", humantime_millis());
}

fn humantime_millis() -> String {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .to_string()
}

struct External {
    connection: Connection,
    remote: pi_env::RemotePlatform,
    known_hosts_dir: PathBuf,
}

impl Drop for External {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.known_hosts_dir);
    }
}

/// The staged connect of upstream's `connected` promise: scan and trust the host key, detect the
/// platform, deploy the daemon (explicit binary), connect, and say hello.
async fn connect() -> Result<External, SshCommandError> {
    let known_hosts = temp_known_hosts();
    let target = target_for(known_hosts.to_string_lossy().into_owned());
    log("scanning the host key");
    accept_host_key(&target, &scan_host_key(&target).await?.0).await?;
    log("detecting the platform");
    let remote = detect_platform(&target).await?;
    log(&format!("detected {remote:?}; deploying"));
    let deployed = deploy_daemon(
        &target,
        &remote,
        PathBuf::from(env!("CARGO_BIN_EXE_pi-env")).as_path(),
    )
    .await?;
    log(&format!("deployed {deployed}; connecting"));
    let log_sink: Arc<dyn Fn(&str) + Send + Sync> =
        Arc::new(|text| log(&format!("daemon: {}", text.trim())));
    let options = SshConnectOptions {
        host: target.host.clone(),
        user: target.user.clone(),
        port: target.port,
        identity_file: target.identity_file.clone(),
        known_hosts_file: target.known_hosts_file.clone(),
        host_key_alias: target.host_key_alias.clone(),
        ssh: target.ssh.clone(),
        config_file: target.config_file.clone(),
        // Explicit binary: this crate ships no packaged cross-target artifacts (see `packaged_daemon`).
        binary: Some(PathBuf::from(env!("CARGO_BIN_EXE_pi-env"))),
        login_shell: false,
        on_log: Some(log_sink),
    };
    let (connection_options, remote_platform) = connect_ssh(options).await?;
    let connection = Connection::new(connection_options);
    log("connected; saying hello");
    let info = connection
        .info()
        .await
        .map_err(|error| SshCommandError::from(error.to_string()))?;
    log(&format!("hello pid {}", info.pid));
    Ok(External {
        connection,
        remote: remote_platform,
        known_hosts_dir: known_hosts.parent().unwrap().to_path_buf(),
    })
}

fn external_runtime() -> &'static Mutex<Option<External>> {
    static EXTERNAL: OnceLock<Mutex<Option<External>>> = OnceLock::new();
    EXTERNAL.get_or_init(|| Mutex::new(None))
}

fn external_init() -> &'static tokio::sync::Mutex<()> {
    static INITIALIZER: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    INITIALIZER.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Take the shared connection, establishing it once per test process.
async fn shared() -> Option<(Connection, pi_env::RemotePlatform)> {
    let _initialization = external_init().lock().await;
    if external_runtime().lock().is_none() {
        let connected = connect().await.ok();
        *external_runtime().lock() = connected;
    }
    external_runtime()
        .lock()
        .as_ref()
        .map(|external| (external.connection.clone(), external.remote.clone()))
}

/// Upstream "reaches the remote system through its default shell".
#[test]
fn reaches_the_remote_system_through_its_default_shell() {
    if host().is_none() {
        eprintln!("skipping: PI_ENV_SSH_HOST is not set");
        return;
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let Some((connection, remote)) = shared().await else {
            panic!("PI_ENV_SSH_HOST is set but the connection failed");
        };
        let info = connection.info().await.unwrap();
        let expected = if cfg!(windows) {
            "windows"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else {
            std::env::consts::OS
        };
        assert_eq!(info.os, expected);
        if cfg!(windows) {
            let expected_shell = std::env::var("PI_ENV_SSH_SHELL").unwrap_or_else(|_| "cmd".into());
            assert_eq!(
                remote.shell.map(|shell| match shell {
                    pi_env::RemoteShell::Cmd => "cmd".to_string(),
                    pi_env::RemoteShell::PowerShell => "powershell".to_string(),
                }),
                Some(expected_shell)
            );
        }
    });
}

/// Upstream's `registerEnvConformance("RemoteExecutionEnv over SSH", ...)`: the whole env conformance
/// suite against the remote host's file system, in a temporary directory of its home.
#[test]
fn passes_the_conformance_suite_over_ssh() {
    if host().is_none() {
        eprintln!("skipping: PI_ENV_SSH_HOST is not set");
        return;
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let with_env: EnvConformanceProvider = Arc::new(|use_env| {
        Box::pin(async move {
            let Some((connection, remote)) = shared().await else {
                panic!("PI_ENV_SSH_HOST is set but the connection failed");
            };
            let home =
                RemoteExecutionEnv::new(connection.clone(), "pi-env:external", remote.home.clone());
            let cwd = home
                .create_temp_dir(Some("pi-env-conformance-"), &BACKGROUND_CONTEXT)
                .await
                .expect("temporary directory on the remote host");
            // Upstream runs the suite with `watch: { pollIntervalMs: 100 }`.
            let env = Arc::new(
                RemoteExecutionEnv::with_options(RemoteExecutionEnvOptions {
                    connection: Some(connection.clone()),
                    id: "pi-env:external".into(),
                    cwd: cwd.clone(),
                    shell_path: None,
                    shell_env: None,
                    watch: RemoteWatchOptions {
                        poll_interval_ms: Some(100),
                        ..Default::default()
                    },
                })
                .expect("env builds"),
            );
            use_env(env.clone()).await;
            home.remove(
                &cwd,
                Some(RemoveOptions {
                    recursive: Some(true),
                    force: Some(true),
                }),
                &BACKGROUND_CONTEXT,
            )
            .await
            .expect("remove the conformance directory");
        })
    });
    let registrations = Arc::new(Mutex::new(Vec::new()));
    let registered = registrations.clone();
    let runner = StorageConformanceRunner {
        describe: Arc::new(|_, register| register()),
        expect: Arc::new(|condition, message| assert!(condition, "{message}")),
        it: Arc::new(move |name, test, _timeout_ms| {
            let handle = tokio::spawn(test());
            registered.lock().push((name.to_owned(), handle));
        }),
    };
    register_env_conformance(&runner, "RemoteExecutionEnv over SSH", with_env, None, None);
    let tests = std::mem::take(&mut *registrations.lock());
    assert!(!tests.is_empty());
    runtime.block_on(async {
        for (name, handle) in tests {
            let result = tokio::time::timeout(std::time::Duration::from_secs(120), handle)
                .await
                .unwrap_or_else(|_| panic!("conformance case timed out: {name}"));
            match result {
                Ok(()) => {}
                Err(error) if error.is_panic() => {
                    let payload = error.into_panic();
                    let message = payload
                        .downcast_ref::<&str>()
                        .map(|message| *message)
                        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or_else(|| "?".into());
                    panic!("case '{name}' panicked: {message}");
                }
                Err(error) => panic!("conformance case {name} did not complete: {error}"),
            }
        }
    });
}
