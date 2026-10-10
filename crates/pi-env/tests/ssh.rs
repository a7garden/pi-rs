//! SSH bootstrap against a disposable local `sshd` (upstream `packages/env/test/ssh.test.ts`): its own
//! host key, client key and home directory; daemon deployment verified by content; host-key trust
//! handled explicitly. Skipped when no `sshd` is available (upstream skips on Windows).

use std::{
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
    time::Duration,
};

use parking_lot::Mutex;
use pi_chord::context::BACKGROUND_CONTEXT;
use pi_durable::env::{FileSystem, Shell};
use pi_env::{
    RemoteExecutionEnv, SshCommandError, SshConnectOptions, accept_host_key, connect_ssh,
    forget_host_key, scan_host_key, ssh_connection,
};
use sha2::{Digest, Sha256};

/// Whether an `sshd` binary is available on this machine.
fn which(program: &str) -> Option<String> {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {program}"))
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|path| !path.is_empty())
}

fn sshd_path() -> Option<String> {
    which("sshd").or_else(|| {
        Path::new("/usr/sbin/sshd")
            .exists()
            .then(|| "/usr/sbin/sshd".to_string())
    })
}

/// A free localhost port for the disposable server.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn wait_for_port(port: u16) {
    for _ in 0..100 {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("sshd did not start");
}

struct Server {
    root: PathBuf,
    target: pi_env::SshTarget,
    port: u16,
    home: PathBuf,
    child: std::process::Child,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn start_server() -> Option<Server> {
    let sshd = sshd_path()?;
    for program in ["ssh-keygen"] {
        if which(program).is_none() {
            return None;
        }
    }
    let root = std::env::temp_dir().join(format!(
        "pi-env-ssh-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let home = root.join("home");
    fs::create_dir_all(&home).unwrap();
    Command::new("ssh-keygen")
        .args(["-q", "-t", "ed25519", "-N", "", "-f"])
        .arg(root.join("host_key"))
        .status()
        .unwrap();
    Command::new("ssh-keygen")
        .args(["-q", "-t", "ed25519", "-N", "", "-f"])
        .arg(root.join("client_key"))
        .status()
        .unwrap();
    fs::write(
        root.join("authorized_keys"),
        fs::read(root.join("client_key.pub")).unwrap(),
    )
    .unwrap();
    let port = free_port();
    fs::write(
        root.join("sshd_config"),
        [
            format!("Port {port}"),
            "ListenAddress 127.0.0.1".to_string(),
            format!("HostKey {}", root.join("host_key").display()),
            format!(
                "AuthorizedKeysFile {}",
                root.join("authorized_keys").display()
            ),
            format!("PidFile {}", root.join("sshd.pid").display()),
            "PasswordAuthentication no".to_string(),
            "KbdInteractiveAuthentication no".to_string(),
            "StrictModes no".to_string(),
            "UsePAM no".to_string(),
            format!("SetEnv HOME={}", home.display()),
            String::new(),
        ]
        .join("\n"),
    )
    .unwrap();
    fs::write(root.join("ssh_config"), "").unwrap();
    let child = Command::new(&sshd)
        .args(["-D", "-e", "-f"])
        .arg(root.join("sshd_config"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_for_port(port);
    Some(Server {
        root: root.clone(),
        target: pi_env::SshTarget {
            host: "127.0.0.1".into(),
            port: Some(port),
            user: Some(
                std::env::var("USER")
                    .ok()
                    .or_else(whoami)
                    .unwrap_or_else(|| "nobody".into()),
            ),
            identity_file: Some(root.join("client_key").to_string_lossy().into_owned()),
            known_hosts_file: root.join("known_hosts").to_string_lossy().into_owned(),
            host_key_alias: "pi-env-test".into(),
            config_file: Some(root.join("ssh_config").to_string_lossy().into_owned()),
            ..Default::default()
        },
        port,
        home,
        child,
    })
}

fn whoami() -> Option<String> {
    let output = Command::new("id").arg("-un").output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn options_with(server: &Server) -> SshConnectOptions {
    SshConnectOptions {
        host: server.target.host.clone(),
        user: server.target.user.clone(),
        port: server.target.port,
        identity_file: server.target.identity_file.clone(),
        known_hosts_file: server.target.known_hosts_file.clone(),
        host_key_alias: server.target.host_key_alias.clone(),
        ssh: server.target.ssh.clone(),
        config_file: server.target.config_file.clone(),
        binary: Some(PathBuf::from(env!("CARGO_BIN_EXE_pi-env"))),
        login_shell: false,
        on_log: None,
    }
}

/// The deployed daemon is named by its content.
fn deployed_name() -> String {
    let bytes = fs::read(env!("CARGO_BIN_EXE_pi-env")).unwrap();
    format!("pi-env-{}", &hex(&Sha256::digest(&bytes))[..32])
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The server and host-key trust are shared by the cases; each test sets them up itself because a
/// failed case must not poison the next one's known-hosts state.
fn trusted_server() -> Option<(Server, tokio::runtime::Runtime)> {
    let server = start_server()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        match scan_host_key(&server.target).await {
            Ok((lines, _fingerprints)) => accept_host_key(&server.target, &lines).await.unwrap(),
            Err(error) => panic!("scan the local sshd's host key: {error}"),
        }
    });
    Some((server, runtime))
}

#[test]
fn ssh_bootstrap_flows() {
    // Upstream gates the whole suite on `sshd` being available.
    let Some((server, runtime)) = trusted_server() else {
        eprintln!("skipping: no sshd or ssh-keygen on this machine");
        return;
    };
    // The suite is one serialized flow; upstream runs the same order in one describe block.
    runtime.block_on(async {
		// "refuses an untrusted host, then deploys and runs the daemon once its key is accepted" —
		// the trust step happened in `trusted_server`; first prove a *fresh* known-hosts refuses.
		let fresh = options_with(&server);
		let fresh_known = format!("{}.untrusted", server.target.known_hosts_file);
		let mut untrusted = fresh.clone();
		untrusted.known_hosts_file = fresh_known.clone();
        match connect_ssh(untrusted).await {
            Err(SshCommandError::HostKeyUnknown(_)) => {}
            other => panic!(
                "untrusted host must fail with HostKeyUnknownError: {}",
                other
                    .err()
                    .map(|error| error.to_string())
                    .unwrap_or_else(|| "unexpected success".to_string())
            ),
        }

		// Pre-seed an old daemon file; deployment replaces it and removes other contents.
		let tools = server.home.join(".pi/mobile/tools");
		fs::create_dir_all(&tools).unwrap();
		fs::write(tools.join("pi-env-0123456789abcdef0123456789abcdef"), b"old").unwrap();
		let deployed = deployed_name();
        let (connection_options, remote) = connect_ssh(options_with(&server)).await.unwrap();
        let connection = pi_env::Connection::new(connection_options);
		assert_eq!(remote.home, server.home.to_string_lossy(), "sshd's SetEnv HOME must reach the session");
		assert_eq!(fs::read(tools.join(&deployed)).unwrap(), fs::read(env!("CARGO_BIN_EXE_pi-env")).unwrap());
		let leftovers: Vec<String> = fs::read_dir(&tools)
			.unwrap()
			.filter_map(|entry| entry.ok())
			.map(|entry| entry.file_name().to_string_lossy().into_owned())
			.filter(|name| name.starts_with("pi-env-"))
			.collect();
		assert_eq!(leftovers, [deployed.clone()]);
		let env = RemoteExecutionEnv::new(connection.clone(), "pi-env:test", server.home.to_string_lossy());
		env.write_file("over-ssh.txt", pi_durable::env::FileContent::from("hello"), &BACKGROUND_CONTEXT).await.unwrap();
		assert_eq!(env.read_text_file("over-ssh.txt", &BACKGROUND_CONTEXT).await.unwrap(), "hello");
		let output = Arc::new(Mutex::new(String::new()));
		let received = output.clone();
		let mut options = pi_durable::env::ShellExecOptions::default();
		options.on_output = Some(Arc::new(move |text, _| received.lock().push_str(text.to_string().as_str())));
		let result = env
			.exec(
				pi_durable::env::ExecCommand::Argv(vec!["sh".into(), "-c".into(), "printf \"%s\" \"$HOME\"".into()]),
				Some(options),
				&BACKGROUND_CONTEXT,
			)
			.await
			.unwrap();
		assert_eq!(result.exit_code, 0);
		assert_eq!(&*output.lock(), &server.home.to_string_lossy().to_string());
		connection.close().await;

		// "reuses a verified daemon and replaces a tampered one".
		let before = fs::metadata(tools.join(&deployed)).unwrap().modified().unwrap();
        let (connection_options, _) = connect_ssh(options_with(&server)).await.unwrap();
        let connection = pi_env::Connection::new(connection_options);
		assert_eq!(fs::metadata(tools.join(&deployed)).unwrap().modified().unwrap(), before);
		connection.close().await;
		fs::write(tools.join(&deployed), b"tampered").unwrap();
        let (connection_options, _) = connect_ssh(options_with(&server)).await.unwrap();
        let connection = pi_env::Connection::new(connection_options);
		assert_eq!(fs::read(tools.join(&deployed)).unwrap(), fs::read(env!("CARGO_BIN_EXE_pi-env")).unwrap());
		assert_eq!(connection.info().await.unwrap().home, server.home.to_string_lossy());
		connection.close().await;

		// "verifies the daemon again before starting it after a lost connection".
        let (connection_options, _) = connect_ssh(options_with(&server)).await.unwrap();
        let connection = pi_env::Connection::new(connection_options);
		let pid = connection.info().await.unwrap().pid;
		std::process::Command::new("kill").args(["-9", &pid.to_string()]).status().unwrap();
		tokio::time::sleep(Duration::from_millis(300)).await;
		fs::write(tools.join(&deployed), b"tampered").unwrap();
		assert_ne!(connection.info().await.unwrap().pid, pid);
		assert_eq!(fs::read(tools.join(&deployed)).unwrap(), fs::read(env!("CARGO_BIN_EXE_pi-env")).unwrap());
		connection.close().await;

		// "starts the daemon through the login shell only when asked".
		fs::write(server.home.join(".profile"), "export PI_ENV_LOGIN=yes\n").unwrap();
		fs::write(server.home.join(".zprofile"), "export PI_ENV_LOGIN=yes\n").unwrap();
		let login = |login_shell: bool| {
			let server = &server;
			async move {
				let mut options = options_with(server);
				options.login_shell = login_shell;
                let (connection_options, _) = connect_ssh(options).await.unwrap();
                let connection = pi_env::Connection::new(connection_options);
				let env = RemoteExecutionEnv::new(connection.clone(), "pi-env:test", server.home.to_string_lossy());
				let output = Arc::new(Mutex::new(String::new()));
				let received = output.clone();
				let mut options = pi_durable::env::ShellExecOptions::default();
				options.on_output = Some(Arc::new(move |text, _| received.lock().push_str(text.to_string().as_str())));
				let result = env
					.exec(
						pi_durable::env::ExecCommand::Argv(vec!["sh".into(), "-c".into(), "printf \"%s\" \"$PI_ENV_LOGIN\"".into()]),
						Some(options),
						&BACKGROUND_CONTEXT,
					)
					.await
					.unwrap();
				connection.close().await;
                if result.exit_code == 0 {
                    output.lock().clone()
                } else {
                    String::new()
                }
			}
		};
		assert_eq!(login(false).await, "");
		assert_eq!(login(true).await, "yes");

		// "connects lazily and reports a failed start as the error of the operation".
		let lazy_known = format!("{}.lazy", server.target.known_hosts_file);
		let mut lazy = options_with(&server);
		lazy.known_hosts_file = lazy_known.clone();
		let (lazy_options, remote_state) = ssh_connection(lazy.clone());
		assert!(remote_state.remote().is_none());
		let lazy_target = lazy.target();
		let lazy_connection = pi_env::Connection::new(lazy_options);
		let lazy_env = RemoteExecutionEnv::new(lazy_connection.clone(), "pi-env:test", server.home.to_string_lossy());
		let read = lazy_env.read_text_file("missing.txt", &BACKGROUND_CONTEXT).await;
		match read {
			Err(error) => assert!(error.message.contains("not trusted"), "untrusted lazy start: {}", error.message),
			Ok(text) => panic!("lazy untrusted start answered: {text}"),
		}
		let command = lazy_env
			.exec(pi_durable::env::ExecCommand::Argv(vec!["sh".into(), "-c".into(), "exit 0".into()]), None, &BACKGROUND_CONTEXT)
			.await;
		assert!(matches!(command, Err(error) if error.code == pi_durable::env::ExecutionErrorCode::SpawnError));
		let (lines, _) = scan_host_key(&lazy_target).await.unwrap();
		accept_host_key(&lazy_target, &lines).await.unwrap();
		let accepted = lazy_env
			.exec(pi_durable::env::ExecCommand::Argv(vec!["sh".into(), "-c".into(), "exit 0".into()]), None, &BACKGROUND_CONTEXT)
			.await;
		assert_eq!(accepted.unwrap().exit_code, 0);
		assert_eq!(remote_state.remote().unwrap().home, server.home.to_string_lossy());
		lazy_connection.close().await;

		// "accepts only host keys for the alias and never replaces a trusted key silently".
		let accept_known = format!("{}.accept", server.target.known_hosts_file);
		let mut scoped = server.target.clone();
		scoped.known_hosts_file = accept_known.clone();
		let first = key_of(&server.root, "accept_first");
		let second = key_of(&server.root, "accept_second");
		assert!(accept_host_key(&scoped, &[format!("other-alias {first}")]).await.is_err());
		assert!(accept_host_key(&scoped, &[format!("@cert-authority pi-env-test {first}")]).await.is_err());
        let accepted_lines = [format!("pi-env-test {first}")];
        let concurrent_lines = accepted_lines.clone();
        let (accepted, _) = tokio::join!(
            accept_host_key(&scoped, &accepted_lines),
            accept_host_key(&scoped, &concurrent_lines)
        );
        accepted.unwrap();
		assert_eq!(fs::read_to_string(&accept_known).unwrap(), format!("pi-env-test {first}\n"));
		assert!(matches!(
			accept_host_key(&scoped, &[format!("pi-env-test {second}")]).await,
			Err(SshCommandError::HostKeyChanged(_))
		));
		forget_host_key(&scoped).await.unwrap();
		accept_host_key(&scoped, &[format!("pi-env-test {second}")]).await.unwrap();
		assert_eq!(fs::read_to_string(&accept_known).unwrap(), format!("pi-env-test {second}\n"));

		// "refuses a changed host key".
		let other = key_of(&server.root, "other_key");
		let changed_known = format!("{}.changed", server.target.known_hosts_file);
		fs::write(&changed_known, format!("pi-env-test {other}\n")).unwrap();
		let mut changed = options_with(&server);
		changed.known_hosts_file = changed_known;
		assert!(matches!(connect_ssh(changed).await, Err(SshCommandError::HostKeyChanged(_))));
	});
}

/// `ssh-keygen -t ed25519 -N ""` and the first two fields of the public key (`type key`).
fn key_of(root: &Path, name: &str) -> String {
    let file = root.join(name);
    let _ = fs::remove_file(&file);
    Command::new("ssh-keygen")
        .args(["-q", "-t", "ed25519", "-N", "", "-f"])
        .arg(&file)
        .status()
        .unwrap();
    let public = fs::read_to_string(root.join(format!("{name}.pub"))).unwrap();
    public
        .split_whitespace()
        .take(2)
        .collect::<Vec<&str>>()
        .join(" ")
}
