//! Remote execution over the system `ssh` (upstream `packages/env/src/ssh.ts`): host keys are checked
//! strictly against the application's own known-hosts file under a fixed alias, the remote system is
//! detected through its login shell (or PowerShell on Windows), and the daemon is deployed there,
//! verified by its SHA-256 before it ever runs.
//!
//! Workspace adaptation: upstream serves the daemon binaries from the npm package's `bin/` directory,
//! which only exists in the published package. This crate ships no packaged cross-target binaries;
//! [`packaged_daemon`] resolves the conventional path under `PI_ENV_PACKAGE_ROOT`, and callers either
//! set that, or pass an explicit `binary` ([`SshConnectOptions::binary`], [`deploy_daemon`]).

use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, LazyLock},
    time::Duration,
};

use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::Command,
};

use crate::connection::{CommandSource, Connection, ConnectionOptions};

/// How long a detection, check or upload over `ssh` may take before it is abandoned.
const SSH_TIMEOUT: Duration = Duration::from_secs(60);
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(300);

/// A remote system the daemon runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Linux,
    Android,
    Darwin,
    Windows,
}

impl Platform {
    pub fn as_str(self) -> &'static str {
        match self {
            Platform::Linux => "linux",
            Platform::Android => "android",
            Platform::Darwin => "darwin",
            Platform::Windows => "windows",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X64,
    Arm64,
}

impl Arch {
    pub fn as_str(self) -> &'static str {
        match self {
            Arch::X64 => "x64",
            Arch::Arm64 => "arm64",
        }
    }
}

/// The shell `sshd` runs commands with on Windows (`DefaultShell`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteShell {
    Cmd,
    PowerShell,
}

/// What was learned about a remote system.
#[derive(Debug, Clone)]
pub struct RemotePlatform {
    pub platform: Platform,
    pub arch: Arch,
    /// The remote home directory, in the remote system's own spelling.
    pub home: String,
    /// Windows only.
    pub shell: Option<RemoteShell>,
    /// Problems worth telling the owner about, such as Termux without `termux-exec`.
    pub warnings: Vec<String>,
}

/// How to reach a machine with the system `ssh`.
#[derive(Debug, Clone, Default)]
pub struct SshTarget {
    /// Host name or `~/.ssh/config` alias.
    pub host: String,
    pub user: Option<String>,
    pub port: Option<u16>,
    pub identity_file: Option<String>,
    /// Host keys this application trusts; `accept_host_key` adds to it.
    pub known_hosts_file: String,
    /// The name keys are stored under, independent of aliases, ports and jump hosts, e.g. `pi-env-<env name>`.
    pub host_key_alias: String,
    /// The `ssh` program; default `ssh`.
    pub ssh: Option<String>,
    /// An `ssh` configuration file instead of `~/.ssh/config` (`-F`).
    pub config_file: Option<String>,
}

/// The remote host's key is not in the known-hosts file; `scan_host_key` shows it so the owner can accept it.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct HostKeyUnknownError(pub String);

/// The remote host's key differs from the one in the known-hosts file; it is never accepted automatically.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct HostKeyChangedError(pub String);

/// An `ssh` invocation that failed for another reason, with its exit code and diagnostics.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct SshError {
    pub message: String,
    pub exit_code: Option<i32>,
    pub stderr: String,
}

/// What an `ssh` invocation can fail with.
#[derive(Debug, thiserror::Error)]
pub enum SshCommandError {
    #[error(transparent)]
    Ssh(#[from] SshError),
    #[error(transparent)]
    HostKeyUnknown(#[from] HostKeyUnknownError),
    #[error(transparent)]
    HostKeyChanged(#[from] HostKeyChangedError),
    /// A plain error, such as an unexpected probe answer or an unsupported architecture.
    #[error("{0}")]
    Other(String),
}

impl From<String> for SshCommandError {
    fn from(message: String) -> SshCommandError {
        SshCommandError::Other(message)
    }
}

/// No leading `-` (it would be read as an `ssh` option) and no whitespace or control characters.
fn check_field(name: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.starts_with('-')
        || value
            .chars()
            .any(|character| character.is_whitespace() || character.is_control())
    {
        return Err(format!("Invalid {name}: {value:?}"));
    }
    Ok(())
}

/// A path for an `ssh` option that expands `%` tokens: quoted, with `%` literal.
fn config_path(name: &str, path: &str) -> Result<String, String> {
    if path.is_empty() || path.contains('"') || path.chars().any(|character| character.is_control())
    {
        return Err(format!("Invalid {name}: {path:?}"));
    }
    Ok(format!("\"{}\"", path.replace('%', "%%")))
}

/// Arguments for `ssh` up to the host: no prompts, no forwarding of any kind, no shared connections, no
/// commands from the configuration, no locale forwarding (the remote uses its own), and host keys checked
/// strictly against the application's own file under a fixed alias.
pub fn ssh_arguments(
    target: &SshTarget,
    strict_host_keys: bool,
    known_hosts_file: Option<&str>,
) -> Result<Vec<String>, String> {
    check_field("host", &target.host)?;
    check_field("host key alias", &target.host_key_alias)?;
    if let Some(user) = &target.user {
        check_field("user", user)?;
    }
    let known_hosts_file = known_hosts_file.unwrap_or(&target.known_hosts_file);
    let option = |setting: String| vec!["-o".to_string(), setting];
    let mut arguments: Vec<String> = Vec::new();
    if let Some(config_file) = &target.config_file {
        arguments.extend(["-F".to_string(), config_file.clone()]);
    }
    arguments.extend(["-T".to_string(), "-a".to_string(), "-x".to_string()]);
    for setting in [
        "BatchMode=yes",
        "ClearAllForwardings=yes",
        "ForwardAgent=no",
        "ForwardX11=no",
        "ControlMaster=no",
        "ControlPath=none",
        "RemoteCommand=none",
        "PermitLocalCommand=no",
        "SendEnv=-*",
        "ServerAliveInterval=15",
    ] {
        arguments.extend(option(setting.to_string()));
    }
    arguments.extend(option(format!(
        "StrictHostKeyChecking={}",
        if strict_host_keys {
            "yes"
        } else {
            "accept-new"
        }
    )));
    arguments.extend(option(format!(
        "UserKnownHostsFile={}",
        config_path("known hosts file", known_hosts_file)?
    )));
    arguments.extend(option("GlobalKnownHostsFile=none".to_string()));
    arguments.extend(option("HashKnownHosts=no".to_string()));
    arguments.extend(option(format!("HostKeyAlias={}", target.host_key_alias)));
    if let Some(user) = &target.user {
        arguments.extend(["-l".to_string(), user.clone()]);
    }
    if let Some(port) = target.port {
        arguments.extend(["-p".to_string(), port.to_string()]);
    }
    if let Some(identity_file) = &target.identity_file {
        arguments.extend(option(format!(
            "IdentityFile={}",
            config_path("identity file", identity_file)?
        )));
        arguments.extend(option("IdentitiesOnly=yes".to_string()));
    }
    arguments.extend(["--".to_string(), target.host.clone()]);
    Ok(arguments)
}

/// Run one remote command over `ssh`, optionally feeding stdin; resolves with stdout, fails on
/// non-zero exit, a lost host key, or the timeout.
async fn run_ssh(
    target: &SshTarget,
    command: &str,
    args: Option<Vec<String>>,
    stdin: Option<Vec<u8>>,
    timeout: Duration,
) -> Result<String, SshCommandError> {
    let arguments = match args {
        Some(arguments) => arguments,
        None => ssh_arguments(target, true, None).map_err(SshCommandError::Other)?,
    };
    let mut child = Command::new(target.ssh.as_deref().unwrap_or("ssh"))
        .args(&arguments)
        .arg(command)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| SshError {
            message: error.to_string(),
            exit_code: None,
            stderr: String::new(),
        })?;
    let mut child_stdin = child.stdin.take().expect("piped stdin");
    let input = stdin.unwrap_or_default();
    let feeder = tokio::spawn(async move {
        let _ = child_stdin.write_all(&input).await;
        let _ = child_stdin.flush().await;
        drop(child_stdin);
    });
    let mut child_stdout = child.stdout.take().expect("piped stdout");
    let mut child_stderr = child.stderr.take().expect("piped stderr");
    let out_reader = tokio::spawn(async move {
        let mut stdout = Vec::new();
        let _ = child_stdout.read_to_end(&mut stdout).await;
        stdout
    });
    let error_reader = tokio::spawn(async move {
        let mut stderr = Vec::new();
        let _ = child_stderr.read_to_end(&mut stderr).await;
        stderr
    });
    // `wait` borrows the child, so the timeout branch can still kill it.
    let waited = tokio::time::timeout(timeout, child.wait()).await;
    let status = match waited {
        Ok(status) => status,
        Err(_) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            let diagnostics =
                String::from_utf8_lossy(&error_reader.await.unwrap_or_default()).into_owned();
            return Err(SshCommandError::Ssh(SshError {
                message: format!(
                    "ssh {} did not finish within {} s: {}",
                    target.host,
                    timeout.as_secs(),
                    diagnostics.trim()
                ),
                exit_code: None,
                stderr: diagnostics,
            }));
        }
    };
    let _ = feeder.await;
    let stdout = String::from_utf8_lossy(&out_reader.await.unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&error_reader.await.unwrap_or_default()).into_owned();
    let code = status.ok().and_then(|status| status.code());
    if code == Some(0) {
        return Ok(stdout);
    }
    if stderr.contains("REMOTE HOST IDENTIFICATION HAS CHANGED") {
        return Err(SshCommandError::HostKeyChanged(HostKeyChangedError(
            format!(
                "The host key of {} changed; remove the old key to continue",
                target.host
            ),
        )));
    }
    if stderr.contains("Host key verification failed") || stderr.contains("host key is known") {
        return Err(SshCommandError::HostKeyUnknown(HostKeyUnknownError(
            format!("The host key of {} is not trusted yet", target.host),
        )));
    }
    Err(SshCommandError::Ssh(SshError {
        message: format!(
            "ssh {} failed with exit code {}: {}",
            target.host,
            code.map(|code| code.to_string())
                .unwrap_or_else(|| "none".into()),
            stderr.trim()
        ),
        exit_code: code,
        stderr: stderr.clone(),
    }))
}

/// Connect once with `accept-new` against a temporary known-hosts file to capture the key the host
/// presents, through the same route (`~/.ssh/config`, jump hosts) as real connections. Returns its
/// known-hosts lines and fingerprints. Showing a fingerprint is not authentication: compare it with one
/// obtained out of band before accepting it.
pub async fn scan_host_key(
    target: &SshTarget,
) -> Result<(Vec<String>, Vec<String>), SshCommandError> {
    let directory =
        mkdtemp("pi-env-hostkey-").map_err(|error| SshCommandError::Other(error.to_string()))?;
    let scanned = directory.join("known_hosts");
    let result = async {
        let arguments = ssh_arguments(target, false, Some(&scanned.to_string_lossy()))
            .map_err(SshCommandError::Other)?;
        // Authentication may fail without the key; the key is recorded before authentication.
        match run_ssh(target, "exit 0", Some(arguments), None, SSH_TIMEOUT).await {
            Ok(_) | Err(SshCommandError::Ssh(_) | SshCommandError::HostKeyUnknown(_)) => {}
            Err(error) => return Err(error),
        }
        let lines = fs::read_to_string(&scanned)
            .map(|content| {
                content
                    .lines()
                    .map(str::to_owned)
                    .filter(|line| !line.trim().is_empty())
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        if lines.is_empty() {
            return Err(SshCommandError::Other(format!(
                "No host key received from {}",
                target.host
            )));
        }
        let output = Command::new("ssh-keygen")
            .args(["-lf", &scanned.to_string_lossy()])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|error| SshCommandError::Other(error.to_string()))?;
        let fingerprints = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::to_owned)
            .collect();
        Ok((lines, fingerprints))
    }
    .await;
    let _ = fs::remove_dir_all(&directory);
    result
}

/// A known-hosts line for `alias`: `alias type key [comment]`, no markers, patterns or hashed names.
fn parse_host_key_line(alias: &str, line: &str) -> Result<(String, String), String> {
    let fields: Vec<&str> = line.trim().split_whitespace().collect();
    let &[host, key_type, key] = fields.as_slice() else {
        return Err(format!("Not a host key line for {alias}: {line:?}"));
    };
    // `^(ssh-[a-z0-9-]+|ecdsa-sha2-[a-z0-9-]+|sk-[a-z0-9@.\-]+)$` and `^[A-Za-z0-9+/]+={0,2}$`.
    let allowed = |character: char, extra: &str| {
        character.is_ascii_lowercase() || character.is_ascii_digit() || extra.contains(character)
    };
    let type_valid = !key_type.is_empty()
        && (if let Some(rest) = key_type.strip_prefix("ssh-") {
            !rest.is_empty() && rest.chars().all(|character| allowed(character, "-"))
        } else if let Some(rest) = key_type.strip_prefix("ecdsa-sha2-") {
            !rest.is_empty() && rest.chars().all(|character| allowed(character, "-"))
        } else if let Some(rest) = key_type.strip_prefix("sk-") {
            !rest.is_empty() && rest.chars().all(|character| allowed(character, "@.-"))
        } else {
            false
        });
    let (body, padding) = match key.find('=') {
        Some(index) => (&key[..index], &key[index..]),
        None => (key, ""),
    };
    let key_valid = !body.is_empty()
        && padding.len() <= 2
        && body
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/');
    if host != alias || !type_valid || !key_valid {
        return Err(format!("Not a host key line for {alias}: {line:?}"));
    }
    Ok((key_type.to_string(), key.to_string()))
}

/// Known-hosts files being changed, so changes to one file happen one at a time.
static KNOWN_HOSTS_CHANGES: LazyLock<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

async fn change_known_hosts(
    file: &str,
    change: impl FnOnce(&[String]) -> Result<Vec<String>, SshCommandError>,
) -> Result<(), SshCommandError> {
    let lock = Arc::clone(
        KNOWN_HOSTS_CHANGES
            .lock()
            .entry(file.to_string())
            .or_default(),
    );
    let _guard = lock.lock().await;
    change_known_hosts_sync(file, change)
}

/// Rewrite the known-hosts file atomically: a new temporary file renamed over the old one.
fn change_known_hosts_sync(
    file: &str,
    change: impl FnOnce(&[String]) -> Result<Vec<String>, SshCommandError>,
) -> Result<(), SshCommandError> {
    if let Some(parent) = Path::new(file)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| SshCommandError::Other(error.to_string()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o700));
        }
    }
    let existing = fs::read_to_string(file)
        .map(|content| {
            content
                .lines()
                .map(str::to_owned)
                .filter(|line| !line.is_empty())
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();
    let lines = change(&existing)?;
    let temporary = format!("{file}.{}.tmp", random_name());
    let mut written = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| SshCommandError::Other(error.to_string()))?;
    let content = if lines.is_empty() {
        Vec::new()
    } else {
        format!("{}\n", lines.join("\n")).into_bytes()
    };
    written
        .write_all(&content)
        .map_err(|error| SshCommandError::Other(error.to_string()))?;
    drop(written);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600));
    }
    if let Err(error) = fs::rename(&temporary, file) {
        let _ = fs::remove_file(&temporary);
        return Err(SshCommandError::Other(error.to_string()));
    }
    Ok(())
}

/// Trust host-key lines from `scan_host_key` by adding them to the target's known-hosts file. Only
/// plain lines for the target's alias are accepted. A key that differs from a trusted key of the same
/// type is refused with `HostKeyChangedError`; `forget_host_key` must remove the old keys first.
pub async fn accept_host_key(target: &SshTarget, lines: &[String]) -> Result<(), SshCommandError> {
    let alias = target.host_key_alias.clone();
    let host = target.host.clone();
    let accepted = lines
        .iter()
        .map(|line| {
            parse_host_key_line(&alias, line)
                .map(|(key_type, key)| (line.trim().to_owned(), key_type, key))
        })
        .collect::<Result<Vec<(String, String, String)>, String>>()
        .map_err(SshCommandError::Other)?;
    change_known_hosts(&target.known_hosts_file, move |existing| {
        let mut trusted: HashMap<String, String> = HashMap::new();
        for line in existing {
            let mut fields = line.trim().split_whitespace();
            if let (Some(host_name), Some(key_type), Some(key)) =
                (fields.next(), fields.next(), fields.next())
            {
                if host_name == alias {
                    trusted.insert(key_type.to_string(), key.to_string());
                }
            }
        }
        let mut added = Vec::new();
        for (line, key_type, key) in accepted {
            match trusted.get(&key_type) {
                Some(known) if *known == key => continue,
                Some(_) => {
                    return Err(SshCommandError::HostKeyChanged(HostKeyChangedError(
                        format!(
                            "The {key_type} host key of {host} changed; forget the old key first"
                        ),
                    )));
                }
                None => {
                    trusted.insert(key_type, key);
                    added.push(line);
                }
            }
        }
        let mut result = existing.to_vec();
        result.extend(added);
        Ok(result)
    })
    .await
}

/// Stop trusting every key stored for the target's alias, e.g. after the owner confirmed a changed host key.
pub async fn forget_host_key(target: &SshTarget) -> Result<(), SshCommandError> {
    let alias = target.host_key_alias.clone();
    change_known_hosts(&target.known_hosts_file, move |existing| {
        Ok(existing
            .iter()
            .filter(|line| line.trim().split_whitespace().next() != Some(alias.as_str()))
            .cloned()
            .collect())
    })
    .await
}

/// POSIX detection, run by the remote login shell; a Windows host answers through PowerShell instead.
const POSIX_PROBE: &str = "sh -c 'echo PI-ENV-PROBE; uname -s; uname -m; uname -o 2>/dev/null || echo -; printf \"%s\\n\" \"$HOME\" \"${TMPDIR:--}\" \"${LD_PRELOAD:--}\"'";

/// A PowerShell script invoked so its output and errors are plain: `-EncodedCommand` carries it as UTF-16LE base64.
fn powershell(script: &str) -> String {
    format!(
        "powershell -NoProfile -NonInteractive -EncodedCommand {}",
        base64_encode(
            &script
                .encode_utf16()
                .flat_map(|unit| unit.to_le_bytes())
                .collect::<Vec<u8>>()
        )
    )
}

const WINDOWS_PROBE: &str = "'PI-ENV-PROBE'; 'Windows'; [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString(); '-'; $HOME; '-'; '-'";

fn normalize_arch(machine: &str) -> Result<Arch, String> {
    let lower = machine.to_lowercase();
    if matches!(lower.as_str(), "x86_64" | "amd64" | "x64") {
        return Ok(Arch::X64);
    }
    if matches!(lower.as_str(), "aarch64" | "arm64") {
        return Ok(Arch::Arm64);
    }
    Err(format!("Unsupported remote architecture: {machine}"))
}

#[derive(Debug, Clone)]
struct Probe {
    system: String,
    machine: String,
    os: String,
    home: String,
    tmpdir: String,
    preload: String,
}

fn parse_probe(target: &SshTarget, output: &str) -> Result<Probe, String> {
    // Login shells may print a banner first.
    let lines: Vec<&str> = output
        .split("\r\n")
        .flat_map(|line| line.split('\n'))
        .collect();
    let start = lines
        .iter()
        .position(|line| line.trim() == "PI-ENV-PROBE")
        .ok_or_else(|| format!("Unexpected answer from {}: {}", target.host, output.trim()))?;
    let field = |index: usize, default: &str| {
        lines
            .get(start + 1 + index)
            .map(|line| line.trim().to_owned())
            .filter(|line| !line.is_empty())
            .unwrap_or_else(|| default.to_string())
    };
    Ok(Probe {
        system: field(0, ""),
        machine: field(1, ""),
        os: field(2, "-"),
        home: field(3, ""),
        tmpdir: field(4, "-"),
        preload: field(5, "-"),
    })
}

/// Termux works only with its own `TMPDIR` and with `termux-exec`, which makes `#!/usr/bin/env` shebangs work.
fn termux_warnings(probe: &Probe) -> Vec<String> {
    let mut warnings = Vec::new();
    if probe.tmpdir == "-" {
        warnings
            .push("TMPDIR is not set; Termux's sshd normally sets it to $PREFIX/tmp.".to_string());
    }
    if !probe.preload.contains("termux-exec") {
        warnings.push("termux-exec is not loaded (LD_PRELOAD); scripts with #!/usr/bin/env shebangs will fail.".to_string());
    }
    warnings.push("Android may suspend Termux; run termux-wake-lock on the device to keep the connection alive.".to_string());
    warnings
}

/// Which system the target runs: `uname` through the login shell, or PowerShell on Windows.
pub async fn detect_platform(target: &SshTarget) -> Result<RemotePlatform, SshCommandError> {
    let mut probe = match run_ssh(target, POSIX_PROBE, None, None, SSH_TIMEOUT).await {
        Ok(output) => parse_probe(target, &output)?,
        Err(error) => {
            // cmd.exe or PowerShell as the remote shell: no `sh`, or one that gets the probe's quotes wrong.
            if matches!(
                error,
                SshCommandError::HostKeyUnknown(_) | SshCommandError::HostKeyChanged(_)
            ) {
                return Err(error);
            }
            parse_probe(
                target,
                &run_ssh(target, &powershell(WINDOWS_PROBE), None, None, SSH_TIMEOUT).await?,
            )?
        }
    };
    // Git Bash as Windows' default SSH shell: ask PowerShell for Windows' own architecture and home spelling.
    if probe.system.starts_with("MINGW")
        || probe.system.starts_with("MSYS")
        || probe.system.starts_with("CYGWIN")
    {
        probe = parse_probe(
            target,
            &run_ssh(target, &powershell(WINDOWS_PROBE), None, None, SSH_TIMEOUT).await?,
        )?;
    }
    let arch = normalize_arch(&probe.machine)?;
    if probe.system == "Windows" {
        // cmd.exe expands %OS%; PowerShell prints it as is.
        let shell = if run_ssh(target, "echo %OS%", None, None, SSH_TIMEOUT)
            .await?
            .contains("Windows_NT")
        {
            RemoteShell::Cmd
        } else {
            RemoteShell::PowerShell
        };
        return Ok(RemotePlatform {
            platform: Platform::Windows,
            arch,
            home: probe.home.clone(),
            shell: Some(shell),
            warnings: Vec::new(),
        });
    }
    if probe.system == "Darwin" {
        return Ok(RemotePlatform {
            platform: Platform::Darwin,
            arch,
            home: probe.home.clone(),
            shell: None,
            warnings: Vec::new(),
        });
    }
    if probe.system == "Linux" {
        let android = probe.os == "Android";
        return Ok(RemotePlatform {
            platform: if android {
                Platform::Android
            } else {
                Platform::Linux
            },
            arch,
            home: probe.home.clone(),
            shell: None,
            warnings: android.then(|| termux_warnings(&probe)).unwrap_or_default(),
        });
    }
    Err(SshCommandError::Other(format!(
        "Unsupported remote system: {}",
        probe.system
    )))
}

/// The daemon binary the package ships for a remote system. This crate has no packaged cross-target
/// binaries: the path only resolves when `PI_ENV_PACKAGE_ROOT` names a directory of the layout
/// `<root>/<pi-env-<platform>-<arch>>/<pi-env[.exe]>`, as the published npm package has it.
pub fn packaged_daemon(remote: &RemotePlatform) -> Option<PathBuf> {
    let root = std::env::var_os("PI_ENV_PACKAGE_ROOT")?;
    let name = format!(
        "pi-env-{}-{}",
        remote.platform.as_str(),
        remote.arch.as_str()
    );
    let binary = if remote.platform == Platform::Windows {
        "pi-env.exe"
    } else {
        "pi-env"
    };
    Some(Path::new(&root).join(name).join(binary))
}

fn quote_posix(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

fn quote_powershell(text: &str) -> String {
    format!("'{}'", text.replace('\'', "''"))
}

/// Where a daemon with this content lives on the remote machine: named by its SHA-256, so versions never collide.
fn daemon_path(remote: &RemotePlatform, sha256: &str) -> String {
    let name = format!("pi-env-{}", &sha256[..32]);
    if remote.platform == Platform::Windows {
        format!("{}\\.pi\\mobile\\tools\\{name}.exe", remote.home)
    } else {
        format!("{}/.pi/mobile/tools/{name}", remote.home)
    }
}

const POSIX_HASH: &str = "hash() { if command -v sha256sum >/dev/null 2>&1; then sha256sum \"$1\" | cut -d' ' -f1; elif command -v shasum >/dev/null 2>&1; then shasum -a 256 \"$1\" | cut -d' ' -f1; elif command -v openssl >/dev/null 2>&1; then openssl dgst -sha256 \"$1\" | sed 's/.*= //'; else echo none; fi; }";

/// Whether the remote file has this content: `present`, `missing`, or (POSIX without a hash tool) `nohash`.
async fn check_daemon(
    target: &SshTarget,
    remote: &RemotePlatform,
    file: &str,
    sha256: &str,
) -> Result<String, SshCommandError> {
    if remote.platform == Platform::Windows {
        let check = powershell(&format!(
            "$f = {}; if ((Test-Path -LiteralPath $f) -and ((Get-FileHash -Algorithm SHA256 -LiteralPath $f).Hash.ToLower() -eq '{}')) {{ 'present' }} else {{ 'missing' }}",
            quote_powershell(file),
            sha256
        ));
        return Ok(
            if run_ssh(target, &check, None, None, SSH_TIMEOUT)
                .await?
                .trim()
                .ends_with("present")
            {
                "present".to_string()
            } else {
                "missing".to_string()
            },
        );
    }
    let check = format!(
        "{POSIX_HASH}; f={}; if [ -f \"$f\" ] && [ \"$(hash \"$f\")\" = {sha256} ]; then echo present; elif [ \"$(hash /dev/null)\" = none ]; then echo nohash; else echo missing; fi",
        quote_posix(file)
    );
    let output = run_ssh(
        target,
        &format!("sh -c {}", quote_posix(&check)),
        None,
        None,
        SSH_TIMEOUT,
    )
    .await?;
    Ok(output
        .trim()
        .lines()
        .last()
        .unwrap_or("missing")
        .to_string())
}

/// Make sure the daemon is on the remote machine, verified by its SHA-256 before it ever runs, and
/// remove daemons of other contents. The upload goes to a new temporary file and is renamed into place
/// only once its hash matches. Returns the remote path of the binary.
pub async fn deploy_daemon(
    target: &SshTarget,
    remote: &RemotePlatform,
    binary: &Path,
) -> Result<String, SshCommandError> {
    if !binary.exists() {
        return Err(SshCommandError::Other(format!(
            "No pi-env daemon for {}-{} at {}; this crate ships no packaged cross-target binaries, so set PI_ENV_PACKAGE_ROOT or pass an explicit binary",
            remote.platform.as_str(),
            remote.arch.as_str(),
            binary.display()
        )));
    }
    let bytes = fs::read(binary).map_err(|error| SshCommandError::Other(error.to_string()))?;
    let sha256 = hex(&Sha256::digest(&bytes));
    let file = daemon_path(remote, &sha256);
    let state = check_daemon(target, remote, &file, &sha256).await?;
    if state == "present" {
        return Ok(file);
    }
    if state == "nohash" {
        return Err(SshCommandError::Other(format!(
            "{} has no sha256sum, shasum or openssl to verify pi-env",
            target.host
        )));
    }
    if remote.platform == Platform::Windows {
        let upload = powershell(&[
			"$ErrorActionPreference = 'Stop'".to_string(),
			format!("$f = {}", quote_powershell(&file)),
			"$d = Split-Path -Parent $f".to_string(),
			"New-Item -ItemType Directory -Force -Path $d | Out-Null".to_string(),
			"$t = Join-Path $d ('.pi-env-' + [guid]::NewGuid().ToString() + '.tmp')".to_string(),
			// PowerShell reads redirected stdin itself, as text lines for `$input`, so the binary comes as
			// base64 lines up to an end marker: Windows' sshd may never pass on the end of stdin.
			"$text = New-Object System.Text.StringBuilder".to_string(),
			"foreach ($line in $input) { if ($line -eq 'PI-ENV-END') { break }; [void]$text.Append($line) }".to_string(),
			"$bytes = [Convert]::FromBase64String($text.ToString())".to_string(),
			"$out = [IO.File]::Open($t, 'CreateNew', 'Write', 'None'); $out.Write($bytes, 0, $bytes.Length); $out.Close()".to_string(),
			format!(
				"if ((Get-FileHash -Algorithm SHA256 -LiteralPath $t).Hash.ToLower() -ne '{sha256}') {{ Remove-Item -LiteralPath $t; throw 'pi-env upload is corrupt' }}"
			),
			// A running daemon or a virus scanner can hold the old file for a moment.
			"for ($i = 0; ; $i++) { try { Move-Item -Force -LiteralPath $t -Destination $f; break } catch { if ($i -ge 20) { throw }; Start-Sleep -Milliseconds 250 } }".to_string(),
			// Older daemons; one that is running stays until it exits.
			"Get-ChildItem -LiteralPath $d -Filter 'pi-env-*.exe' | Where-Object { $_.FullName -ne $f } | ForEach-Object { Remove-Item -LiteralPath $_.FullName -ErrorAction SilentlyContinue }".to_string(),
			"'deployed'".to_string(),
		]
		.join("; "));
        let lines = base64_lines(&bytes);
        let mut stdin_payload = String::new();
        for line in lines {
            stdin_payload.push_str(&line);
            stdin_payload.push('\n');
        }
        stdin_payload.push_str("PI-ENV-END\n");
        run_ssh(
            target,
            &upload,
            None,
            Some(stdin_payload.into_bytes()),
            UPLOAD_TIMEOUT,
        )
        .await?;
        return Ok(file);
    }
    let upload = [
		"set -e".to_string(),
		POSIX_HASH.to_string(),
		format!("f={}", quote_posix(&file)),
		"d=$(dirname \"$f\")".to_string(),
		"mkdir -p \"$d\"".to_string(),
		"chmod 700 \"$d\"".to_string(),
		"t=$(mktemp \"$d/.pi-env.XXXXXX\")".to_string(),
		"cat > \"$t\"".to_string(),
		format!("if [ \"$(hash \"$t\")\" != {sha256} ]; then rm -f \"$t\"; echo \"pi-env upload is corrupt\" >&2; exit 1; fi"),
		"chmod 700 \"$t\"".to_string(),
		"mv -f \"$t\" \"$f\"".to_string(),
		// Older daemons; running ones keep their file open until they exit.
		"for old in \"$d\"/pi-env-*; do [ \"$old\" = \"$f\" ] || rm -f \"$old\"; done".to_string(),
		"echo deployed".to_string(),
	]
	.join("\n");
    run_ssh(
        target,
        &format!("sh -c {}", quote_posix(&upload)),
        None,
        Some(bytes),
        UPLOAD_TIMEOUT,
    )
    .await?;
    Ok(file)
}

/// Options for `ssh_connection` and `connect_ssh`: the target, plus the binary to deploy. Without
/// `binary`, deployment looks for the packaged daemon (`PI_ENV_PACKAGE_ROOT`); see [`packaged_daemon`].
#[derive(Clone, Default)]
pub struct SshConnectOptions {
    pub host: String,
    pub user: Option<String>,
    pub port: Option<u16>,
    pub identity_file: Option<String>,
    pub known_hosts_file: String,
    pub host_key_alias: String,
    pub ssh: Option<String>,
    pub config_file: Option<String>,
    pub binary: Option<PathBuf>,
    /// POSIX: start the daemon through the user's login shell (`$SHELL -l`), so commands see the
    /// environment of `~/.profile` and similar files. Off by default: `ssh` runs commands without a
    /// login shell.
    pub login_shell: bool,
    pub on_log: Option<Arc<dyn Fn(&str) + Send + Sync>>,
}

impl SshConnectOptions {
    /// The [`SshTarget`] these options describe.
    pub fn target(&self) -> SshTarget {
        SshTarget {
            host: self.host.clone(),
            user: self.user.clone(),
            port: self.port,
            identity_file: self.identity_file.clone(),
            known_hosts_file: self.known_hosts_file.clone(),
            host_key_alias: self.host_key_alias.clone(),
            ssh: self.ssh.clone(),
            config_file: self.config_file.clone(),
        }
    }
}

/// The remote command that starts the daemon at `file`, before the `serve` arguments.
fn launch_command(remote: &RemotePlatform, file: &str, login_shell: bool) -> String {
    if remote.platform == Platform::Windows {
        // PowerShell runs a quoted path only with the call operator; cmd.exe keeps one pair of quotes around a program.
        if remote.shell == Some(RemoteShell::PowerShell) {
            return format!("& {}", quote_powershell(file));
        }
        return if file.contains(' ') {
            format!("\"{file}\"")
        } else {
            file.to_string()
        };
    }
    // `$0` is the daemon, `"$@"` its arguments; `$SHELL` is the login shell's own name for itself.
    if login_shell {
        return format!(
            "exec \"$SHELL\" -lc 'exec \"$0\" \"$@\"' {}",
            quote_posix(file)
        );
    }
    quote_posix(file)
}

/// What an SSH connection learned about its remote system, and the daemon it verified last.
#[derive(Default)]
struct SshState {
    remote: Option<RemotePlatform>,
    verified: Option<String>,
}

/// A handle on the remote system an [`ssh_connection`] has detected so far.
#[derive(Clone, Default)]
pub struct SshRemote {
    state: Arc<Mutex<SshState>>,
}

impl SshRemote {
    /// The detected system, once known.
    pub fn remote(&self) -> Option<RemotePlatform> {
        self.state.lock().remote.clone()
    }
}

fn ssh_provider(
    options: SshConnectOptions,
    state: Arc<Mutex<SshState>>,
    log: Option<Arc<dyn Fn(&str) + Send + Sync>>,
) -> ConnectionOptions {
    let command_log = log.clone();
    ConnectionOptions {
        command: CommandSource::Computed(Arc::new(move || {
            let options = options.clone();
            let state = state.clone();
            let log = command_log.clone();
            Box::pin(async move {
                let target = options.target();
                let remote = {
                    let existing = state.lock().remote.clone();
                    match existing {
                        Some(remote) => remote,
                        None => {
                            let remote = detect_platform(&target)
                                .await
                                .map_err(|error| error.to_string())?;
                            if let Some(log) = &log {
                                for warning in &remote.warnings {
                                    log(&format!("{warning}\n"));
                                }
                            }
                            state.lock().remote = Some(remote.clone());
                            remote
                        }
                    }
                };
                // A deployment that just verified the binary counts for the first start. The stored
                // flag is read before the match, so its lock guard cannot be held across the await.
                let verified = state.lock().verified.take();
                let file = match verified {
                    Some(verified) => verified,
                    None => {
                        let binary = options
							.binary
							.clone()
							.or_else(|| packaged_daemon(&remote))
							.ok_or_else(|| "No pi-env daemon binary: set PI_ENV_PACKAGE_ROOT or pass an explicit binary".to_string())?;
                        deploy_daemon(&target, &remote, &binary)
                            .await
                            .map_err(|error| error.to_string())?
                    }
                };
                let arguments =
                    ssh_arguments(&target, true, None).map_err(|error| error.to_string())?;
                let mut command = vec![target.ssh.clone().unwrap_or_else(|| "ssh".to_string())];
                command.extend(arguments);
                command.push(launch_command(&remote, &file, options.login_shell));
                Ok(command)
            })
        })),
        on_log: log,
    }
}

/// A `Connection` builder for the target that does nothing until its first request. Each start detects
/// the remote system (the first time), verifies the daemon and deploys it if it is missing or changed,
/// then starts it over `ssh`. A failure (no network, an untrusted or changed host key) fails the
/// requests waiting for that start with code `spawn_error` and the `ssh` diagnostics as message; the
/// next request tries again. `remote()` is the detected system, once known.
pub fn ssh_connection(options: SshConnectOptions) -> (ConnectionOptions, SshRemote) {
    let log = options.on_log.clone();
    let state = Arc::new(Mutex::new(SshState::default()));
    (
        ssh_provider(options, state.clone(), log),
        SshRemote { state },
    )
}

/// Detect the remote system and deploy the daemon now, then return connection options that start it
/// over `ssh`, like `ssh_connection`. Failures reject here: host keys must already be trusted
/// (`scan_host_key`, `accept_host_key`), otherwise this fails with `HostKeyUnknownError`.
pub async fn connect_ssh(
    options: SshConnectOptions,
) -> Result<(ConnectionOptions, RemotePlatform), SshCommandError> {
    let target = options.target();
    let remote = detect_platform(&target).await?;
    let binary = options
        .binary
        .clone()
        .or_else(|| packaged_daemon(&remote))
        .ok_or_else(|| {
            SshCommandError::Other(
                "No pi-env daemon binary: set PI_ENV_PACKAGE_ROOT or pass an explicit binary"
                    .to_string(),
            )
        })?;
    let verified = deploy_daemon(&target, &remote, &binary).await?;
    if let Some(log) = &options.on_log {
        for warning in &remote.warnings {
            log(&format!("{warning}\n"));
        }
    }
    let state = Arc::new(Mutex::new(SshState {
        remote: Some(remote.clone()),
        verified: Some(verified),
    }));
    let log = options.on_log.clone();
    Ok((ssh_provider(options, state, log), remote))
}

fn random_name() -> String {
    let mut bytes = [0u8; 6];
    getrandom::getrandom(&mut bytes).expect("system randomness");
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn mkdtemp(prefix: &str) -> std::io::Result<PathBuf> {
    let mut directory;
    let mut attempts = 0;
    loop {
        directory = std::env::temp_dir().join(format!("{prefix}{}", random_name()));
        match fs::create_dir(&directory) {
            Ok(()) => break,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists && attempts < 1000 => {
                attempts += 1
            }
            Err(error) => return Err(error),
        }
    }
    Ok(directory)
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding, as `Buffer.toString("base64")` produces.
fn base64_encode(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let values = [
            chunk[0] as u32,
            chunk.get(1).copied().unwrap_or(0) as u32,
            chunk.get(2).copied().unwrap_or(0) as u32,
        ];
        let group = (values[0] << 16) | (values[1] << 8) | values[2];
        encoded.push(BASE64[(group >> 18) as usize & 0x3f] as char);
        encoded.push(BASE64[(group >> 12) as usize & 0x3f] as char);
        encoded.push(if chunk.len() > 1 {
            BASE64[(group >> 6) as usize & 0x3f] as char
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            BASE64[group as usize & 0x3f] as char
        } else {
            '='
        });
    }
    encoded
}

/// Base64 wrapped at 76 characters, as the Windows upload feeds it line by line.
fn base64_lines(bytes: &[u8]) -> Vec<String> {
    let encoded = base64_encode(bytes);
    encoded
        .as_bytes()
        .chunks(76)
        .map(|chunk| String::from_utf8_lossy(chunk).into_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn parses_and_rejects_host_key_lines() {
        let alias = "pi-env-test";
        let (key_type, key) = parse_host_key_line(
            alias,
            "pi-env-test ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample+Key/Body==",
        )
        .unwrap();
        assert_eq!(
            (key_type.as_str(), key.as_str()),
            ("ssh-ed25519", "AAAAC3NzaC1lZDI1NTE5AAAAIExample+Key/Body==")
        );
        assert!(parse_host_key_line(alias, "other-alias ssh-ed25519 AAAA").is_err());
        assert!(
            parse_host_key_line(alias, "@cert-authority pi-env-test ssh-ed25519 AAAA").is_err()
        );
        assert!(parse_host_key_line(alias, "pi-env-test ssh-ed25519 not base64!!").is_err());
    }

    #[test]
    fn validates_ssh_arguments() {
        let mut target = SshTarget {
            host: "127.0.0.1".into(),
            port: Some(2222),
            user: Some("garden".into()),
            known_hosts_file: "/tmp/known_hosts".into(),
            host_key_alias: "pi-env-test".into(),
            config_file: Some("/tmp/config".into()),
            identity_file: Some("/tmp/id_ed25519".into()),
            ..Default::default()
        };
        let arguments = ssh_arguments(&target, true, None).unwrap();
        assert!(arguments.contains(&"BatchMode=yes".to_string()));
        assert_eq!(*arguments.last().unwrap(), "127.0.0.1");
        assert!(
            arguments
                .iter()
                .any(|argument| argument.contains("UserKnownHostsFile="))
        );
        assert!(
            !arguments
                .iter()
                .any(|argument| argument.contains("accept-new"))
        );
        let lenient = ssh_arguments(&target, false, Some("/tmp/scan")).unwrap();
        assert!(
            lenient
                .iter()
                .any(|argument| argument.contains("accept-new"))
        );
        assert!(
            lenient
                .iter()
                .any(|argument| argument.contains("/tmp/scan"))
        );
        target.host = "-oProxyCommand=evil".into();
        assert!(ssh_arguments(&target, true, None).is_err());
        target.host = "127.0.0.1".into();
        target.port = Some(0);
        assert!(ssh_arguments(&target, true, None).is_err());
    }

    #[test]
    fn launches_daemons_per_platform() {
        let posix = RemotePlatform {
            platform: Platform::Linux,
            arch: Arch::X64,
            home: "/home/garden".into(),
            shell: None,
            warnings: Vec::new(),
        };
        assert_eq!(
            launch_command(&posix, "/home/garden/.pi/mobile/tools/pi-env-x", false),
            "'/home/garden/.pi/mobile/tools/pi-env-x'"
        );
        assert_eq!(
            launch_command(&posix, "/home/garden/x", true),
            "exec \"$SHELL\" -lc 'exec \"$0\" \"$@\"' '/home/garden/x'"
        );
        let windows = RemotePlatform {
            platform: Platform::Windows,
            arch: Arch::Arm64,
            home: "C:\\Users\\g".into(),
            shell: Some(RemoteShell::Cmd),
            warnings: Vec::new(),
        };
        assert_eq!(
            launch_command(&windows, "C:\\a b\\pi-env.exe", false),
            "\"C:\\a b\\pi-env.exe\""
        );
        let powershell_host = RemotePlatform {
            platform: Platform::Windows,
            arch: Arch::Arm64,
            home: "C:\\Users\\g".into(),
            shell: Some(RemoteShell::PowerShell),
            warnings: Vec::new(),
        };
        assert_eq!(
            launch_command(&powershell_host, "C:\\a b\\pi-env.exe", false),
            "& 'C:\\a b\\pi-env.exe'"
        );
    }
}
