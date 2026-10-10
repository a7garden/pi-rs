//! Port of `packages/coding-agent/src/utils/shell.ts` — shell resolution,
//! sanitization, detached-child tracking and cross-platform process-tree
//! kills.
//!
//! Deviations (crate docs, D13): upstream uses `cross-spawn` on Windows to
//! resolve `.cmd`/`.bat` shims; Rust `Command` has no such resolution, so the
//! same `Command` is used on every platform. Signal delivery to process
//! groups uses the `libc` crate (dependency request: `libc`).

use parking_lot::Mutex;
use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

use crate::config::get_agent_dir;
use crate::utils::paths::join;

/// Shell invocation configuration (upstream `ShellConfig`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellConfig {
    pub shell: String,
    pub args: Vec<String>,
    /// How the command string reaches the shell; `None` = argv (`-c`).
    pub command_transport: Option<CommandTransport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandTransport {
    /// Feed the command on stdin (`bash -s`).
    Stdin,
    /// Pass the command as an argv element (`bash -c <cmd>`).
    Argv,
}

/// Upstream `POWERSHELL_ARGS`.
pub const POWERSHELL_ARGS: [&str; 5] = [
    "-NoProfile",
    "-NonInteractive",
    "-ExecutionPolicy",
    "Bypass",
    "-Command",
];

/// `/^[a-z]:\\windows\\(?:system32|sysnative)\\bash\.exe$/` against the
/// forward-slash-normalized, lowercased path (upstream `isLegacyWslBashPath`).
fn is_legacy_wsl_bash_path(path: &str) -> bool {
    let normalized = path.replace('/', "\\").to_ascii_lowercase();
    // <drive>:\windows\{system32|sysnative}\bash.exe
    let bytes = normalized.as_bytes();
    if bytes.len() < 3 || !bytes[0].is_ascii_lowercase() || bytes[1] != b':' {
        return false;
    }
    let rest = &normalized[2..];
    [
        "\\windows\\system32\\bash.exe",
        "\\windows\\sysnative\\bash.exe",
    ]
    .iter()
    .any(|suffix| rest == *suffix)
}

fn get_bash_shell_config(shell: &str) -> ShellConfig {
    if is_legacy_wsl_bash_path(shell) {
        ShellConfig {
            shell: shell.to_string(),
            args: vec!["-s".to_string()],
            command_transport: Some(CommandTransport::Stdin),
        }
    } else {
        ShellConfig {
            shell: shell.to_string(),
            args: vec!["-c".to_string()],
            command_transport: None,
        }
    }
}

/// Run a helper lookup command with a hard 5s timeout (upstream
/// `spawnSync(..., { timeout: 5000 })`).
fn spawn_sync_with_timeout(command: &str, args: &[&str]) -> Option<(Option<i32>, String)> {
    let mut child = std::process::Command::new(command)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(5000);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(10)),
            Err(_) => return None,
        }
    };
    let mut stdout = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        let _ = pipe.read_to_string(&mut stdout);
    }
    Some((exit_code(status), stdout))
}

fn exit_code(status: std::process::ExitStatus) -> Option<i32> {
    status.code().or_else(|| signal_code(status))
}

#[cfg(unix)]
fn signal_code(status: std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal().map(|signal| 128 + signal)
}

#[cfg(not(unix))]
fn signal_code(_status: std::process::ExitStatus) -> Option<i32> {
    None
}

/// Find an executable on PATH (upstream `findExecutableOnPath`): `where` +
/// existence check on Windows, `which` on unix.
fn find_executable_on_path(executable: &str) -> Option<String> {
    if cfg!(windows) {
        let (status, stdout) = spawn_sync_with_timeout("where", &[executable])?;
        if status == Some(0) && !stdout.trim().is_empty() {
            let first_match = stdout.trim().split("\r\n").next()?.to_string();
            if !first_match.is_empty() && Path::new(&first_match).exists() {
                return Some(first_match);
            }
        }
        return None;
    }

    let (status, stdout) = spawn_sync_with_timeout("which", &[executable])?;
    if status == Some(0) && !stdout.trim().is_empty() {
        let first_match = stdout.trim().split('\n').next()?;
        if !first_match.is_empty() {
            return Some(first_match.to_string());
        }
    }
    None
}

/// Resolve shell configuration based on platform and an optional explicit
/// shell path (`getShellConfig`). Resolution order:
/// 1. User-specified `shell_path`
/// 2. On Windows: Git Bash in known locations, then bash on PATH
/// 3. On Unix: /bin/bash, then bash on PATH, then fallback to sh
pub fn get_shell_config(custom_shell_path: Option<&str>) -> Result<ShellConfig, String> {
    // 1. Check user-specified shell path
    if let Some(custom_shell_path) = custom_shell_path {
        if Path::new(custom_shell_path).exists() {
            return Ok(get_bash_shell_config(custom_shell_path));
        }
        return Err(format!("Custom shell path not found: {custom_shell_path}"));
    }

    if cfg!(windows) {
        // 2. Try Git Bash in known locations
        let mut paths: Vec<String> = Vec::new();
        if let Ok(program_files) = std::env::var("ProgramFiles") {
            paths.push(format!("{program_files}\\Git\\bin\\bash.exe"));
        }
        if let Ok(program_files_x86) = std::env::var("ProgramFiles(x86)") {
            paths.push(format!("{program_files_x86}\\Git\\bin\\bash.exe"));
        }

        for path in &paths {
            if Path::new(path).exists() {
                return Ok(get_bash_shell_config(path));
            }
        }

        // 3. Fallback: search bash.exe on PATH (Cygwin, MSYS2, WSL, etc.)
        if let Some(bash_on_path) = find_executable_on_path("bash.exe") {
            return Ok(get_bash_shell_config(&bash_on_path));
        }

        return Err(format!(
            "No bash shell found. Options:\n  \
			 1. Install Git for Windows: https://git-scm.com/download/win\n  \
			 2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n\n  \
			 Searched Git Bash in:\n{}",
            paths
                .iter()
                .map(|p| format!("  {p}"))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    // Unix: try /bin/bash, then bash on PATH, then fallback to sh
    if Path::new("/bin/bash").exists() {
        return Ok(get_bash_shell_config("/bin/bash"));
    }

    if let Some(bash_on_path) = find_executable_on_path("bash") {
        return Ok(get_bash_shell_config(&bash_on_path));
    }

    Ok(ShellConfig {
        shell: "sh".to_string(),
        args: vec!["-c".to_string()],
        command_transport: None,
    })
}

/// Resolve PowerShell on Windows, preferring PowerShell 7 when available
/// (`getPowerShellConfig`).
pub fn get_powershell_config() -> Result<ShellConfig, String> {
    if !cfg!(windows) {
        return Err("The powershell tool is only available on Windows.".to_string());
    }

    let shell =
        find_executable_on_path("pwsh.exe").or_else(|| find_executable_on_path("powershell.exe"));
    let Some(shell) = shell else {
        return Err(
			"No PowerShell executable found. Install PowerShell or add powershell.exe/pwsh.exe to PATH.".to_string(),
		);
    };

    Ok(ShellConfig {
        shell,
        args: POWERSHELL_ARGS.iter().map(|s| s.to_string()).collect(),
        command_transport: None,
    })
}

/// Upstream `getShellEnv`: the process env with the managed bin directory
/// prepended to PATH (deduplicated, case-insensitive PATH key).
pub fn get_shell_env() -> Vec<(String, String)> {
    let bin_dir = join(&get_agent_dir(), "bin");
    let delimiter = if cfg!(windows) { ';' } else { ':' };
    let path_key = std::env::vars_os()
        .filter_map(|(k, _)| k.into_string().ok())
        .find(|key| key.eq_ignore_ascii_case("PATH"))
        .unwrap_or_else(|| "PATH".to_string());
    let current_path = std::env::var(&path_key).unwrap_or_default();
    let path_entries: Vec<&str> = current_path
        .split(delimiter)
        .filter(|e| !e.is_empty())
        .collect();
    let has_bin_dir = path_entries.iter().any(|entry| *entry == bin_dir);
    let updated_path = if has_bin_dir {
        current_path
    } else {
        [bin_dir.as_str(), current_path.as_str()]
            .iter()
            .filter(|entry| !entry.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(&delimiter.to_string())
    };

    std::env::vars_os()
        .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
        .map(|(k, v)| {
            if k.eq_ignore_ascii_case("PATH") {
                (path_key.clone(), updated_path.clone())
            } else {
                (k, v)
            }
        })
        .collect()
}

/// Sanitize binary output for display/storage (`sanitizeBinaryOutput`).
/// Removes control characters (except tab/newline/CR) and the Unicode
/// interlinear annotation characters U+FFF9..U+FFFB that crash string-width.
pub fn sanitize_binary_output(s: &str) -> String {
    s.chars()
		.filter(|&c| {
			let strip = matches!(c, '\0'..='\x08' | '\x0B' | '\x0C' | '\x0E'..='\x1F' | '\u{FFF9}'..='\u{FFFB}');
			!strip
		})
		.collect()
}

static TRACKED_DETACHED_CHILD_PIDS: Mutex<BTreeSet<u32>> = Mutex::new(BTreeSet::new());

/// Track a detached child so it can be killed on parent shutdown signals.
pub fn track_detached_child_pid(pid: u32) {
    TRACKED_DETACHED_CHILD_PIDS.lock().insert(pid);
}

pub fn untrack_detached_child_pid(pid: u32) {
    TRACKED_DETACHED_CHILD_PIDS.lock().remove(&pid);
}

pub fn kill_tracked_detached_children() {
    let pids = std::mem::take(&mut *TRACKED_DETACHED_CHILD_PIDS.lock());
    for pid in pids {
        kill_process_tree(pid);
    }
}

/// Kill a process and all its children, cross-platform (`killProcessTree`).
pub fn kill_process_tree(pid: u32) {
    if cfg!(windows) {
        // Use the trusted System32 executable so cleanup does not depend on
        // PATH. Errors are ignored (upstream consumes the async error event).
        let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
        let _ = std::process::Command::new(join(&system_root, "System32\\taskkill.exe"))
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    } else {
        // SIGKILL the process group, falling back to the child alone.
        #[cfg(unix)]
        unsafe {
            let i32_pid = pid as i32;
            if libc::kill(-i32_pid, libc::SIGKILL) != 0 {
                let _ = libc::kill(i32_pid, libc::SIGKILL);
            }
        }
        #[cfg(not(unix))]
        let _ = pid;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_wsl_bash_path_detection() {
        assert!(is_legacy_wsl_bash_path("C:\\Windows\\System32\\bash.exe"));
        assert!(is_legacy_wsl_bash_path("c:/windows/sysnative/bash.EXE"));
        assert!(!is_legacy_wsl_bash_path(
            "C:\\Program Files\\Git\\bin\\bash.exe"
        ));
        assert!(!is_legacy_wsl_bash_path("D:\\windows\\system32\\cmd.exe"));
        assert!(!is_legacy_wsl_bash_path("/usr/bin/bash"));
    }

    #[test]
    fn bash_shell_config_default_uses_argv_transport() {
        let config = get_bash_shell_config("/bin/bash");
        assert_eq!(config.shell, "/bin/bash");
        assert_eq!(config.args, vec!["-c"]);
        assert_eq!(config.command_transport, None);
    }

    #[test]
    fn legacy_wsl_bash_uses_stdin_transport() {
        let config = get_bash_shell_config("C:\\Windows\\System32\\bash.exe");
        assert_eq!(config.args, vec!["-s"]);
        assert_eq!(config.command_transport, Some(CommandTransport::Stdin));
    }

    #[test]
    fn unix_shell_config_prefers_bash() {
        // /bin/bash exists on macOS; on minimal environments the PATH search
        // or sh fallback applies — all three resolve successfully.
        let config = get_shell_config(None).unwrap();
        if Path::new("/bin/bash").exists() {
            assert_eq!(config.shell, "/bin/bash");
        } else {
            assert!(!config.shell.is_empty());
        }
    }

    #[test]
    fn custom_shell_path_missing_errors_with_message() {
        let err = get_shell_config(Some("/nonexistent/pi/shell/xyz")).unwrap_err();
        assert_eq!(
            err,
            "Custom shell path not found: /nonexistent/pi/shell/xyz"
        );
    }

    #[test]
    fn powershell_rejected_off_windows() {
        if !cfg!(windows) {
            assert_eq!(
                get_powershell_config().unwrap_err(),
                "The powershell tool is only available on Windows."
            );
        }
    }

    #[test]
    fn sanitize_binary_output_strips_control_and_annotation_chars() {
        assert_eq!(
            sanitize_binary_output("a\u{0}b\u{8}c\u{B}d\u{C}e\u{F}f"),
            "abcdef"
        );
        assert_eq!(
            sanitize_binary_output("keep\ttab\nnl\rcr"),
            "keep\ttab\nnl\rcr"
        );
        assert_eq!(
            sanitize_binary_output("x\u{FFF9}y\u{FFFA}z\u{FFFB}w"),
            "xyzw"
        );
        assert_eq!(sanitize_binary_output("del\u{7F}kept"), "del\u{7F}kept");
    }

    #[test]
    fn tracked_pids_round_trip() {
        track_detached_child_pid(4242);
        untrack_detached_child_pid(4242);
        untrack_detached_child_pid(4242); // delete of absent pid is a no-op
    }
}
