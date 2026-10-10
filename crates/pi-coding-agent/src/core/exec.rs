//! Port of `packages/coding-agent/src/core/exec.ts` — shared command
//! execution utilities for extensions and custom tools.
//!
//! Deviations (numbered in the crate docs):
//! - **D116 — the upstream 5s SIGKILL escalation actually fires here.**
//!   Node's `subprocess.killed` is already `true` immediately after the
//!   SIGTERM `kill()` call, so upstream's `if (!proc.killed)` guard never
//!   passes and the escalation is dead code; the port sends SIGKILL 5s after
//!   the SIGTERM when the child has not exited, matching the documented
//!   intent ("Force kill after 5 seconds if SIGTERM doesn't work"). On
//!   non-unix platforms Node maps every `kill()` to unconditional process
//!   termination, so "SIGTERM" and "SIGKILL" collapse to
//!   [`tokio::process::Child::start_kill`]. Spawn failures (e.g. ENOENT,
//!   which upstream surfaces through the `waitForChildProcess` rejection)
//!   return the upstream catch-path result `{ code: 1, killed: false }`.
//! - The abort/timeout bookkeeping uses [`tokio::select!`] over the
//!   pure-Rust [`AbortSignal::aborted_future`] (crate D3 signal) instead of
//!   `addEventListener`; the wait+drain body mirrors
//!   [`crate::utils::child_process::wait_for_child_process`] with kill
//!   decisions interleaved.

use std::process::Stdio;
use std::time::Duration;

use crate::utils::abort::AbortSignal;
use serde::{Deserialize, Serialize};

/// Grace period between SIGTERM and the SIGKILL escalation.
const FORCE_KILL_AFTER_MS: u64 = 5_000;

/// Post-exit idle grace for stdio pipes held open by detached descendants
/// (mirrors `utils::child_process`).
const EXIT_STDIO_GRACE_MS: u64 = 100;

/// Options for executing shell commands (`ExecOptions`).
///
/// Upstream keeps the unused `cwd` member on the options object; the working
/// directory is the dedicated `exec_command` argument (same as upstream's
/// `execCommand(command, args, cwd, options)`).
#[derive(Debug, Clone, Default)]
pub struct ExecOptions {
    /// Signal to cancel the command.
    pub signal: Option<AbortSignal>,
    /// Timeout in milliseconds.
    pub timeout: Option<u64>,
    /// Working directory (unused by [`exec_command`], upstream parity).
    pub cwd: Option<String>,
}

/// Result of executing a shell command (`ExecResult`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub code: i32,
    pub killed: bool,
}

enum KillRequest {
    Terminate,
    ForceKill,
}

/// Execute a shell command and return stdout/stderr/code. Supports timeout
/// and abort signal (`execCommand`).
///
/// The child spawns with `shell: false, stdio: ["ignore", "pipe", "pipe"]`;
/// a timeout or abort sends SIGTERM (process termination on non-unix) and,
/// if the child is still running 5s later, SIGKILL (D116). A spawn failure
/// resolves with `code: 1` like the upstream rejection path.
pub async fn exec_command(
    command: &str,
    args: &[&str],
    cwd: &str,
    options: ExecOptions,
) -> ExecResult {
    let mut spawn = tokio::process::Command::new(command);
    spawn
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match spawn.spawn() {
        Ok(child) => child,
        Err(_) => {
            return ExecResult {
                code: 1,
                ..ExecResult::default()
            }
        }
    };
    let pid = child.id();

    // Kill decisions run on the outside; the child wait/drain loop applies
    // them from `kill_rx` so the handle stays reachable (non-unix SIGTERM).
    let (kill_tx, mut kill_rx) = tokio::sync::mpsc::unbounded_channel();
    let run = async {
        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();
        let status = match loop {
            tokio::select! {
                status = child.wait() => break status,
                Some(request) = kill_rx.recv() => apply_kill(&mut child, pid, request),
            }
        } {
            Ok(status) => status,
            // Upstream catch path: `{ stdout, stderr, code: 1, killed }`.
            Err(_) => return Err(()),
        };
        // Post-exit: drain both pipes until EOF or the idle grace elapses with
        // no new data (a detached descendant holding the handle, pi#5303).
        let (stdout, stderr) = drain(&mut stdout_pipe, &mut stderr_pipe).await;
        Ok((
            status.code().or_else(|| status_signal(&status)),
            String::from_utf8_lossy(&stdout).into_owned(),
            String::from_utf8_lossy(&stderr).into_owned(),
        ))
    };
    tokio::pin!(run);

    let mut killed = false;
    let mut force_sent = false;
    let mut escalation_at: Option<tokio::time::Instant> = None;
    let timeout_at = options
        .timeout
        .filter(|timeout| *timeout > 0)
        .map(|timeout| tokio::time::Instant::now() + Duration::from_millis(timeout));
    let abort_future = options
        .signal
        .as_ref()
        .map(|signal| signal.aborted_future());
    let abort_wait = async {
        match abort_future {
            Some(future) => {
                let _ = future.await;
            }
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(abort_wait);

    let outcome = loop {
        let timeout_wait = async {
            match timeout_at {
                Some(deadline) => tokio::time::sleep_until(deadline).await,
                None => std::future::pending::<()>().await,
            }
        };
        let escalation_wait = async {
            match escalation_at {
                Some(deadline) => tokio::time::sleep_until(deadline).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::select! {
            outcome = &mut run => break outcome,
            _ = &mut abort_wait, if !killed => request_termination(&mut killed, &mut escalation_at, &kill_tx),
            _ = timeout_wait, if !killed => request_termination(&mut killed, &mut escalation_at, &kill_tx),
            _ = escalation_wait, if !force_sent && escalation_at.is_some() => {
                force_sent = true;
                let _ = kill_tx.send(KillRequest::ForceKill);
            }
        }
    };

    let (code, stdout, stderr) = outcome.unwrap_or((Some(1), String::new(), String::new()));
    ExecResult {
        stdout,
        stderr,
        // Upstream: `code ?? 0` on resolve, `1` on the wait rejection.
        code: code.unwrap_or(0),
        killed,
    }
}

fn request_termination(
    killed: &mut bool,
    escalation_at: &mut Option<tokio::time::Instant>,
    kill_tx: &tokio::sync::mpsc::UnboundedSender<KillRequest>,
) {
    *killed = true;
    *escalation_at = Some(tokio::time::Instant::now() + Duration::from_millis(FORCE_KILL_AFTER_MS));
    let _ = kill_tx.send(KillRequest::Terminate);
}

fn apply_kill(child: &mut tokio::process::Child, pid: Option<u32>, request: KillRequest) {
    match request {
        KillRequest::Terminate => {
            #[cfg(unix)]
            if let Some(pid) = pid {
                let _ = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
                return;
            }
            // Non-unix signals terminate unconditionally (Node parity, D116);
            // a reaped unix child falls back to the direct kill no-op.
            let _ = child.start_kill();
        }
        KillRequest::ForceKill => {
            let _ = child.start_kill();
        }
    }
}

/// Drain both pipes until EOF or [`EXIT_STDIO_GRACE_MS`] of silence; a chunk
/// on either pipe re-arms the grace timer (same shape as
/// [`crate::utils::child_process::wait_for_child_process`]).
async fn drain(
    stdout_pipe: &mut Option<impl tokio::io::AsyncRead + Unpin>,
    stderr_pipe: &mut Option<impl tokio::io::AsyncRead + Unpin>,
) -> (Vec<u8>, Vec<u8>) {
    use tokio::io::AsyncReadExt;

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut stdout_done = stdout_pipe.is_none();
    let mut stderr_done = stderr_pipe.is_none();
    let grace = Duration::from_millis(EXIT_STDIO_GRACE_MS);
    while !stdout_done || !stderr_done {
        let read_both = async {
            let mut chunk = [0u8; 8192];
            if !stdout_done {
                if let Some(pipe) = stdout_pipe.as_mut() {
                    match pipe.read(&mut chunk).await {
                        Ok(0) => stdout_done = true,
                        Ok(n) => {
                            stdout.extend_from_slice(&chunk[..n]);
                            return;
                        }
                        Err(_) => stdout_done = true,
                    }
                }
            }
            if !stderr_done {
                if let Some(pipe) = stderr_pipe.as_mut() {
                    match pipe.read(&mut chunk).await {
                        Ok(0) => stderr_done = true,
                        Ok(n) => stderr.extend_from_slice(&chunk[..n]),
                        Err(_) => stderr_done = true,
                    }
                }
            }
        };
        if tokio::time::timeout(grace, read_both).await.is_err() {
            break;
        }
    }
    (stdout, stderr)
}

#[cfg(unix)]
fn status_signal(status: &std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal().map(|signal| 128 + signal)
}

#[cfg(not(unix))]
fn status_signal(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::abort::AbortController;

    #[tokio::test]
    async fn captures_stdout_stderr_and_exit_code() {
        let result = exec_command(
            "/bin/sh",
            &["-c", "echo out; echo err 1>&2; exit 3"],
            "/",
            ExecOptions::default(),
        )
        .await;
        assert_eq!(result.stdout.trim(), "out");
        assert_eq!(result.stderr.trim(), "err");
        assert_eq!(result.code, 3);
        assert!(!result.killed);
    }

    #[tokio::test]
    async fn timeout_terminates_the_child_and_reports_killed() {
        let started = std::time::Instant::now();
        let result = exec_command(
            "/bin/sh",
            &["-c", "sleep 30"],
            "/",
            ExecOptions {
                timeout: Some(50),
                ..Default::default()
            },
        )
        .await;
        assert!(result.killed);
        assert!(started.elapsed() < std::time::Duration::from_secs(10));
        assert_ne!(result.code, 0);
    }

    #[tokio::test]
    async fn abort_signal_terminates_the_child() {
        let controller = AbortController::new();
        let signal = controller.signal().clone();
        let aborter = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            controller.abort();
        });
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            exec_command(
                "/bin/sh",
                &["-c", "sleep 30"],
                "/",
                ExecOptions {
                    signal: Some(signal),
                    ..Default::default()
                },
            ),
        )
        .await
        .expect("completes after abort");
        aborter.await.unwrap();
        assert!(result.killed);
    }

    #[tokio::test]
    async fn pre_aborted_signal_kills_immediately() {
        let controller = AbortController::new();
        controller.abort();
        let result = exec_command(
            "/bin/sh",
            &["-c", "sleep 30"],
            "/",
            ExecOptions {
                signal: Some(controller.signal().clone()),
                ..Default::default()
            },
        )
        .await;
        assert!(result.killed);
    }

    #[tokio::test]
    async fn spawn_failure_resolves_with_code_one() {
        let result = exec_command(
            "/nonexistent/pi-exec-command",
            &[],
            "/",
            ExecOptions::default(),
        )
        .await;
        assert_eq!(result.code, 1);
        assert!(!result.killed);
        assert!(result.stdout.is_empty());
    }

    #[tokio::test]
    async fn zero_timeout_disables_the_timer() {
        let result = exec_command(
            "/bin/sh",
            &["-c", "echo done"],
            "/",
            ExecOptions {
                timeout: Some(0),
                ..Default::default()
            },
        )
        .await;
        assert_eq!(result.stdout.trim(), "done");
        assert!(!result.killed);
    }
}
