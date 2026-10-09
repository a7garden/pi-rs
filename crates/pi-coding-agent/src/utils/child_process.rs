//! Port of `packages/coding-agent/src/utils/child-process.ts` — spawn helpers
//! plus the post-exit stdio-drain wait (`waitForChildProcess`).
//!
//! Deviations (crate docs, D13): upstream selects `cross-spawn` on Windows to
//! execute `.cmd`/`.bat` shims; Rust has no equivalent, so
//! [`std::process::Command`]/[`tokio::process::Command`] are used on every
//! platform. `waitForChildProcess` returns the drained stdout/stderr instead
//! of leaving the streams attached to the child object.

use std::process::Stdio;

use tokio::io::AsyncReadExt;

const EXIT_STDIO_GRACE_MS: u64 = 100;

/// Result of [`spawn_process_sync`] (upstream
/// `SpawnSyncReturns<string>` with `encoding: "utf-8"`).
#[derive(Debug, Clone, Default)]
pub struct SyncOutput {
	pub status: Option<i32>,
	pub stdout: String,
	pub stderr: String,
}

/// Spawn a child with piped stdout/stderr (`spawnProcess` with
/// `stdio: ["ignore", "pipe", "pipe"]`).
///
/// Windows note: upstream routes through `cross-spawn` to execute `.cmd`/
/// `.bat` shims (D13); here the plain command is spawned on all platforms.
pub fn spawn_process(command: &str, args: &[&str]) -> std::io::Result<tokio::process::Child> {
	tokio::process::Command::new(command)
		.args(args)
		.stdin(Stdio::null())
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.spawn()
}

/// Run a child to completion with UTF-8 output (`spawnProcessSync` with
/// `encoding: "utf-8"`).
pub fn spawn_process_sync(command: &str, args: &[&str]) -> std::io::Result<SyncOutput> {
	let output = std::process::Command::new(command)
		.args(args)
		.stdin(Stdio::null())
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.output()?;
	Ok(SyncOutput {
		status: exit_code(output.status),
		stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
		stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
	})
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

/// Drained child output plus exit code (`waitForChildProcess` result).
#[derive(Debug, Clone, Default)]
pub struct ChildResult {
	pub code: Option<i32>,
	pub stdout: Vec<u8>,
	pub stderr: Vec<u8>,
}

/// Wait for a child process to terminate without hanging on inherited stdio
/// handles (`waitForChildProcess`).
///
/// A short-lived child can exit while a detached descendant keeps its
/// stdout/stderr pipe open. After `exit` we wait for the pipes to fall idle:
/// the grace timer is re-armed on every chunk, so an actively writing
/// descendant keeps us reading, while a quiet inherited handle still releases
/// us after [`EXIT_STDIO_GRACE_MS`] elapses (earendil-works/pi#5303).
pub async fn wait_for_child_process(mut child: tokio::process::Child) -> std::io::Result<ChildResult> {
	let mut stdout_pipe = child.stdout.take();
	let mut stderr_pipe = child.stderr.take();
	let mut stdout = Vec::new();
	let mut stderr = Vec::new();
	let mut stdout_done = stdout_pipe.is_none();
	let mut stderr_done = stderr_pipe.is_none();

	let status = child.wait().await?;

	// Post-exit: keep draining until both pipes hit EOF, or the idle grace
	// elapses with no new data (a detached descendant holding the pipe). A
	// chunk on either pipe re-arms the grace timer (upstream `armIdleTimer`).
	let grace = std::time::Duration::from_millis(EXIT_STDIO_GRACE_MS);
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
			break; // idle past the grace window: release like upstream
		}
	}

	Ok(ChildResult { code: status.code().or_else(|| status_signal(&status)), stdout, stderr })
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

	#[tokio::test]
	async fn spawn_process_captures_output() {
		let _env_guard = crate::utils::test_env_lock();
		let child = spawn_process("/bin/sh", &["-c", "echo out; echo err 1>&2"]).unwrap();
		let result = wait_for_child_process(child).await.unwrap();
		assert_eq!(result.code, Some(0));
		assert_eq!(String::from_utf8_lossy(&result.stdout).trim(), "out");
		assert_eq!(String::from_utf8_lossy(&result.stderr).trim(), "err");
	}

	#[tokio::test]
	async fn wait_returns_after_exit_even_with_detached_grandchild_holding_pipe() {
		let _env_guard = crate::utils::test_env_lock();
		// A short-lived child that leaves a detached descendant writing to the
		// (inherited) pipe: the grace timer must release us, not hang forever.
		let child = spawn_process(
			"/bin/sh",
			&["-c", "echo parent-out; (sleep 30; echo late) >/dev/null 2>&1 & echo done; exit 0"],
		)
		.unwrap();
		let started = std::time::Instant::now();
		let result = wait_for_child_process(child).await.unwrap();
		assert_eq!(result.code, Some(0));
		assert!(started.elapsed() < std::time::Duration::from_secs(10));
		assert!(String::from_utf8_lossy(&result.stdout).contains("parent-out"));
	}

	#[tokio::test]
	async fn keeps_draining_output_written_after_exit() {
		// The child exits after a detached descendant writes to the shared
		// pipe; chunks arriving after exit must not be truncated.
		let child = spawn_process(
			"/bin/sh",
			&["-c", r#"(sleep 0.05; echo tail; ) >/dev/fd/1 2>/dev/null & sleep 0.2; exit 0"#],
		)
		.unwrap();
		let result = wait_for_child_process(child).await.unwrap();
		assert_eq!(result.code, Some(0));
		assert!(String::from_utf8_lossy(&result.stdout).contains("tail"));
	}

	#[test]
	fn sync_spawn_captures_utf8_output() {
		let output = spawn_process_sync("/bin/sh", &["-c", "printf 'héllo'; exit 3"]).unwrap();
		assert_eq!(output.status, Some(3));
		assert_eq!(output.stdout, "héllo");
	}
}
