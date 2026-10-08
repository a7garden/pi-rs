//! Port of `packages/coding-agent/src/utils/clipboard-command.ts` — running
//! clipboard helper commands with input/timeout/buffer caps.
//!
//! Deviations (crate docs, deviation D16): upstream is async over
//! `child_process.spawn`; this port is synchronous over `std::process` and is
//! driven from blocking contexts. A hidden override hook
//! ([`set_command_runner_override`]) replaces the spawned process in tests,
//! standing in for upstream's vi.mock-based tests.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

type CommandRunner = dyn Fn(&str, &[String], Option<&str>, u64, u64) -> Option<Vec<u8>> + Send + Sync;

static OVERRIDE: parking_lot::RwLock<Option<Box<CommandRunner>>> = parking_lot::RwLock::new(None);

/// Test seam: replace the spawned-process implementation.
#[doc(hidden)]
#[cfg(test)]
pub fn set_command_runner_override(runner: Box<CommandRunner>) {
	*OVERRIDE.write() = Some(runner);
}

#[cfg(test)]
pub(crate) fn clear_command_runner_override() {
	*OVERRIDE.write() = None;
}

/// Options mirroring upstream's `{ input?, timeoutMs?, maxBufferBytes? }`.
#[derive(Debug, Clone, Default)]
pub struct RunClipboardCommandOptions<'a> {
	pub input: Option<&'a str>,
	pub timeout_ms: Option<u64>,
	pub max_buffer_bytes: Option<u64>,
}

/// Undefined means the command failed; an empty buffer is a successful result
/// (`runClipboardCommand`).
pub fn run_clipboard_command(
	command: &str,
	args: &[String],
	options: Option<RunClipboardCommandOptions<'_>>,
) -> Option<Vec<u8>> {
	let options = options.unwrap_or(RunClipboardCommandOptions::default());
	let timeout_ms = options.timeout_ms.unwrap_or(3000);
	let max_buffer_bytes = options.max_buffer_bytes.unwrap_or(50 * 1024 * 1024);

	if let Some(runner) = OVERRIDE.read().as_ref() {
		return runner(command, args, options.input, timeout_ms, max_buffer_bytes);
	}

	// Clipboard writers can daemonize. Do not give them output pipes to retain.
	let mut child = Command::new(command)
		.args(args)
		.stdin(Stdio::piped())
		.stdout(if options.input.is_none() { Stdio::piped() } else { Stdio::null() })
		.stderr(Stdio::null())
		.spawn()
		.ok()?;

	let deadline = Instant::now() + Duration::from_millis(timeout_ms);

	// Drain stdout on a separate thread so a chatty child cannot deadlock us.
	let mut stdout = child.stdout.take();
	let reader = std::thread::spawn(move || {
		let mut buffer = Vec::new();
		if let Some(stdout) = stdout.as_mut() {
			let mut chunk = [0u8; 8192];
			loop {
				match stdout.read(&mut chunk) {
					Ok(0) | Err(_) => break,
					Ok(read) => {
						buffer.extend_from_slice(&chunk[..read]);
						if buffer.len() as u64 > max_buffer_bytes {
							break;
						}
					}
				}
			}
		}
		buffer
	});

	// A writer may exit before consuming all input.
	if let Some(input) = options.input {
		if let Some(mut stdin) = child.stdin.take() {
			let _ = stdin.write_all(input.as_bytes());
		}
	} else {
		drop(child.stdin.take());
	}

	let mut timed_out = false;
	loop {
		match child.try_wait() {
			Ok(Some(_status)) => break,
			Ok(None) => {
				if Instant::now() >= deadline {
					timed_out = true;
					break;
				}
				std::thread::sleep(Duration::from_millis(5));
			}
			Err(_) => {
				let _ = reader.join();
				return None;
			}
		}
	}

	if timed_out {
		let _ = child.kill();
		let _ = reader.join();
		return None;
	}

	let output = match reader.join() {
		Ok(output) => output,
		Err(_) => return None,
	};

	if output.len() as u64 > max_buffer_bytes {
		let _ = child.kill();
		let _ = child.wait();
		return None;
	}

	let status = child.wait().ok()?;
	if status.success() {
		Some(output)
	} else {
		None
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn sh(script: &str) -> (String, Vec<String>) {
		("sh".to_string(), vec!["-c".to_string(), script.to_string()])
	}

	#[test]
	fn preserves_binary_output_and_distinguishes_empty_success_from_failure() {
		let (command, args) = sh("printf '\\x00\\xff\\x0a'");
		assert_eq!(run_clipboard_command(&command, &args, None), Some(vec![0, 255, 10]));

		let (command, args) = sh("");
		assert_eq!(run_clipboard_command(&command, &args, None), Some(Vec::new()));

		let (command, args) = sh("exit 1");
		assert_eq!(run_clipboard_command(&command, &args, None), None);

		assert_eq!(run_clipboard_command("pi-clipboard-command-does-not-exist", &[], None), None);
	}

	#[test]
	fn sends_unicode_input_to_clipboard_writers() {
		let (command, args) = sh("text=$(cat); [ \"$text\" = 'café 日本語' ]");
		let options = RunClipboardCommandOptions { input: Some("café 日本語"), timeout_ms: None, max_buffer_bytes: None };
		assert_eq!(run_clipboard_command(&command, &args, Some(options)), Some(Vec::new()));
	}

	#[test]
	fn times_out_without_blocking() {
		let (command, args) = sh("sleep 30");
		let options = RunClipboardCommandOptions { input: None, timeout_ms: Some(200), max_buffer_bytes: None };
		let started = Instant::now();
		assert_eq!(run_clipboard_command(&command, &args, Some(options)), None);
		assert!(started.elapsed() < Duration::from_secs(5));
	}

	#[test]
	fn rejects_output_above_the_buffer_limit() {
		let (command, args) = sh("dd if=/dev/zero bs=1024 count=1 2>/dev/null");
		let options = RunClipboardCommandOptions { input: None, timeout_ms: None, max_buffer_bytes: Some(16) };
		assert_eq!(run_clipboard_command(&command, &args, Some(options)), None);
	}
}
