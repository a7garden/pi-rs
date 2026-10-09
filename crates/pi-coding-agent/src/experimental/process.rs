//! Port of `packages/coding-agent/src/experimental/process.ts` — consistent
//! spawning/termination of detached Pi-owned internal processes.
//!
//! # Deviations
//!
//! - **D138 — process seam.** Upstream spawns `process.execPath` with a
//!   `--import` source resolver (source-run) or a bundled
//!   `dist/bundle/{coordinator,cli}.js` entry (`isBundledNode`) and locates
//!   entries relative to `import.meta.url`. The Rust port always spawns the
//!   running executable (`std::env::current_exe`) with
//!   `InternalProcessSpawnOptions::entry_override` as an optional explicit
//!   program replacement; `entryUrl`, `isBunBinary`/`isBundledNode`
//!   detection and `isDirectInternalProcessEntry` (`import.meta.url`
//!   comparison) have no Rust counterpart and are not ported. Node
//!   `detached` + `child.unref()` becomes a new process group
//!   (`process_group(0)`) with `kill_on_drop(false)`; `stdio: "ignore"`
//!   becomes null stdio.
//! - **D138a — `terminateInternalProcess`** escalates for real: tokio's
//!   `start_kill` sends SIGKILL and `wait` joins the reaper, matching the
//!   upstream kill-then-await observable behavior.

use std::collections::HashMap;
use std::process::Stdio;

use tokio::process::{Child, Command};

/// Env var marking a Pi-owned internal process (`__PI_INTERNAL_SPAWN`).
pub const INTERNAL_PROCESS_ENV: &str = "__PI_INTERNAL_SPAWN";

/// Maximum encoded size of one coordinator control line (128 MiB).
pub const MAX_CONTROL_LINE_BYTES: usize = 128 * 1024 * 1024;

/// Internal process roles (`InternalProcessRole`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InternalProcessRole {
	Coordinator,
	Server,
	SessionWorker,
}

impl InternalProcessRole {
	pub fn as_str(self) -> &'static str {
		match self {
			Self::Coordinator => "coordinator",
			Self::Server => "server",
			Self::SessionWorker => "session-worker",
		}
	}

	fn from_str(value: &str) -> Option<Self> {
		match value {
			"coordinator" => Some(Self::Coordinator),
			"server" => Some(Self::Server),
			"session-worker" => Some(Self::SessionWorker),
			_ => None,
		}
	}
}

/// Read and validate an internal process role without consuming it.
///
/// `Err` carries the upstream `Unsupported internal process role: …` message.
pub fn get_internal_process_role() -> Result<Option<InternalProcessRole>, String> {
	match std::env::var(INTERNAL_PROCESS_ENV) {
		Ok(role) => InternalProcessRole::from_str(&role)
			.map(Some)
			.ok_or_else(|| format!("Unsupported internal process role: {role}")),
		Err(std::env::VarError::NotPresent) => Ok(None),
		Err(std::env::VarError::NotUnicode(_)) => {
			Err(format!("Unsupported internal process role: <non-unicode>"))
		}
	}
}

/// Read, validate, and remove the role so descendants do not inherit it.
pub fn consume_internal_process_role() -> Result<Option<InternalProcessRole>, String> {
	let role = get_internal_process_role();
	std::env::remove_var(INTERNAL_PROCESS_ENV);
	role
}

/// `InternalProcessSpawnOptions`. `entry_override` replaces the spawned
/// program entirely (D138 — upstream `entryUrl` only selects the JS entry of
/// the same interpreter).
#[derive(Debug, Clone, Default)]
pub struct InternalProcessSpawnOptions {
	pub entry_override: Option<std::path::PathBuf>,
	pub env: HashMap<String, String>,
}

/// Spawn a detached Pi-owned internal process consistently across runtimes.
pub fn spawn_internal_process(
	role: InternalProcessRole,
	args: &[String],
	options: &InternalProcessSpawnOptions,
) -> std::io::Result<Child> {
	let program = match &options.entry_override {
		Some(program) => program.clone(),
		None => std::env::current_exe()?,
	};
	let mut command = Command::new(&program);
	command
		.args(args)
		.env(INTERNAL_PROCESS_ENV, role.as_str())
		.stdin(Stdio::null())
		.stdout(Stdio::null())
		.stderr(Stdio::null())
		.process_group(0)
		.kill_on_drop(false);
	for (key, value) in &options.env {
		command.env(key, value);
	}
	Ok(command.spawn()?)
}

/// Force a spawned internal process to exit and wait until it can no longer
/// take ownership.
pub async fn terminate_internal_process(mut child: Child) {
	let _ = child.start_kill();
	let _ = child.wait().await;
}

/// Encode one control message as a single JSON line. Errors mirror the
/// upstream `Internal control message is too large` throw.
pub fn encode_control_line(message: &serde_json::Value) -> Result<String, String> {
	let mut line = serde_json::to_string(message).expect("control messages serialize");
	line.push('\n');
	if line.len() > MAX_CONTROL_LINE_BYTES {
		return Err("Internal control message is too large".into());
	}
	Ok(line)
}

/// Decode one JSON control line.
pub fn decode_control_line(line: &str) -> Result<serde_json::Value, String> {
	serde_json::from_str(line).map_err(|_| "Coordinator sent invalid JSON".to_string())
}

/// Incremental newline-framed reader shared by the coordinator client and
/// process (`attachJsonLineReader`). Destroys (drops) the connection when a
/// line exceeds [`MAX_CONTROL_LINE_BYTES`] or is invalid JSON.
pub struct LineReader {
	buffer: Vec<u8>,
}

impl LineReader {
	pub fn new() -> Self {
		Self { buffer: Vec::new() }
	}

	/// Feed one chunk; returns every complete line decoded as `Value`.
	/// `Err` means the connection must be destroyed (message too large or
	/// invalid JSON).
	pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<serde_json::Value>, String> {
		self.buffer.extend_from_slice(chunk);
		if self.buffer.len() > MAX_CONTROL_LINE_BYTES {
			return Err("Coordinator message is too large".into());
		}
		let mut messages = Vec::new();
		while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
			let line: Vec<u8> = self.buffer.drain(..=newline).collect();
			let line = &line[..line.len() - 1];
			let message: serde_json::Value = serde_json::from_slice(line)
				.map_err(|_| "Coordinator sent invalid JSON".to_string())?;
			messages.push(message);
		}
		Ok(messages)
	}
}

impl Default for LineReader {
	fn default() -> Self {
		Self::new()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn role_round_trip() {
		for role in [
			InternalProcessRole::Coordinator,
			InternalProcessRole::Server,
			InternalProcessRole::SessionWorker,
		] {
			assert_eq!(InternalProcessRole::from_str(role.as_str()), Some(role));
		}
		assert_eq!(InternalProcessRole::from_str("other"), None);
	}

	#[test]
	fn get_role_rejects_unknown_value() {
		let _guard = crate::experimental::TEST_GUARD.lock();
		// SAFETY: single-threaded test touching the same env var.
		std::env::set_var(INTERNAL_PROCESS_ENV, "bogus");
		let error = get_internal_process_role().unwrap_err();
		assert_eq!(error, "Unsupported internal process role: bogus");
		std::env::remove_var(INTERNAL_PROCESS_ENV);
		assert_eq!(get_internal_process_role().unwrap(), None);
	}

	#[test]
	fn encode_control_line_appends_newline() {
		let line = encode_control_line(&serde_json::json!({"type": "ping"})).unwrap();
		assert_eq!(line, "{\"type\":\"ping\"}\n");
	}

	#[test]
	fn line_reader_frames_and_rejects() {
		let mut reader = LineReader::new();
		let messages = reader.push(b"{\"a\":1}\n{\"b\":2}\n{\"c").unwrap();
		assert_eq!(messages.len(), 2);
		// `{"c:3}` is invalid JSON — the line rejects on completion.
		assert_eq!(reader.push(b":3}\n").unwrap_err(), "Coordinator sent invalid JSON");
		assert_eq!(reader.push(b"not json\n").unwrap_err(), "Coordinator sent invalid JSON");
		assert_eq!(
			reader.push(b"{}\n").unwrap(),
			vec![serde_json::json!({})]
		);
	}

	#[test]
	fn oversize_control_line_rejected() {
		let message = serde_json::Value::String("x".repeat(MAX_CONTROL_LINE_BYTES));
		assert_eq!(
			encode_control_line(&message).unwrap_err(),
			"Internal control message is too large"
		);
	}
}
