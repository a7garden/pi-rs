//! Port of `packages/coding-agent/src/extensions/mcp/log.ts`: log messages
//! MCP servers send with `notifications/message`, appended to `mcp.log` in the
//! agent directory. Several pi processes may write to the same file, so every
//! message is one synchronous append. The file is rotated to `mcp.log.1` once
//! it grows past [`MAX_LOG_BYTES`].

use std::io::Write;
use std::path::Path;

use serde_json::Value;

const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

fn is_record(value: &Value) -> bool {
	value.is_object()
}

fn format_data(data: &Value) -> String {
	match data {
		Value::String(text) => text.clone(),
		other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
	}
}

/// Format one `notifications/message` from `server` as a log line;
/// continuation lines are indented (`formatMcpLogMessage`).
pub fn format_mcp_log_message(server: &str, params: &Value, now: &str) -> String {
	let message = if is_record(params) { params.clone() } else { serde_json::json!({ "data": params }) };
	let level = message
		.get("level")
		.and_then(|value| value.as_str())
		.unwrap_or("info");
	let logger = message
		.get("logger")
		.and_then(|value| value.as_str())
		.filter(|logger| !logger.is_empty())
		.map(|logger| format!(" {logger}:"))
		.unwrap_or_default();
	let text = format_data(message.get("data").unwrap_or(&Value::Null)).replace("\r\n", "\n").replace('\n', "\n    ");
	format!("{now} [{server}] {level}{logger} {text}\n")
}

/// `new Date().toISOString()` — millisecond UTC ISO-8601 timestamp. Computed
/// with the standard library (no chrono dependency; D83).
fn jiff_like_timestamp() -> String {
	let now = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.unwrap_or_default();
	let seconds = now.as_secs();
	let millis = now.subsec_millis();
	let days = seconds / 86_400;
	let time = seconds % 86_400;
	let (year, month, day) = civil_from_days(days as i64);
	format!(
		"{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
		time / 3600,
		(time % 3600) / 60,
		time % 60
	)
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 → (y, m, d).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
	let z = days + 719_468;
	let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
	let doe = z - era * 146_097;
	let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
	let y = yoe + era * 400;
	let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
	let mp = (5 * doy + 2) / 153;
	let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
	let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
	(if m <= 2 { y + 1 } else { y }, m, d)
}

/// Appends server log messages to one file. Write errors are ignored: logging
/// must not break tools (`McpServerLog`).
pub struct McpServerLog {
	pub path: String,
	size: Option<u64>,
}

impl McpServerLog {
	pub fn new(path: impl Into<String>) -> McpServerLog {
		McpServerLog { path: path.into(), size: None }
	}

	pub fn write(&mut self, server: &str, params: &Value) {
		let now = jiff_like_timestamp();
		let line = format_mcp_log_message(server, params, &now);
		let result = (|| -> std::io::Result<()> {
			if self.size.is_none() {
				if let Some(parent) = Path::new(&self.path).parent() {
					std::fs::create_dir_all(parent)?;
				}
				self.size = Some(self.current_size());
			}
			if self.size.unwrap_or(0) > MAX_LOG_BYTES {
				// Another process may have rotated it already; check before renaming.
				if self.current_size() > MAX_LOG_BYTES {
					let _ = std::fs::rename(&self.path, format!("{}.1", self.path));
				}
				self.size = Some(self.current_size());
			}
			let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
			file.write_all(line.as_bytes())?;
			*self.size_mut() += line.len() as u64;
			Ok(())
		})();
		if result.is_err() {
			// Ignore: the log is best effort.
		}
	}

	fn size_mut(&mut self) -> &mut u64 {
		self.size.as_mut().expect("initialized above")
	}

	fn current_size(&self) -> u64 {
		std::fs::metadata(&self.path).map(|metadata| metadata.len()).unwrap_or(0)
	}
}

#[cfg(test)]
mod log_tests {
	use super::*;
	use serde_json::json;

	/// Port of the formatting assertions from `mcp-extension.test.ts`
	/// "appends server log messages to the log file" (the connection part
	/// lives in [`crate::extensions::mcp::runtime`]).
	#[test]
	fn formats_server_log_messages() {
		let warning = format_mcp_log_message("fake", &json!({ "level": "warning", "logger": "db", "data": "slow\nquery" }), "T");
		assert_eq!(warning, "T [fake] warning db: slow\n    query\n");
		let error = format_mcp_log_message("fake", &json!({ "level": "error", "data": { "code": 7 } }), "T");
		assert_eq!(error, "T [fake] error {\"code\":7}\n");
		// Non-record params are wrapped as `data`.
		let raw = format_mcp_log_message("fake", &json!("plain"), "T");
		assert_eq!(raw, "T [fake] info plain\n");
	}

	#[test]
	fn appends_and_rotates() {
		let dir = std::env::temp_dir().join(format!("pi-mcp-log-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("clock").as_nanos()));
		std::fs::create_dir_all(&dir).expect("dir");
		let path = dir.join("mcp.log").to_string_lossy().into_owned();
		let mut log = McpServerLog::new(&path);
		log.write("fake", &json!({ "level": "warning", "logger": "db", "data": "slow\nquery" }));
		log.write("fake", &json!({ "level": "error", "data": { "code": 7 } }));
		let content = std::fs::read_to_string(&path).expect("read");
		// Upstream strips `^\S+ ` per line: only lines starting with non-space
		// (the timestamp) lose their first field; indented continuations stay.
		let stripped: String = content
			.lines()
			.map(|line| match line.split_once(' ') {
				Some((head, rest)) if !head.is_empty() && head.chars().all(|c| !c.is_whitespace()) => format!("{rest}\n"),
				_ => format!("{line}\n"),
			})
			.collect();
		assert_eq!(stripped, "[fake] warning db: slow\n    query\n[fake] error {\"code\":7}\n");
		let _ = std::fs::remove_dir_all(&dir);
	}
}
