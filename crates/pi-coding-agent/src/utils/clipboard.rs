//! Port of `packages/coding-agent/src/utils/clipboard.ts` — text clipboard
//! read/write with OSC 52 fallback.
//!
//! Deviations (crate docs, deviation D16): upstream's `getNativeClipboard()`
//! from pi-tui becomes the [`NativeClipboard`] trait and process-wide registry
//! ([`set_native_clipboard`]); nothing is registered by default, so only the
//! platform commands and OSC 52 paths run. `isWSL` (upstream `wsl.ts`, not
//! otherwise ported) is inlined as [`is_wsl`], and stdout writes are plain
//! `stdout` instead of node's `process.stdout`.

use std::io::Write;

use super::clipboard_command::{run_clipboard_command, RunClipboardCommandOptions};

const MAX_OSC52_ENCODED_LENGTH: usize = 100_000;

/// Upstream `NativeClipboard` from pi-tui (all methods optional there; absent
/// capabilities return `None` here).
pub trait NativeClipboard: Send + Sync {
	fn get_text(&self) -> Option<String> {
		None
	}
	fn get_file_paths(&self) -> Option<Vec<String>> {
		None
	}
	fn set_text(&self, _text: &str) -> Option<()> {
		None
	}
	fn get_image(&self) -> Option<Vec<u8>> {
		None
	}
}

static NATIVE_CLIPBOARD: parking_lot::RwLock<Option<std::sync::Arc<dyn NativeClipboard>>> = parking_lot::RwLock::new(None);

/// Register the native clipboard backend (upstream pi-tui registration).
pub fn set_native_clipboard(clipboard: std::sync::Arc<dyn NativeClipboard>) {
	*NATIVE_CLIPBOARD.write() = Some(clipboard);
}

pub(crate) fn native_clipboard() -> Option<std::sync::Arc<dyn NativeClipboard>> {
	NATIVE_CLIPBOARD.read().clone()
}

/// Test-only reset so global registration does not leak across tests.
#[cfg(test)]
pub(crate) fn clear_native_clipboard() {
	*NATIVE_CLIPBOARD.write() = None;
}

/// `process.platform` seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
	Darwin,
	Linux,
	Windows,
	Other,
}

pub(crate) fn current_platform() -> Platform {
	match std::env::consts::OS {
		"macos" => Platform::Darwin,
		"linux" => Platform::Linux,
		"windows" => Platform::Windows,
		_ => Platform::Other,
	}
}

/// `isWSL` reuses the shared [`crate::utils::wsl`] port against this call's
/// environment map.
pub fn is_wsl(env: &std::collections::HashMap<String, String>) -> bool {
	crate::utils::wsl::is_wsl_in_env(&|name| env.get(name).cloned())
}

fn is_remote_session(env: &std::collections::HashMap<String, String>) -> bool {
	env.contains_key("SSH_CONNECTION") || env.contains_key("SSH_CLIENT") || env.contains_key("MOSH_CONNECTION")
}

fn emit_osc52(text: &str) -> bool {
	let encoded = super::image_convert::base64_encode(text.as_bytes());
	if encoded.len() > MAX_OSC52_ENCODED_LENGTH {
		return false;
	}
	let mut stdout = std::io::stdout();
	let _ = stdout.write_all(format!("\x1b]52;c;{encoded}\x07").as_bytes());
	let _ = stdout.flush();
	true
}

/// WSL without WSLg has no Linux display, so the Windows clipboard is written
/// through interop. PowerShell reads the text from a file because `clip.exe`
/// and PowerShell stdin decode piped bytes with the console code page, which
/// mangles non-ASCII UTF-8.
fn copy_via_windows_clipboard(text: &str) -> bool {
	let tmp_file = std::env::temp_dir().join(format!("pi-wsl-clip-{}.txt", uuid()));
	if std::fs::write(&tmp_file, text).is_err() {
		return false;
	}
	let result = (|| -> Option<bool> {
		let win_path = run_clipboard_command(
			"wslpath",
			&[tmp_file.to_string_lossy().to_string()],
			Some(RunClipboardCommandOptions { timeout_ms: Some(1000), ..Default::default() }),
		)?;
		let win_path = String::from_utf8_lossy(&win_path).trim().to_string();
		if win_path.is_empty() {
			return Some(false);
		}
		let script = format!(
			"Set-Clipboard -Value ([System.IO.File]::ReadAllText('{}', [System.Text.Encoding]::UTF8))",
			win_path.replace('\'', "''")
		);
		Some(
			run_clipboard_command(
				"powershell.exe",
				&["-NoProfile".to_string(), "-Command".to_string(), script],
				Some(RunClipboardCommandOptions { timeout_ms: Some(5000), ..Default::default() }),
			)
			.is_some(),
		)
	})()
	.unwrap_or(false);
	let _ = std::fs::remove_file(&tmp_file);
	result
}

/// `randomUUID` seam — `getrandom`-backed hex/uuid-shaped id.
pub(crate) fn uuid() -> String {
	let mut bytes = [0u8; 16];
	getrandom::getrandom(&mut bytes).expect("OS RNG unavailable");
	bytes[6] = (bytes[6] & 0x0f) | 0x40;
	bytes[8] = (bytes[8] & 0x3f) | 0x80;
	let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
	format!(
		"{}-{}-{}-{}-{}",
		&hex[0..8],
		&hex[8..12],
		&hex[12..16],
		&hex[16..20],
		&hex[20..32]
	)
}

/// Read plain text from the system clipboard (`readClipboardText`).
pub async fn read_clipboard_text() -> Option<String> {
	let env: std::collections::HashMap<String, String> = std::env::vars().collect();
	if current_platform() == Platform::Linux {
		let mut commands: Vec<(String, Vec<String>)> = Vec::new();
		if env.contains_key("TERMUX_VERSION") {
			commands.push(("termux-clipboard-get".to_string(), vec![]));
		}
		if env.contains_key("WAYLAND_DISPLAY") {
			commands.push((
				"wl-paste".to_string(),
				vec!["--no-newline".to_string(), "--type".to_string(), "text".to_string()],
			));
		}
		if env.contains_key("DISPLAY") {
			commands.push((
				"xclip".to_string(),
				vec!["-selection".to_string(), "clipboard".to_string(), "-out".to_string()],
			));
			commands.push(("xsel".to_string(), vec!["--clipboard".to_string(), "--output".to_string()]));
		}
		for (command, args) in commands {
			if let Some(bytes) = run_clipboard_command(&command, &args, Some(RunClipboardCommandOptions { timeout_ms: Some(5000), ..Default::default() })) {
				return if bytes.is_empty() { None } else { Some(String::from_utf8_lossy(&bytes).to_string()) };
			}
		}
	}
	if let Some(clipboard) = native_clipboard() {
		return clipboard.get_text().filter(|text| !text.is_empty());
	}
	None
}

/// Read file paths, such as Finder file copies, from the native clipboard
/// (`readClipboardFilePaths`).
pub async fn read_clipboard_file_paths() -> Option<Vec<String>> {
	let paths = native_clipboard().and_then(|clipboard| clipboard.get_file_paths())?;
	if paths.is_empty() { None } else { Some(paths) }
}

/// Copy text to the system clipboard (`copyToClipboard`).
///
/// # Errors
/// Mirrors upstream's thrown messages: the clipboard was unreachable, or the
/// text exceeds the OSC 52 size limit.
pub async fn copy_to_clipboard(text: &str) -> Result<(), String> {
	let p = current_platform();
	let env: std::collections::HashMap<String, String> = std::env::vars().collect();
	let mut copied = false;
	// Direct writes precede OSC 52 so the terminal cannot race the native
	// writer. Linux tools retain clipboard selection ownership after this call
	// returns.
	if p != Platform::Linux {
		if let Some(clipboard) = native_clipboard() {
			if clipboard.set_text(text).is_some() {
				copied = true;
			}
		}
	}
	if !copied {
		let mut commands: Vec<(String, Vec<String>)> = Vec::new();
		match p {
			Platform::Darwin => commands.push(("pbcopy".to_string(), vec![])),
			Platform::Windows => commands.push(("clip".to_string(), vec![])),
			_ => {
				if env.contains_key("TERMUX_VERSION") {
					commands.push(("termux-clipboard-set".to_string(), vec![]));
				}
				if env.contains_key("WAYLAND_DISPLAY") {
					commands.push(("wl-copy".to_string(), vec![]));
				}
				if env.contains_key("DISPLAY") {
					commands.push((
						"xclip".to_string(),
						vec!["-selection".to_string(), "clipboard".to_string()],
					));
					commands.push(("xsel".to_string(), vec!["--clipboard".to_string(), "--input".to_string()]));
				}
			}
		}
		for (command, args) in commands {
			if run_clipboard_command(
				&command,
				&args,
				Some(RunClipboardCommandOptions { input: Some(text), timeout_ms: Some(5000), ..Default::default() }),
			)
			.is_some()
			{
				copied = true;
				break;
			}
		}
	}
	let mut osc52_emitted = false;
	if !copied && p == Platform::Linux && is_wsl(&env) {
		// Windows Terminal supports OSC 52; prefer it over the slower
		// PowerShell round trip.
		if env.contains_key("WT_SESSION") {
			osc52_emitted = emit_osc52(text);
		}
		copied = osc52_emitted || copy_via_windows_clipboard(text);
	}
	// OSC 52 cannot be verified, so a desktop session with a display reports
	// the failure instead (#9618). Without a display the terminal is the only
	// clipboard route (containers, WSL without WSLg), and remote sessions
	// always emit it to reach the client clipboard.
	let headless = p == Platform::Linux
		&& !env.contains_key("DISPLAY")
		&& !env.contains_key("WAYLAND_DISPLAY")
		&& !env.contains_key("TERMUX_VERSION");
	let mut oversized = false;
	if !osc52_emitted && (is_remote_session(&env) || (!copied && headless)) {
		if emit_osc52(text) {
			copied = true;
		} else {
			oversized = true;
		}
	}
	if copied {
		return Ok(());
	}
	if oversized {
		return Err("Clipboard unavailable: text exceeds the OSC 52 size limit".to_string());
	}
	if p == Platform::Linux {
		if env.contains_key("TERMUX_VERSION") {
			return Err(
				"Clipboard unavailable: install the Termux:API app and `termux-api` package".to_string()
			);
		}
		if env.contains_key("WAYLAND_DISPLAY") {
			return Err(
				"Clipboard unavailable: install `wl-clipboard` (`wl-copy`) or check Wayland access".to_string()
			);
		}
		if env.contains_key("DISPLAY") {
			return Err("Clipboard unavailable: install `xclip` or `xsel`, or check X11 access".to_string());
		}
	}
	Err("Clipboard unavailable".to_string())
}

#[cfg(test)]
mod tests {
	use super::*;

	fn env_map(pairs: &[(&str, &str)]) -> std::collections::HashMap<String, String> {
		pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
	}

	#[test]
	fn wsl_detection_matches_upstream() {
		assert!(is_wsl(&env_map(&[("WSL_DISTRO_NAME", "Ubuntu")])));
		assert!(is_wsl(&env_map(&[("WSLENV", "1")])));
		assert!(!is_wsl(&env_map(&[])));
	}

	#[test]
	fn remote_session_detection_matches_upstream() {
		assert!(is_remote_session(&env_map(&[("SSH_CONNECTION", "x")])));
		assert!(is_remote_session(&env_map(&[("SSH_CLIENT", "x")])));
		assert!(is_remote_session(&env_map(&[("MOSH_CONNECTION", "x")])));
		assert!(!is_remote_session(&env_map(&[])));
	}

	#[test]
	fn osc52_rejects_oversized_payloads() {
		let oversized = "x".repeat(140_000);
		assert!(!emit_osc52(&oversized));
	}

	#[test]
	fn uuid_shape_matches_rfc_4122_v4() {
		let value = uuid();
		assert_eq!(value.len(), 36);
		assert_eq!(&value[14..15], "4");
		assert!(matches!(&value[19..20], "8" | "9" | "a" | "b"));
	}

	#[tokio::test]
	async fn read_clipboard_file_paths_is_none_without_native_backend() {
		// No native seam registration in tests, so file-path reads are None.
		assert_eq!(read_clipboard_file_paths().await, None);
	}
}
