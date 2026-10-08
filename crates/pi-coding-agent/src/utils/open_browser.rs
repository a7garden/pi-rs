//! Port of `packages/coding-agent/src/utils/open-browser.ts` — open a URL or
//! file with the platform default handler, never through a shell.
//!
//! Deviation (crate docs, D13): upstream spawns detached and unrefs; Rust
//! spawns and immediately detaches by dropping the [`std::process::Child`]
//! (no zombie reap on unix — the child is reparented to init).

use std::process::{Command, Stdio};

/// Open a URL or file in the platform browser/default handler (`openBrowser`).
///
/// This intentionally never invokes a shell. On Windows, do not use
/// `cmd /c start`: cmd.exe re-parses metacharacters (&, |, ^, ...) before
/// `start` runs, which would make attacker-controlled URLs injectable.
///
/// Launcher failures (for example, a missing `xdg-open`) are swallowed:
/// browser launch is best-effort — callers still present the target to the
/// user, so the failure must not become a process crash.
pub fn open_browser(target: &str) {
	let mut cmd = if cfg!(target_os = "macos") {
		let mut cmd = Command::new("open");
		cmd.arg(target);
		cmd
	} else if cfg!(windows) {
		let mut cmd = Command::new("rundll32");
		cmd.arg("url.dll,FileProtocolHandler").arg(target);
		cmd
	} else {
		let mut cmd = Command::new("xdg-open");
		cmd.arg(target);
		cmd
	};

	cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
	// Best-effort: a failed spawn (or failed detach) is ignored, matching the
	// upstream `.on("error", () => {})`.
	#[cfg(unix)]
	{
		use std::os::unix::process::CommandExt;
		// Detach: setsid so the browser survives us and is not reaped as our
		// child (upstream `.unref()`).
		unsafe {
			let _ = cmd.pre_exec(|| {
				// Ignore the already-group-leader failure; detaching is a no-op.
				let _ = libc::setsid();
				Ok(())
			});
		}
	}
	if let Ok(child) = cmd.spawn() {
		drop(child);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn open_browser_never_panics_on_missing_launcher() {
		// xdg-open is absent in most CI environments; the call must stay
		// best-effort and silent (upstream swallows the spawn error event).
		open_browser("https://example.invalid/");
	}
}
