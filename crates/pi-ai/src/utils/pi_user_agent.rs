//! Port of `packages/ai/src/utils/pi-user-agent.ts`.

/// `pi (<platform> <release>; <arch>)` (upstream `getPiUserAgent`). The
/// upstream Node `os` module values map to `std::env::consts`; the browser
/// fallback is not applicable and never occurs.
pub fn get_pi_user_agent() -> String {
	let platform = platform_name();
	let arch = match std::env::consts::ARCH {
		"aarch64" => "arm64".to_string(),
		other => other.to_string(),
	};
	format!("pi ({} {}; {})", platform, os_release(), arch)
}

fn platform_name() -> String {
	match std::env::consts::OS {
		"macos" => "darwin".to_string(),
		other => other.to_string(),
	}
}

/// Node `os.release()` equivalent: Darwin kernel version on macOS, `uname -r`
/// elsewhere.
fn os_release() -> String {
	#[cfg(target_os = "macos")]
	{
		if let Ok(output) = std::process::Command::new("sw_vers").arg("-productVersion").output() {
			if output.status.success() {
				let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
				if !version.is_empty() {
					// Node reports the Darwin kernel version; map the common majors.
					let major: u32 = version.split('.').next().and_then(|v| v.parse().ok()).unwrap_or(0);
					let darwin_major = major + 4;
					return format!("{darwin_major}.0.0");
				}
			}
		}
	}
	if let Ok(output) = std::process::Command::new("uname").arg("-r").output() {
		if output.status.success() {
			return String::from_utf8_lossy(&output.stdout).trim().to_string();
		}
	}
	std::env::consts::OS.to_string()
}
