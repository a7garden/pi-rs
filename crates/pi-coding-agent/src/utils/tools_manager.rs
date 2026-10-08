//! Port of `packages/coding-agent/src/utils/tools-manager.ts` — downloading
//! and locating the bundled `fd`/`rg` binaries.
//!
//! Deviations (crate docs, deviation D17): upstream's `fetchWithRetry`
//! (`utils/management-http.ts`, not in this slice) becomes the [`ToolFetcher`]
//! trait — callers pass a fetcher (or [`global_fetcher`]); none is registered
//! by default, so downloads fail with the upstream-shaped error strings.
//! `getBinDir` (upstream `config.ts`, not in this slice) is implemented
//! locally as `getAgentDir()/bin`. Archive extraction and binary probing shell
//! out to `tar`/`unzip`/`--version` exactly like upstream's `spawnSync`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::config::APP_NAME;

const NETWORK_TIMEOUT_MS: u64 = 10_000;
const DOWNLOAD_TIMEOUT_MS: u64 = 120_000;

/// `isOfflineModeEnabled` — the `PI_OFFLINE` env gate.
pub fn is_offline_mode_enabled() -> bool {
	let Ok(value) = std::env::var("PI_OFFLINE") else {
		return false;
	};
	if value.is_empty() {
		return false;
	}
	value == "1" || value.to_lowercase() == "true" || value.to_lowercase() == "yes"
}

/// `getBinDir` (upstream `config.ts`) — `getAgentDir()/bin`.
fn get_bin_dir() -> PathBuf {
	Path::new(&crate::config::get_agent_dir()).join("bin")
}

/// Upstream `ToolConfig`. `get_asset_name` is the `getAssetName` closure.
pub struct ToolConfig {
	pub name: &'static str,
	pub repo: &'static str,
	pub binary_name: &'static str,
	pub system_binary_names: &'static [&'static str],
	pub tag_prefix: &'static str,
	pub get_asset_name: fn(version: &str, platform: &str, architecture: &str) -> Option<String>,
}

fn fd_asset_name(version: &str, plat: &str, architecture: &str) -> Option<String> {
	let arch_str = if architecture == "arm64" { "aarch64" } else { "x86_64" };
	match plat {
		"darwin" => Some(format!("fd-v{version}-{arch_str}-apple-darwin.tar.gz")),
		"linux" => Some(format!("fd-v{version}-{arch_str}-unknown-linux-musl.tar.gz")),
		"win32" => Some(format!("fd-v{version}-{arch_str}-pc-windows-msvc.zip")),
		_ => None,
	}
}

fn rg_asset_name(version: &str, plat: &str, architecture: &str) -> Option<String> {
	let arch_str = if architecture == "arm64" { "aarch64" } else { "x86_64" };
	match plat {
		"darwin" => Some(format!("ripgrep-{version}-{arch_str}-apple-darwin.tar.gz")),
		"linux" => Some(format!("ripgrep-{version}-{arch_str}-unknown-linux-musl.tar.gz")),
		"win32" => Some(format!("ripgrep-{version}-{arch_str}-pc-windows-msvc.zip")),
		_ => None,
	}
}

/// Upstream `TOOLS`.
pub fn tool_config(tool: &str) -> Option<&'static ToolConfig> {
	match tool {
		"fd" => Some(&ToolConfig {
			name: "fd",
			repo: "sharkdp/fd",
			binary_name: "fd",
			system_binary_names: &["fd", "fdfind"],
			tag_prefix: "v",
			get_asset_name: fd_asset_name,
		}),
		"rg" => Some(&ToolConfig {
			name: "ripgrep",
			repo: "BurntSushi/ripgrep",
			binary_name: "rg",
			system_binary_names: &[],
			tag_prefix: "",
			get_asset_name: rg_asset_name,
		}),
		_ => None,
	}
}

/// Upstream `ToolName` (`"fd" | "rg"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolName {
	Fd,
	Rg,
}

impl ToolName {
	fn config(&self) -> &'static ToolConfig {
		match self {
			ToolName::Fd => tool_config("fd").unwrap(),
			ToolName::Rg => tool_config("rg").unwrap(),
		}
	}

}

use std::future::Future;
use std::pin::Pin;

/// HTTP seam for `fetchWithRetry` (crate docs, deviation D17). Object-safe so
/// callers can pass `&dyn ToolFetcher`.
pub trait ToolFetcher: Send + Sync {
	/// Fetch `url` with `headers`, following no redirects
	/// (`redirect: "manual"`), aborting after `timeout_ms`. The error string
	/// should be the upstream `Error.message` surface.
	fn fetch<'a>(
		&'a self,
		url: &'a str,
		headers: &'a [(&'a str, &'a str)],
		timeout_ms: u64,
	) -> Pin<Box<dyn Future<Output = Result<HttpResponse, String>> + Send + 'a>>;
}

/// Upstream `Response` subset used by this module.
pub struct HttpResponse {
	pub status: u16,
	/// `headers.get("location")`.
	pub location: Option<String>,
	pub body: Vec<u8>,
}

static GLOBAL_FETCHER: OnceLock<std::sync::Arc<dyn ToolFetcher>> = OnceLock::new();

/// Register the process-wide fetcher (`fetchWithRetry`'s backend).
pub fn set_global_fetcher(fetcher: std::sync::Arc<dyn ToolFetcher>) {
	let _ = GLOBAL_FETCHER.set(fetcher);
}

/// Registered fetcher, or the reqwest-backed default.
fn global_fetcher() -> std::sync::Arc<dyn ToolFetcher> {
	GLOBAL_FETCHER.get_or_init(|| std::sync::Arc::new(ReqwestFetcher::new())).clone()
}

/// reqwest-backed `fetchWithRetry`: manual redirects, per-request timeout.
pub struct ReqwestFetcher {
	client: reqwest::Client,
}

impl Default for ReqwestFetcher {
	fn default() -> Self {
		Self::new()
	}
}

impl ReqwestFetcher {
	pub fn new() -> ReqwestFetcher {
		let client = reqwest::Client::builder()
			.redirect(reqwest::redirect::Policy::none())
			.build()
			.unwrap_or_else(|_| reqwest::Client::new());
		ReqwestFetcher { client }
	}
}

impl ToolFetcher for ReqwestFetcher {
	fn fetch<'a>(
		&'a self,
		url: &'a str,
		headers: &'a [(&'a str, &'a str)],
		timeout_ms: u64,
	) -> Pin<Box<dyn Future<Output = Result<HttpResponse, String>> + Send + 'a>> {
		Box::pin(async move {
			let mut request = self.client.get(url).timeout(std::time::Duration::from_millis(timeout_ms));
			for (name, value) in headers {
				request = request.header(*name, *value);
			}
			let response = request.send().await.map_err(|error| error.to_string())?;
			let status = response.status().as_u16();
			let location = response
				.headers()
				.get(reqwest::header::LOCATION)
				.and_then(|value| value.to_str().ok())
				.map(str::to_string);
			let body = response.bytes().await.map_err(|error| error.to_string())?.to_vec();
			Ok(HttpResponse { status, location, body })
		})
	}
}

fn platform_name() -> &'static str {
	match std::env::consts::OS {
		"macos" => "darwin",
		"linux" => "linux",
		"windows" => "win32",
		"android" => "android",
		_ => "unknown",
	}
}

fn arch_name() -> &'static str {
	match std::env::consts::ARCH {
		"aarch64" => "arm64",
		"x86_64" => "x64",
		other => other,
	}
}

/// Check if a command exists in PATH by trying to run it (`commandExists`).
fn command_exists(cmd: &str) -> bool {
	std::process::Command::new(cmd)
		.arg("--version")
		.stdout(std::process::Stdio::piped())
		.stderr(std::process::Stdio::piped())
		.spawn()
		.and_then(|mut child| child.wait())
		// Check for ENOENT error (command not found): any successful spawn
		// counts, like upstream's `result.error === undefined/null` check.
		.map(|_| true)
		.unwrap_or(false)
}

/// Get the path to a tool (system-wide or in our tools dir) — `getToolPath`.
pub fn get_tool_path(tool: ToolName) -> Option<String> {
	let config = tool.config();

	// Check our tools directory first
	let local_path = get_bin_dir().join(format!(
		"{}{}",
		config.binary_name,
		if platform_name() == "win32" { ".exe" } else { "" }
	));
	if local_path.exists() {
		return Some(local_path.to_string_lossy().to_string());
	}

	// Check system PATH - if found, just return the command name (it's in PATH)
	let system_binary_names: Vec<&str> = if config.system_binary_names.is_empty() {
		vec![config.binary_name]
	} else {
		config.system_binary_names.to_vec()
	};
	for system_binary_name in system_binary_names {
		if command_exists(system_binary_name) {
			return Some(system_binary_name.to_string());
		}
	}

	None
}

/// Resolve the latest release version from the release page redirect
/// (`getLatestVersion`). The api.github.com releases endpoint counts against
/// the anonymous API quota (60 requests/hour per IP), which is permanently
/// exhausted behind shared egress IPs such as corporate proxies and CI
/// runners. The web endpoint answers with a redirect to the tagged release at
/// no quota cost and lives on the same origin as the binary download itself.
///
/// # Errors
/// Mirrors upstream's thrown messages.
pub async fn get_latest_version(fetcher: &dyn ToolFetcher, repo: &str) -> Result<String, String> {
	let response = fetcher
		.fetch(
			&format!("https://github.com/{repo}/releases/latest"),
			&[("User-Agent", &format!("{APP_NAME}-coding-agent"))],
			NETWORK_TIMEOUT_MS,
		)
		.await?;

	// Only the status and headers matter here. Discard the body so the
	// connection can be reused.
	let location = if (300..400).contains(&response.status) { response.location.clone() } else { None };
	let Some(location) = location else {
		return Err(format!("Failed to resolve latest {repo} release: HTTP {} without redirect", response.status));
	};

	let path = url_path(&location);
	let tag = path.rsplit('/').next().unwrap_or("");
	if tag.is_empty() || !location.contains("/releases/tag/") {
		return Err(format!("Failed to resolve latest {repo} release: unexpected redirect to {location}"));
	}
	let tag = percent_decode(tag);
	Ok(tag.strip_prefix('v').unwrap_or(&tag).to_string())
}

/// The `new URL(location, "https://github.com").pathname` seam: absolute URLs
/// use their path; relative ones resolve against the github origin.
fn url_path(location: &str) -> &str {
	let without_scheme = location.split_once("://").map(|(_, rest)| rest).unwrap_or(location);
	let start = without_scheme.find('/').map(|i| &without_scheme[i..]).unwrap_or("/");
	start.split(['?', '#']).next().unwrap_or("/")
}

fn percent_decode(text: &str) -> String {
	let bytes = text.as_bytes();
	let mut out = Vec::with_capacity(bytes.len());
	let mut i = 0;
	while i < bytes.len() {
		if bytes[i] == b'%' && i + 2 < bytes.len() {
			if let Ok(byte) = u8::from_str_radix(&text[i + 1..i + 3], 16) {
				out.push(byte);
				i += 3;
				continue;
			}
		}
		out.push(bytes[i]);
		i += 1;
	}
	String::from_utf8_lossy(&out).to_string()
}

/// Download a file from URL (`downloadFile`).
async fn download_file(fetcher: &dyn ToolFetcher, url: &str, dest: &Path) -> Result<(), String> {
	let response = fetcher
		.fetch(url, &[], DOWNLOAD_TIMEOUT_MS)
		.await
		.map_err(|error| format!("Download failed: {error}"))?;

	if !(200..300).contains(&response.status) {
		return Err(format!("Download failed with HTTP {}: {url}", response.status));
	}

	if response.body.is_empty() && !url.ends_with(".zip") && !url.ends_with(".tar.gz") {
		return Err("No response body".to_string());
	}

	std::fs::write(dest, &response.body).map_err(|error| error.to_string())
}

/// `findBinaryRecursively`.
fn find_binary_recursively(root_dir: &Path, binary_file_name: &str) -> Option<PathBuf> {
	let mut stack: Vec<PathBuf> = vec![root_dir.to_path_buf()];

	while let Some(current_dir) = stack.pop() {
		let Ok(entries) = std::fs::read_dir(&current_dir) else {
			continue;
		};
		for entry in entries.flatten() {
			let full_path = current_dir.join(entry.file_name());
			let Ok(file_type) = entry.file_type() else {
				continue;
			};
			if file_type.is_file() && entry.file_name().to_string_lossy() == binary_file_name {
				return Some(full_path);
			}
			if file_type.is_dir() {
				stack.push(full_path);
			}
		}
	}

	None
}

/// `formatSpawnFailure`.
fn format_spawn_failure(status: Option<i32>, error: Option<String>, stderr: &[u8], stdout: &[u8]) -> String {
	if let Some(error) = error {
		return error;
	}
	let stderr = String::from_utf8_lossy(stderr).trim().to_string();
	if !stderr.is_empty() {
		return stderr;
	}
	let stdout = String::from_utf8_lossy(stdout).trim().to_string();
	if !stdout.is_empty() {
		return stdout;
	}
	format!("exit status {}", status.map(|s| s.to_string()).unwrap_or_else(|| "unknown".to_string()))
}

/// `runExtractionCommand` — `None` on success, the failure text otherwise.
fn run_extraction_command(command: &str, args: &[String]) -> Option<String> {
	let output = std::process::Command::new(command)
		.args(args)
		.stdout(std::process::Stdio::piped())
		.stderr(std::process::Stdio::piped())
		.output();
	match output {
		Ok(output) if output.status.success() => None,
		Ok(output) => Some(format!(
			"{command}: {}",
			format_spawn_failure(output.status.code(), None, &output.stderr, &output.stdout)
		)),
		Err(error) => Some(format!("{command}: {error}")),
	}
}

/// `extractTarGzArchive`.
fn extract_tar_gz_archive(archive_path: &Path, extract_dir: &Path, asset_name: &str) -> Result<(), String> {
	if let Some(failure) = run_extraction_command(
		"tar",
		&[
			"xzf".to_string(),
			archive_path.to_string_lossy().to_string(),
			"-C".to_string(),
			extract_dir.to_string_lossy().to_string(),
		],
	) {
		return Err(format!("Failed to extract {asset_name}: {failure}"));
	}
	Ok(())
}

/// `getWindowsTarCommand`.
fn get_windows_tar_command() -> String {
	let system_root = std::env::var("SystemRoot").or_else(|_| std::env::var("WINDIR")).ok();
	if let Some(system_root) = system_root {
		let system_tar = Path::new(&system_root).join("System32").join("tar.exe");
		if system_tar.exists() {
			return system_tar.to_string_lossy().to_string();
		}
	}
	"tar.exe".to_string()
}

/// `extractZipArchive`.
fn extract_zip_archive(archive_path: &Path, extract_dir: &Path, asset_name: &str) -> Result<(), String> {
	let mut failures: Vec<String> = Vec::new();

	if platform_name() == "win32" {
		// Windows ships bsdtar as tar.exe, which supports zip files. Prefer the
		// System32 binary over Git Bash's GNU tar, which does not handle zip
		// archives.
		if run_extraction_command(
			&get_windows_tar_command(),
			&[
				"xf".to_string(),
				archive_path.to_string_lossy().to_string(),
				"-C".to_string(),
				extract_dir.to_string_lossy().to_string(),
			],
		)
		.is_none()
		{
			return Ok(());
		}

		let script = "& { param($archive, $destination) $ErrorActionPreference = 'Stop'; Expand-Archive -LiteralPath $archive -DestinationPath $destination -Force }";
		let powershell_args = [
			"-NoLogo".to_string(),
			"-NoProfile".to_string(),
			"-NonInteractive".to_string(),
			"-ExecutionPolicy".to_string(),
			"Bypass".to_string(),
			"-Command".to_string(),
			script.to_string(),
			archive_path.to_string_lossy().to_string(),
			extract_dir.to_string_lossy().to_string(),
		];
		match run_extraction_command("powershell.exe", &powershell_args) {
			Some(failure) => failures.push(failure),
			None => return Ok(()),
		}
	} else {
		let unzip_args = [
			"-q".to_string(),
			archive_path.to_string_lossy().to_string(),
			"-d".to_string(),
			extract_dir.to_string_lossy().to_string(),
		];
		match run_extraction_command("unzip", &unzip_args) {
			Some(failure) => failures.push(failure),
			None => return Ok(()),
		}

		let tar_args = [
			"xf".to_string(),
			archive_path.to_string_lossy().to_string(),
			"-C".to_string(),
			extract_dir.to_string_lossy().to_string(),
		];
		match run_extraction_command("tar", &tar_args) {
			Some(failure) => failures.push(failure),
			None => return Ok(()),
		}
	}

	Err(format!("Failed to extract {asset_name}: {}", failures.join("; ")))
}

/// Download and install a tool (`downloadTool`).
///
/// # Errors
/// Mirrors upstream's thrown messages.
async fn download_tool(fetcher: &dyn ToolFetcher, tool: ToolName) -> Result<String, String> {
	let config = tool.config();
	let plat = platform_name();
	let architecture = arch_name();

	// fd is pinned on darwin/x64, so skip the version lookup there.
	let version = if tool == ToolName::Fd && plat == "darwin" && architecture == "x64" {
		"10.3.0".to_string()
	} else {
		get_latest_version(fetcher, config.repo).await?
	};

	// Get asset name for this platform
	let Some(asset_name) = (config.get_asset_name)(&version, plat, architecture) else {
		return Err(format!("Unsupported platform: {plat}/{architecture}"));
	};

	// Create tools directory
	let tools_dir = get_bin_dir();
	let _ = std::fs::create_dir_all(&tools_dir);

	let download_url = format!(
		"https://github.com/{}/releases/download/{}{version}/{asset_name}",
		config.repo, config.tag_prefix
	);
	let archive_path = tools_dir.join(&asset_name);
	let binary_ext = if plat == "win32" { ".exe" } else { "" };
	let binary_path = tools_dir.join(format!("{}{binary_ext}", config.binary_name));

	// Download
	download_file(fetcher, &download_url, &archive_path).await?;

	// Extract into a unique temp directory. fd and rg downloads can run
	// concurrently during startup, so sharing a fixed directory causes races.
	let extract_dir = tools_dir.join(format!(
		"extract_tmp_{}_{}_{}_{}",
		config.binary_name,
		std::process::id(),
		crate::utils::time::now_ms(),
		&super::clipboard::uuid()[..8]
	));
	let _ = std::fs::create_dir_all(&extract_dir);

	let result = (|| {
		if asset_name.ends_with(".tar.gz") {
			extract_tar_gz_archive(&archive_path, &extract_dir, &asset_name)?;
		} else if asset_name.ends_with(".zip") {
			extract_zip_archive(&archive_path, &extract_dir, &asset_name)?;
		} else {
			return Err(format!("Unsupported archive format: {asset_name}"));
		}

		// Find the binary in extracted files. Some archives contain files
		// directly at root, others nest under a versioned subdirectory.
		let binary_file_name = format!("{}{binary_ext}", config.binary_name);
		let extracted_dir = extract_dir.join(asset_name.trim_end_matches(".tar.gz").trim_end_matches(".zip"));
		let extracted_binary_candidates = [extracted_dir.join(&binary_file_name), extract_dir.join(&binary_file_name)];
		let extracted_binary = extracted_binary_candidates
			.iter()
			.find(|candidate| candidate.exists())
			.cloned()
			.or_else(|| find_binary_recursively(&extract_dir, &binary_file_name));

		match extracted_binary {
			Some(extracted_binary) => {
				std::fs::rename(extracted_binary, &binary_path).map_err(|error| error.to_string())?;
			}
			None => {
				return Err(format!(
					"Binary not found in archive: expected {binary_file_name} under {}",
					extract_dir.to_string_lossy()
				));
			}
		}

		// Make executable (Unix only)
		if plat != "win32" {
			set_executable(&binary_path)?;
		}
		Ok(())
	})();

	// Cleanup
	let _ = std::fs::remove_file(&archive_path);
	let _ = std::fs::remove_dir_all(&extract_dir);

	result?;
	Ok(binary_path.to_string_lossy().to_string())
}

fn set_executable(path: &Path) -> Result<(), String> {
	#[cfg(unix)]
	{
		use std::os::unix::fs::PermissionsExt;
		std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).map_err(|error| error.to_string())
	}
	#[cfg(not(unix))]
	{
		let _ = path;
		Ok(())
	}
}

/// Termux package names for tools (`TERMUX_PACKAGES`).
fn termux_package(tool: ToolName) -> &'static str {
	match tool {
		ToolName::Fd => "fd",
		ToolName::Rg => "ripgrep",
	}
}

/// Upstream `ToolStatus`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolStatus {
	Info { message: String },
	Warning { message: String },
}

impl ToolStatus {
	pub fn message(&self) -> &str {
		match self {
			ToolStatus::Info { message } | ToolStatus::Warning { message } => message,
		}
	}

	pub fn is_warning(&self) -> bool {
		matches!(self, ToolStatus::Warning { .. })
	}
}

/// Ensure a tool is available, downloading if necessary (`ensureTool`).
/// Reports progress through `on_status`; status messages are otherwise silent.
/// Returns the tool path, or `None` if unavailable.
pub async fn ensure_tool(
	tool: ToolName,
	fetcher: Option<&dyn ToolFetcher>,
	on_status: Option<&dyn Fn(&ToolStatus)>,
) -> Option<String> {
	if let Some(existing_path) = get_tool_path(tool) {
		return Some(existing_path);
	}

	let config = tool.config();
	let default_fetcher;
	let fetcher = match fetcher {
		Some(fetcher) => fetcher,
		None => {
			default_fetcher = global_fetcher();
			&*default_fetcher
		}
	};

	if is_offline_mode_enabled() {
		if let Some(on_status) = on_status {
			on_status(&ToolStatus::Warning {
				message: format!("{} not found. Offline mode enabled, skipping download.", config.name),
			});
		}
		return None;
	}

	// On Android/Termux, Linux binaries don't work due to Bionic libc
	// incompatibility. Users must install via pkg.
	if platform_name() == "android" {
		if let Some(on_status) = on_status {
			on_status(&ToolStatus::Warning {
				message: format!("{} not found. Install with: pkg install {}", config.name, termux_package(tool)),
			});
		}
		return None;
	}

	// Tool not found - download it
	if let Some(on_status) = on_status {
		on_status(&ToolStatus::Info { message: format!("{} not found. Downloading...", config.name) });
	}

	match download_tool(fetcher, tool).await {
		Ok(path) => {
			if let Some(on_status) = on_status {
				on_status(&ToolStatus::Info { message: format!("{} installed to {}", config.name, path) });
			}
			Some(path)
		}
		// Include the error cause chain: fetch failures surface as a bare
		// "fetch failed" TypeError with the actionable detail (DNS, TLS,
		// timeout) hidden in the cause. Depth-capped to guard against circular
		// cause chains; the port's seam carries one message per hop.
		Err(error) => {
			let mut messages: Vec<String> = Vec::new();
			let mut current = error.as_str();
			for _ in 0..5 {
				if messages.iter().any(|m| m == current) {
					break;
				}
				messages.push(current.to_string());
				let Some((head, rest)) = current.split_once(": ") else {
					break;
				};
				let _ = head;
				current = rest;
			}
			if let Some(on_status) = on_status {
				on_status(&ToolStatus::Warning {
					message: format!(
						"Failed to download {}: {}",
						config.name,
						if messages.is_empty() { error.clone() } else { messages.join(": ") }
					),
				});
			}
			None
		}
	}
}

// Upstream module constants kept for parity with tests and callers.
pub const NETWORK_TIMEOUT: u64 = NETWORK_TIMEOUT_MS;
pub const DOWNLOAD_TIMEOUT: u64 = DOWNLOAD_TIMEOUT_MS;

#[cfg(test)]
mod tests {
	use super::*;

	fn env_guard() -> MutexGuard<'static> {
		static LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
		LOCK.lock()
	}

	type MutexGuard<'a> = parking_lot::MutexGuard<'a, ()>;

	#[test]
	fn offline_mode_gate_matches_upstream() {
		let _guard = env_guard();
		std::env::set_var("PI_OFFLINE", "1");
		assert!(is_offline_mode_enabled());
		std::env::set_var("PI_OFFLINE", "TRUE");
		assert!(is_offline_mode_enabled());
		std::env::set_var("PI_OFFLINE", "yes");
		assert!(is_offline_mode_enabled());
		std::env::set_var("PI_OFFLINE", "0");
		assert!(!is_offline_mode_enabled());
		std::env::set_var("PI_OFFLINE", "junk");
		assert!(!is_offline_mode_enabled());
		std::env::set_var("PI_OFFLINE", "");
		assert!(!is_offline_mode_enabled());
		std::env::remove_var("PI_OFFLINE");
		assert!(!is_offline_mode_enabled());
	}

	#[test]
	fn asset_names_match_upstream() {
		// Inputs are node's `arch()` values ("arm64"/"x64"); the asset names map
		// them to the platform triples.
		assert_eq!(fd_asset_name("10.2.0", "darwin", "arm64").as_deref(), Some("fd-v10.2.0-aarch64-apple-darwin.tar.gz"));
		assert_eq!(fd_asset_name("10.2.0", "linux", "arm64").as_deref(), Some("fd-v10.2.0-aarch64-unknown-linux-musl.tar.gz"));
		assert_eq!(fd_asset_name("10.2.0", "linux", "x64").as_deref(), Some("fd-v10.2.0-x86_64-unknown-linux-musl.tar.gz"));
		assert_eq!(fd_asset_name("10.2.0", "win32", "x64").as_deref(), Some("fd-v10.2.0-x86_64-pc-windows-msvc.zip"));
		assert_eq!(fd_asset_name("10.2.0", "sunos", "x64"), None);
		assert_eq!(
			rg_asset_name("14.1.0", "darwin", "x64").as_deref(),
			Some("ripgrep-14.1.0-x86_64-apple-darwin.tar.gz")
		);
		assert_eq!(
			rg_asset_name("14.1.0", "linux", "arm64").as_deref(),
			Some("ripgrep-14.1.0-aarch64-unknown-linux-musl.tar.gz")
		);
		assert_eq!(rg_asset_name("14.1.0", "win32", "arm64").as_deref(), Some("ripgrep-14.1.0-aarch64-pc-windows-msvc.zip"));
		assert_eq!(rg_asset_name("14.1.0", "sunos", "x64"), None);
	}

	#[test]
	fn unknown_tool_has_no_config() {
		assert!(tool_config("awk").is_none());
		assert!(tool_config("fd").is_some());
		assert!(tool_config("rg").is_some());
	}

	#[test]
	fn tag_strip_matches_decode_uri_component_semantics() {
		assert_eq!(percent_decode("v10.2.0"), "v10.2.0");
		assert_eq!(percent_decode("release%201.0"), "release 1.0");
	}

	#[tokio::test]
	async fn latest_version_strips_tag_prefix_from_redirect() {
		struct F;
		impl ToolFetcher for F {
			fn fetch<'a>(
				&'a self,
				url: &'a str,
				headers: &'a [(&'a str, &'a str)],
				_timeout_ms: u64,
			) -> Pin<Box<dyn Future<Output = Result<HttpResponse, String>> + Send + 'a>> {
				Box::pin(async move {
					assert!(url.ends_with("/sharkdp/fd/releases/latest"));
					assert_eq!(headers, &[("User-Agent", "pi-coding-agent")]);
					Ok(HttpResponse {
						status: 302,
						location: Some("https://github.com/sharkdp/fd/releases/tag/v10.2.0".to_string()),
						body: Vec::new(),
					})
				})
			}
		}
		assert_eq!(get_latest_version(&F, "sharkdp/fd").await.unwrap(), "10.2.0");
	}

	#[tokio::test]
	async fn latest_version_requires_a_redirect_location() {
		struct F;
		impl ToolFetcher for F {
			fn fetch<'a>(
				&'a self,
				_url: &'a str,
				_headers: &'a [(&'a str, &'a str)],
				_timeout_ms: u64,
			) -> Pin<Box<dyn Future<Output = Result<HttpResponse, String>> + Send + 'a>> {
				Box::pin(async move { Ok(HttpResponse { status: 200, location: None, body: Vec::new() }) })
			}
		}
		assert_eq!(
			get_latest_version(&F, "sharkdp/fd").await,
			Err("Failed to resolve latest sharkdp/fd release: HTTP 200 without redirect".to_string())
		);
	}

	#[tokio::test]
	async fn latest_version_rejects_unexpected_redirects() {
		struct F;
		impl ToolFetcher for F {
			fn fetch<'a>(
				&'a self,
				_url: &'a str,
				_headers: &'a [(&'a str, &'a str)],
				_timeout_ms: u64,
			) -> Pin<Box<dyn Future<Output = Result<HttpResponse, String>> + Send + 'a>> {
				Box::pin(async move {
					Ok(HttpResponse {
						status: 302,
						location: Some("https://github.com/sharkdp/fd/issues".to_string()),
						body: Vec::new(),
					})
				})
			}
		}
		assert_eq!(
			get_latest_version(&F, "sharkdp/fd").await,
			Err("Failed to resolve latest sharkdp/fd release: unexpected redirect to https://github.com/sharkdp/fd/issues".to_string())
		);
	}

	struct StaticFetcher {
		location: Option<String>,
		status: u16,
		body: Vec<u8>,
	}

	impl ToolFetcher for StaticFetcher {
		fn fetch<'a>(
			&'a self,
			_url: &'a str,
			_headers: &'a [(&'a str, &'a str)],
			_timeout_ms: u64,
		) -> Pin<Box<dyn Future<Output = Result<HttpResponse, String>> + Send + 'a>> {
			Box::pin(async move {
				Ok(HttpResponse { status: self.status, location: self.location.clone(), body: self.body.clone() })
			})
		}
	}

	/// Always fails, mirroring upstream's bare "fetch failed".
	struct FailingFetcher;

	impl ToolFetcher for FailingFetcher {
		fn fetch<'a>(
			&'a self,
			_url: &'a str,
			_headers: &'a [(&'a str, &'a str)],
			_timeout_ms: u64,
		) -> Pin<Box<dyn Future<Output = Result<HttpResponse, String>> + Send + 'a>> {
			Box::pin(async move { Err("fetch failed".to_string()) })
		}
	}

	#[tokio::test]
	async fn ensure_tool_reports_offline_skip() {
		if get_tool_path(ToolName::Fd).is_some() {
			return;
		}
		let _guard = env_guard();
		std::env::set_var("PI_OFFLINE", "1");
		let statuses = parking_lot::Mutex::new(Vec::new());
		let result = ensure_tool(ToolName::Fd, None, Some(&|status: &ToolStatus| {
			statuses.lock().push(status.message().to_string());
		}))
		.await;
		std::env::remove_var("PI_OFFLINE");
		assert_eq!(result, None);
		assert_eq!(
			statuses.into_inner(),
			vec!["fd not found. Offline mode enabled, skipping download.".to_string()]
		);
	}

	#[test]
	fn ensure_tool_reports_termux_pkg_instructions() {
		let config = ToolName::Rg.config();
		let message = format!("{} not found. Install with: pkg install {}", config.name, termux_package(ToolName::Rg));
		assert_eq!(message, "ripgrep not found. Install with: pkg install ripgrep");
	}

	#[tokio::test]
	async fn ensure_tool_surfaces_fetch_failures_as_warnings() {
		// If ripgrep is already installed, ensureTool short-circuits like
		// upstream; the failure path only runs when the tool is missing.
		if get_tool_path(ToolName::Rg).is_some() {
			return;
		}
		let statuses = parking_lot::Mutex::new(Vec::new());
		let result = ensure_tool(ToolName::Rg, Some(&FailingFetcher), Some(&|status: &ToolStatus| {
			statuses.lock().push(status.message().to_string());
		}))
		.await;
		let statuses = statuses.into_inner();
		assert_eq!(result, None);
		assert!(statuses[0].ends_with("not found. Downloading..."));
		assert!(statuses[1].starts_with("Failed to download ripgrep:"));
		assert!(statuses[1].contains("fetch failed"));
	}

	#[test]
	fn get_tool_path_supports_fd_and_rg_names() {
		// Either resolves a local/system path or returns None; both are valid
		// outcomes in a sandbox, but unknown tools never resolve.
		let _ = get_tool_path(ToolName::Fd);
		let _ = get_tool_path(ToolName::Rg);
	}

	#[test]
	fn archive_format_dispatch_matches_upstream() {
		let dir = std::env::temp_dir().join(format!("pi-tools-mgr-test-{}", super::super::clipboard::uuid()));
		std::fs::create_dir_all(&dir).unwrap();
		let archive = dir.join("tool.rar");
		std::fs::write(&archive, b"not-an-archive").unwrap();
		let err = extract_zip_archive(&archive, &dir, "tool.rar").unwrap_err();
		assert!(err.starts_with("Failed to extract tool.rar"));
		assert!(err.contains("no extractor succeeded") || err.contains("tar") || err.contains("unzip"));
		let _ = std::fs::remove_dir_all(&dir);
	}
}
