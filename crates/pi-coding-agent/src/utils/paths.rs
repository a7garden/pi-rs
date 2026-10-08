//! Port of `packages/coding-agent/src/utils/paths.ts` — the subset
//! session-manager depends on: [`normalize_path`] and [`resolve_path`] (plus
//! the [`PathInputOptions`] surface they take).
//!
//! `normalizePath`'s `file://` expansion (JS `fileURLToPath`) is a JS URL seam
//! and is not applied (crate docs, deviation D3).
//!
//! Also ported here: `isLocalPath` and `markPathIgnoredByCloudSync` (used by
//! `core/package-manager.ts`).

use std::path::{Component, Path, PathBuf};

/// `[\u00A0\u2000-\u200A\u202F\u205F\u3000]` (upstream `UNICODE_SPACES`).
fn is_unicode_space(c: char) -> bool {
	matches!(c, '\u{00A0}' | '\u{202F}' | '\u{205F}' | '\u{3000}') || ('\u{2000}'..='\u{200A}').contains(&c)
}

/// Options mirroring upstream `PathInputOptions`.
#[derive(Debug, Clone, Default)]
pub struct PathInputOptions {
	/// Trim leading/trailing whitespace before normalization.
	pub trim: bool,
	/// Expand leading `~` to a home directory. Defaults to true.
	pub expand_tilde: Option<bool>,
	/// Home directory used for `~` expansion. Defaults to the OS home dir.
	pub home_dir: Option<String>,
	/// Strip a leading `@`, used for CLI @file paths.
	pub strip_at_prefix: bool,
	/// Normalize unicode space variants to regular spaces.
	pub normalize_unicode_spaces: bool,
}

/// `os.homedir()` seam: `$HOME` on unix, `USERPROFILE` on windows.
pub fn home_dir() -> String {
	std::env::var(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).unwrap_or_default()
}

/// Convert Git Bash, MSYS, Cygwin, and WSL drive paths to a form native
/// Windows APIs accept (`normalizeWindowsShellPath`).
fn normalize_windows_shell_path(file_path: &str) -> String {
	if !file_path.starts_with('/') || file_path.starts_with("//") || file_path.contains('\\') {
		return file_path.to_string();
	}
	let rest = file_path.strip_prefix('/').unwrap_or(file_path);
	let rest = rest.strip_prefix("mnt/").or_else(|| rest.strip_prefix("cygdrive/")).unwrap_or(rest);
	let mut chars = rest.chars();
	let Some(drive) = chars.next() else { return file_path.to_string() };
	if !drive.is_ascii_alphabetic() {
		return file_path.to_string();
	}
	let suffix = chars.as_str().replace('/', "\\");
	format!("{}:\\{}", drive.to_ascii_uppercase(), suffix)
}

/// Normalize a path: optional trim/unicode-space/at-prefix handling, windows
/// shell-path conversion, `~` expansion (`normalizePath`).
///
/// Deviation: `file://` URLs are returned unchanged (JS `fileURLToPath` seam).
pub fn normalize_path(input: &str, options: &PathInputOptions) -> String {
	let mut normalized = if options.trim { input.trim().to_string() } else { input.to_string() };
	if options.normalize_unicode_spaces {
		normalized = normalized.chars().map(|c| if is_unicode_space(c) { ' ' } else { c }).collect();
	}
	if options.strip_at_prefix && normalized.starts_with('@') {
		normalized = normalized[1..].to_string();
	}

	if cfg!(windows) {
		normalized = normalize_windows_shell_path(&normalized);
	}

	if options.expand_tilde.unwrap_or(true) {
		if normalized == "~" {
			return options.home_dir.clone().unwrap_or_else(home_dir);
		}
		let tilde_slash = normalized.starts_with("~/");
		let tilde_back = cfg!(windows) && normalized.starts_with("~\\");
		if tilde_slash || tilde_back {
			let home = options.home_dir.clone().unwrap_or_else(home_dir);
			return join(&home, &normalized[2..]);
		}
	}

	normalized
}

/// Resolve a path against `base_dir` (default: cwd), lexically normalizing
/// `.`/`..` segments like node's `path.resolve` (`resolvePath`). A relative
/// base resolves against the process cwd, exactly like node.
pub fn resolve_path(input: &str, base_dir: &str) -> String {
	let normalized = normalize_path(input, &PathInputOptions::default());
	let normalized_base = normalize_path(base_dir, &PathInputOptions::default());
	let base = if Path::new(&normalized_base).is_absolute() {
		normalized_base
	} else {
		join(&current_dir_string(), &normalized_base)
	};
	if Path::new(&normalized).is_absolute() {
		lexically_clean(&normalized)
	} else {
		lexically_clean(join(&base, &normalized))
	}
}

/// Canonicalize a path with symlinks resolved, returning the input unchanged
/// if canonicalization fails (upstream `canonicalizePath`).
pub fn canonicalize_path(path: &str) -> String {
	std::fs::canonicalize(path)
		.map(|p| p.to_string_lossy().into_owned())
		.unwrap_or_else(|_| path.to_string())
}

/// `resolvePath(input)` — base directory defaults to the process cwd.
pub fn resolve_path_cwd(input: &str) -> String {
	resolve_path(input, &current_dir_string())
}

fn current_dir_string() -> String {
	std::env::current_dir()
		.map(|dir| dir.to_string_lossy().into_owned())
		.unwrap_or_else(|_| ".".to_string())
}

/// `path.join`-style join (separator per platform, no normalization).
pub fn join(base: &str, rest: &str) -> String {
	let mut path = PathBuf::from(base);
	path.push(rest);
	path.to_string_lossy().into_owned()
}

/// Returns true if the value is NOT a package source (`npm:`, `git:`, etc.),
/// a built-in extension (`builtin:`), or a remote URL protocol. Bare names,
/// relative paths, and `file:` URLs are considered local (upstream
/// `isLocalPath`).
pub fn is_local_path(value: &str) -> bool {
	let trimmed = value.trim();
	// Known non-local prefixes. file: URLs are local paths and are intentionally
	// resolved by resolvePath().
	for prefix in ["npm:", "git:", "github:", "http:", "https:", "ssh:", "builtin:"] {
		if trimmed.starts_with(prefix) {
			return false;
		}
	}
	true
}

/// Mark a path as ignored by cloud-sync providers (Dropbox / Apple File
/// Provider) via platform xattr helpers, ignoring failures (upstream
/// `markPathIgnoredByCloudSync`).
pub fn mark_path_ignored_by_cloud_sync(path: &str) {
	let attrs: &[&str] = if cfg!(target_os = "macos") {
		&["com.dropbox.ignored", "com.apple.fileprovider.ignore#P"]
	} else if cfg!(target_os = "linux") {
		&["user.com.dropbox.ignored"]
	} else {
		&[]
	};
	for attr in attrs {
		if cfg!(target_os = "macos") {
			let _ = crate::utils::child_process::spawn_process_sync("xattr", &["-w", attr, "1", path]);
		} else {
			let _ = crate::utils::child_process::spawn_process_sync("setfattr", &["-n", attr, "-v", "1", path]);
		}
	}
}

/// Cheap file-identity stamp for change detection (upstream `getFileRevision`):
/// `dev:ino:size:mtimeNs:ctimeNs`; `None` when the file does not exist.
pub fn get_file_revision(path: &str) -> Option<String> {
	let metadata = std::fs::metadata(path).ok()?;
	Some(file_revision_from_metadata(&metadata))
}

#[cfg(unix)]
fn file_revision_from_metadata(metadata: &std::fs::Metadata) -> String {
	use std::os::unix::fs::MetadataExt;
	format!(
		"{}:{}:{}:{}:{}",
		metadata.dev(),
		metadata.ino(),
		metadata.size(),
		metadata.mtime_nsec(),
		metadata.ctime_nsec()
	)
}

/// Best-effort stand-in without unix inode metadata: size plus wall-clock
/// nanos (upstream's bigint `stat` fields; only used as a cache key).
#[cfg(not(unix))]
fn file_revision_from_metadata(metadata: &std::fs::Metadata) -> String {
	fn epoch_nanos(time: std::io::Result<std::time::SystemTime>) -> u128 {
		time.ok().and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok()).map(|duration| duration.as_nanos()).unwrap_or(0)
	}
	format!(
		"0:0:{}:{}:{}",
		metadata.len(),
		epoch_nanos(metadata.modified()),
		epoch_nanos(metadata.created())
	)
}

/// Lexically remove `.` and resolve `..` segments (node resolve semantics).
fn lexically_clean(path: impl AsRef<Path>) -> String {
	let mut cleaned = PathBuf::new();
	for component in path.as_ref().components() {
		match component {
			Component::CurDir => {}
			Component::ParentDir => {
				// Upstream resolve never escapes the (absolute) base; without an
				// absolute base there is nothing to pop, so keep the `..`.
				if !cleaned.pop() {
					cleaned.push(component);
				}
			}
			component => cleaned.push(component),
		}
	}
	cleaned.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
	use super::*;

	// Ported from upstream `test/paths.test.ts` (the cases covering the ported
	// surface; the file-URL and Windows-only cases stay unported with the rest
	// of the module).

	#[test]
	fn resolve_path_expands_only_home_tilde_shortcuts() {
		let home = home_dir();
		assert_eq!(resolve_path("~/file", &home), format!("{home}/file"));
		assert_eq!(resolve_path("~/~/file", &home), format!("{home}/~/file"));
		assert_eq!(resolve_path("some~/file", &home), format!("{home}/some~/file"));
	}

	#[test]
	fn resolve_path_resolves_relative_paths_against_the_base_directory() {
		assert_eq!(resolve_path("foo", "/bar"), "/bar/foo");
		assert_eq!(resolve_path("./foo", "/bar"), "/bar/foo");
		assert_eq!(resolve_path("../foo", "/bar/baz"), "/bar/foo");
		assert_eq!(
			resolve_path("foo", "bar"),
			lexically_clean(join(&std::env::current_dir().unwrap().to_string_lossy(), "bar/foo"))
		);
	}

	#[test]
	fn normalize_path_expands_tilde() {
		let home = home_dir();
		assert_eq!(normalize_path("~", &PathInputOptions::default()), home);
		assert_eq!(normalize_path("~/notes", &PathInputOptions::default()), format!("{home}/notes"));
		assert_eq!(
			normalize_path("~", &PathInputOptions { home_dir: Some("/custom".into()), ..Default::default() }),
			"/custom"
		);
	}
}
