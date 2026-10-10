//! Port of `packages/coding-agent/src/utils/windows-self-update.ts` — Windows
//! native-addon quarantine during self-update.
//!
//! Windows self-updates must not leave loaded `.node` native addons in place
//! while the new build is swapped in: the addons are "quarantined" (renamed
//! out of the package, copied back) so a restarting process loads fresh files.
//!
//! Deviations (crate docs, D13): upstream reads loaded shared objects from
//! `process.report.getReport()`; the Rust port reads `/proc/self/maps` on
//! Linux and reports none elsewhere (macOS/Windows have no portable
//! equivalent, so quarantine is a no-op there, like a report-less node).
//! `toNamespacedPath` has no std counterpart and is omitted.

use std::path::{Path, PathBuf};

use crate::utils::paths::{join, resolve_path, resolve_path_cwd};

const QUARANTINE_DIR_NAME: &str = ".pi-native-quarantine";

fn normalize_path(path: &str) -> String {
    // Upstream: toNamespacedPath(resolve(path)) — resolve + (windows) \\?\
    // prefix; the namespace prefix is omitted (D13).
    resolve_path_cwd(path)
}

/// Upstream `getCwdRelativePath` (paths.ts) — a relative path from `cwd` to
/// `file_path` when inside, `None` otherwise.
fn get_cwd_relative_path(file_path: &str, cwd: &str) -> Option<String> {
    let resolved_cwd = resolve_path(
        cwd,
        &std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy(),
    );
    let resolved_path = resolve_path(file_path, &resolved_cwd);
    let relative = lexically_relative(&resolved_path, &resolved_cwd)?;
    let is_inside_cwd = relative.is_empty()
        || (relative != ".."
            && !relative.starts_with("../")
            && !Path::new(&relative).is_absolute());
    if is_inside_cwd {
        Some(if relative.is_empty() {
            ".".to_string()
        } else {
            relative
        })
    } else {
        None
    }
}

/// node `path.relative` (lexical, separator per host).
fn lexically_relative(path: &str, base: &str) -> Option<String> {
    let path_parts: Vec<&str> = path
        .split(['/', '\\'])
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();
    let base_parts: Vec<&str> = base
        .split(['/', '\\'])
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();
    let mut common = 0;
    while common < path_parts.len()
        && common < base_parts.len()
        && path_parts[common].eq_ignore_ascii_case(base_parts[common])
    {
        common += 1;
    }
    let mut result: Vec<String> = Vec::new();
    for _ in common..base_parts.len() {
        result.push("..".to_string());
    }
    result.extend(path_parts[common..].iter().map(|s| s.to_string()));
    Some(result.join(std::path::MAIN_SEPARATOR_STR))
}

fn get_quarantine_root(package_dir: &str) -> Option<String> {
    let mut current = resolve_path_cwd(package_dir);
    loop {
        if basename_lower(&current) == "node_modules" {
            return Some(join(&current, QUARANTINE_DIR_NAME));
        }
        match parent_dir(&current) {
            Some(parent) => current = parent,
            None => return None,
        }
    }
}

fn basename_lower(path: &str) -> String {
    path.rsplit(['/', '\\'])
        .next()
        .unwrap_or(path)
        .to_ascii_lowercase()
}

fn parent_dir(path: &str) -> Option<String> {
    let trimmed = path.trim_end_matches(['/', '\\']);
    let parent = match trimmed.rfind(['/', '\\']) {
        Some(0) => trimmed[..1].to_string(), // "/a" → "/"
        Some(i) => trimmed[..i].to_string(),
        None => return None,
    };
    if parent == trimmed {
        None
    } else {
        Some(parent)
    }
}

/// Loaded shared-object paths under `package_dir` (upstream
/// `getLoadedSharedObjectsInPackageDir`). Empty when the runtime cannot
/// report loaded objects (D13).
fn get_loaded_shared_objects_in_package_dir(package_dir: &str) -> Vec<String> {
    let Some(shared_objects) = read_shared_objects() else {
        return Vec::new();
    };

    let root = normalize_path(package_dir).to_ascii_lowercase();
    let mut seen = std::collections::BTreeSet::new();
    let mut loaded_files = Vec::new();
    for value in shared_objects {
        let file_path = normalize_path(&value);
        let comparison_path = file_path.to_ascii_lowercase();
        if get_cwd_relative_path(&comparison_path, &root).is_none()
            || seen.contains(&comparison_path)
        {
            continue;
        }
        seen.insert(comparison_path);
        loaded_files.push(file_path);
    }
    loaded_files
}

#[cfg(target_os = "linux")]
fn read_shared_objects() -> Option<Vec<String>> {
    // /proc/self/maps mirrors node's report.sharedObjects.
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    let mut out = Vec::new();
    for line in maps.lines() {
        // `… perm offset dev inode pathname`
        if let Some(pathname) = line.split_whitespace().nth(5) {
            out.push(pathname.to_string());
        }
    }
    Some(out)
}

#[cfg(not(target_os = "linux"))]
fn read_shared_objects() -> Option<Vec<String>> {
    None
}

/// Remove a previous run's quarantine directory
/// (`cleanupWindowsSelfUpdateQuarantine`). Errors are swallowed: a previous pi
/// process may still be exiting and holding a native addon.
pub fn cleanup_windows_self_update_quarantine(package_dir: &str) {
    let Some(quarantine_root) = get_quarantine_root(package_dir) else {
        return;
    };
    let _ = std::fs::remove_dir_all(quarantine_root);
}

/// Quarantine loaded native dependencies before a self-update
/// (`quarantineWindowsNativeDependencies`): each loaded `.node` file is
/// renamed into the quarantine run dir and copied back, so the on-disk bytes
/// can be replaced while the current process keeps its mapped copy.
pub fn quarantine_windows_native_dependencies(package_dir: &str) {
    let resolved_package_dir = normalize_path(package_dir);
    let Some(quarantine_root) = get_quarantine_root(&resolved_package_dir) else {
        return;
    };

    let loaded_files = get_loaded_shared_objects_in_package_dir(&resolved_package_dir);
    if loaded_files.is_empty() {
        return;
    }

    let quarantine_run_dir = join(
        &quarantine_root,
        &format!("{}-{}-{}", now_millis(), std::process::id(), random_uuid(),),
    );
    for loaded_file in loaded_files {
        if !Path::new(&loaded_file).exists() {
            continue;
        }
        let relative = lexically_relative(&loaded_file, &resolved_package_dir).unwrap_or_default();
        let quarantine_path: PathBuf = Path::new(&quarantine_run_dir).join(relative);
        if let Some(parent) = quarantine_path.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                continue;
            }
        }
        if std::fs::rename(&loaded_file, &quarantine_path).is_err() {
            continue;
        }
        let _ = std::fs::copy(&quarantine_path, &loaded_file);
    }
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// `randomUUID()` via the OS RNG (D13).
fn random_uuid() -> String {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes).expect("OS RNG unavailable");
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // RFC 4122 variant
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarantine_root_walks_up_to_node_modules() {
        let root = std::env::temp_dir().join(format!("pi-wsu-{:p}", &QUARANTINE_DIR_NAME));
        let package_dir = root.join("node_modules/@earendil/pi-coding-agent");
        std::fs::create_dir_all(&package_dir).unwrap();
        let resolved = resolve_path_cwd(&package_dir.to_string_lossy());
        let quarantine = get_quarantine_root(&resolved).unwrap();
        assert_eq!(
            quarantine,
            join(
                &resolve_path_cwd(&root.join("node_modules").to_string_lossy()),
                QUARANTINE_DIR_NAME
            )
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn no_quarantine_root_outside_node_modules() {
        assert!(get_quarantine_root("/tmp").is_none() || get_quarantine_root(".").is_none());
    }

    #[test]
    fn cleanup_is_a_safe_noop_without_quarantine_root() {
        cleanup_windows_self_update_quarantine("/definitely/not/node_modules/anything");
    }

    #[test]
    fn quarantine_noops_when_no_loaded_shared_objects_reported() {
        // On non-Linux the shared-object report is always empty, so quarantine
        // must return without creating anything (matches report-less node).
        let dir = std::env::temp_dir().join(format!("pi-wsu-noop-{:p}", &QUARANTINE_DIR_NAME));
        let package_dir = dir.join("node_modules/pkg");
        std::fs::create_dir_all(&package_dir).unwrap();
        quarantine_windows_native_dependencies(&package_dir.to_string_lossy());
        let quarantine_root =
            get_quarantine_root(&resolve_path_cwd(&package_dir.to_string_lossy())).unwrap();
        assert!(!Path::new(&quarantine_root).exists() || cfg!(target_os = "linux"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cwd_relative_path_semantics() {
        let base = "/base/dir";
        assert_eq!(
            get_cwd_relative_path("/base/dir/a/b.txt", base).as_deref(),
            Some("a/b.txt")
        );
        assert_eq!(
            get_cwd_relative_path("/base/dir", base).as_deref(),
            Some(".")
        );
        assert_eq!(get_cwd_relative_path("/elsewhere/x", base), None);
        assert_eq!(get_cwd_relative_path("/base/dir/../x", base), None);
    }

    #[test]
    fn random_uuid_is_v4_shaped() {
        let uuid = random_uuid();
        assert_eq!(uuid.len(), 36);
        assert_eq!(&uuid[14..15], "4");
    }
}
