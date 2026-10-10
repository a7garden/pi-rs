//! Extension discovery, ported from upstream
//! `packages/coding-agent/src/core/extensions/loader.ts`.
//!
//! Discovery rules (upstream parity):
//! - Direct files in a directory: `*.ts` / `*.js` load directly.
//! - Subdirectories load their entry: `package.json` `pi.extensions` field
//!   first, then `index.ts`, then `index.js`. No recursion beyond one level.
//! - Reading a `pi` manifest follows upstream `readPiManifest`: the `pi` field
//!   of `package.json` may declare `extensions`, `skills`, `prompts`, and
//!   `themes` string arrays.

use std::path::{Path, PathBuf};

/// Upstream `isExtensionFile`.
pub fn is_extension_file(name: &str) -> bool {
    name.ends_with(".ts") || name.ends_with(".js")
}

/// The `pi` field of a `package.json` (upstream `PiManifest`).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PiManifest {
    pub extensions: Vec<String>,
    pub skills: Vec<String>,
    pub prompts: Vec<String>,
    pub themes: Vec<String>,
}

/// Port of upstream `readPiManifest`: parse the `pi` field of a
/// `package.json`, keeping string-array resource fields. Returns `None` when
/// the file is missing, unparsable, or has no `pi` object.
pub fn read_pi_manifest(package_json_path: &Path) -> Option<PiManifest> {
    let raw = std::fs::read_to_string(package_json_path).ok()?;
    let raw = strip_bom(&raw);
    let package: serde_json::Value = serde_json::from_str(raw).ok()?;
    let pi = package.get("pi")?.as_object()?;
    let mut manifest = PiManifest::default();
    for (field, target) in [
        ("extensions", &mut manifest.extensions),
        ("skills", &mut manifest.skills),
        ("prompts", &mut manifest.prompts),
        ("themes", &mut manifest.themes),
    ] {
        if let Some(entries) = pi.get(field).and_then(serde_json::Value::as_array) {
            if entries.iter().all(|entry| entry.is_string()) {
                *target = entries
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_string)
                    .collect();
            }
        }
    }
    Some(manifest)
}

fn strip_bom(input: &str) -> &str {
    input.strip_prefix('\u{feff}').unwrap_or(input)
}

/// Resolve extension entry points from one extension package directory
/// (upstream `resolveExtensionEntries`): `package.json` `pi.extensions` paths
/// first, then `index.ts`, then `index.js`. `None` when no entry exists.
pub fn resolve_extension_entries(dir: &Path) -> Option<Vec<PathBuf>> {
    let package_json_path = dir.join("package.json");
    if package_json_path.is_file() {
        if let Some(manifest) = read_pi_manifest(&package_json_path) {
            if !manifest.extensions.is_empty() {
                let entries: Vec<PathBuf> = manifest
                    .extensions
                    .iter()
                    .map(|extension_path| dir.join(extension_path))
                    .filter(|resolved| resolved.exists())
                    .collect();
                if !entries.is_empty() {
                    return Some(entries);
                }
            }
        }
    }
    let index_ts = dir.join("index.ts");
    if index_ts.exists() {
        return Some(vec![index_ts]);
    }
    let index_js = dir.join("index.js");
    if index_js.exists() {
        return Some(vec![index_js]);
    }
    None
}

/// Discover extension entries in a directory (upstream
/// `discoverExtensionsInDir`): direct `*.ts`/`*.js` files, plus subdirectory
/// entries (manifest-declared or index). No recursion beyond one level.
pub fn discover_extensions_in_dir(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut discovered = Vec::new();
    for entry in entries.flatten() {
        let entry_path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };
        // 1. Direct files: *.ts or *.js (symlinks included).
        if (file_type.is_file() || file_type.is_symlink())
            && is_extension_file(&entry.file_name().to_string_lossy())
        {
            discovered.push(entry_path);
            continue;
        }
        // 2 & 3. Subdirectories.
        if file_type.is_dir() || file_type.is_symlink() {
            if let Some(entries) = resolve_extension_entries(&entry_path) {
                discovered.extend(entries);
            }
        }
    }
    discovered
}

/// Deduplicate already-visited paths while preserving order (upstream
/// `addPaths`/`seen` set behavior in `discoverAndLoadExtensions`).
pub fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = std::collections::HashSet::new();
    paths
        .into_iter()
        .filter(|path| seen.insert(path.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_extension_file_matches_upstream_extensions() {
        assert!(is_extension_file("a.ts"));
        assert!(is_extension_file("a.js"));
        assert!(!is_extension_file("a.mjs"));
        assert!(!is_extension_file("a.json"));
    }

    #[test]
    fn reads_pi_manifest_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("package.json");
        std::fs::write(
            &path,
            "{\"pi\": {\"extensions\": [\"./x.ts\"], \"skills\": [\"s\"], \"themes\": 4}}",
        )
        .unwrap();
        let manifest = read_pi_manifest(&path).unwrap();
        assert_eq!(manifest.extensions, vec!["./x.ts".to_string()]);
        assert_eq!(manifest.skills, vec!["s".to_string()]);
        assert_eq!(manifest.prompts, Vec::<String>::new());
        assert_eq!(manifest.themes, Vec::<String>::new());
    }

    #[test]
    fn manifest_without_pi_field_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("package.json");
        std::fs::write(&path, "{\"name\": \"x\"}").unwrap();
        assert!(read_pi_manifest(&path).is_none());
    }
}
