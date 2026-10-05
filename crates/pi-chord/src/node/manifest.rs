//! Port of `packages/chord/src/node/manifest.ts`, plus the manifest helpers
//! (`validateManifest`, `resolveBundleFile`, `parseIntegrity`) from
//! `packages/chord/src/node/bundle-loader.ts`.
//!
//! Wire structs mirror the upstream JSON shape exactly (`camelCase` field
//! names, optional fields omitted when absent).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::json::JsonValue;
use crate::services::errors::{error, type_error, AnyError};

pub const FACET_BUNDLE_FORMAT: &str = "chord.facet-bundle";
pub const FACET_BUNDLE_FORMAT_VERSION: u64 = 2;
pub const FACET_BUNDLE_MANIFEST_FILE: &str = "chord-facets.json";
pub const FACET_BUNDLE_ARTIFACT_FORMAT: &str = "chord.facet-bundle-artifact";
pub const FACET_BUNDLE_ARTIFACT_FORMAT_VERSION: u64 = 2;

/// Upstream `FacetBundlePlugin`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FacetBundlePlugin {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Upstream `FacetBundleEntry`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FacetBundleEntry {
    /// Content-addressed CommonJS filename relative to the manifest.
    pub file: String,
    /// SHA-256 subresource-integrity value for the JavaScript file.
    pub integrity: String,
    /// Imports intentionally left for the loading application to resolve.
    pub external_imports: Vec<String>,
    /// Source map filename relative to the manifest, when emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_map: Option<String>,
}

/// Upstream `FacetBundleManifest`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FacetBundleManifest {
    pub format: String,
    pub format_version: u64,
    pub plugin: FacetBundlePlugin,
    /// Entry name → entry. Serialized in sorted key order, matching the
    /// upstream bundle writer's locale-sorted insertion order.
    pub entries: BTreeMap<String, FacetBundleEntry>,
}

/// One self-contained manifest entry suitable for storage or transport to
/// another Node host (upstream `FacetBundleArtifact`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FacetBundleArtifact {
    pub plugin: FacetBundlePlugin,
    pub entry_name: String,
    pub entry: FacetBundleEntry,
    pub source: String,
    pub source_map_contents: Option<String>,
}

/// Resolve one entry file relative to a manifest directory.
///
/// Port of upstream `resolveBundleFile`: the file must be a bare filename.
pub fn resolve_bundle_file(manifest_path: &Path, file: &str, label: &str) -> Result<PathBuf, AnyError> {
    let invalid = file.is_empty()
        || Path::new(file).is_absolute()
        || Path::new(file).file_name().and_then(|name| name.to_str()) != Some(file)
        || file == "."
        || file == "..";
    if invalid {
        return Err(error(format!(
            "Facet bundle {label} must be a filename relative to its manifest"
        )));
    }
    Ok(manifest_path.parent().unwrap_or(Path::new(".")).join(file))
}

/// Split a `sha256-<base64>` subresource integrity value; port of upstream
/// `parseIntegrity`.
pub fn parse_integrity(integrity: &str) -> Result<String, AnyError> {
    const PREFIX: &str = "sha256-";
    if !integrity.starts_with(PREFIX) || integrity.len() == PREFIX.len() {
        return Err(error("Facet bundle entry has an invalid SHA-256 integrity value"));
    }
    Ok(integrity[PREFIX.len()..].to_string())
}

/// Validate an arbitrary parsed JSON value as a facet bundle manifest; port of
/// upstream `validateManifest` (exact error messages).
pub fn validate_manifest(value: &JsonValue, path: &str) -> Result<FacetBundleManifest, AnyError> {
    let Some(obj) = value.as_object() else {
        return Err(error(format!("Invalid facet bundle manifest format in {path}")));
    };
    if obj.get("format").and_then(JsonValue::as_str) != Some(FACET_BUNDLE_FORMAT) {
        return Err(error(format!("Invalid facet bundle manifest format in {path}")));
    }
    if obj.get("formatVersion").map(facet_version_matches) != Some(true) {
        let shown = format_version_display(obj.get("formatVersion"));
        return Err(error(format!(
            "Unsupported facet bundle manifest version in {path}: {shown}"
        )));
    }
    let plugin = obj.get("plugin").ok_or_else(|| {
        error(format!("Facet bundle manifest has an invalid plugin identity in {path}"))
    })?;
    let plugin_id = plugin
        .get("id")
        .and_then(JsonValue::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| error(format!("Facet bundle manifest has an invalid plugin identity in {path}")))?;
    let plugin_version = match plugin.get("version") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::String(version)) if !version.is_empty() => Some(version.clone()),
        Some(_) => {
            return Err(error(format!(
                "Facet bundle manifest has an invalid plugin version in {path}"
            )))
        }
    };
    let raw_entries = obj.get("entries").and_then(JsonValue::as_object).ok_or_else(|| {
        error(format!("Facet bundle manifest has no entries in {path}"))
    })?;
    if raw_entries.is_empty() {
        return Err(error(format!("Facet bundle manifest has no entries in {path}")));
    }
    let mut entries = BTreeMap::new();
    for (name, candidate) in raw_entries {
        if name.is_empty() || !candidate.is_object() {
            return Err(error(format!("Facet bundle manifest has an invalid entry in {path}")));
        }
        let file = candidate
            .get("file")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| error(format!("Facet bundle entry {name} has no file")))?;
        resolve_bundle_file(Path::new(path), file, &format!("entry {name}"))?;
        let integrity = candidate
            .get("integrity")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| error(format!("Facet bundle entry {name} has no integrity")))?;
        parse_integrity(integrity)?;
        let mut external_imports = Vec::new();
        match candidate.get("externalImports").and_then(JsonValue::as_array) {
            Some(items) => {
                for item in items {
                    match item.as_str() {
                        Some(specifier) => external_imports.push(specifier.to_string()),
                        None => {
                            return Err(error(format!(
                                "Facet bundle entry {name} has invalid external imports"
                            )))
                        }
                    }
                }
            }
            None => {
                return Err(error(format!(
                    "Facet bundle entry {name} has invalid external imports"
                )))
            }
        }
        if external_imports.iter().collect::<std::collections::BTreeSet<_>>().len() != external_imports.len() {
            return Err(error(format!(
                "Facet bundle entry {name} has duplicate external imports"
            )));
        }
        let source_map = match candidate.get("sourceMap") {
            None | Some(JsonValue::Null) => None,
            Some(JsonValue::String(map)) => {
                resolve_bundle_file(Path::new(path), map, &format!("entry {name} source map"))?;
                Some(map.clone())
            }
            Some(_) => {
                return Err(error(format!(
                    "Facet bundle entry {name} has an invalid source map"
                )))
            }
        };
        entries.insert(
            name.clone(),
            FacetBundleEntry {
                file: file.to_string(),
                integrity: integrity.to_string(),
                external_imports,
                source_map,
            },
        );
    }
    Ok(FacetBundleManifest {
        format: FACET_BUNDLE_FORMAT.to_string(),
        format_version: FACET_BUNDLE_FORMAT_VERSION,
        plugin: FacetBundlePlugin {
            id: plugin_id.to_string(),
            version: plugin_version,
        },
        entries,
    })
}

/// `typeof value.formatVersion === "number" && value.formatVersion === 2`.
fn facet_version_matches(value: &JsonValue) -> bool {
    value.as_u64() == Some(FACET_BUNDLE_FORMAT_VERSION)
}

/// Upstream `String(value.formatVersion)` for the error message; `undefined`
/// prints as the literal `undefined`.
fn format_version_display(value: Option<&JsonValue>) -> String {
    match value {
        None | Some(JsonValue::Null) => "undefined".to_string(),
        Some(JsonValue::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// Reject option-object misuse the way upstream's TypeScript types do at
/// compile time; kept as a runtime guard for callers constructing options from
/// dynamic data.
pub fn assert_option_string(name: &str, value: &str) -> Result<(), AnyError> {
    if value.is_empty() {
        return Err(type_error(format!("{name} must not be empty")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_manifest() -> JsonValue {
        json!({
            "format": "chord.facet-bundle",
            "formatVersion": 2,
            "plugin": { "id": "test-plugin", "version": "1.0.0" },
            "entries": {
                "worker": {
                    "file": "facet-aaaaaaaaaaaaaaaaaaaaaaaa.cjs",
                    "integrity": "sha256-K7gNI5FIskWC1ainSAtUmO3wUrrel5cAWSJymesj9",
                    "externalImports": ["@earendil-works/chord"],
                    "sourceMap": "facet-aaaaaaaaaaaaaaaaaaaaaaaa.cjs.map"
                }
            }
        })
    }

    #[test]
    fn accepts_a_valid_manifest() {
        let manifest = validate_manifest(&valid_manifest(), "/tmp/chord-facets.json").unwrap();
        assert_eq!(manifest.plugin.id, "test-plugin");
        assert_eq!(manifest.plugin.version.as_deref(), Some("1.0.0"));
        let entry = manifest.entries.get("worker").unwrap();
        assert_eq!(entry.file, "facet-aaaaaaaaaaaaaaaaaaaaaaaa.cjs");
        assert_eq!(entry.external_imports, ["@earendil-works/chord"]);
        assert_eq!(entry.source_map.as_deref(), Some("facet-aaaaaaaaaaaaaaaaaaaaaaaa.cjs.map"));
    }

    #[test]
    fn rejects_invalid_format() {
        let mut value = valid_manifest();
        value["format"] = json!("other");
        let err = validate_manifest(&value, "/x.json").unwrap_err().to_string();
        assert_eq!(err, "Invalid facet bundle manifest format in /x.json");

        let err = validate_manifest(&json!([]), "/x.json").unwrap_err().to_string();
        assert_eq!(err, "Invalid facet bundle manifest format in /x.json");
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut value = valid_manifest();
        value["formatVersion"] = json!(1);
        let err = validate_manifest(&value, "/x.json").unwrap_err().to_string();
        assert_eq!(err, "Unsupported facet bundle manifest version in /x.json: 1");

        value["formatVersion"] = json!(null);
        let err = validate_manifest(&value, "/x.json").unwrap_err().to_string();
        assert_eq!(err, "Unsupported facet bundle manifest version in /x.json: undefined");
    }

    #[test]
    fn rejects_invalid_plugin_identity_and_version() {
        let mut value = valid_manifest();
        value["plugin"] = json!({ "id": "" });
        let err = validate_manifest(&value, "/x.json").unwrap_err().to_string();
        assert_eq!(err, "Facet bundle manifest has an invalid plugin identity in /x.json");

        value["plugin"] = json!({ "id": "p", "version": "" });
        let err = validate_manifest(&value, "/x.json").unwrap_err().to_string();
        assert_eq!(err, "Facet bundle manifest has an invalid plugin version in /x.json");
    }

    #[test]
    fn rejects_missing_or_invalid_entries() {
        let mut value = valid_manifest();
        value["entries"] = json!({});
        let err = validate_manifest(&value, "/x.json").unwrap_err().to_string();
        assert_eq!(err, "Facet bundle manifest has no entries in /x.json");

        value["entries"] = json!([]);
        assert_eq!(
            validate_manifest(&value, "/x.json").unwrap_err().to_string(),
            "Facet bundle manifest has no entries in /x.json"
        );

        value["entries"] = json!({ "worker": "nope" });
        assert_eq!(
            validate_manifest(&value, "/x.json").unwrap_err().to_string(),
            "Facet bundle manifest has an invalid entry in /x.json"
        );

        value["entries"] = json!({ "": { "file": "a.cjs", "integrity": "sha256-x", "externalImports": [] } });
        assert_eq!(
            validate_manifest(&value, "/x.json").unwrap_err().to_string(),
            "Facet bundle manifest has an invalid entry in /x.json"
        );
    }

    #[test]
    fn rejects_entries_without_file_or_integrity() {
        let mut value = valid_manifest();
        value["entries"]["worker"].as_object_mut().unwrap().remove("file");
        assert_eq!(
            validate_manifest(&value, "/x.json").unwrap_err().to_string(),
            "Facet bundle entry worker has no file"
        );

        let mut value = valid_manifest();
        value["entries"]["worker"].as_object_mut().unwrap().remove("integrity");
        assert_eq!(
            validate_manifest(&value, "/x.json").unwrap_err().to_string(),
            "Facet bundle entry worker has no integrity"
        );
    }

    #[test]
    fn rejects_invalid_integrity() {
        let mut value = valid_manifest();
        value["entries"]["worker"]["integrity"] = json!("md5-abc");
        assert_eq!(
            validate_manifest(&value, "/x.json").unwrap_err().to_string(),
            "Facet bundle entry has an invalid SHA-256 integrity value"
        );
        value["entries"]["worker"]["integrity"] = json!("sha256-");
        assert_eq!(
            validate_manifest(&value, "/x.json").unwrap_err().to_string(),
            "Facet bundle entry has an invalid SHA-256 integrity value"
        );
        assert_eq!(parse_integrity("sha256-abc").unwrap(), "abc");
    }

    #[test]
    fn rejects_invalid_or_duplicate_external_imports() {
        let mut value = valid_manifest();
        value["entries"]["worker"]["externalImports"] = json!(["ok", 3]);
        assert_eq!(
            validate_manifest(&value, "/x.json").unwrap_err().to_string(),
            "Facet bundle entry worker has invalid external imports"
        );
        value["entries"]["worker"]["externalImports"] = json!(["a", "a"]);
        assert_eq!(
            validate_manifest(&value, "/x.json").unwrap_err().to_string(),
            "Facet bundle entry worker has duplicate external imports"
        );
    }

    #[test]
    fn rejects_invalid_source_map() {
        let mut value = valid_manifest();
        value["entries"]["worker"]["sourceMap"] = json!(42);
        assert_eq!(
            validate_manifest(&value, "/x.json").unwrap_err().to_string(),
            "Facet bundle entry worker has an invalid source map"
        );
        value["entries"]["worker"]["sourceMap"] = json!("../escape.map");
        assert_eq!(
            validate_manifest(&value, "/x.json").unwrap_err().to_string(),
            "Facet bundle entry worker source map must be a filename relative to its manifest"
        );
    }

    #[test]
    fn resolve_bundle_file_rejects_escapes() {
        let manifest_path = Path::new("/bundle/chord-facets.json");
        assert_eq!(
            resolve_bundle_file(manifest_path, "a.cjs", "entry").unwrap(),
            PathBuf::from("/bundle/a.cjs")
        );
        for bad in ["", "/abs/a.cjs", "dir/a.cjs", ".", ".."] {
            let err = resolve_bundle_file(manifest_path, bad, "entry").unwrap_err().to_string();
            assert_eq!(err, "Facet bundle entry must be a filename relative to its manifest");
        }
    }
}
