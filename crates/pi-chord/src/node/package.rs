//! Port of `packages/chord/src/node/package.ts`: build a plugin package using
//! package.json metadata and application-provided facet conventions.

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::json::JsonValue;
use crate::node::bundle::{bundle_facets, BundleFacetsOptions, FacetCompiler};
use crate::services::errors::{error, AnyError};

/// Upstream `BundleFacetPackageOptions`.
pub struct BundleFacetPackageOptions {
    /// Plugin package directory or its package.json path.
    pub package_path: PathBuf,
    pub outdir: PathBuf,
    /// Application conventions applied when the corresponding source file exists.
    pub default_facets: BTreeMap<String, String>,
}

/// Upstream `BundleFacetPackageResult`.
#[derive(Debug)]
pub struct BundleFacetPackageResult {
    pub manifest: crate::node::manifest::FacetBundleManifest,
    pub manifest_path: PathBuf,
    pub package_directory: PathBuf,
    pub package_json_path: PathBuf,
}

#[derive(Debug)]
struct FacetPackageMetadata {
    package_directory: PathBuf,
    package_json_path: PathBuf,
    name: String,
    version: String,
    peer_dependencies: Vec<String>,
    configured_facets: BTreeMap<String, Option<String>>,
    external: Vec<String>,
    source_map: bool,
}

/// Build a plugin package using package.json metadata and application-provided
/// facet conventions. Port of upstream `bundleFacetPackage`; takes the
/// [`FacetCompiler`] used by [`bundle_facets`].
pub async fn bundle_facet_package(
    options: BundleFacetPackageOptions,
    compiler: &dyn FacetCompiler,
) -> Result<BundleFacetPackageResult, AnyError> {
    let metadata = read_facet_package_metadata(&options.package_path)?;
    let entries = resolve_facet_entries(&metadata, &options.default_facets)?;
    let mut external = Vec::new();
    for specifier in metadata.peer_dependencies.iter().chain(metadata.external.iter()) {
        external.push(specifier.clone());
        external.push(format!("{specifier}/*"));
    }
    let result = bundle_facets(
        BundleFacetsOptions {
            plugin: crate::node::manifest::FacetBundlePlugin {
                id: metadata.name.clone(),
                version: Some(metadata.version.clone()),
            },
            entries,
            outdir: options.outdir,
            working_directory: Some(metadata.package_directory.clone()),
            external,
            source_map: metadata.source_map,
            minify: false,
            define: BTreeMap::new(),
            platform: None,
            target: vec![],
        },
        compiler,
    )
    .await?;
    Ok(BundleFacetPackageResult {
        manifest: result.manifest,
        manifest_path: result.manifest_path,
        package_directory: metadata.package_directory,
        package_json_path: metadata.package_json_path,
    })
}

/// Port of upstream `readFacetPackageMetadata`.
fn read_facet_package_metadata(package_path: &Path) -> Result<FacetPackageMetadata, AnyError> {
    if package_path.as_os_str().is_empty() {
        return Err(error("Facet package path must not be empty"));
    }
    let candidate = absolute(package_path);
    let candidate_stats = match fs::metadata(&candidate) {
        Ok(stats) => stats,
        Err(e) => {
            return Err(error(format!(
                "Could not access facet package {}: {e}",
                candidate.display()
            )))
        }
    };
    let (package_directory, package_json_path) = if candidate_stats.is_dir() {
        let real = fs::canonicalize(&candidate)
            .map_err(|e| error(format!("Could not resolve facet package {}: {e}", candidate.display())))?;
        let manifest = real.join("package.json");
        (real, manifest)
    } else if candidate_stats.is_file() && candidate.file_name().and_then(|n| n.to_str()) == Some("package.json") {
        let real = fs::canonicalize(&candidate)
            .map_err(|e| error(format!("Could not resolve facet package {}: {e}", candidate.display())))?;
        let dir = real.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        (dir, real)
    } else {
        return Err(error(format!(
            "Facet package path must name a directory or package.json: {}",
            candidate.display()
        )));
    };

    let raw = match fs::read_to_string(&package_json_path) {
        Ok(raw) => raw,
        Err(e) => {
            return Err(error(format!(
                "Could not read facet package metadata {}: {e}",
                package_json_path.display()
            )))
        }
    };
    let parsed: JsonValue = match serde_json::from_str(&raw) {
        Ok(parsed) => parsed,
        Err(e) => {
            return Err(error(format!(
                "Could not read facet package metadata {}: {e}",
                package_json_path.display()
            )))
        }
    };
    if !parsed.is_object() {
        return Err(error(format!(
            "Facet package metadata must be an object: {}",
            package_json_path.display()
        )));
    }
    let name = parsed
        .get("name")
        .and_then(JsonValue::as_str)
        .filter(|n| !n.is_empty())
        .ok_or_else(|| {
            error(format!(
                "Facet package must have a non-empty name: {}",
                package_json_path.display()
            ))
        })?;
    let version = parsed
        .get("version")
        .and_then(JsonValue::as_str)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            error(format!(
                "Facet package must have a non-empty version: {}",
                package_json_path.display()
            ))
        })?;
    let peer_dependencies = parse_peer_dependencies(parsed.get("peerDependencies"), &package_json_path)?;
    let chord = parse_chord_configuration(parsed.get("chord"), &package_json_path)?;
    Ok(FacetPackageMetadata {
        package_directory,
        package_json_path,
        name: name.to_string(),
        version: version.to_string(),
        peer_dependencies,
        configured_facets: chord.facets,
        external: chord.external,
        source_map: chord.source_map,
    })
}

fn parse_peer_dependencies(value: Option<&JsonValue>, package_json_path: &Path) -> Result<Vec<String>, AnyError> {
    let Some(value) = non_null(value) else {
        return Ok(Vec::new());
    };
    let Some(object) = value.as_object() else {
        return Err(error(format!(
            "Facet package has invalid peerDependencies: {}",
            package_json_path.display()
        )));
    };
    let valid = object
        .iter()
        .all(|(name, version)| !name.is_empty() && version.is_string());
    if !valid {
        return Err(error(format!(
            "Facet package has invalid peerDependencies: {}",
            package_json_path.display()
        )));
    }
    let mut names: Vec<String> = object.keys().cloned().collect();
    names.sort();
    Ok(names)
}

struct ChordConfiguration {
    facets: BTreeMap<String, Option<String>>,
    external: Vec<String>,
    source_map: bool,
}

fn parse_chord_configuration(value: Option<&JsonValue>, package_json_path: &Path) -> Result<ChordConfiguration, AnyError> {
    let Some(value) = non_null(value) else {
        return Ok(ChordConfiguration {
            facets: BTreeMap::new(),
            external: Vec::new(),
            source_map: true,
        });
    };
    if !value.is_object() {
        return Err(error(format!(
            "Facet package chord configuration must be an object: {}",
            package_json_path.display()
        )));
    }
    for key in value.as_object().unwrap().keys() {
        if key != "facets" && key != "external" && key != "sourceMap" {
            return Err(error(format!(
                "Facet package chord configuration has an unknown field: {}",
                package_json_path.display()
            )));
        }
    }
    let mut facets = BTreeMap::new();
    if let Some(raw_facets) = non_null(value.get("facets")) {
        let Some(object) = raw_facets.as_object() else {
            return Err(error(format!(
                "Facet package chord.facets must be an object: {}",
                package_json_path.display()
            )));
        };
        for (name, source) in object {
            // `false` deletes an entry; otherwise a non-empty string path.
            let configured = match source {
                JsonValue::Bool(false) => None,
                JsonValue::String(path) if !path.is_empty() => Some(path.clone()),
                _ => {
                    return Err(error(format!(
                        "Facet package has an invalid chord.facets entry: {}",
                        package_json_path.display()
                    )))
                }
            };
            facets.insert(name.clone(), configured);
        }
    }
    let mut external: Vec<String> = Vec::new();
    if let Some(raw_external) = non_null(value.get("external")) {
        let Some(items) = raw_external.as_array() else {
            return Err(error(format!(
                "Facet package chord.external must contain non-empty strings: {}",
                package_json_path.display()
            )));
        };
        for item in items {
            match item.as_str() {
                Some(specifier) if !specifier.is_empty() => external.push(specifier.to_string()),
                _ => {
                    return Err(error(format!(
                        "Facet package chord.external must contain non-empty strings: {}",
                        package_json_path.display()
                    )))
                }
            }
        }
        external.sort();
        external.dedup();
    }
    let source_map = match non_null(value.get("sourceMap")) {
        None => true,
        Some(JsonValue::Bool(flag)) => *flag,
        Some(_) => {
            return Err(error(format!(
                "Facet package chord.sourceMap must be a boolean: {}",
                package_json_path.display()
            )))
        }
    };
    Ok(ChordConfiguration {
        facets,
        external,
        source_map,
    })
}

/// Port of upstream `resolveFacetEntries`: defaults first (missing files
/// skipped), configured entries second (`false` deletes), then the
/// at-least-one-entry invariant.
fn resolve_facet_entries(
    metadata: &FacetPackageMetadata,
    default_facets: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>, AnyError> {
    let mut entries: BTreeMap<String, String> = BTreeMap::new();
    for (name, source) in default_facets {
        validate_facet_mapping(name, source, "default")?;
        let path = resolve_package_entry(&metadata.package_directory, source, name)?;
        match stat_file(&path) {
            Ok(canonical_path) => {
                validate_canonical_package_entry(&metadata.package_directory, &canonical_path, name)?;
                entries.insert(name.clone(), canonical_path.to_string_lossy().into_owned());
            }
            Err(EntryError::Missing) => continue,
            Err(EntryError::Other(e)) => {
                return Err(error(format!(
                    "Default facet entry {name} is not a file: {}: {e}",
                    path.display()
                )))
            }
        }
    }
    for (name, source) in &metadata.configured_facets {
        let Some(source) = source else {
            entries.remove(name);
            continue;
        };
        validate_facet_mapping(name, source, "configured")?;
        let path = resolve_package_entry(&metadata.package_directory, source, name)?;
        let canonical_path = stat_file(&path).map_err(|e| match e {
            EntryError::Missing => {
                error(format!(
                    "Could not access configured facet entry {name}: {}",
                    path.display()
                ))
            }
            EntryError::Other(err) => error(format!(
                "Configured facet entry {name} is not a file: {}: {err}",
                path.display()
            )),
        })?;
        validate_canonical_package_entry(&metadata.package_directory, &canonical_path, name)?;
        entries.insert(name.clone(), canonical_path.to_string_lossy().into_owned());
    }
    if entries.is_empty() {
        return Err(error(format!(
            "Facet package {} has no configured or conventional facet entries",
            metadata.name
        )));
    }
    Ok(entries)
}

enum EntryError {
    Missing,
    Other(std::io::Error),
}

/// `stat(path)` + `isFile()` + `realpath(path)`.
fn stat_file(path: &Path) -> Result<PathBuf, EntryError> {
    match fs::metadata(path) {
        Ok(stats) if stats.is_file() => fs::canonicalize(path).map_err(EntryError::Other),
        Ok(_) => Err(EntryError::Other(std::io::Error::new(
            ErrorKind::Other,
            "not a file",
        ))),
        Err(e) if e.kind() == ErrorKind::NotFound => Err(EntryError::Missing),
        Err(e) => Err(EntryError::Other(e)),
    }
}

fn validate_facet_mapping(name: &str, source: &str, kind: &str) -> Result<(), AnyError> {
    if name.is_empty() {
        return Err(error(format!(
            "Facet package {kind} entry name must not be empty"
        )));
    }
    if source.is_empty() {
        return Err(error(format!(
            "Facet package {kind} entry {name} must have a source path"
        )));
    }
    Ok(())
}

fn resolve_package_entry(package_directory: &Path, source: &str, name: &str) -> Result<PathBuf, AnyError> {
    let source_path = Path::new(source);
    if source_path.is_absolute() {
        return Err(error(format!(
            "Facet package entry {name} must be relative to the package directory"
        )));
    }
    let path = package_directory.join(source_path);
    let relative_path = path
        .strip_prefix(package_directory)
        .map_err(|_| error(format!("Facet package entry {name} escapes the package directory")))?;
    let relative_str = relative_path.to_string_lossy();
    if relative_str.is_empty()
        || relative_str == ".."
        || relative_str.starts_with("../")
        || relative_path.is_absolute()
    {
        return Err(error(format!(
            "Facet package entry {name} escapes the package directory"
        )));
    }
    Ok(path)
}

fn validate_canonical_package_entry(package_directory: &Path, path: &Path, name: &str) -> Result<(), AnyError> {
    let escape = match path.strip_prefix(package_directory) {
        Ok(relative_path) => {
            let relative_str = relative_path.to_string_lossy();
            relative_str == ".." || relative_str.starts_with("../")
        }
        Err(_) => true,
    };
    if escape {
        return Err(error(format!(
            "Facet package entry {name} resolves outside the package directory"
        )));
    }
    Ok(())
}

/// `value === undefined` check on JSON input (`null` is also treated as absent,
/// matching how serde models missing fields).
fn non_null(value: Option<&JsonValue>) -> Option<&JsonValue> {
    match value {
        None | Some(JsonValue::Null) => None,
        Some(other) => Some(other),
    }
}

/// `path.resolve(packagePath)`.
fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// Never reached by these tests: entry resolution fails first.
    struct UnusedCompiler;
    #[async_trait]
    impl FacetCompiler for UnusedCompiler {
        async fn build(&self, _request: crate::node::bundle::FacetBuildRequest) -> Result<crate::node::bundle::FacetBuildOutput, AnyError> {
            Err(error("compiler must not be used"))
        }
    }

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pi-chord-package-test-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn rejects_configured_entries_escaping_the_package_directory() {
        let dir = tempdir();
        fs::write(
            dir.join("package.json"),
            r#"{"name":"invalid-plugin","version":"1.0.0","chord":{"facets":{"tui":"../outside.ts"}}}"#,
        )
        .unwrap();
        let err = bundle_facet_package(
            BundleFacetPackageOptions {
                package_path: dir.clone(),
                outdir: dir.join("build"),
                default_facets: BTreeMap::new(),
            },
            &UnusedCompiler,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("escapes the package directory"), "{err}");
    }

    #[test]
    fn rejects_absolute_configured_entries() {
        let dir = tempdir();
        let err = resolve_package_entry(&dir, "/abs/entry.ts", "tui").unwrap_err().to_string();
        assert_eq!(
            err,
            "Facet package entry tui must be relative to the package directory"
        );
        let err = resolve_package_entry(&dir, "../outside.ts", "tui").unwrap_err().to_string();
        assert_eq!(
            err,
            "Facet package entry tui escapes the package directory"
        );
    }

    #[test]
    fn rejects_empty_metadata_fields() {
        let dir = tempdir();
        fs::write(dir.join("package.json"), r#"{"name":"","version":"1.0.0"}"#).unwrap();
        let err = read_facet_package_metadata(&dir).unwrap_err().to_string();
        assert!(err.starts_with("Facet package must have a non-empty name: "), "{err}");
    }
}
