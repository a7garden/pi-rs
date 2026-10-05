//! Port of `packages/chord/src/node/bundle-loader.ts`.
//!
//! Everything except CommonJS evaluation is real: manifest reading/validation,
//! entry resolution, integrity verification, artifact validation and
//! materialization, and loader orchestration. The JavaScript evaluation half of
//! upstream (`executeCommonJsModule` via `node:vm`/`createRequire`) is behind
//! the [`FacetBundleModuleLoader`] trait for the future `pi-runtime` host;
//! [`UnsupportedModuleLoader`] is the honest boundary implementation.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use sha2::{Digest, Sha256};

use crate::json::JsonValue;
use crate::node::manifest::{
    parse_integrity, resolve_bundle_file, validate_manifest, FacetBundleArtifact, FacetBundleEntry,
    FacetBundleManifest, FACET_BUNDLE_ARTIFACT_FORMAT, FACET_BUNDLE_ARTIFACT_FORMAT_VERSION,
    FACET_BUNDLE_MANIFEST_FILE,
};
use crate::services::errors::{error, type_error, AggregateError, AnyError};
use crate::types::{BoxFuture, Facet, FacetLoader, LoadedFacets};

/// Upstream `FacetBundleLoaderOptions`. The upstream `resolveExternal` callback
/// is a JavaScript-resolver boundary and lives with the module loader in the
/// runtime host.
pub struct FacetBundleLoaderOptions {
    pub manifest_path: PathBuf,
    pub entry: String,
    /// Verify the entry's SHA-256 integrity before evaluating it. Defaults to true.
    pub verify_integrity: Option<bool>,
}

/// Upstream `FacetBundleArtifactLoaderOptions`.
pub struct FacetBundleArtifactLoaderOptions {
    /// An untrusted parsed artifact value (upstream: `artifact: unknown`).
    pub artifact: JsonValue,
    /// Parent directory for materialized module generations. Defaults to the
    /// operating system temp directory.
    pub temporary_directory: Option<PathBuf>,
}

/// The JavaScript/CommonJS evaluation boundary (upstream
/// `executeCommonJsModule` + `compileFunction` + `createRequire`). Given the
/// verified module file and its declared external imports, evaluate the module
/// and convert its `facetsFromModule`-validated export shape into [`Facet`]
/// values. Implementations live in the future `pi-runtime` JS host.
#[async_trait]
pub trait FacetBundleModuleLoader: Send + Sync {
    async fn load_entry(&self, module_path: &Path, external_imports: &[String]) -> Result<Vec<Facet>, AnyError>;
}

/// Boundary implementation used until a JavaScript runtime host exists.
pub struct UnsupportedModuleLoader;

#[async_trait]
impl FacetBundleModuleLoader for UnsupportedModuleLoader {
    async fn load_entry(&self, _module_path: &Path, _external_imports: &[String]) -> Result<Vec<Facet>, AnyError> {
        Err(error("facet bundle module evaluation requires a JavaScript runtime host"))
    }
}

/// Read and validate a versioned facet bundle manifest; port of upstream
/// `readFacetBundleManifest` (upstream keeps the cause out of the message).
pub fn read_facet_bundle_manifest(path: &Path) -> Result<FacetBundleManifest, AnyError> {
    let raw = fs::read_to_string(path)
        .map_err(|_| error(format!("Could not read facet bundle manifest {}", path.display())))?;
    let parsed: JsonValue = serde_json::from_str(&raw)
        .map_err(|_| error(format!("Could not read facet bundle manifest {}", path.display())))?;
    validate_manifest(&parsed, &path.to_string_lossy())
}

/// Read and verify one transportable entry from a facet bundle on disk; port
/// of upstream `readFacetBundleArtifact`.
pub fn read_facet_bundle_artifact(options: &FacetBundleArtifactReadOptions) -> Result<FacetBundleArtifact, AnyError> {
    if options.entry.is_empty() {
        return Err(type_error("Facet bundle entry name must not be empty"));
    }
    let manifest_path = absolute(&options.manifest_path);
    let manifest = read_facet_bundle_manifest(&manifest_path)?;
    let entry = manifest
        .entries
        .get(&options.entry)
        .cloned()
        .ok_or_else(|| {
            error(format!(
                "Facet bundle {} has no entry named {}",
                manifest.plugin.id, options.entry
            ))
        })?;
    let module_path = resolve_bundle_file(&manifest_path, &entry.file, "entry")?;
    let source = fs::read_to_string(&module_path)
        .map_err(|e| error(format!("Could not read facet bundle entry {}: {e}", module_path.display())))?;
    verify_source(&source, &entry)?;
    let source_map_contents = match &entry.source_map {
        None => None,
        Some(map) => Some(read_string(resolve_bundle_file(&manifest_path, map, "source map")?)?),
    };
    Ok(FacetBundleArtifact {
        plugin: manifest.plugin,
        entry_name: options.entry.clone(),
        entry,
        source,
        source_map_contents,
    })
}

/// Argument shape of upstream's inline options for `readFacetBundleArtifact`.
pub struct FacetBundleArtifactReadOptions {
    pub manifest_path: PathBuf,
    pub entry: String,
}

/// Materialize a transported artifact and create a fresh loader generation per
/// load; port of upstream `createFacetBundleArtifactLoader`.
pub fn create_facet_bundle_artifact_loader(
    options: FacetBundleArtifactLoaderOptions,
    module_loader: Arc<dyn FacetBundleModuleLoader>,
) -> FacetBundleArtifactLoader {
    let artifact = validate_artifact(&options.artifact).expect("validated artifact");
    FacetBundleArtifactLoader {
        artifact,
        temporary_parent: absolute(
            &options
                .temporary_directory
                .clone()
                .unwrap_or_else(default_temp_dir),
        ),
        module_loader,
    }
}

fn default_temp_dir() -> PathBuf {
    std::env::temp_dir()
}

/// A loader that materializes the artifact into a fresh temp directory per
/// load and delegates to [`FacetBundleLoader`]; implements the upstream
/// `FacetLoader` shape.
pub struct FacetBundleArtifactLoader {
    artifact: FacetBundleArtifact,
    temporary_parent: PathBuf,
    module_loader: Arc<dyn FacetBundleModuleLoader>,
}

impl FacetLoader for FacetBundleArtifactLoader {
    fn load(&self) -> BoxFuture<Result<LoadedFacets, AnyError>> {
        let artifact = self.artifact.clone();
        let temporary_parent = self.temporary_parent.clone();
        let module_loader = Arc::clone(&self.module_loader);
        Box::pin(async move {
            fs::create_dir_all(&temporary_parent)
                .map_err(|e| error(format!("Could not create facet artifact parent directory: {e}")))?;
            let directory = temporary_parent.join(format!("chord-facet-{}", unique_suffix()));
            fs::create_dir(&directory)
                .map_err(|e| error(format!("Could not create facet artifact directory: {e}")))?;
            let result: Result<LoadedFacets, AnyError> = async {
                materialize_artifact(&directory, &artifact)?;
                create_facet_bundle_loader(
                    FacetBundleLoaderOptions {
                        manifest_path: directory.join(FACET_BUNDLE_MANIFEST_FILE),
                        entry: artifact.entry_name.clone(),
                        verify_integrity: None,
                    },
                    module_loader,
                )
                .load()
                .await
            }
            .await;
            match result {
                Ok(mut loaded) => {
                    let artifact_directory = directory.clone();
                    loaded.dispose = Some(Box::new(move || {
                        Box::pin(async move {
                            let mut errors: Vec<AnyError> = Vec::new();
                            if let Err(e) = fs::remove_dir_all(&artifact_directory) {
                                errors.push(error(format!(
                                    "Could not remove facet bundle artifact directory: {e}"
                                )));
                            }
                            match errors.len() {
                                0 => Ok(()),
                                1 => Err(errors.pop().unwrap()),
                                _ => Err(AggregateError::new(
                                    "Failed to dispose facet bundle artifact",
                                    errors,
                                )),
                            }
                        })
                    }));
                    Ok(loaded)
                }
                Err(load_error) => {
                    if let Err(cleanup_error) = fs::remove_dir_all(&directory) {
                        return Err(AggregateError::new(
                            "Facet bundle artifact loading and cleanup failed",
                            vec![load_error, error(cleanup_error.to_string())],
                        ));
                    }
                    Err(load_error)
                }
            }
        })
    }
}

/// Create a reusable loader for one opaque entry in a facet bundle manifest;
/// port of upstream `createFacetBundleLoader`.
pub fn create_facet_bundle_loader(
    options: FacetBundleLoaderOptions,
    module_loader: Arc<dyn FacetBundleModuleLoader>,
) -> FacetBundleLoader {
    // Upstream throws a TypeError here for an empty entry name; the factory
    // cannot return Result, so the same error is raised at load time.
    FacetBundleLoader {
        manifest_path: absolute(&options.manifest_path),
        entry: options.entry,
        verify_integrity: options.verify_integrity.unwrap_or(true),
        module_loader,
    }
}

pub struct FacetBundleLoader {
    manifest_path: PathBuf,
    entry: String,
    verify_integrity: bool,
    module_loader: Arc<dyn FacetBundleModuleLoader>,
}

impl FacetLoader for FacetBundleLoader {
    fn load(&self) -> BoxFuture<Result<LoadedFacets, AnyError>> {
        let manifest_path = self.manifest_path.clone();
        let entry_name = self.entry.clone();
        let verify_integrity = self.verify_integrity;
        let module_loader = Arc::clone(&self.module_loader);
        Box::pin(async move {
            if entry_name.is_empty() {
                return Err(type_error("Facet bundle entry name must not be empty"));
            }
            let manifest = read_facet_bundle_manifest(&manifest_path)?;
            let entry = manifest.entries.get(&entry_name).cloned().ok_or_else(|| {
                error(format!(
                    "Facet bundle {} has no entry named {}",
                    manifest.plugin.id, entry_name
                ))
            })?;
            let result: Result<Vec<Facet>, AnyError> = (|| async {
                let module_path = resolve_bundle_file(&manifest_path, &entry.file, "entry")?;
                let source = fs::read_to_string(&module_path)
                    .map_err(|e| error(format!("Could not read facet bundle entry {}: {e}", module_path.display())))?;
                if verify_integrity {
                    verify_source(&source, &entry)?;
                }
                module_loader.load_entry(&module_path, &entry.external_imports).await
            })()
            .await;
            match result {
                Ok(facets) => Ok(LoadedFacets {
                    facets,
                    dispose: Some(Box::new(|| Box::pin(async { Ok(()) }))),
                }),
                Err(inner) => {
                    let detail = format!(": {inner}");
                    Err(error(format!(
                        "Could not load facet bundle entry {}/{entry_name}{detail}",
                        manifest.plugin.id
                    )))
                }
            }
        })
    }
}

/// Port of upstream `verifySource` (SHA-256 over the UTF-8 source).
pub fn verify_source(source: &str, entry: &FacetBundleEntry) -> Result<(), AnyError> {
    let expected = parse_integrity(&entry.integrity)?;
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    let actual = base64::engine::general_purpose::STANDARD.encode(hasher.finalize());
    if actual != expected {
        return Err(error(format!(
            "Facet bundle integrity check failed for {}",
            entry.file
        )));
    }
    Ok(())
}

/// Port of upstream `facetsFromModule`, operating on the parsed JSON export
/// shape so the future runtime host can reuse it after evaluating CommonJS.
///
/// JSON cannot carry functions, so `setup` is required to be a present,
/// non-null member; the runtime host substitutes the actual callable when
/// building real [`Facet`] values.
pub fn facets_from_module(
    imported: &JsonValue,
    plugin_id: &str,
    entry_name: &str,
) -> Result<Vec<FacetModuleExport>, AnyError> {
    let label = format!("Facet bundle entry {plugin_id}/{entry_name}");
    let Some(record) = imported.as_object() else {
        return Err(error(format!("{label} did not export a module")));
    };
    let exported = record.get("default").unwrap_or(&JsonValue::Null);
    let candidates: Vec<&JsonValue> = match exported {
        JsonValue::Array(items) => items.iter().collect(),
        other => vec![other],
    };
    if candidates.is_empty() {
        return Err(error(format!("{label} exported no facets")));
    }
    let mut facets = Vec::new();
    for candidate in candidates {
        let id = candidate
            .get("id")
            .and_then(JsonValue::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| error(format!("{label} has a facet with an invalid ID")))?;
        let has_setup = matches!(candidate.get("setup"), Some(v) if !v.is_null());
        if !has_setup {
            return Err(error(format!("{label} facet {id} has no setup function")));
        }
        facets.push(FacetModuleExport {
            id: id.to_string(),
            setup: candidate.get("setup").cloned().unwrap_or(JsonValue::Null),
        });
    }
    let ids: std::collections::BTreeSet<&str> = facets.iter().map(|f| f.id.as_str()).collect();
    if ids.len() != facets.len() {
        return Err(error(format!("{label} exports duplicate facet IDs")));
    }
    Ok(facets)
}

/// One validated facet export from a module (runtime-host input to [`Facet`]).
#[derive(Debug)]
pub struct FacetModuleExport {
    pub id: String,
    /// Placeholder for the evaluated `setup` function (runtime-host boundary).
    pub setup: JsonValue,
}

/// Port of upstream `validateArtifact` (exact messages); input is the parsed
/// artifact value.
pub fn validate_artifact(value: &JsonValue) -> Result<FacetBundleArtifact, AnyError> {
    let record = value.as_object();
    let valid = match record {
        None => false,
        Some(record) => {
            record.get("format").and_then(JsonValue::as_str) == Some(FACET_BUNDLE_ARTIFACT_FORMAT)
                && artifact_version_matches(value)
                && value
                    .get("entryName")
                    .and_then(JsonValue::as_str)
                    .map(|name| !name.is_empty())
                    .unwrap_or(false)
                && value.get("source").and_then(JsonValue::as_str).is_some()
        }
    };
    if !valid {
        return Err(error("Invalid facet bundle artifact"));
    }
    build_validated_artifact(value)
}

fn artifact_version_matches(value: &JsonValue) -> bool {
    value.get("formatVersion").and_then(JsonValue::as_u64) == Some(FACET_BUNDLE_ARTIFACT_FORMAT_VERSION)
}

fn build_validated_artifact(value: &JsonValue) -> Result<FacetBundleArtifact, AnyError> {
    let synthetic = serde_json::json!({
        "format": crate::node::manifest::FACET_BUNDLE_FORMAT,
        "formatVersion": crate::node::manifest::FACET_BUNDLE_FORMAT_VERSION,
        "plugin": value.get("plugin").cloned().unwrap_or(JsonValue::Null),
        "entries": {
            value.get("entryName").and_then(JsonValue::as_str).unwrap_or(""): {
                "file": value.get("entry").and_then(|e| e.get("file")).cloned().unwrap_or(JsonValue::Null),
                "integrity": value.get("entry").and_then(|e| e.get("integrity")).cloned().unwrap_or(JsonValue::Null),
                "externalImports": value.get("entry").and_then(|e| e.get("externalImports")).cloned().unwrap_or(JsonValue::Null),
                "sourceMap": value.get("entry").and_then(|e| e.get("sourceMap")).cloned().unwrap_or(JsonValue::Null),
            }
        },
    });
    let manifest = validate_manifest(&synthetic, "facet bundle artifact")?;
    let entry_name = value.get("entryName").and_then(JsonValue::as_str).unwrap().to_string();
    let entry = manifest.entries.get(&entry_name).cloned().unwrap();
    if entry.source_map.is_none() {
        if value.get("sourceMapContents").map(|v| !v.is_null()).unwrap_or(false) {
            return Err(error("Facet bundle artifact has source map contents without a source map"));
        }
    } else if value.get("sourceMapContents").and_then(JsonValue::as_str).is_none() {
        return Err(error("Facet bundle artifact is missing its source map contents"));
    }
    verify_source(value.get("source").and_then(JsonValue::as_str).unwrap(), &entry)?;
    Ok(FacetBundleArtifact {
        plugin: manifest.plugin,
        entry_name,
        entry,
        source: value.get("source").and_then(JsonValue::as_str).unwrap().to_string(),
        source_map_contents: value
            .get("sourceMapContents")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
    })
}

/// Port of upstream `materializeArtifact`.
pub fn materialize_artifact(directory: &Path, artifact: &FacetBundleArtifact) -> Result<(), AnyError> {
    let manifest = FacetBundleManifest {
        format: crate::node::manifest::FACET_BUNDLE_FORMAT.to_string(),
        format_version: crate::node::manifest::FACET_BUNDLE_FORMAT_VERSION,
        plugin: artifact.plugin.clone(),
        entries: std::iter::once((artifact.entry_name.clone(), artifact.entry.clone())).collect(),
    };
    let serialized = format!("{}\n", serde_json::to_string_pretty(&manifest).map_err(|e| error(e.to_string()))?);
    fs::write(directory.join(&artifact.entry.file), &artifact.source)
        .map_err(|e| error(format!("Could not write facet bundle entry: {e}")))?;
    fs::write(directory.join(FACET_BUNDLE_MANIFEST_FILE), serialized)
        .map_err(|e| error(format!("Could not write facet bundle manifest: {e}")))?;
    if let Some(map) = &artifact.entry.source_map {
        fs::write(
            directory.join(map),
            artifact.source_map_contents.as_deref().unwrap_or_default(),
        )
        .map_err(|e| error(format!("Could not write facet bundle source map: {e}")))?;
    }
    Ok(())
}

/// Port of upstream `validatePackageSpecifier` (pure; exact message).
pub fn validate_package_specifier(specifier: &str) -> Result<(), AnyError> {
    let segments: Vec<&str> = specifier.split('/').collect();
    let package_parts = if specifier.starts_with('@') { 2 } else { 1 };
    let invalid = specifier.starts_with('.')
        || specifier.starts_with('/')
        || specifier.starts_with('#')
        || specifier.contains(':')
        || specifier.contains('\\')
        || segments.len() < package_parts
        || segments
            .iter()
            .any(|part| part.is_empty() || *part == "." || *part == "..");
    if invalid {
        return Err(error(format!(
            "Facet bundle artifact has an unsupported external import: {specifier}"
        )));
    }
    Ok(())
}

fn read_string(path: PathBuf) -> Result<String, AnyError> {
    fs::read_to_string(&path)
        .map_err(|e| error(format!("Could not read facet bundle file {}: {e}", path.display())))
}

/// `path.resolve(...)`.
fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

/// Stand-in for `mkdtemp`'s random suffix.
fn unique_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or_default();
    format!("{:x}", nanos ^ (std::process::id() as u64).rotate_left(32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use base64::engine::general_purpose::STANDARD as B64;

    fn integrity_for(contents: &str) -> String {
        format!("sha256-{}", B64.encode(Sha256::digest(contents.as_bytes())))
    }

    /// Builds a real on-disk facet bundle (manifest + entry + optional source
    /// map) and returns (directory, manifest_path, entry_name, entry_source).
    fn write_bundle(dir: &Path, with_source_map: bool, entry_source: &str) -> (PathBuf, PathBuf, String, String) {
        let file = "facet-0123456789abcdef-deadbeef.cjs";
        let mut entry = serde_json::json!({
            "file": file,
            "integrity": integrity_for(entry_source),
            "externalImports": ["@earendil-works/chord"],
        });
        if with_source_map {
            entry["sourceMap"] = serde_json::json!(format!("{file}.map"));
        }
        let manifest = serde_json::json!({
            "format": "chord.facet-bundle",
            "formatVersion": 2,
            "plugin": { "id": "test-bundle", "version": "1" },
            "entries": { "worker": entry },
        });
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join(file), entry_source).unwrap();
        if with_source_map {
            fs::write(dir.join(format!("{file}.map")), "{\"version\":3}").unwrap();
        }
        let manifest_path = dir.join(FACET_BUNDLE_MANIFEST_FILE);
        fs::write(&manifest_path, format!("{}\n", serde_json::to_string_pretty(&manifest).unwrap())).unwrap();
        (dir.to_path_buf(), manifest_path, "worker".to_string(), entry_source.to_string())
    }

    #[test]
    fn reads_and_validates_a_manifest_from_disk() {
        let dir = std::env::temp_dir().join(format!("pi-chord-bl-{}-{}", std::process::id(), unique_suffix()));
        fs::create_dir_all(&dir).unwrap();
        let (_, manifest_path, _, _) = write_bundle(&dir, false, "module.exports = {};");
        let manifest = read_facet_bundle_manifest(&manifest_path).unwrap();
        assert_eq!(manifest.plugin.id, "test-bundle");
        assert!(manifest.entries.contains_key("worker"));
    }

    #[test]
    fn reports_unreadable_manifests() {
        let err = read_facet_bundle_manifest(Path::new("/nonexistent/chord-facets.json"))
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "Could not read facet bundle manifest /nonexistent/chord-facets.json"
        );
    }

    #[test]
    fn reads_a_verified_artifact_round_trip() {
        let dir = std::env::temp_dir().join(format!("pi-chord-bl-{}-{}", std::process::id(), unique_suffix()));
        let source = "module.exports = { default: { id: \"worker\", setup: function () {} } };";
        let (_, manifest_path, entry_name, _) = write_bundle(&dir, true, source);
        let artifact = read_facet_bundle_artifact(&FacetBundleArtifactReadOptions {
            manifest_path,
            entry: entry_name,
        })
        .unwrap();
        assert_eq!(artifact.entry_name, "worker");
        assert_eq!(artifact.source, source);
        assert_eq!(artifact.source_map_contents.as_deref(), Some("{\"version\":3}"));
        assert_eq!(artifact.entry.file, "facet-0123456789abcdef-deadbeef.cjs");

        // validateArtifact round-trips the serialized artifact.
        let serialized = serde_json::json!({
            "format": "chord.facet-bundle-artifact",
            "formatVersion": 2,
            "plugin": artifact.plugin,
            "entryName": artifact.entry_name,
            "entry": artifact.entry,
            "source": artifact.source,
            "sourceMapContents": artifact.source_map_contents,
        });
        let validated = validate_artifact(&serialized).unwrap();
        assert_eq!(validated, artifact);
    }

    #[test]
    fn rejects_artifacts_with_bad_integrity_or_shape() {
        let dir = std::env::temp_dir().join(format!("pi-chord-bl-{}-{}", std::process::id(), unique_suffix()));
        let (_, manifest_path, entry_name, _) = write_bundle(&dir, false, "version-one");
        // Tamper with the entry file: integrity check must fail.
        let file = "facet-0123456789abcdef-deadbeef.cjs";
        fs::write(dir.join(file), "tampered").unwrap();
        let err = read_facet_bundle_artifact(&FacetBundleArtifactReadOptions {
            manifest_path,
            entry: entry_name,
        })
        .unwrap_err()
        .to_string();
        assert_eq!(err, format!("Facet bundle integrity check failed for {file}"));

        let err = validate_artifact(&serde_json::json!({ "format": "nope" }))
            .unwrap_err()
            .to_string();
        assert_eq!(err, "Invalid facet bundle artifact");

        let err = validate_artifact(&serde_json::json!(null)).unwrap_err().to_string();
        assert_eq!(err, "Invalid facet bundle artifact");
    }

    #[test]
    fn enforces_source_map_consistency_rules() {
        let dir = std::env::temp_dir().join(format!("pi-chord-bl-{}-{}", std::process::id(), unique_suffix()));
        let (_, manifest_path, entry_name, _source) = write_bundle(&dir, false, &source_line());
        let artifact = read_facet_bundle_artifact(&FacetBundleArtifactReadOptions {
            manifest_path: manifest_path.clone(),
            entry: entry_name.clone(),
        })
        .unwrap();
        // Contents without a source map.
        let err = validate_artifact(&artifact_json(&artifact, Some("{\"version\":3}")))
            .unwrap_err()
            .to_string();
        assert_eq!(err, "Facet bundle artifact has source map contents without a source map");

        // Source map declared but contents missing.
        let (_, manifest_path, entry_name, _source) = write_bundle(&dir, true, &source_line());
        let artifact = read_facet_bundle_artifact(&FacetBundleArtifactReadOptions {
            manifest_path,
            entry: entry_name,
        })
        .unwrap();
        let err = validate_artifact(&artifact_json_no_maps(&artifact))
            .unwrap_err()
            .to_string();
        assert_eq!(err, "Facet bundle artifact is missing its source map contents");
    }

    fn source_line() -> String {
        "module.exports = { default: [{ id: \"a\", setup: function () {} }] };".to_string()
    }

    fn artifact_json(artifact: &FacetBundleArtifact, source_map_contents: Option<&str>) -> JsonValue {
        let mut value = serde_json::json!({
            "format": "chord.facet-bundle-artifact",
            "formatVersion": 2,
            "plugin": artifact.plugin,
            "entryName": artifact.entry_name,
            "entry": artifact.entry,
            "source": artifact.source,
        });
        if let Some(contents) = source_map_contents {
            value["sourceMapContents"] = serde_json::json!(contents);
        }
        value
    }

    fn artifact_json_no_maps(artifact: &FacetBundleArtifact) -> JsonValue {
        serde_json::json!({
            "format": "chord.facet-bundle-artifact",
            "formatVersion": 2,
            "plugin": artifact.plugin,
            "entryName": artifact.entry_name,
            "entry": artifact.entry,
            "source": artifact.source,
        })
    }

    #[test]
    fn rejects_unknown_entries() {
        let dir = std::env::temp_dir().join(format!("pi-chord-bl-{}-{}", std::process::id(), unique_suffix()));
        let (_, manifest_path, _, _) = write_bundle(&dir, false, "x");
        let err = read_facet_bundle_artifact(&FacetBundleArtifactReadOptions {
            manifest_path,
            entry: "missing".to_string(),
        })
        .unwrap_err()
        .to_string();
        assert_eq!(err, "Facet bundle test-bundle has no entry named missing");
    }

    #[tokio::test]
    async fn artifact_loader_materializes_and_cleans_up() {
        let dir = std::env::temp_dir().join(format!("pi-chord-bl-{}-{}", std::process::id(), unique_suffix()));
        let materialized = dir.join("materialized");
        let (_, manifest_path, entry_name, _) = write_bundle(&dir, true, &source_line());
        let artifact = read_facet_bundle_artifact(&FacetBundleArtifactReadOptions {
            manifest_path,
            entry: entry_name,
        })
        .unwrap();
        let serialized = serde_json::json!({
            "format": "chord.facet-bundle-artifact",
            "formatVersion": 2,
            "plugin": artifact.plugin,
            "entryName": artifact.entry_name,
            "entry": artifact.entry,
            "source": artifact.source,
            "sourceMapContents": artifact.source_map_contents,
        });
        let loader = create_facet_bundle_artifact_loader(
            FacetBundleArtifactLoaderOptions {
                artifact: serialized,
                temporary_directory: Some(materialized.clone()),
            },
            Arc::new(UnsupportedModuleLoader),
        );
        // load() materializes the bundle, then fails at the JS boundary; the
        // failed generation's directory is cleaned up and the error wrapped.
        let err = match loader.load().await {
            Ok(_) => panic!("expected load failure at the JS boundary"),
            Err(e) => e.to_string(),
        };
        assert!(err.starts_with("Could not load facet bundle entry test-bundle/worker"), "{err}");
        assert!(err.contains("requires a JavaScript runtime host"), "{err}");
        let leftovers = fs::read_dir(&materialized).unwrap().count();
        assert_eq!(leftovers, 0);
    }

    #[test]
    fn validates_package_specifiers() {
        assert!(validate_package_specifier("@scope/pkg/sub").is_ok());
        assert!(validate_package_specifier("pkg").is_ok());
        assert_eq!(
            validate_package_specifier("../escape").unwrap_err().to_string(),
            "Facet bundle artifact has an unsupported external import: ../escape"
        );
        assert!(validate_package_specifier("pkg/../sub").is_err());
        assert!(validate_package_specifier("has:colon").is_err());
        assert!(validate_package_specifier("back\\slash").is_err());
        assert!(validate_package_specifier("@incomplete").is_err());
    }

    #[test]
    fn validates_export_shapes() {
        // Upstream requires `typeof candidate.setup === "function"`; the JSON
        // export stand-in carries a non-null setup payload for valid facets.
        let label_ok = serde_json::json!({
            "default": [{ "id": "a", "setup": { "js": "setup" } }, { "id": "b", "setup": { "js": "setup" } }]
        });
        let facets = facets_from_module(&label_ok, "plugin", "entry").unwrap();
        assert_eq!(facets.iter().map(|f| f.id.as_str()).collect::<Vec<_>>(), ["a", "b"]);

        assert_eq!(
            facets_from_module(&serde_json::json!("str"), "plugin", "entry")
                .unwrap_err()
                .to_string(),
            "Facet bundle entry plugin/entry did not export a module"
        );
        assert_eq!(
            facets_from_module(&serde_json::json!({ "default": [] }), "plugin", "entry")
                .unwrap_err()
                .to_string(),
            "Facet bundle entry plugin/entry exported no facets"
        );
        assert_eq!(
            facets_from_module(&serde_json::json!({ "default": [{ "id": "" }] }), "plugin", "entry")
                .unwrap_err()
                .to_string(),
            "Facet bundle entry plugin/entry has a facet with an invalid ID"
        );
        assert_eq!(
            facets_from_module(&serde_json::json!({ "default": [{ "id": "a" }] }), "plugin", "entry")
                .unwrap_err()
                .to_string(),
            "Facet bundle entry plugin/entry facet a has no setup function"
        );
        assert_eq!(
            facets_from_module(
                &serde_json::json!({ "default": [
                    { "id": "a", "setup": {} }, { "id": "a", "setup": {} }
                ] }),
                "plugin",
                "entry"
            )
            .unwrap_err()
            .to_string(),
            "Facet bundle entry plugin/entry exports duplicate facet IDs"
        );
    }

    /// Module loader standing in for the future runtime host: returns no
    /// facets so the full load path (materialize → read → verify → delegate)
    /// succeeds without JS evaluation.
    struct EmptyModuleLoader;

    #[async_trait]
    impl FacetBundleModuleLoader for EmptyModuleLoader {
        async fn load_entry(&self, _module_path: &Path, _external_imports: &[String]) -> Result<Vec<Facet>, AnyError> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn artifact_loader_disposes_by_removing_the_materialized_directory() {
        let dir = std::env::temp_dir().join(format!("pi-chord-bl-{}-{}", std::process::id(), unique_suffix()));
        let materialized = dir.join("materialized");
        let (_, manifest_path, entry_name, _) = write_bundle(&dir, true, &source_line());
        let artifact = read_facet_bundle_artifact(&FacetBundleArtifactReadOptions {
            manifest_path,
            entry: entry_name,
        })
        .unwrap();
        let serialized = serde_json::json!({
            "format": "chord.facet-bundle-artifact",
            "formatVersion": 2,
            "plugin": artifact.plugin,
            "entryName": artifact.entry_name,
            "entry": artifact.entry,
            "source": artifact.source,
            "sourceMapContents": artifact.source_map_contents,
        });
        let mut loaded = create_facet_bundle_artifact_loader(
            FacetBundleArtifactLoaderOptions {
                artifact: serialized,
                temporary_directory: Some(materialized.clone()),
            },
            Arc::new(EmptyModuleLoader),
        )
        .load()
        .await
        .unwrap();
        // Upstream materializes into a fresh mkdtemp-style generation directory
        // inside the temporary parent, so exactly one entry exists before
        // disposal and none after.
        assert_eq!(fs::read_dir(&materialized).unwrap().count(), 1);
        loaded.dispose().await.unwrap();
        assert_eq!(fs::read_dir(&materialized).unwrap().count(), 0);
    }

    #[allow(dead_code)]
    fn unused(_: BTreeMap<String, String>) {}
}
