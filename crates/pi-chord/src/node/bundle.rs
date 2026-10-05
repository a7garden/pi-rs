//! Port of `packages/chord/src/node/bundle.ts`.
//!
//! Everything except esbuild itself is ported verbatim (validation, temp-dir
//! staging, atomic directory replacement with backup + rollback, manifest
//! serialization, entry post-validation, integrity hashing). The actual
//! bundling step is behind the [`FacetCompiler`] trait: evaluating esbuild is a
//! JavaScript-runtime capability that the future `pi-runtime` host provides.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine as _;
use sha2::{Digest, Sha256};

use crate::node::manifest::{
    FacetBundleEntry, FacetBundleManifest, FacetBundlePlugin, FACET_BUNDLE_MANIFEST_FILE,
};
use crate::services::errors::{error, type_error, AnyError};

/// Upstream `FacetBundlePlatform = "node" | "browser" | "neutral"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FacetBundlePlatform {
    Node,
    Browser,
    Neutral,
}

impl FacetBundlePlatform {
    /// Upstream string form (also the esbuild `--platform` value).
    pub fn as_str(&self) -> &'static str {
        match self {
            FacetBundlePlatform::Node => "node",
            FacetBundlePlatform::Browser => "browser",
            FacetBundlePlatform::Neutral => "neutral",
        }
    }
}

impl Default for FacetBundlePlatform {
    fn default() -> Self {
        FacetBundlePlatform::Node
    }
}

/// One esbuild-style build request for a single facet entry; port of the
/// `BuildOptions` upstream constructs in `bundleEntry`.
pub struct FacetBuildRequest {
    pub entry_points: Vec<PathBuf>,
    pub abs_working_dir: PathBuf,
    pub outdir: PathBuf,
    /// Upstream `entryNames` template, e.g. `facet-<hash>-[hash]`.
    pub entry_names: String,
    pub external: Vec<String>,
    pub define: BTreeMap<String, String>,
    pub minify: bool,
    pub platform: FacetBundlePlatform,
    pub source_map: bool,
    /// Upstream `target` (string or string[]); empty means the compiler default.
    pub target: Vec<String>,
    /// Always `"cjs"` in upstream.
    pub format: String,
    /// Upstream `banner.js` (`"use strict";`).
    pub banner: String,
}

/// Result of one successful compiler build. All `output_files` paths are
/// relative to the request's `outdir` (upstream reports metafile paths and
/// reads them back from disk; the compiler hands us the bytes directly).
pub struct FacetBuildOutput {
    pub output_files: Vec<(PathBuf, Vec<u8>)>,
    /// Path of the single `.cjs` entry output, relative to `outdir`
    /// (upstream: the metafile output with an `entryPoint`).
    pub entry_output: String,
    /// Externally-resolved import specifiers from the build metadata.
    pub external_imports: Vec<String>,
    /// Path of the emitted source map, relative to `outdir`, when any.
    pub source_map_path: Option<String>,
    /// Human-readable diagnostics, used to format build-failure messages.
    pub messages: Vec<String>,
}

/// Diagnostic list attached to a build failure; upstream reads
/// `error.errors` (esbuild `Message[]`) off the thrown error.
#[derive(Debug)]
pub struct FacetBuildMessages {
    pub messages: Vec<String>,
}

impl std::fmt::Display for FacetBuildMessages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.messages.join("\n"))
    }
}

impl std::error::Error for FacetBuildMessages {}

/// The esbuild boundary. Implementations live in the future `pi-runtime` JS
/// host; tests provide in-memory compilers.
#[async_trait]
pub trait FacetCompiler: Send + Sync {
    async fn build(&self, request: FacetBuildRequest) -> Result<FacetBuildOutput, AnyError>;
}

/// Upstream `BundleFacetsOptions`.
pub struct BundleFacetsOptions {
    pub plugin: FacetBundlePlugin,
    /// Opaque application-selected entry names mapped to source files.
    pub entries: BTreeMap<String, String>,
    pub outdir: PathBuf,
    pub working_directory: Option<PathBuf>,
    /// Additional package imports intentionally left for the loading application to resolve.
    pub external: Vec<String>,
    pub source_map: bool,
    pub minify: bool,
    pub define: BTreeMap<String, String>,
    pub platform: Option<FacetBundlePlatform>,
    /// Upstream `target?: string | readonly string[]`.
    pub target: Vec<String>,
}

/// Upstream `BundleFacetsResult`.
#[derive(Debug)]
pub struct BundleFacetsResult {
    pub manifest: FacetBundleManifest,
    pub manifest_path: PathBuf,
}

/// Bundle each opaque facet entry into an independent content-addressed
/// CommonJS file. Port of upstream `bundleFacets`.
pub async fn bundle_facets(options: BundleFacetsOptions, compiler: &dyn FacetCompiler) -> Result<BundleFacetsResult, AnyError> {
    validate_options(&options)?;
    let working_directory = match &options.working_directory {
        Some(dir) => fs::canonicalize(dir)
            .map_err(|e| error(format!("Could not resolve facet bundle working directory {}: {e}", dir.display())))?,
        None => std::env::current_dir().map_err(|e| error(format!("Could not resolve process working directory: {e}")))?,
    };
    // Upstream `resolve(workingDirectory, options.outdir)`; the outdir need not
    // exist yet, so join lexically.
    let output_directory = resolve_lexical(&working_directory, &options.outdir);
    let output_parent = output_directory
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&output_parent)
        .map_err(|e| error(format!("Could not create facet bundle output parent {}: {e}", output_parent.display())))?;
    let temporary_directory =
        output_parent.join(format!(".{}.tmp-{}", file_name(&output_directory), unique_suffix()));
    fs::create_dir(&temporary_directory).map_err(|e| {
        error(format!(
            "Could not create facet bundle staging directory {}: {e}",
            temporary_directory.display()
        ))
    })?;
    let result = bundle_into(
        &options,
        compiler,
        &working_directory,
        &output_directory,
        &temporary_directory,
    )
    .await;
    if let Err(err) = result {
        let _ = fs::remove_dir_all(&temporary_directory);
        return Err(err);
    }
    let (manifest, manifest_path) = result.unwrap();
    Ok(BundleFacetsResult {
        manifest,
        manifest_path,
    })
}

async fn bundle_into(
    options: &BundleFacetsOptions,
    compiler: &dyn FacetCompiler,
    working_directory: &Path,
    output_directory: &Path,
    temporary_directory: &Path,
) -> Result<(FacetBundleManifest, PathBuf), AnyError> {
    let mut entries = BTreeMap::new();
    // BTreeMap iteration is sorted by key, matching upstream's localeCompare
    // ordering of `Object.entries` (byte order differs only for non-ASCII).
    for (entry_name, source) in &options.entries {
        let entry = bundle_entry(
            entry_name,
            &resolve_lexical(working_directory, Path::new(source)),
            temporary_directory,
            working_directory,
            options,
            compiler,
        )
        .await?;
        entries.insert(entry_name.clone(), entry);
    }
    let manifest = FacetBundleManifest {
        format: crate::node::manifest::FACET_BUNDLE_FORMAT.to_string(),
        format_version: crate::node::manifest::FACET_BUNDLE_FORMAT_VERSION,
        plugin: options.plugin.clone(),
        entries,
    };
    let serialized = format!("{}\n", serde_json::to_string_pretty(&manifest).map_err(|e| error(e.to_string()))?);
    fs::write(temporary_directory.join(FACET_BUNDLE_MANIFEST_FILE), serialized)
        .map_err(|e| error(format!("Could not write facet bundle manifest: {e}")))?;
    replace_directory(temporary_directory, output_directory)?;
    Ok((
        manifest,
        output_directory.join(FACET_BUNDLE_MANIFEST_FILE),
    ))
}

/// Port of upstream `bundleEntry`: run the compiler, then post-validate the
/// single entry output and hash it.
async fn bundle_entry(
    entry_name: &str,
    source: &Path,
    temporary_directory: &Path,
    working_directory: &Path,
    options: &BundleFacetsOptions,
    compiler: &dyn FacetCompiler,
) -> Result<FacetBundleEntry, AnyError> {
    let entry_prefix = format!("facet-{}", short_hash(entry_name));
    let requested_platform = options.platform.unwrap_or(FacetBundlePlatform::Node);
    let mut external = BTreeSet::new();
    external.insert("@earendil-works/chord".to_string());
    external.insert("@earendil-works/chord/*".to_string());
    external.extend(options.external.iter().cloned());
    let target = if !options.target.is_empty() {
        options.target.clone()
    } else if requested_platform == FacetBundlePlatform::Node {
        vec!["node22.19".to_string()]
    } else {
        vec!["es2022".to_string()]
    };
    let request = FacetBuildRequest {
        entry_points: vec![source.to_path_buf()],
        abs_working_dir: working_directory.to_path_buf(),
        outdir: temporary_directory.to_path_buf(),
        entry_names: format!("{entry_prefix}-[hash]"),
        external: external.into_iter().collect(),
        define: options.define.clone(),
        minify: options.minify,
        platform: requested_platform,
        source_map: options.source_map,
        target,
        format: "cjs".to_string(),
        banner: "\"use strict\";".to_string(),
    };
    let result = compiler.build(request).await;
    let output = match result {
        Ok(output) => output,
        Err(build_error) => {
            let diagnostics = build_error
                .downcast_ref::<FacetBuildMessages>()
                .map(|m| m.messages.clone())
                .unwrap_or_default();
            let detail = if diagnostics.is_empty() {
                String::new()
            } else {
                format!("\n{}", diagnostics.join("\n"))
            };
            return Err(error(format!("Could not bundle facet entry {entry_name}{detail}")));
        }
    };
    // Upstream runs esbuild with `write: true`, so the build itself materializes
    // the bundle (and source map) inside the temporary directory. The Rust
    // compiler trait returns bytes, so materialize them here before the atomic
    // directory swap publishes them.
    for (path, contents) in &output.output_files {
        let destination = temporary_directory.join(path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                error(format!(
                    "Could not write facet entry {entry_name} output {}: {e}",
                    destination.display()
                ))
            })?;
        }
        fs::write(&destination, contents).map_err(|e| {
            error(format!(
                "Could not write facet entry {entry_name} output {}: {e}",
                destination.display()
            ))
        })?;
    }
    // Exactly one entry-point `.cjs` output (upstream filters the metafile).
    let cjs_outputs: Vec<&(PathBuf, Vec<u8>)> = output
        .output_files
        .iter()
        .filter(|(path, _)| path.extension().and_then(|e| e.to_str()) == Some("cjs"))
        .collect();
    if cjs_outputs.len() != 1 {
        return Err(error(format!(
            "Facet entry {entry_name} did not produce exactly one JavaScript file"
        )));
    }
    let file = output.entry_output.clone();
    let invalid_path = file.is_empty()
        || file == ".."
        || file.starts_with("../")
        || file.starts_with('/')
        || Path::new(&file).file_name().and_then(|n| n.to_str()) != Some(file.as_str());
    if invalid_path {
        return Err(error(format!(
            "Facet entry {entry_name} produced an invalid output path"
        )));
    }
    let source_map = if options.source_map {
        Some(format!("{file}.map"))
    } else {
        None
    };
    let mut allowed: BTreeSet<PathBuf> = BTreeSet::from([PathBuf::from(&file)]);
    if let Some(map) = &source_map {
        allowed.insert(PathBuf::from(map));
    }
    let unexpected: Vec<&PathBuf> = output
        .output_files
        .iter()
        .map(|(path, _)| path)
        .filter(|path| !allowed.contains(*path))
        .collect();
    if !unexpected.is_empty() {
        return Err(error(format!(
            "Facet entry {entry_name} produced files other than its JavaScript bundle and source map"
        )));
    }
    let contents = &cjs_outputs[0].1;
    // Upstream `stat(join(temporaryDirectory, sourceMap))`: the map must exist.
    if let Some(map) = &source_map {
        if !output.output_files.iter().any(|(path, _)| path == map) {
            return Err(error(format!(
                "Facet entry {entry_name} produced no source map"
            )));
        }
    }
    let mut hasher = Sha256::new();
    hasher.update(contents);
    let integrity = format!("sha256-{}", base64::engine::general_purpose::STANDARD.encode(hasher.finalize()));
    let mut external_imports: Vec<String> = output.external_imports.clone();
    external_imports.sort();
    external_imports.dedup();
    let mut entry = FacetBundleEntry {
        file,
        integrity,
        external_imports,
        source_map: None,
    };
    if let Some(map) = source_map {
        entry.source_map = Some(map);
    }
    Ok(entry)
}

/// Port of upstream `validateOptions` (exact messages, upstream `TypeError`s).
fn validate_options(options: &BundleFacetsOptions) -> Result<(), AnyError> {
    if options.plugin.id.is_empty() {
        return Err(type_error("Facet bundle plugin ID must not be empty"));
    }
    if let Some(version) = &options.plugin.version {
        if version.is_empty() {
            return Err(type_error("Facet bundle plugin version must not be empty"));
        }
    }
    if options.entries.is_empty() {
        return Err(type_error("Facet bundle must contain at least one entry"));
    }
    for (name, source) in &options.entries {
        if name.is_empty() {
            return Err(type_error("Facet bundle entry name must not be empty"));
        }
        if source.is_empty() {
            return Err(type_error(format!(
                "Facet bundle entry {name} must have a source path"
            )));
        }
    }
    for external in &options.external {
        if external.is_empty() {
            return Err(type_error("Facet bundle external import must not be empty"));
        }
    }
    Ok(())
}

/// Port of upstream `replaceDirectory`: atomic swap with backup + rollback.
fn replace_directory(temporary_directory: &Path, output_directory: &Path) -> Result<(), AnyError> {
    let backup_directory = PathBuf::from(format!(
        "{}.old-{}",
        output_directory.display(),
        unique_suffix()
    ));
    let mut moved_existing = false;
    match fs::rename(output_directory, &backup_directory) {
        Ok(()) => moved_existing = true,
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => {
            return Err(error(format!(
                "Could not back up facet bundle output directory {}: {e}",
                output_directory.display()
            )))
        }
    }
    if let Err(e) = fs::rename(temporary_directory, output_directory) {
        if moved_existing {
            fs::rename(&backup_directory, output_directory).map_err(|rollback| {
                error(format!(
                    "Could not restore facet bundle output directory after failure: {rollback} (original error: {e})"
                ))
            })?;
        }
        return Err(error(format!(
            "Could not publish facet bundle output directory {}: {e}",
            output_directory.display()
        )));
    }
    if moved_existing {
        let _ = fs::remove_dir_all(&backup_directory);
    }
    Ok(())
}

/// `createHash("sha256").update(value).digest("hex").slice(0, 12)`.
pub fn short_hash(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    hex[..12].to_string()
}

/// Stand-in for `randomUUID()`: a collision-resistant-enough suffix for
/// staging/backup directory names within one machine.
fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or_default();
    format!("{:x}", nanos ^ (process::id() as u64).rotate_left(32))
}

/// Upstream `resolve(workingDirectory, source)` for sources that may not exist
/// yet: absolute sources win, otherwise lexical join.
fn resolve_lexical(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct TestCompiler {
        fail_worker: AtomicBool,
    }

    /// Deterministic stand-in for esbuild: one `.cjs` output plus an optional
    /// source map, content-addressed by plugin version + entry name.
    impl TestCompiler {
        fn output_name(&self, request: &FacetBuildRequest) -> String {
            let version_tag = request
                .abs_working_dir
                .to_string_lossy()
                .into_owned();
            format!(
                "{}-{}.cjs",
                request.entry_names.replace("-[hash]", ""),
                short_hash(&version_tag)
            )
        }
    }

    #[async_trait]
    impl FacetCompiler for TestCompiler {
        async fn build(&self, request: FacetBuildRequest) -> Result<FacetBuildOutput, AnyError> {
            let entry_name = request
                .entry_points
                .first()
                .and_then(|p| p.file_stem())
                .and_then(|s| s.to_str())
                .unwrap_or("entry")
                .to_string();
            if self.fail_worker.load(Ordering::SeqCst) {
                return Err(error_with_messages(format!(
                    "{entry_name}.ts:1:1: syntax error"
                )));
            }
            let name = self.output_name(&request);
            let mut output_files = vec![(
                PathBuf::from(&name),
                format!("module.exports = {{ id: \"{entry_name}\" }};").into_bytes(),
            )];
            let mut source_map_path = None;
            if request.source_map {
                let map = format!("{name}.map");
                output_files.push((PathBuf::from(&map), b"{}".to_vec()));
                source_map_path = Some(map);
            }
            Ok(FacetBuildOutput {
                output_files,
                entry_output: name,
                external_imports: request.external.clone(),
                source_map_path,
                messages: Vec::new(),
            })
        }
    }

    fn error_with_messages(messages: String) -> AnyError {
        // Attach diagnostics the way upstream esbuild errors carry
        // `error.errors`; bundle_entry recovers them for message formatting.
        FacetBuildMessages {
            messages: vec![messages],
        }
        .into()
    }

    fn options(entries: &[(&str, &str)], outdir: &Path) -> BundleFacetsOptions {
        BundleFacetsOptions {
            plugin: FacetBundlePlugin {
                id: "test-bundle".to_string(),
                version: Some("1".to_string()),
            },
            entries: entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            outdir: outdir.to_path_buf(),
            working_directory: None,
            external: vec![],
            source_map: false,
            minify: false,
            define: BTreeMap::new(),
            platform: None,
            target: vec![],
        }
    }

    fn write_source(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, contents).unwrap();
        path
    }

    #[tokio::test]
    async fn bundles_entries_into_a_content_addressed_manifest() {
        let dir = tempdir();
        let source = write_source(&dir, "worker.ts", "export default {};");
        let outdir = dir.join("bundle");
        let mut opts = options(&[("worker", &source.to_string_lossy())], &outdir);
        opts.source_map = true;
        let result = bundle_facets(opts, &TestCompiler { fail_worker: AtomicBool::new(false) })
            .await
            .unwrap();
        assert_eq!(result.manifest_path, outdir.join(FACET_BUNDLE_MANIFEST_FILE));
        let entry = result.manifest.entries.get("worker").unwrap();
        assert!(entry.file.starts_with("facet-") && entry.file.ends_with(".cjs"));
        assert_eq!(entry.source_map.as_deref(), Some(&*format!("{}.map", entry.file)));
        // Manifest file is pretty-printed with a trailing newline.
        let raw = fs::read_to_string(&result.manifest_path).unwrap();
        assert!(raw.ends_with("\n"));
        let reparsed: FacetBundleManifest = serde_json::from_str(&raw).unwrap();
        assert_eq!(reparsed, result.manifest);
        // Files actually materialized on disk.
        assert!(outdir.join(&entry.file).exists());
        assert!(outdir.join(format!("{}.map", entry.file)).exists());
        // Second identical build replaces the directory atomically with equal output.
        let mut second_options = options(&[("worker", &source.to_string_lossy())], &outdir);
        second_options.source_map = true;
        let second = bundle_facets(
            second_options,
            &TestCompiler { fail_worker: AtomicBool::new(false) },
        )
        .await
        .unwrap();
        assert_eq!(second.manifest.entries.get("worker").unwrap(), entry);
        // No staging directories left behind.
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(".bundle.tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[tokio::test]
    async fn replaces_an_existing_output_directory() {
        let dir = tempdir();
        let source = write_source(&dir, "worker.ts", "a");
        let outdir = dir.join("bundle");
        fs::create_dir_all(&outdir).unwrap();
        fs::write(outdir.join("stale.txt"), b"old").unwrap();
        bundle_facets(
            options(&[("worker", &source.to_string_lossy())], &outdir),
            &TestCompiler { fail_worker: AtomicBool::new(false) },
        )
        .await
        .unwrap();
        let names: Vec<_> = fs::read_dir(&outdir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(!names.iter().any(|n| n == "stale.txt"), "{names:?}");
        assert!(outdir.join(FACET_BUNDLE_MANIFEST_FILE).exists());
    }

    #[tokio::test]
    async fn validates_options_with_upstream_messages() {
        let dir = tempdir();
        let outdir = dir.join("bundle");
        let compiler = TestCompiler { fail_worker: AtomicBool::new(false) };
        let mut opts = options(&[("worker", "w.ts")], &outdir);
        opts.plugin.id = String::new();
        assert_eq!(
            bundle_facets(opts, &compiler).await.unwrap_err().to_string(),
            "Facet bundle plugin ID must not be empty"
        );

        let mut opts = options(&[("worker", "w.ts")], &outdir);
        opts.plugin.version = Some(String::new());
        assert_eq!(
            bundle_facets(opts, &compiler).await.unwrap_err().to_string(),
            "Facet bundle plugin version must not be empty"
        );

        let opts = options(&[], &outdir);
        assert_eq!(
            bundle_facets(opts, &compiler).await.unwrap_err().to_string(),
            "Facet bundle must contain at least one entry"
        );

        let mut opts = options(&[("", "w.ts")], &outdir);
        opts.entries.insert("worker".to_string(), String::new());
        let opts = BundleFacetsOptions {
            entries: BTreeMap::from([("".to_string(), "w.ts".to_string())]),
            ..opts
        };
        assert_eq!(
            bundle_facets(opts, &compiler).await.unwrap_err().to_string(),
            "Facet bundle entry name must not be empty"
        );

        let mut opts = options(&[("worker", "w.ts")], &outdir);
        opts.external = vec![String::new()];
        assert_eq!(
            bundle_facets(opts, &compiler).await.unwrap_err().to_string(),
            "Facet bundle external import must not be empty"
        );

        let mut opts = options(&[("worker", "")], &outdir);
        opts.plugin.id = "p".to_string();
        assert_eq!(
            bundle_facets(opts, &compiler).await.unwrap_err().to_string(),
            "Facet bundle entry worker must have a source path"
        );
    }

    #[tokio::test]
    async fn formats_compiler_failures_with_diagnostics() {
        let dir = tempdir();
        let source = write_source(&dir, "worker.ts", "bad");
        let outdir = dir.join("bundle");
        let compiler = TestCompiler {
            fail_worker: AtomicBool::new(true),
        };
        let err = bundle_facets(
            options(&[("worker", &source.to_string_lossy())], &outdir),
            &compiler,
        )
        .await
        .unwrap_err()
        .to_string();
        assert_eq!(
            err,
            "Could not bundle facet entry worker\nworker.ts:1:1: syntax error"
        );
        // The staging directory is cleaned up on failure.
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(".bundle.tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[tokio::test]
    async fn rejects_wrong_output_shapes() {
        struct TwoEntriesCompiler;
        #[async_trait]
        impl FacetCompiler for TwoEntriesCompiler {
            async fn build(&self, _request: FacetBuildRequest) -> Result<FacetBuildOutput, AnyError> {
                Ok(FacetBuildOutput {
                    output_files: vec![
                        (PathBuf::from("a.cjs"), b"1".to_vec()),
                        (PathBuf::from("b.cjs"), b"2".to_vec()),
                    ],
                    entry_output: "a.cjs".to_string(),
                    external_imports: vec![],
                    source_map_path: None,
                    messages: vec![],
                })
            }
        }
        struct EscapedCompiler;
        #[async_trait]
        impl FacetCompiler for EscapedCompiler {
            async fn build(&self, _request: FacetBuildRequest) -> Result<FacetBuildOutput, AnyError> {
                Ok(FacetBuildOutput {
                    output_files: vec![(PathBuf::from("../escape.cjs"), b"1".to_vec())],
                    entry_output: "../escape.cjs".to_string(),
                    external_imports: vec![],
                    source_map_path: None,
                    messages: vec![],
                })
            }
        }
        struct ExtraFileCompiler;
        #[async_trait]
        impl FacetCompiler for ExtraFileCompiler {
            async fn build(&self, _request: FacetBuildRequest) -> Result<FacetBuildOutput, AnyError> {
                Ok(FacetBuildOutput {
                    output_files: vec![
                        (PathBuf::from("a.cjs"), b"1".to_vec()),
                        (PathBuf::from("extra.txt"), b"2".to_vec()),
                    ],
                    entry_output: "a.cjs".to_string(),
                    external_imports: vec![],
                    source_map_path: None,
                    messages: vec![],
                })
            }
        }

        let dir = tempdir();
        let source = write_source(&dir, "worker.ts", "a");
        let outdir = dir.join("bundle");
        let err = bundle_facets(
            options(&[("worker", &source.to_string_lossy())], &outdir),
            &TwoEntriesCompiler,
        )
        .await
        .unwrap_err()
        .to_string();
        assert_eq!(err, "Facet entry worker did not produce exactly one JavaScript file");

        let err = bundle_facets(
            options(&[("worker", &source.to_string_lossy())], &outdir),
            &EscapedCompiler,
        )
        .await
        .unwrap_err()
        .to_string();
        assert_eq!(err, "Facet entry worker produced an invalid output path");

        let err = bundle_facets(
            options(&[("worker", &source.to_string_lossy())], &outdir),
            &ExtraFileCompiler,
        )
        .await
        .unwrap_err()
        .to_string();
        assert_eq!(
            err,
            "Facet entry worker produced files other than its JavaScript bundle and source map"
        );
    }

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pi-chord-bundle-test-{}-{:x}",
            process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
