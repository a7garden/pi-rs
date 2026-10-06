//! Integration tests: loading facet bundles through the pi-chord loader
//! traits over the embedded `JsRuntime`.
//!
//! Setups are exercised through the real `FacetKernel` (`Facet::setup` is
//! `pub(crate)` to pi-chord), which is exactly how the loaded facets run in
//! production.

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use pi_chord::bundler::FacetCompiler;
use pi_chord::node::bundle::bundle_facets;
use pi_chord::node::bundle_loader::{
	create_facet_bundle_loader, FacetBundleLoaderOptions, FacetBundleModuleLoader,
};
use pi_chord::node::manifest::{FacetBundleEntry, FacetBundlePlugin};
use pi_chord::services::errors::AnyError;
use pi_chord::types::{Facet, FacetLoader, FacetOptions};
use pi_runtime::{FacetBundleCompiler, FacetRuntimeModuleLoader, FacetRuntimeOptions};

/// A tiny facet entry in TypeScript, split over two modules, importing the
/// chord SDK external (registered as a JSON facade by the test).
const ENTRY_TS: &str = r#"
import { defineFacet, greeting } from "./lib.ts";
import { defineFacet as defineSdkFacet } from "@earendil-works/chord";

const sdk: unknown = defineSdkFacet;
void defineFacet;

export default [
	{
		id: "test/echo",
		setup(): void {
			greeting("from setup");
		},
	},
	{
		id: sdk ? "test/sdk" : "test/other",
		setup(env: unknown): void {
			const environment = env as {
				own: (fn: () => void) => void;
				onActivate: (fn: () => void) => void;
			};
			environment.own(() => undefined);
			environment.onActivate(() => undefined);
		},
	},
];
"#;

const LIB_TS: &str = r#"
export function defineFacet(definition: unknown): unknown {
	return definition;
}

export function greeting(name: string): string {
	return name;
}
"#;

fn futures_block<F: Future>(future: F) -> F::Output {
	tokio::runtime::Builder::new_current_thread()
		.enable_all()
		.build()
		.unwrap()
		.block_on(future)
}

struct BundleFixture {
	_dir: tempfile::TempDir,
	manifest_path: std::path::PathBuf,
	entry: FacetBundleEntry,
}

/// Bundle a TS entry through pi-chord's real bundling flow with the runtime
/// compiler, producing a manifest + integrity-verified artifact on disk.
fn bundle_fixture(
	entry_name: &str,
	entry_source: &str,
	external: Vec<String>,
) -> BundleFixture {
	let dir = tempfile::tempdir().unwrap();
	let entry_path = dir.path().join("entry.ts");
	std::fs::write(&entry_path, entry_source).unwrap();
	if entry_source.contains("./lib.ts") {
		std::fs::write(dir.path().join("lib.ts"), LIB_TS).unwrap();
	}
	let options = pi_chord::node::bundle::BundleFacetsOptions {
		plugin: FacetBundlePlugin { id: "test-plugin".to_string(), version: None },
		entries: BTreeMap::from([(
			entry_name.to_string(),
			entry_path.to_string_lossy().to_string(),
		)]),
		outdir: dir.path().join("bundle"),
		working_directory: Some(dir.path().to_path_buf()),
		external: external.clone(),
		source_map: true,
		minify: false,
		define: BTreeMap::new(),
		platform: None,
		target: Vec::new(),
	};
	let result = std::thread::spawn(move || futures_block(bundle_facets(options, &FacetBundleCompiler)))
		.join()
		.unwrap()
		.unwrap();
	let bundled = result.manifest.entries[entry_name].clone();
	if !external.is_empty() {
		assert_eq!(bundled.external_imports, external);
	}
	let manifest_path = dir.path().join("bundle").join("chord-facets.json");
	assert!(manifest_path.exists());
	BundleFixture { _dir: dir, manifest_path, entry: bundled }
}

fn module_loader_for(
	externals: HashMap<String, serde_json::Value>,
) -> Arc<FacetRuntimeModuleLoader> {
	Arc::new(FacetRuntimeModuleLoader::new(FacetRuntimeOptions {
		plugin_id: "test-plugin".to_string(),
		entry_name: "test-entry".to_string(),
		externals,
	}))
}

async fn load_facets(
	manifest_path: &Path,
	module_loader: Arc<FacetRuntimeModuleLoader>,
) -> Result<Vec<Facet>, AnyError> {
	let loader = create_facet_bundle_loader(
		FacetBundleLoaderOptions {
			manifest_path: manifest_path.to_path_buf(),
			entry: "test-entry".to_string(),
			verify_integrity: Some(true),
		},
		module_loader,
	);
	FacetLoader::load(&loader).await.map(|loaded| loaded.facets)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundles_and_loads_a_typescript_facet_bundle() {
	let fixture = bundle_fixture(
		"test-entry",
		ENTRY_TS,
		vec!["@earendil-works/chord".to_string()],
	);
	assert_eq!(fixture.entry.external_imports, vec!["@earendil-works/chord"]);

	let mut externals = HashMap::new();
	externals.insert(
		"@earendil-works/chord".to_string(),
		serde_json::json!({ "defineFacet": { "kind": "facade" } }),
	);
	let facets = load_facets(&fixture.manifest_path, module_loader_for(externals))
		.await
		.unwrap();
	assert_eq!(facets.len(), 2);
	assert_eq!(facets[0].id(), "test/echo");
	assert_eq!(facets[1].id(), "test/sdk");

	// Run the setups through the real kernel: the JS env shim's `own` and
	// `onActivate` registrations must apply cleanly.
	let mut kernel = pi_chord::facets::host::FacetKernel::new(FacetOptions {
		facets: facets.clone(),
		service_sources: Vec::new(),
		on_error: None,
	})
	.unwrap();
	kernel.activate().await.unwrap();
	kernel.dispose().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn facet_setup_errors_surface_from_the_kernel() {
	let source = r#"
export default [
	{
		id: "test/throwing",
		setup(): void {
			throw new Error("setup failed on purpose");
		},
	},
];
"#;
	let fixture = bundle_fixture("test-entry", source, Vec::new());
	let facets = load_facets(&fixture.manifest_path, module_loader_for(HashMap::new()))
		.await
		.unwrap();
	assert_eq!(facets.len(), 1);
	let mut kernel = pi_chord::facets::host::FacetKernel::new(FacetOptions {
		facets,
		service_sources: Vec::new(),
		on_error: None,
	})
	.unwrap();
	let error = kernel.activate().await.unwrap_err();
	let message = error.to_string();
	assert!(
		message.contains("setup failed on purpose"),
		"unexpected error: {message}"
	);
}

/// A hand-written CJS bundle evaluated straight through `load_entry` — the
/// upstream `node:vm` path exercised over the embedded runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn evaluates_a_plain_cjs_facet_module() {
	let dir = tempfile::tempdir().unwrap();
	let module_path = dir.path().join("facet.cjs");
	std::fs::write(
		&module_path,
		r#"module.exports = { default: { id: "worker", setup: function () {} } };"#,
	)
	.unwrap();

	let loader = FacetRuntimeModuleLoader::new(FacetRuntimeOptions::new("plugin", "entry"));
	let facets = loader.load_entry(&module_path, &[]).await.unwrap();
	assert_eq!(facets.len(), 1);
	assert_eq!(facets[0].id(), "worker");
}

/// Validation errors from `facetsFromModule` keep their upstream messages.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rejects_a_bundle_with_no_setup_function() {
	let dir = tempfile::tempdir().unwrap();
	let module_path = dir.path().join("bad.cjs");
	std::fs::write(&module_path, "module.exports = { default: [{ id: \"x\" }] };").unwrap();
	let loader = FacetRuntimeModuleLoader::new(FacetRuntimeOptions::new("plugin", "entry"));
	let error = loader.load_entry(&module_path, &[]).await.unwrap_err();
	assert!(
		error.to_string().contains("has no setup function"),
		"unexpected error: {error}"
	);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rejects_a_bundle_that_did_not_export_a_module() {
	let dir = tempfile::tempdir().unwrap();
	let module_path = dir.path().join("nope.cjs");
	std::fs::write(&module_path, "module.exports = null;").unwrap();
	let loader = FacetRuntimeModuleLoader::new(FacetRuntimeOptions::new("plugin", "entry"));
	let error = loader.load_entry(&module_path, &[]).await.unwrap_err();
	assert!(
		error.to_string().contains("did not export a module"),
		"unexpected error: {error}"
	);
}

/// The compiler is also usable directly: one `.cjs` output, template hashing.
#[tokio::test]
async fn compiler_produces_one_cjs_output() {
	let dir = tempfile::tempdir().unwrap();
	let entry = dir.path().join("entry.ts");
	std::fs::write(&entry, "export default { id: \"x\", setup() {} };").unwrap();
	let request = pi_chord::node::bundle::FacetBuildRequest {
		entry_points: vec![entry],
		abs_working_dir: dir.path().to_path_buf(),
		outdir: dir.path().to_path_buf(),
		entry_names: "facet-abc123-[hash]".to_string(),
		external: vec![],
		define: BTreeMap::new(),
		minify: false,
		platform: Default::default(),
		source_map: false,
		target: Vec::new(),
		format: "cjs".to_string(),
		banner: "\"use strict\";".to_string(),
	};
	let output = FacetBundleCompiler.build(request).await.unwrap();
	assert_eq!(output.output_files.len(), 1);
	let (path, contents) = &output.output_files[0];
	assert_eq!(path.extension().and_then(|e| e.to_str()), Some("cjs"));
	assert!(path.to_string_lossy().starts_with("facet-abc123-"));
	let source = std::str::from_utf8(contents).unwrap();
	assert!(source.starts_with("\"use strict\";"));
	assert!(source.contains("module.exports = {"));
	assert_eq!(output.entry_output, path.to_string_lossy());
}
