//! Port of `packages/chord/src/bundler.ts` (pure re-exports).
//!
//! The compiler/module-loader traits (`FacetCompiler`, `FacetBundleModuleLoader`)
//! have no upstream counterpart — they carry the esbuild / CommonJS evaluation
//! boundaries for the future `pi-runtime` host.

pub use crate::node::bundle::{
    bundle_facets, BundleFacetsOptions, BundleFacetsResult, FacetBuildMessages, FacetBuildOutput,
    FacetBuildRequest, FacetBundlePlatform, FacetCompiler,
};
pub use crate::node::manifest::{FacetBundleArtifact, FacetBundleEntry, FacetBundleManifest};
pub use crate::node::package::{
    bundle_facet_package, BundleFacetPackageOptions, BundleFacetPackageResult,
};
