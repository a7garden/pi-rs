//! Wiring for `packages/chord/src/node/*` (upstream has no `node/index.ts`;
//! members are imported directly).

pub mod bundle;
pub mod bundle_loader;
pub mod manifest;
pub mod package;

// Mirrors upstream `src/node.ts` (the Node entry point re-exports).
pub use bundle::{bundle_facets, BundleFacetsOptions, BundleFacetsResult, FacetBundlePlatform};
pub use manifest::{FacetBundleEntry, FacetBundleManifest};
pub use package::{bundle_facet_package, BundleFacetPackageOptions, BundleFacetPackageResult};
