//! Port of `packages/coding-agent/src/core/source-info.ts` — provenance
//! metadata for loaded resources.
//!
//! Deviation D25 — upstream `SourceInfo.createSourceInfo` takes `PathMetadata`
//! from `core/package-manager.ts`, which is not ported yet; the struct is
//! defined here (fields identical, including `packageRoot`) and can move to
//! `core/package_manager.rs` when that module lands.

use serde::Serialize;

/// Where a resource path came from (upstream `SourceScope`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceScope {
    User,
    Project,
    Temporary,
}

/// Whether the resource came from an installed package (upstream
/// `SourceOrigin`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceOrigin {
    Package,
    #[serde(rename = "top-level")]
    TopLevel,
}

/// Provenance of a loaded resource (upstream `SourceInfo`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceInfo {
    pub path: String,
    pub source: String,
    pub scope: SourceScope,
    pub origin: SourceOrigin,
    pub base_dir: Option<String>,
}

/// Metadata attached to a resolved path (upstream `PathMetadata` from
/// `core/package-manager.ts`; see the module-level deviation D25).
#[derive(Debug, Clone, PartialEq)]
pub struct PathMetadata {
    pub source: String,
    pub scope: SourceScope,
    pub origin: SourceOrigin,
    pub base_dir: Option<String>,
    pub package_root: Option<String>,
}

/// Prefix of built-in tool and extension paths, such as `builtin:read` or
/// `builtin:mcp` (upstream `BUILTIN_PATH_PREFIX`).
pub const BUILTIN_PATH_PREFIX: &str = "builtin:";

/// Source of a path that names no file: `builtin` for `builtin:<name>`, or
/// the prefix of an angle-bracket path such as `inline` for `<inline:name>`.
/// `None` for file paths (upstream `getSyntheticPathSource`).
pub fn get_synthetic_path_source(path: &str) -> Option<String> {
    if path.starts_with(BUILTIN_PATH_PREFIX) {
        return Some("builtin".to_string());
    }
    if path.starts_with('<') && path.ends_with('>') {
        let inner = &path[1..path.len() - 1];
        return Some(match inner.split(':').next() {
            Some(prefix) if !prefix.is_empty() => prefix.to_string(),
            _ => "temporary".to_string(),
        });
    }
    None
}

/// Whether a path names no file (`builtin:*` or `<...>`) — upstream
/// `isSyntheticPath`.
pub fn is_synthetic_path(path: &str) -> bool {
    path.starts_with(BUILTIN_PATH_PREFIX) || path.starts_with('<')
}

/// Build a [`SourceInfo`] from resolved path metadata (upstream
/// `createSourceInfo`).
pub fn create_source_info(path: &str, metadata: &PathMetadata) -> SourceInfo {
    SourceInfo {
        path: path.to_string(),
        source: metadata.source.clone(),
        scope: metadata.scope,
        origin: metadata.origin,
        base_dir: metadata.base_dir.clone(),
    }
}

/// Build a [`SourceInfo`] for a synthetic path with explicit fields
/// (upstream `createSyntheticSourceInfo`).
pub fn create_synthetic_source_info(
    path: &str,
    source: &str,
    scope: Option<SourceScope>,
    origin: Option<SourceOrigin>,
    base_dir: Option<&str>,
) -> SourceInfo {
    SourceInfo {
        path: path.to_string(),
        source: source.to_string(),
        scope: scope.unwrap_or(SourceScope::Temporary),
        origin: origin.unwrap_or(SourceOrigin::TopLevel),
        base_dir: base_dir.map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_path_sources() {
        assert_eq!(
            get_synthetic_path_source("builtin:read").as_deref(),
            Some("builtin")
        );
        assert_eq!(
            get_synthetic_path_source("<inline:foo>").as_deref(),
            Some("inline")
        );
        assert_eq!(get_synthetic_path_source("<bare>").as_deref(), Some("bare"));
        assert_eq!(
            get_synthetic_path_source("<:x>").as_deref(),
            Some("temporary")
        );
        assert_eq!(get_synthetic_path_source("/tmp/x"), None);
        assert!(is_synthetic_path("builtin:read"));
        assert!(is_synthetic_path("<inline:foo>"));
        assert!(!is_synthetic_path("/tmp/x"));
    }

    #[test]
    fn synthetic_source_info_defaults() {
        let info = create_synthetic_source_info("/p", "test", None, None, None);
        assert_eq!(info.scope, SourceScope::Temporary);
        assert_eq!(info.origin, SourceOrigin::TopLevel);
        assert_eq!(info.base_dir, None);
        let info =
            create_synthetic_source_info("/p", "local", Some(SourceScope::User), None, Some("/p"));
        assert_eq!(info.scope, SourceScope::User);
        assert_eq!(info.base_dir.as_deref(), Some("/p"));
    }
}
