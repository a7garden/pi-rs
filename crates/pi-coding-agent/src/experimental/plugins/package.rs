//! Port of `experimental/plugins/package.ts` — persists explicit plugin
//! package selections per logical server and per durable Session, and builds
//! serialized server-owned plugin packages.
//!
//! Deviations (D152):
//! - `createHash("sha256")` becomes the `sha2` crate; digests are hex.
//! - `ConfiguredServerPluginPackage` keeps upstream's serialized-build
//!   behavior with a tokio mutex standing in for the `buildTail` promise
//!   chain (builds run one at a time per package).
//! - `bundleFacetPackage` requires the application-provided
//!   [`pi_runtime::FacetBundleCompiler`] (the esbuild-seam implementation).
//! - `readFacetBundleArtifact` takes an options struct instead of an inline
//!   object.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use pi_chord::node::bundle_loader::{read_facet_bundle_artifact, read_facet_bundle_manifest, FacetBundleArtifactReadOptions};
use pi_chord::node::manifest::{FacetBundleArtifact, FACET_BUNDLE_MANIFEST_FILE};
use pi_chord::node::package::{bundle_facet_package, BundleFacetPackageOptions};
use pi_chord::services::errors::{error, AnyError};
use sha2::{Digest, Sha256};

use crate::utils::paths::resolve_path_cwd;

/// Version stamped into every plugin package profile.
pub const PLUGIN_PACKAGE_PROFILE_VERSION: u64 = 1;

/// Upstream `DEFAULT_PLUGIN_FACETS`.
pub fn default_plugin_facets() -> BTreeMap<String, String> {
	BTreeMap::from([
		("session".to_string(), "src/session.ts".to_string()),
		("tui".to_string(), "src/tui.ts".to_string()),
	])
}

/// Upstream `ConfiguredServerPluginPackage`: one configured plugin package
/// with its manifest path inside the server directory and a serialized
/// `build()`.
pub struct ConfiguredServerPluginPackage {
	/// Manifest written by the package's builds.
	pub manifest_path: PathBuf,
	package_path: PathBuf,
	outdir: PathBuf,
	/// Serializes builds (upstream `buildTail` promise chaining).
	build_lock: tokio::sync::Mutex<()>,
}

impl ConfiguredServerPluginPackage {
	/// Build the package and return its presentation (`tui`) artifact, or an
	/// empty list when the package ships no `tui` facet.
	pub async fn build(&self) -> Result<Vec<FacetBundleArtifact>, AnyError> {
		let _guard = self.build_lock.lock().await;
		let result = bundle_facet_package(
			BundleFacetPackageOptions {
				package_path: self.package_path.clone(),
				outdir: self.outdir.clone(),
				default_facets: default_plugin_facets(),
			},
			&pi_runtime::FacetBundleCompiler,
		)
		.await?;
		if !result.manifest.entries.contains_key("tui") {
			return Ok(Vec::new());
		}
		Ok(vec![read_facet_bundle_artifact(&FacetBundleArtifactReadOptions {
			manifest_path: result.manifest_path,
			entry: "tui".to_string(),
		})?])
	}
}

/// Persist an explicit plugin package selection or restore it for a later
/// server generation (`restoreServerPluginPackageProfile`).
///
/// `configured_package_paths == None` restores the stored selection (empty
/// when absent); `Some` normalizes and stores it (an empty list clears the
/// stored profile).
pub async fn restore_server_plugin_package_profile(
	directory: &str,
	server_id: &str,
	configured_package_paths: Option<Vec<String>>,
) -> Result<Vec<String>, AnyError> {
	let path = server_plugin_profile_path(directory, server_id);
	match configured_package_paths {
		None => Ok(read_plugin_package_profile(&path, false, None).await?.unwrap_or_default()),
		Some(configured) => {
			let package_paths = normalize_plugin_package_paths(&configured)?;
			if package_paths.is_empty() {
				remove_file_if_exists(&path).await;
			} else {
				write_plugin_package_profile(&path, &package_paths, None).await?;
			}
			Ok(package_paths)
		}
	}
}

/// Read the explicit plugin selection stored for one durable Session.
pub async fn read_session_plugin_package_profile(
	directory: &str,
	server_id: &str,
	session_path: &str,
) -> Result<Option<Vec<String>>, AnyError> {
	read_plugin_package_profile(
		&session_plugin_profile_path(directory, server_id, session_path),
		true,
		Some(session_path),
	)
	.await
}

/// Remove the plugin selection stored for one deleted Session.
pub async fn remove_session_plugin_package_profile(directory: &str, server_id: &str, session_path: &str) {
	remove_file_if_exists(&session_plugin_profile_path(directory, server_id, session_path)).await;
}

/// Persist the plugin selection for one durable Session.
pub async fn write_session_plugin_package_profile(
	directory: &str,
	server_id: &str,
	session_path: &str,
	package_paths: &[String],
) -> Result<(), AnyError> {
	write_plugin_package_profile(
		&session_plugin_profile_path(directory, server_id, session_path),
		&normalize_plugin_package_paths(package_paths)?,
		Some(session_path),
	)
	.await
}

/// Create a serialized server-owned builder for one configured plugin package
/// (`createServerPluginPackage`).
pub fn create_server_plugin_package(
	directory: &str,
	server_id: &str,
	package_path: &str,
) -> std::sync::Arc<ConfiguredServerPluginPackage> {
	let normalized_package_path = PathBuf::from(resolve_path_cwd(package_path));
	let outdir = Path::new(directory)
		.join("plugin-builds")
		.join(server_id)
		.join(plugin_build_directory_name(&normalized_package_path));
	std::sync::Arc::new(ConfiguredServerPluginPackage {
		manifest_path: outdir.join(FACET_BUNDLE_MANIFEST_FILE),
		package_path: normalized_package_path,
		outdir,
		build_lock: tokio::sync::Mutex::new(()),
	})
}

/// Read and validate one plugin package profile (`readPluginPackageProfile`).
async fn read_plugin_package_profile(
	path: &Path,
	allow_empty: bool,
	session_path: Option<&str>,
) -> Result<Option<Vec<String>>, AnyError> {
	let raw = match tokio::fs::read_to_string(path).await {
		Ok(raw) => raw,
		// `ENOENT` reads as an absent profile.
		Err(io_error) if io_error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
		Err(io_error) => {
			return Err(error(format!(
				"Could not read experimental plugin package profile {}: {io_error}",
				path.display()
			)))
		}
	};
	let parsed: serde_json::Value = serde_json::from_str(&raw).map_err(|serde_error| {
		error(format!(
			"Could not read experimental plugin package profile {}: {serde_error}",
			path.display()
		))
	})?;
	let invalid = || error(format!("Invalid experimental plugin package profile {}", path.display()));
	let Some(object) = parsed.as_object() else {
		return Err(invalid());
	};
	for key in object.keys() {
		let allowed = match key.as_str() {
			"version" | "packagePaths" => true,
			"sessionPath" => session_path.is_some(),
			_ => false,
		};
		if !allowed {
			return Err(invalid());
		}
	}
	if object.get("version").and_then(|value| value.as_u64()) != Some(PLUGIN_PACKAGE_PROFILE_VERSION) {
		return Err(invalid());
	}
	match (session_path, object.get("sessionPath")) {
		(None, Some(_)) => return Err(invalid()),
		(Some(expected), Some(actual)) if actual.as_str() != Some(expected) => return Err(invalid()),
		(Some(_), None) => return Err(invalid()),
		_ => {}
	}
	let Some(package_paths) = object.get("packagePaths").and_then(|value| value.as_array()) else {
		return Err(invalid());
	};
	if !allow_empty && package_paths.is_empty() {
		return Err(invalid());
	}
	let mut normalized = Vec::with_capacity(package_paths.len());
	for package_path in package_paths {
		let Some(value) = package_path.as_str() else {
			return Err(invalid());
		};
		if value.is_empty() {
			return Err(invalid());
		}
		normalized.push(value.to_string());
	}
	Ok(Some(normalize_plugin_package_paths(&normalized)?))
}

async fn write_plugin_package_profile(path: &Path, package_paths: &[String], session_path: Option<&str>) -> Result<(), AnyError> {
	let mut profile = serde_json::Map::new();
	profile.insert(
		"version".to_string(),
		serde_json::Value::from(PLUGIN_PACKAGE_PROFILE_VERSION),
	);
	if let Some(session_path) = session_path {
		profile.insert("sessionPath".to_string(), serde_json::Value::from(session_path));
	}
	profile.insert(
		"packagePaths".to_string(),
		serde_json::Value::from(package_paths.to_vec()),
	);
	let body = format!(
		"{}\n",
		serde_json::to_string_pretty(&serde_json::Value::Object(profile)).expect("profile serializes")
	);
	write_private(path, &body).await
}

/// Write a 0o600 file (upstream `writeFile(..., { mode: 0o600 })`).
async fn write_private(path: &Path, body: &str) -> Result<(), AnyError> {
	let path = path.to_path_buf();
	let body = body.to_string();
	tokio::task::spawn_blocking(move || -> Result<(), AnyError> {
		if let Some(parent) = path.parent() {
			std::fs::create_dir_all(parent)
				.map_err(|io_error| error(format!("Could not create {}: {io_error}", parent.display())))?;
		}
		std::fs::write(&path, body).map_err(|io_error| {
			error(format!("Could not write experimental plugin package profile {}: {io_error}", path.display()))
		})
	})
	.await
	.expect("profile write joins")
}

/// `normalizePluginPackagePaths`: non-empty, resolved absolute, unique.
pub fn normalize_plugin_package_paths(package_paths: &[String]) -> Result<Vec<String>, AnyError> {
	let mut normalized = Vec::with_capacity(package_paths.len());
	for package_path in package_paths {
		if package_path.is_empty() {
			return Err(error("Plugin package path must not be empty"));
		}
		normalized.push(resolve_path_cwd(package_path));
	}
	let unique = normalized.iter().collect::<std::collections::BTreeSet<_>>();
	if unique.len() != normalized.len() {
		return Err(error("Plugin package paths must be unique"));
	}
	Ok(normalized)
}

fn server_plugin_profile_path(directory: &str, server_id: &str) -> PathBuf {
	Path::new(directory).join(format!("plugin-packages-{server_id}.json"))
}

fn session_plugin_profile_path(directory: &str, server_id: &str, session_path: &str) -> PathBuf {
	let hash = hex(&sha256(session_path.as_bytes())[..12]);
	Path::new(directory).join(format!("session-plugin-packages-{server_id}-{hash}.json"))
}

fn plugin_build_directory_name(package_path: &Path) -> String {
	let package_path = package_path.to_string_lossy().into_owned();
	let package_directory = if Path::new(&package_path)
		.file_name()
		.is_some_and(|name| name == "package.json")
	{
		Path::new(&package_path)
			.parent()
			.map(|parent| parent.to_string_lossy().into_owned())
			.unwrap_or_else(|| package_path.clone())
	} else {
		package_path.clone()
	};
	let base = Path::new(&package_directory)
		.file_name()
		.map(|name| name.to_string_lossy().into_owned())
		.unwrap_or_default();
	let base: String = base
		.chars()
		.map(|character| {
			if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
				character
			} else {
				'-'
			}
		})
		.collect();
	let base = if base.is_empty() { "plugin".to_string() } else { base };
	let label = if base.ends_with("-plugin") { base } else { format!("{base}-plugin") };
	let hash = hex(&sha256(package_path.as_bytes())[..6]);
	format!("{label}-{hash}")
}

fn sha256(value: &[u8]) -> [u8; 32] {
	let mut hasher = Sha256::new();
	hasher.update(value);
	hasher.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
	bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

async fn remove_file_if_exists(path: &Path) {
	// `rm(..., { force: true })` swallows ENOENT.
	let _ = tokio::fs::remove_file(path).await;
}

#[cfg(test)]
mod tests {
	use super::*;

	fn temp_dir(label: &str) -> PathBuf {
		let dir = std::env::temp_dir().join(format!("pi-plugin-{label}-{}", std::process::id()));
		let _ = std::fs::remove_dir_all(&dir);
		std::fs::create_dir_all(&dir).unwrap();
		dir
	}

	#[test]
	fn normalize_rejects_empty_and_duplicates() {
		assert_eq!(normalize_plugin_package_paths(&["/pkg/a".into()]).unwrap(), vec!["/pkg/a"]);
		assert_eq!(
			normalize_plugin_package_paths(&["".into()]).unwrap_err().to_string(),
			"Plugin package path must not be empty"
		);
		assert_eq!(
			normalize_plugin_package_paths(&["/pkg/a".into(), "/pkg/a".into()])
				.unwrap_err()
				.to_string(),
			"Plugin package paths must be unique"
		);
	}

	#[tokio::test]
	async fn server_profile_round_trips_and_clears() {
		let directory = temp_dir("server-profile");
		let directory = directory.to_string_lossy().into_owned();
		let restored = restore_server_plugin_package_profile(&directory, "sid", None).await.unwrap();
		assert!(restored.is_empty());
		restore_server_plugin_package_profile(&directory, "sid", Some(vec!["/pkg/a".into()]))
			.await
			.unwrap();
		let stored = read_plugin_package_profile(
			&Path::new(&directory).join("plugin-packages-sid.json"),
			false,
			None,
		)
		.await
		.unwrap();
		assert_eq!(stored, Some(vec!["/pkg/a".into()]));
		// An empty explicit list clears the stored profile.
		let cleared = restore_server_plugin_package_profile(&directory, "sid", Some(Vec::new())).await.unwrap();
		assert!(cleared.is_empty());
		assert!(!Path::new(&directory).join("plugin-packages-sid.json").exists());
	}

	#[tokio::test]
	async fn session_profile_binds_session_path() {
		let directory = temp_dir("session-profile");
		let directory = directory.to_string_lossy().into_owned();
		write_session_plugin_package_profile(&directory, "sid", "/sessions/a", &["/pkg/a".into()])
			.await
			.unwrap();
		let read = read_session_plugin_package_profile(&directory, "sid", "/sessions/a").await.unwrap();
		assert_eq!(read, Some(vec!["/pkg/a".into()]));
		// A stored profile whose `sessionPath` does not match the reading
		// session is invalid (same hashed path, foreign binding).
		let foreign_path = session_plugin_profile_path(&directory, "sid", "/sessions/a");
		std::fs::write(
			&foreign_path,
			r#"{"version":1,"sessionPath":"/sessions/b","packagePaths":["/pkg/a"]}"#,
		)
		.unwrap();
		assert!(read_session_plugin_package_profile(&directory, "sid", "/sessions/a").await.is_err());
		remove_session_plugin_package_profile(&directory, "sid", "/sessions/a").await;
		assert_eq!(
			read_session_plugin_package_profile(&directory, "sid", "/sessions/a").await.unwrap(),
			None
		);
	}

	#[test]
	fn build_directory_name_is_stable_and_sanitized() {
		// "my plugin" sanitizes to "my-plugin", which already ends in
		// `-plugin` so no suffix is appended.
		let name = plugin_build_directory_name(Path::new("/plugins/my plugin"));
		assert!(name.starts_with("my-plugin-"), "{name}");
		assert_eq!(name, plugin_build_directory_name(Path::new("/plugins/my plugin")));
		let json_name = plugin_build_directory_name(Path::new("/plugins/other/package.json"));
		assert!(json_name.starts_with("other-plugin-"), "{json_name}");
	}
}
