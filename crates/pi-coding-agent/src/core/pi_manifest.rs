//! Port of `packages/coding-agent/src/core/pi-manifest.ts` — the `pi` section
//! of a package.json manifest declaring package resources.

use crate::utils::text::strip_bom;

/// Upstream `PiManifest`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PiManifest {
    pub extensions: Option<Vec<String>>,
    pub skills: Option<Vec<String>>,
    pub prompts: Option<Vec<String>>,
    pub themes: Option<Vec<String>>,
}

const RESOURCE_FIELDS: [&str; 4] = ["extensions", "skills", "prompts", "themes"];

/// Upstream `readPiManifest` — parse the `pi` section of a package.json.
/// Returns `None` when the file is missing, unparsable, or has no usable
/// `pi` manifest.
pub fn read_pi_manifest(package_json_path: &str) -> Option<PiManifest> {
    let content = std::fs::read_to_string(package_json_path).ok()?;
    let pkg: serde_json::Value = serde_json::from_str(strip_bom(&content)).ok()?;
    let pi = pkg.get("pi")?;

    let mut manifest = PiManifest::default();
    for field in RESOURCE_FIELDS {
        let entries = pi.get(field);
        let Some(entries) = entries.and_then(|value| value.as_array()) else {
            continue;
        };
        if entries.iter().all(|entry| entry.is_string()) {
            let list: Vec<String> = entries
                .iter()
                .filter_map(|entry| entry.as_str().map(str::to_string))
                .collect();
            match field {
                "extensions" => manifest.extensions = Some(list),
                "skills" => manifest.skills = Some(list),
                "prompts" => manifest.prompts = Some(list),
                "themes" => manifest.themes = Some(list),
                _ => {}
            }
        }
    }
    Some(manifest)
}

#[cfg(test)]
mod pi_manifest_tests {
    use super::*;

    #[test]
    fn reads_pi_section_when_present() {
        let dir = fresh_dir("pi-manifest-present");
        let path = format!("{dir}/package.json");
        std::fs::write(
			&path,
			r#"{"name":"pkg","pi":{"extensions":["./src/index.ts"],"skills":["./skills"],"prompts":1,"themes":["a", 2]}}"#,
		)
		.unwrap();
        let manifest = read_pi_manifest(&path).unwrap();
        assert_eq!(
            manifest.extensions,
            Some(vec!["./src/index.ts".to_string()])
        );
        assert_eq!(manifest.skills, Some(vec!["./skills".to_string()]));
        // Non-string arrays are dropped (upstream `every typeof string` check).
        assert_eq!(manifest.prompts, None);
        assert_eq!(manifest.themes, None);
    }

    #[test]
    fn returns_none_for_missing_or_invalid_files() {
        let dir = fresh_dir("pi-manifest-missing");
        assert!(read_pi_manifest(&format!("{dir}/package.json")).is_none());
        let path = format!("{dir}/package.json");
        std::fs::write(&path, "not json").unwrap();
        assert!(read_pi_manifest(&path).is_none());
        std::fs::write(&path, r#"{"name":"pkg"}"#).unwrap();
        assert!(read_pi_manifest(&path).is_none());
    }

    fn fresh_dir(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("pm-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.to_string_lossy().into_owned()
    }
}
