//! Port of `packages/coding-agent/src/utils/changelog.ts` — CHANGELOG.md
//! parsing and GitHub link normalization for update notifications.

use std::path::{Path, PathBuf};

use crate::utils::paths::join;

/// Parsed changelog entry (upstream `ChangelogEntry`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangelogEntry {
    pub major: i64,
    pub minor: i64,
    pub patch: i64,
    pub content: String,
}

const GITHUB_REPO: &str = "earendil-works/pi";
const CHANGELOG_LINK_BASE_PATH: &str = "packages/coding-agent";

fn entry_version(entry: &ChangelogEntry) -> String {
    format!("{}.{}.{}", entry.major, entry.minor, entry.patch)
}

fn normalize_tag(version: impl AsRef<str>) -> String {
    let version = version.as_ref();
    if version.starts_with('v') {
        version.to_string()
    } else {
        format!("v{version}")
    }
}

fn split_local_target(target: &str) -> (String, String, String) {
    let hash_index = target.find('#');
    let (before_hash, fragment) = match hash_index {
        None => (target.to_string(), String::new()),
        Some(i) => (target[..i].to_string(), target[i..].to_string()),
    };
    match before_hash.find('?') {
        None => (fragment, before_hash, String::new()),
        Some(query_index) => (
            fragment,
            before_hash[..query_index].to_string(),
            before_hash[query_index..].to_string(),
        ),
    }
}

fn normalize_path_part(value: &str) -> String {
    value.replace('\\', "/")
}

fn is_url_scheme(target: &str) -> bool {
    // Upstream `URL_SCHEME_RE`: /^[a-z][a-z0-9+.-]*:/i
    let mut chars = target.char_indices();
    match chars.next() {
        Some((_, c)) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    for (_, c) in chars {
        if c == ':' {
            return true;
        }
        if !(c.is_ascii_alphanumeric() || c == '+' || c == '.' || c == '-') {
            return false;
        }
    }
    false
}

/// Strip leading `/` segments then lexically resolve `.`/`..`
/// (node `path.posix.normalize`).
fn posix_normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|last| *last != "..") {
                    parts.pop();
                } else {
                    parts.push("..");
                }
            }
            other => parts.push(other),
        }
    }
    let joined = parts.join("/");
    if path.ends_with('/') {
        format!("{joined}/")
    } else if joined.is_empty() {
        ".".to_string()
    } else {
        joined
    }
}

fn resolve_repository_path(target_path: &str) -> Option<String> {
    let normalized_target = normalize_path_part(target_path);
    let joined = if normalized_target.starts_with('/') {
        let stripped = normalized_target.trim_start_matches('/');
        posix_normalize(stripped)
    } else {
        posix_normalize(&join_posix(CHANGELOG_LINK_BASE_PATH, &normalized_target))
    };

    if joined == "." || joined.starts_with("../") || joined == ".." {
        return None;
    }
    Some(joined)
}

fn join_posix(base: &str, rest: &str) -> String {
    if base.is_empty() {
        rest.to_string()
    } else if base.ends_with('/') {
        format!("{base}{rest}")
    } else {
        format!("{base}/{rest}")
    }
}

fn is_directory_target(original_path: &str, repository_path: &str) -> bool {
    if original_path.ends_with('/') {
        return true;
    }
    let basename = repository_path
        .rsplit('/')
        .next()
        .unwrap_or(repository_path);
    !basename.contains('.')
}

fn rewrite_legacy_repo(target: &str) -> String {
    // Upstream LEGACY_REPO_RE: /^https:\/\/github\.com\/(?:badlogic|earendil-works)\/pi-mono(?=\/|$)/
    let prefix = "https://github.com/";
    for owner in ["badlogic", "earendil-works"] {
        let legacy = format!("{prefix}{owner}/pi-mono");
        if let Some(rest) = target.strip_prefix(&legacy) {
            if rest.is_empty() || rest.starts_with('/') {
                return format!("{prefix}{GITHUB_REPO}{rest}");
            }
        }
    }
    target.to_string()
}

fn normalize_changelog_link_target(target: &str, tag: &str) -> String {
    let mut canonical_target = rewrite_legacy_repo(target);
    let repo_url = format!("{prefix}{GITHUB_REPO}", prefix = "https://github.com/");

    for route in ["blob", "tree"] {
        for branch in ["main", "master"] {
            let floating_ref_prefix = format!("{repo_url}/{route}/{branch}/");
            if let Some(rest) = canonical_target.strip_prefix(&floating_ref_prefix) {
                canonical_target = format!("{repo_url}/{route}/{tag}/{rest}");
            }
        }
    }

    if canonical_target.starts_with('#')
        || canonical_target.starts_with("//")
        || is_url_scheme(&canonical_target)
    {
        return canonical_target;
    }

    let (fragment, path_part, query) = split_local_target(&canonical_target);
    if path_part.is_empty() {
        return canonical_target;
    }

    let Some(repository_path) = resolve_repository_path(&path_part) else {
        return canonical_target;
    };

    let route = if is_directory_target(&path_part, &repository_path) {
        "tree"
    } else {
        "blob"
    };
    format!(
        "{repo_url}/{route}/{tag}/{}{query}{fragment}",
        encode_uri(&repository_path)
    )
}

/// `encodeURI`: percent-encode everything outside
/// `A-Za-z0-9;,/?:@&=+$-_.!~*'()#`.
fn encode_uri(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        let keep = byte.is_ascii_alphanumeric() || b";/?:@&=+$-_.!~*'()#".contains(&byte);
        if keep {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// Rewrite inline markdown links in changelog markdown so relative targets
/// `string | ChangelogEntry` (upstream parameter union).
#[derive(Debug, Clone)]
pub enum VersionRef<'a> {
    Str(&'a str),
    Entry(&'a ChangelogEntry),
}

impl<'a> From<&'a str> for VersionRef<'a> {
    fn from(value: &'a str) -> Self {
        VersionRef::Str(value)
    }
}

impl<'a> From<&'a ChangelogEntry> for VersionRef<'a> {
    fn from(value: &'a ChangelogEntry) -> Self {
        VersionRef::Entry(value)
    }
}

/// point at the tag-pinned GitHub source tree (`normalizeChangelogLinks`).
///
/// The upstream regex is
/// `(!?\[[^\]\n]+\]\()([^\s)]+)((?:\s+[^)]*)?\))` — an optional `!`, a
/// bracketed label without `]`/newline, a `(`, the target (no whitespace or
/// `)`), an optional title, and the closing `)`.
pub fn normalize_changelog_links<'a>(markdown: &str, version: impl Into<VersionRef<'a>>) -> String {
    let tag = normalize_tag(match version.into() {
        VersionRef::Str(v) => v.to_string(),
        VersionRef::Entry(entry) => entry_version(entry),
    });
    let bytes = markdown.as_bytes();
    let mut out = String::with_capacity(markdown.len());
    let mut i = 0;

    while i < bytes.len() {
        // Find the next link candidate: `[` (optionally preceded by `!`).
        let bracket = match markdown[i..].find('[') {
            Some(rel) => i + rel,
            None => break,
        };
        let label_start = bracket + 1;
        // Optional `!` immediately before the bracket.
        let bang = if bracket > 0 && bytes[bracket - 1] == b'!' {
            Some(bracket - 1)
        } else {
            None
        };
        // Label: no `]`, no newline.
        let label_end = bytes[label_start..]
            .iter()
            .position(|&b| b == b']' || b == b'\n')
            .map(|p| label_start + p);
        let Some(label_end) = label_end else { break };
        if label_end >= bytes.len() || bytes[label_end] != b']' {
            out.push_str(&markdown[i..=label_end]);
            i = label_end + 1;
            continue;
        }
        // `(` must immediately follow `]`.
        if label_end + 1 >= bytes.len() || bytes[label_end + 1] != b'(' {
            out.push_str(&markdown[i..=label_end]);
            i = label_end + 1;
            continue;
        }
        let target_start = label_end + 2;
        // Target: no whitespace, no `)`; then optional `\s+[^)]*` title; then `)`.
        let target_end = bytes[target_start..]
            .iter()
            .position(|&b| b == b')' || b.is_ascii_whitespace())
            .map(|p| target_start + p);
        let Some(target_end) = target_end else { break };
        let target = &markdown[target_start..target_end];
        // Find the closing paren, allowing `\s+title` between.
        let mut close: Option<usize> = None;
        if bytes[target_end] == b')' {
            close = Some(target_end);
        } else {
            // Whitespace, then non-`)` title chars, then `)`.
            let mut j = target_end;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j > target_end {
                while j < bytes.len() && bytes[j] != b')' {
                    j += 1;
                }
                if j < bytes.len() {
                    close = Some(j);
                }
            }
        }
        let Some(close) = close else {
            out.push_str(&markdown[i..target_end]);
            i = target_end;
            continue;
        };
        // Emit any preceding text, then prefix (incl. bang), rewritten target,
        // suffix (title + `)`).
        let prefix_start = bang.unwrap_or(bracket);
        out.push_str(&markdown[i..prefix_start]);
        out.push_str(&markdown[prefix_start..bracket]);
        out.push('[');
        out.push_str(&markdown[label_start..label_end]);
        out.push_str("](");
        out.push_str(&normalize_changelog_link_target(target, &tag));
        out.push_str(&markdown[target_end..=close]);
        i = close + 1;
    }
    if i < bytes.len() {
        out.push_str(&markdown[i..]);
    }
    out
}

/// Parse changelog entries from CHANGELOG.md (`parseChangelog`). Scans for
/// `##` lines and collects content until the next `##` or EOF. Read/parse
/// failures return an empty list (upstream logs a warning).
pub fn parse_changelog(changelog_path: impl AsRef<Path>) -> Vec<ChangelogEntry> {
    if !changelog_path.as_ref().exists() {
        return Vec::new();
    }

    let Ok(content) = std::fs::read_to_string(changelog_path.as_ref()) else {
        eprintln!("Warning: Could not parse changelog: read error");
        return Vec::new();
    };

    let lines = content.split('\n');
    let mut entries: Vec<ChangelogEntry> = Vec::new();
    let mut current_lines: Vec<&str> = Vec::new();
    let mut current_version: Option<(i64, i64, i64)> = None;

    for line in lines {
        if let Some(rest) = line.strip_prefix("## ") {
            if let Some((major, minor, patch)) = current_version {
                if !current_lines.is_empty() {
                    entries.push(ChangelogEntry {
                        major,
                        minor,
                        patch,
                        content: current_lines.join("\n").trim().to_string(),
                    });
                }
            }
            // Upstream: /##\s+\[?(\d+)\.(\d+)\.(\d+)\]?/ against the full line.
            match parse_version_header(line) {
                Some(v) => {
                    current_version = Some(v);
                    current_lines = vec![line];
                }
                None => {
                    current_version = None;
                    current_lines = Vec::new();
                }
            }
            let _ = rest;
        } else if current_version.is_some() {
            current_lines.push(line);
        }
    }

    if let Some((major, minor, patch)) = current_version {
        if !current_lines.is_empty() {
            entries.push(ChangelogEntry {
                major,
                minor,
                patch,
                content: current_lines.join("\n").trim().to_string(),
            });
        }
    }

    entries
}

/// `/##\s+\[?(\d+)\.(\d+)\.(\d+)\]?/`
fn parse_version_header(line: &str) -> Option<(i64, i64, i64)> {
    let rest = line.strip_prefix("##")?;
    let rest = rest.trim_start_matches([' ', '\t']);
    let rest = rest.strip_prefix('[').unwrap_or(rest);
    let mut parts = rest.splitn(3, '.');
    let major = parts.next()?;
    let minor_etc = parts.next()?;
    let patch_etc = parts.next()?;
    let parse_num = |s: &str| -> Option<i64> {
        let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            None
        } else {
            digits.parse().ok()
        }
    };
    let major = parse_num(major)?;
    let minor = parse_num(minor_etc)?;
    let patch = parse_num(patch_etc)?;
    Some((major, minor, patch))
}

/// Compare versions. Returns -1 / 0 / 1 (`compareVersions`).
pub fn compare_versions(v1: &ChangelogEntry, v2: &ChangelogEntry) -> i32 {
    if v1.major != v2.major {
        return (v1.major - v2.major).signum() as i32;
    }
    if v1.minor != v2.minor {
        return (v1.minor - v2.minor).signum() as i32;
    }
    (v1.patch - v2.patch).signum() as i32
}

/// Get entries newer than `lastVersion` (`getNewEntries`).
pub fn get_new_entries(entries: &[ChangelogEntry], last_version: &str) -> Vec<ChangelogEntry> {
    let parts: Vec<&str> = last_version.split('.').collect();
    let num = |i: usize| -> i64 {
        parts
            .get(i)
            .and_then(|p| p.parse::<i64>().ok())
            .unwrap_or(0)
    };
    let last = ChangelogEntry {
        major: num(0),
        minor: num(1),
        patch: num(2),
        content: String::new(),
    };
    entries
        .iter()
        .filter(|entry| compare_versions(entry, &last) > 0)
        .cloned()
        .collect()
}

/// Path to CHANGELOG.md (upstream re-exports `getChangelogPath` from
/// `config.ts`; the package-dir detection is a seam here — deviation, crate
/// docs D13: `PI_PACKAGE_DIR` override wins, else the executable directory).
pub fn get_changelog_path() -> String {
    let base: PathBuf = std::env::var("PI_PACKAGE_DIR")
        .ok()
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(Path::to_path_buf))
                .unwrap_or_else(|| PathBuf::from("."))
        });
    join(&base.to_string_lossy(), "CHANGELOG.md")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(major: i64, minor: i64, patch: i64) -> ChangelogEntry {
        ChangelogEntry {
            major,
            minor,
            patch,
            content: String::new(),
        }
    }

    #[test]
    fn rewrites_package_relative_links_to_tag_pinned_github() {
        let markdown = [
            "[Project Trust](README.md#project-trust)",
            "[Extensions](docs/extensions.md#project_trust)",
            "[Examples](examples/extensions/)",
            "[Root README](../../README.md#supply-chain-hardening)",
        ]
        .join("\n");

        assert_eq!(
			normalize_changelog_links(&markdown, &entry(0, 79, 0)),
			[
				"[Project Trust](https://github.com/earendil-works/pi/blob/v0.79.0/packages/coding-agent/README.md#project-trust)",
				"[Extensions](https://github.com/earendil-works/pi/blob/v0.79.0/packages/coding-agent/docs/extensions.md#project_trust)",
				"[Examples](https://github.com/earendil-works/pi/tree/v0.79.0/packages/coding-agent/examples/extensions/)",
				"[Root README](https://github.com/earendil-works/pi/blob/v0.79.0/README.md#supply-chain-hardening)",
			]
			.join("\n")
		);
    }

    #[test]
    fn canonicalizes_old_repository_urls_without_changing_external_links() {
        let markdown = [
			"[#5167](https://github.com/earendil-works/pi-mono/pull/5167)",
			"[#4163](https://github.com/badlogic/pi-mono/issues/4163)",
			"[Agent README](https://github.com/badlogic/pi-mono/blob/main/packages/agent/README.md)",
			"[External](https://example.com/docs)",
			"[Local anchor](#settings)",
		]
		.join("\n");

        assert_eq!(
			normalize_changelog_links(&markdown, "0.79.0"),
			[
				"[#5167](https://github.com/earendil-works/pi/pull/5167)",
				"[#4163](https://github.com/earendil-works/pi/issues/4163)",
				"[Agent README](https://github.com/earendil-works/pi/blob/v0.79.0/packages/agent/README.md)",
				"[External](https://example.com/docs)",
				"[Local anchor](#settings)",
			]
			.join("\n")
		);
    }

    #[test]
    fn parse_changelog_scops_version_sections() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("CHANGELOG-test-{:p}.md", &dir));
        std::fs::write(
			&path,
			"# Changelog\n\n## [0.79.0]\n\nFirst release notes.\n\n## 0.78.1\n\nHotfix\nmore\n\n## Unrelated\n\nno version\n",
		)
		.unwrap();
        let entries = parse_changelog(&path);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].major, 0);
        assert_eq!(entries[0].minor, 79);
        assert_eq!(entries[0].patch, 0);
        assert_eq!(entries[0].content, "## [0.79.0]\n\nFirst release notes.");
        assert_eq!(
            entries[1],
            ChangelogEntry {
                major: 0,
                minor: 78,
                patch: 1,
                content: "## 0.78.1\n\nHotfix\nmore".to_string(),
            }
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn parse_changelog_missing_file_is_empty() {
        assert!(parse_changelog("/nonexistent/pi/CHANGELOG.md").is_empty());
    }

    #[test]
    fn compare_and_filter_versions() {
        assert_eq!(compare_versions(&entry(0, 79, 0), &entry(0, 78, 9)), 1);
        assert_eq!(compare_versions(&entry(1, 0, 0), &entry(1, 0, 0)), 0);
        assert_eq!(compare_versions(&entry(0, 1, 0), &entry(0, 2, 0)), -1);

        let entries = vec![entry(1, 0, 0), entry(0, 79, 0), entry(0, 70, 5)];
        let new_entries = get_new_entries(&entries, "0.78.0");
        assert_eq!(new_entries, vec![entry(1, 0, 0), entry(0, 79, 0)]);

        // Non-numeric components read as 0 (upstream `Number(...) || 0`).
        assert_eq!(get_new_entries(&entries, "beta"), entries);
    }
}
