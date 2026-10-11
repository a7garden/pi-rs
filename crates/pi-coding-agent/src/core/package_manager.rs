//! Port of `packages/coding-agent/src/core/package-manager.ts` — package
//! source parsing, npm/git install management, and resource resolution for
//! extensions/skills/prompts/themes.
//!
//! Deviations:
//! - **D55** — dependency stand-ins and seams for this module:
//!   - `isStdoutTakenOver` gates the inherit-spawn stdio; it reads
//!     [`crate::core::output_guard::is_stdout_taken_over`] (D91 state flag).
//!   - Upstream tests mock private methods via `vi.spyOn`; here process
//!     execution lives behind the [`CommandRunner`] trait
//!     ([`ProcessCommandRunner`] is the real implementation, tests inject a
//!     recording fake through [`DefaultPackageManager::with_runner`]).
//!   - `resolve`'s `onMissing` callback is synchronous in Rust (upstream
//!     awaits a `Promise<MissingSourceAction>`); install decisions are
//!     identical.
//! - **D55a** — upstream npm `minimatch` semantics are approximated with the
//!   `glob` crate's `Pattern` (brace expansion unsupported; `*` also matches
//!   `/` with the default options). The gitignore matcher mirrors
//!   `core/skills.rs`'s `IgnoreMatcher` (D24).
//! - **D55b** — npm/git command execution is synchronous-through-async via
//!   [`tokio::process`]; upstream's `.cmd`/`.bat` `cross-spawn` shim does not
//!   apply (same rationale as utils D13).

use std::collections::HashSet;
use std::future::Future;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::join_all;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use indexmap::IndexMap;
use parking_lot::Mutex;

use sha2::Digest;

use crate::config::CONFIG_DIR_NAME;
use crate::core::pi_manifest::{read_pi_manifest, PiManifest};
use crate::core::settings_manager::{PackageSource, Settings, SettingsManager};
use crate::core::source_info::{PathMetadata, SourceOrigin, SourceScope, BUILTIN_PATH_PREFIX};
use crate::utils::child_process::SyncOutput;
use crate::utils::git::{parse_git_url, GitSource};
use crate::utils::paths::{
    canonicalize_path, is_local_path, mark_path_ignored_by_cloud_sync, normalize_path,
    resolve_path, PathInputOptions,
};
use crate::utils::text::strip_bom;

const NETWORK_TIMEOUT_MS: u64 = 10000;
const UPDATE_CHECK_CONCURRENCY: usize = 4;
const GIT_UPDATE_CONCURRENCY: usize = 4;

const RESOURCE_TYPES: [ResourceType; 4] = [
    ResourceType::Extensions,
    ResourceType::Skills,
    ResourceType::Prompts,
    ResourceType::Themes,
];

/// Upstream `ResourceType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Extensions,
    Skills,
    Prompts,
    Themes,
}

impl ResourceType {
    fn key(self) -> &'static str {
        match self {
            ResourceType::Extensions => "extensions",
            ResourceType::Skills => "skills",
            ResourceType::Prompts => "prompts",
            ResourceType::Themes => "themes",
        }
    }

    #[allow(dead_code)]
    fn parse(key: &str) -> Option<ResourceType> {
        match key {
            "extensions" => Some(ResourceType::Extensions),
            "skills" => Some(ResourceType::Skills),
            "prompts" => Some(ResourceType::Prompts),
            "themes" => Some(ResourceType::Themes),
            _ => None,
        }
    }
}

fn file_pattern_matches(resource_type: ResourceType, name: &str) -> bool {
    match resource_type {
        ResourceType::Extensions => name.ends_with(".ts") || name.ends_with(".js"),
        ResourceType::Skills | ResourceType::Prompts => name.ends_with(".md"),
        ResourceType::Themes => name.ends_with(".json"),
    }
}

/// Whether a manifest/settings entry is an include/exclude glob (`isPattern`).
fn is_pattern(s: &str) -> bool {
    s.starts_with('!')
        || s.starts_with('+')
        || s.starts_with('-')
        || s.contains('*')
        || s.contains('?')
}

/// Whether an entry is an explicit `!`/`+`/`-` override (`isOverridePattern`).
fn is_override_pattern(s: &str) -> bool {
    s.starts_with('!') || s.starts_with('+') || s.starts_with('-')
}

fn has_glob_pattern(s: &str) -> bool {
    s.contains('*') || s.contains('?')
}

/// A resolved resource entry (upstream `ResolvedResource`; metadata is
/// `core/source_info.rs`'s `PathMetadata`, which mirrors upstream
/// `package-manager.ts`'s `PathMetadata`).
#[derive(Debug, Clone)]
pub struct ResolvedResource {
    pub path: String,
    pub enabled: bool,
    pub metadata: PathMetadata,
}

/// Upstream `ResolvedPaths`.
#[derive(Debug, Clone, Default)]
pub struct ResolvedPaths {
    pub extensions: Vec<ResolvedResource>,
    pub skills: Vec<ResolvedResource>,
    pub prompts: Vec<ResolvedResource>,
    pub themes: Vec<ResolvedResource>,
}

impl ResolvedPaths {
    /// All resources across the four types (test/parent convenience).
    pub fn all(&self) -> impl Iterator<Item = &ResolvedResource> {
        self.extensions
            .iter()
            .chain(self.skills.iter())
            .chain(self.prompts.iter())
            .chain(self.themes.iter())
    }
}

/// Upstream `MissingSourceAction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingSourceAction {
    Install,
    Skip,
    Error,
}

/// Upstream `ProgressEvent`.
#[derive(Debug, Clone)]
pub struct ProgressEvent {
    pub event_type: ProgressEventType,
    pub action: ProgressAction,
    pub source: String,
    pub message: Option<String>,
}

/// Upstream `ProgressEvent["type"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressEventType {
    Start,
    Progress,
    Complete,
    Error,
}

/// Upstream `ProgressEvent["action"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressAction {
    Install,
    Remove,
    Update,
    Clone,
    Pull,
}

pub type ProgressCallback = Arc<dyn Fn(&ProgressEvent) + Send + Sync>;

/// Upstream `PackageUpdate`.
#[derive(Debug, Clone)]
pub struct PackageUpdate {
    pub source: String,
    pub display_name: String,
    pub update_type: PackageUpdateType,
    pub scope: SourceScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageUpdateType {
    Npm,
    Git,
}

/// Upstream `ConfiguredPackage`.
#[derive(Debug, Clone)]
pub struct ConfiguredPackage {
    pub source: String,
    pub scope: SourceScope,
    pub filtered: bool,
    pub installed_path: Option<String>,
}

/// Upstream `PackageManagerOptions`.
pub struct PackageManagerOptions {
    pub cwd: String,
    pub agent_dir: String,
    pub settings_manager: Arc<Mutex<SettingsManager>>,
    /// Names of built-in extensions, resolved as `builtin:<name>` extension
    /// resources.
    pub builtin_extensions: Vec<String>,
}

/// Upstream `PackageFilter`.
#[derive(Debug, Clone, Default)]
pub struct PackageFilter {
    pub autoload: Option<bool>,
    pub extensions: Option<Vec<String>>,
    pub skills: Option<Vec<String>>,
    pub prompts: Option<Vec<String>>,
    pub themes: Option<Vec<String>>,
}

impl PackageFilter {
    fn from_source(pkg: &PackageSource) -> Option<PackageFilter> {
        match pkg {
            PackageSource::Simple(_) => None,
            PackageSource::Filtered {
                autoload,
                extensions,
                skills,
                prompts,
                themes,
                ..
            } => Some(PackageFilter {
                autoload: *autoload,
                extensions: extensions.clone(),
                skills: skills.clone(),
                prompts: prompts.clone(),
                themes: themes.clone(),
            }),
        }
    }

    fn patterns(&self, resource_type: ResourceType) -> Option<&Vec<String>> {
        match resource_type {
            ResourceType::Extensions => self.extensions.as_ref(),
            ResourceType::Skills => self.skills.as_ref(),
            ResourceType::Prompts => self.prompts.as_ref(),
            ResourceType::Themes => self.themes.as_ref(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NpmSource {
    pub spec: String,
    pub name: String,
    pub version: Option<String>,
    pub range: Option<String>,
    pub pinned: bool,
}

#[derive(Debug, Clone)]
pub struct LocalSource {
    pub path: String,
}

/// Upstream `ParsedSource` discriminated union.
#[derive(Debug, Clone)]
pub enum ParsedSource {
    Npm(NpmSource),
    Git(GitSource),
    Local(LocalSource),
}

/// `SourceScope` restricted to installed scopes (upstream
/// `InstalledSourceScope`).
type InstalledSourceScope = SourceScope;

struct ConfiguredUpdateSource {
    source: String,
    scope: InstalledSourceScope,
}

struct NpmUpdateTarget {
    entry: ConfiguredUpdateSource,
    parsed: NpmSource,
}

struct GitUpdateTarget {
    entry: ConfiguredUpdateSource,
    parsed: GitSource,
}

#[derive(Debug, Clone)]
struct ResourceEntry {
    metadata: PathMetadata,
    enabled: bool,
}

#[derive(Debug, Default)]
struct ResourceAccumulator {
    extensions: IndexMap<String, ResourceEntry>,
    skills: IndexMap<String, ResourceEntry>,
    prompts: IndexMap<String, ResourceEntry>,
    themes: IndexMap<String, ResourceEntry>,
}

/// Compute a numeric precedence rank for a resource based on its metadata.
/// Lower rank = higher precedence. Used to sort resolved resources so that
/// name-collision resolution ("first wins") produces the correct outcome.
///
/// Precedence (highest to lowest):
///   0  project + settings entry (source: "local", scope: "project")
///   1  project + auto-discovered (source: "auto", scope: "project")
///   2  user + settings entry (source: "local", scope: "user")
///   3  user + auto-discovered (source: "auto", scope: "user")
///   4  package resource (origin: "package")
fn resource_precedence_rank(m: &PathMetadata) -> i32 {
    // Built-in extensions load after file and package extensions, whichever scope enables them.
    if m.source == "builtin" {
        return 5;
    }
    if m.origin == SourceOrigin::Package {
        return 4;
    }
    let scope_base = if m.scope == SourceScope::Project {
        0
    } else {
        2
    };
    scope_base + if m.source == "local" { 0 } else { 1 }
}

const IGNORE_FILE_NAMES: [&str; 3] = [".gitignore", ".ignore", ".fdignore"];

fn to_posix_path(p: &str) -> String {
    if cfg!(windows) {
        p.replace('\\', "/")
    } else {
        p.to_string()
    }
}

fn get_home_dir() -> String {
    crate::utils::paths::home_dir()
}

/// Upstream `getExtensionTempFolder` — `<agentDir>/tmp/extensions`, created
/// with owner-only permissions.
pub fn get_extension_temp_folder(agent_dir: &str) -> String {
    let temp_folder = Path::new(agent_dir).join("tmp").join("extensions");
    let _ = std::fs::create_dir_all(&temp_folder);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&temp_folder, std::fs::Permissions::from_mode(0o700));
    }
    temp_folder.to_string_lossy().into_owned()
}

fn prefix_ignore_pattern(line: &str, prefix: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('#') && !trimmed.starts_with("\\#") {
        return None;
    }

    let mut pattern = line;
    let mut negated = false;

    if let Some(rest) = pattern.strip_prefix('!') {
        negated = true;
        pattern = rest;
    } else if let Some(rest) = pattern.strip_prefix("\\!") {
        pattern = rest;
    }

    pattern = pattern.strip_prefix('/').unwrap_or(pattern);

    let prefixed = if prefix.is_empty() {
        pattern.to_string()
    } else {
        format!("{prefix}{pattern}")
    };
    Some(if negated {
        format!("!{prefixed}")
    } else {
        prefixed
    })
}

/// Accumulated gitignore rules rooted at a scan root (module docs D55a).
#[derive(Debug, Clone)]
struct IgnoreMatcher {
    root: String,
    lines: Vec<String>,
}

impl IgnoreMatcher {
    fn new(root: &str) -> Self {
        IgnoreMatcher {
            root: root.to_string(),
            lines: Vec::new(),
        }
    }

    fn add_dir_rules(&mut self, dir: &str) {
        let relative_dir = node_relative_path(&self.root, dir);
        let prefix = if relative_dir.is_empty() {
            String::new()
        } else {
            format!("{}/", to_posix_path(&relative_dir))
        };

        for filename in IGNORE_FILE_NAMES {
            let ignore_path = Path::new(dir).join(filename);
            if !ignore_path.exists() {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&ignore_path) else {
                continue;
            };
            for line in content.split(|c| c == '\n' || c == '\r') {
                if let Some(pattern) = prefix_ignore_pattern(line, &prefix) {
                    self.lines.push(pattern);
                }
            }
        }
    }

    fn build(&self) -> Gitignore {
        let mut builder = GitignoreBuilder::new(&self.root);
        for line in &self.lines {
            let _ = builder.add_line(Some(PathBuf::from(&self.root)), line);
        }
        builder.build().unwrap_or_else(|_| Gitignore::empty())
    }

    fn ignores(&self, gitignore: &Gitignore, rel: &str, is_dir: bool) -> bool {
        // The ignore crate strips the builder root from the candidate path;
        // relative inputs don't match when the root is absolute, so always
        // pass a root-joined path — and a trailing `/` makes the match fail,
        // so it is stripped (dir-ness is carried by `is_dir`).
        let rel = rel.trim_end_matches('/');
        matches!(
            gitignore.matched(&Path::new(&self.root).join(rel), is_dir),
            ignore::Match::Ignore(_)
        )
    }
}

/// node `path.relative(from, to)` — lexical, emits `..` segments.
fn node_relative_path(from: &str, to: &str) -> String {
    let split = |p: &str| -> Vec<String> {
        Path::new(p)
            .components()
            .filter_map(|c| match c {
                std::path::Component::RootDir | std::path::Component::Prefix(_) => None,
                std::path::Component::CurDir => None,
                other => Some(other.as_os_str().to_string_lossy().into_owned()),
            })
            .collect()
    };
    let from_parts = split(from);
    let to_parts = split(to);
    let mut common = 0;
    while common < from_parts.len()
        && common < to_parts.len()
        && from_parts[common] == to_parts[common]
    {
        common += 1;
    }
    let mut parts: Vec<String> = from_parts[common..]
        .iter()
        .map(|_| "..".to_string())
        .collect();
    parts.extend_from_slice(&to_parts[common..]);
    if parts.is_empty() {
        return String::new();
    }
    parts.join("/")
}

/// Glob entries discover visible paths; exact entries can target dot paths or
/// symlinked trees (upstream `expandPackageGlob`).
fn expand_package_glob(pattern: &str, root: &str) -> Vec<String> {
    let normalized = pattern.strip_prefix("./").unwrap_or(pattern);
    let full = format!("{}/{}", root.trim_end_matches('/'), normalized);
    let Ok(entries) = glob::glob(&full) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.to_string_lossy().into_owned())
        .filter(|path| {
            node_relative_path(root, path)
                .split('/')
                .all(|segment| segment == ".." || !segment.starts_with('.'))
        })
        .collect();
    out.sort();
    out
}

fn split_patterns(entries: &[String]) -> (Vec<String>, Vec<String>) {
    let mut plain = Vec::new();
    let mut patterns = Vec::new();
    for entry in entries {
        if is_pattern(entry) {
            patterns.push(entry.clone());
        } else {
            plain.push(entry.clone());
        }
    }
    (plain, patterns)
}

/// minimatch approximation (module docs D55a).
fn minimatch(text: &str, pattern: &str) -> bool {
    match glob::Pattern::new(pattern) {
        Ok(compiled) => compiled.matches(text),
        Err(_) => false,
    }
}

fn collect_files(
    dir: &str,
    resource_type: ResourceType,
    skip_node_modules: bool,
    ignore_matcher: Option<&IgnoreMatcher>,
    root_dir: Option<&str>,
) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    if !Path::new(dir).exists() {
        return files;
    }

    let root = root_dir.unwrap_or(dir).to_string();
    let mut fresh_matcher: Option<IgnoreMatcher> = None;
    let ig: &IgnoreMatcher = match ignore_matcher {
        Some(existing) => existing,
        None => {
            let mut matcher = IgnoreMatcher::new(&root);
            matcher.add_dir_rules(dir);
            fresh_matcher.insert(matcher)
        }
    };
    let gitignore = ig.build();

    let Ok(entries) = std::fs::read_dir(dir) else {
        return files;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if skip_node_modules && name == "node_modules" {
            continue;
        }

        let full_path = entry.path().to_string_lossy().into_owned();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };
        let mut is_dir = file_type.is_dir();
        let mut is_file = file_type.is_file();

        if file_type.is_symlink() {
            match std::fs::metadata(&full_path) {
                Ok(stats) => {
                    is_dir = stats.is_dir();
                    is_file = stats.is_file();
                }
                Err(_) => continue,
            }
        }

        let rel_path = to_posix_path(&node_relative_path(&root, &full_path));
        let ignore_path = if is_dir {
            format!("{rel_path}/")
        } else {
            rel_path.clone()
        };
        if ig.ignores(&gitignore, &ignore_path, is_dir) {
            continue;
        }

        if is_dir {
            files.extend(collect_files(
                &full_path,
                resource_type,
                skip_node_modules,
                Some(ig),
                Some(&root),
            ));
        } else if is_file && file_pattern_matches(resource_type, &name) {
            files.push(full_path);
        }
    }

    files
}

/// Skill discovery mode (upstream `SkillDiscoveryMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkillDiscoveryMode {
    Pi,
    Agents,
}

fn collect_skill_entries(
    dir: &str,
    mode: SkillDiscoveryMode,
    ignore_matcher: Option<&IgnoreMatcher>,
    root_dir: Option<&str>,
) -> Vec<String> {
    let mut entries: Vec<String> = Vec::new();
    if !Path::new(dir).exists() {
        return entries;
    }

    let root = root_dir.unwrap_or(dir).to_string();
    let matcher = match ignore_matcher {
        Some(existing) => existing.clone(),
        None => {
            let mut matcher = IgnoreMatcher::new(&root);
            matcher.add_dir_rules(dir);
            matcher
        }
    };
    let gitignore = matcher.build();

    let Ok(dir_entries) = std::fs::read_dir(dir) else {
        return entries;
    };
    let dir_entries: Vec<_> = dir_entries.flatten().collect();

    for entry in &dir_entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name != "SKILL.md" {
            continue;
        }

        let full_path = entry.path().to_string_lossy().into_owned();
        let mut is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
        if entry.file_type().map(|t| t.is_symlink()).unwrap_or(false) {
            match std::fs::metadata(&full_path) {
                Ok(stats) => is_file = stats.is_file(),
                Err(_) => continue,
            }
        }

        let rel_path = to_posix_path(&node_relative_path(&root, &full_path));
        if is_file && !matcher.ignores(&gitignore, &rel_path, false) {
            entries.push(full_path);
            return entries;
        }
    }

    for entry in &dir_entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if name == "node_modules" {
            continue;
        }

        let full_path = entry.path().to_string_lossy().into_owned();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };
        let mut is_dir = file_type.is_dir();
        let mut is_file = file_type.is_file();

        if file_type.is_symlink() {
            match std::fs::metadata(&full_path) {
                Ok(stats) => {
                    is_dir = stats.is_dir();
                    is_file = stats.is_file();
                }
                Err(_) => continue,
            }
        }

        let rel_path = to_posix_path(&node_relative_path(&root, &full_path));
        let should_include_markdown_file = is_file
            && name.ends_with(".md")
            && !matcher.ignores(&gitignore, &rel_path, false)
            && ((mode == SkillDiscoveryMode::Pi && dir == root)
                || (mode == SkillDiscoveryMode::Agents && dir != root));
        if should_include_markdown_file {
            entries.push(full_path);
            continue;
        }

        if !is_dir {
            continue;
        }
        if matcher.ignores(&gitignore, &format!("{rel_path}/"), true) {
            continue;
        }

        entries.extend(collect_skill_entries(
            &full_path,
            mode,
            Some(&matcher),
            Some(&root),
        ));
    }

    entries
}

fn collect_auto_skill_entries(dir: &str, mode: SkillDiscoveryMode) -> Vec<String> {
    collect_skill_entries(dir, mode, None, None)
}

fn find_git_repo_root(start_dir: &str) -> Option<String> {
    let mut dir = resolve_path(
        start_dir,
        &std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy(),
    );
    loop {
        if Path::new(&dir).join(".git").exists() {
            return Some(dir);
        }
        let parent = match Path::new(&dir).parent() {
            Some(parent) => parent.to_string_lossy().into_owned(),
            None => return None,
        };
        if parent == dir {
            return None;
        }
        dir = parent;
    }
}

fn collect_ancestor_agents_skill_dirs(start_dir: &str) -> Vec<String> {
    let mut skill_dirs: Vec<String> = Vec::new();
    let resolved_start_dir = resolve_path(
        start_dir,
        &std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy(),
    );
    let git_repo_root = find_git_repo_root(&resolved_start_dir);

    let mut dir = resolved_start_dir;
    loop {
        skill_dirs.push(
            Path::new(&dir)
                .join(".agents")
                .join("skills")
                .to_string_lossy()
                .into_owned(),
        );
        if let Some(root) = &git_repo_root {
            if &dir == root {
                break;
            }
        }
        let parent = match Path::new(&dir).parent() {
            Some(parent) => parent.to_string_lossy().into_owned(),
            None => break,
        };
        if parent == dir {
            break;
        }
        dir = parent;
    }

    skill_dirs
}

fn collect_simple_dir_entries(dir: &str, extension_check: impl Fn(&str) -> bool) -> Vec<String> {
    let mut entries: Vec<String> = Vec::new();
    if !Path::new(dir).exists() {
        return entries;
    }

    let mut matcher = IgnoreMatcher::new(dir);
    matcher.add_dir_rules(dir);
    let gitignore = matcher.build();

    let Ok(dir_entries) = std::fs::read_dir(dir) else {
        return entries;
    };
    for entry in dir_entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if name == "node_modules" {
            continue;
        }

        let full_path = entry.path().to_string_lossy().into_owned();
        let mut is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
        if entry.file_type().map(|t| t.is_symlink()).unwrap_or(false) {
            match std::fs::metadata(&full_path) {
                Ok(stats) => is_file = stats.is_file(),
                Err(_) => continue,
            }
        }

        let rel_path = to_posix_path(&node_relative_path(dir, &full_path));
        if matcher.ignores(&gitignore, &rel_path, false) {
            continue;
        }

        if is_file && extension_check(&name) {
            entries.push(full_path);
        }
    }

    entries
}

fn collect_auto_prompt_entries(dir: &str) -> Vec<String> {
    collect_simple_dir_entries(dir, |name| name.ends_with(".md"))
}

fn collect_auto_theme_entries(dir: &str) -> Vec<String> {
    collect_simple_dir_entries(dir, |name| name.ends_with(".json"))
}

/// Upstream `resolveExtensionEntries` — explicit entries (package.json
/// `pi.extensions`, then `index.ts`/`index.js`), or `None`.
fn resolve_extension_entries(dir: &str) -> Option<Vec<String>> {
    let package_json_path = Path::new(dir).join("package.json");
    if package_json_path.exists() {
        let manifest = read_pi_manifest(&package_json_path.to_string_lossy());
        if let Some(extensions) = manifest.as_ref().and_then(|m| m.extensions.as_ref()) {
            if !extensions.is_empty() {
                let mut entries: Vec<String> = Vec::new();
                for ext_path in extensions {
                    // Upstream uses plain `path.resolve(dir, extPath)` — no tilde
                    // expansion (package-relative `~` directories are literal).
                    let joined = crate::utils::paths::join(dir, ext_path);
                    let resolved_ext_path = resolve_path(&joined, dir);
                    if Path::new(&resolved_ext_path).exists() {
                        entries.push(resolved_ext_path);
                    }
                }
                if !entries.is_empty() {
                    return Some(entries);
                }
            }
        }
    }

    let index_ts = Path::new(dir).join("index.ts");
    if index_ts.exists() {
        return Some(vec![index_ts.to_string_lossy().into_owned()]);
    }
    let index_js = Path::new(dir).join("index.js");
    if index_js.exists() {
        return Some(vec![index_js.to_string_lossy().into_owned()]);
    }

    None
}

fn collect_auto_extension_entries(dir: &str) -> Vec<String> {
    let mut entries: Vec<String> = Vec::new();
    if !Path::new(dir).exists() {
        return entries;
    }

    // First check if this directory itself has explicit extension entries (package.json or index)
    if let Some(root_entries) = resolve_extension_entries(dir) {
        return root_entries;
    }

    // Otherwise, discover extensions from directory contents
    let mut matcher = IgnoreMatcher::new(dir);
    matcher.add_dir_rules(dir);
    let gitignore = matcher.build();

    let Ok(dir_entries) = std::fs::read_dir(dir) else {
        return entries;
    };
    for entry in dir_entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if name == "node_modules" {
            continue;
        }

        let full_path = entry.path().to_string_lossy().into_owned();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };
        let mut is_dir = file_type.is_dir();
        let mut is_file = file_type.is_file();

        if file_type.is_symlink() {
            match std::fs::metadata(&full_path) {
                Ok(stats) => {
                    is_dir = stats.is_dir();
                    is_file = stats.is_file();
                }
                Err(_) => continue,
            }
        }

        let rel_path = to_posix_path(&node_relative_path(dir, &full_path));
        let ignore_path = if is_dir {
            format!("{rel_path}/")
        } else {
            rel_path.clone()
        };
        if matcher.ignores(&gitignore, &ignore_path, is_dir) {
            continue;
        }

        if is_file && (name.ends_with(".ts") || name.ends_with(".js")) {
            entries.push(full_path);
        } else if is_dir {
            if let Some(resolved_entries) = resolve_extension_entries(&full_path) {
                entries.extend(resolved_entries);
            }
        }
    }

    entries
}

/// Collect resource files from a directory based on resource type.
/// Extensions use smart discovery (index.ts in subdirs), others use recursive collection.
fn collect_resource_files(dir: &str, resource_type: ResourceType) -> Vec<String> {
    match resource_type {
        ResourceType::Skills => collect_skill_entries(dir, SkillDiscoveryMode::Pi, None, None),
        ResourceType::Extensions => collect_auto_extension_entries(dir),
        ResourceType::Prompts | ResourceType::Themes => {
            collect_files(dir, resource_type, true, None, None)
        }
    }
}

fn matches_any_pattern(file_path: &str, patterns: &[String], base_dir: &str) -> bool {
    let rel = to_posix_path(&node_relative_path(base_dir, file_path));
    let name = Path::new(file_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let file_path_posix = to_posix_path(file_path);
    let is_skill_file = name == "SKILL.md";
    let parent_dir = if is_skill_file {
        Path::new(file_path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
    } else {
        None
    };
    let parent_rel = parent_dir
        .as_ref()
        .map(|dir| to_posix_path(&node_relative_path(base_dir, dir)));
    let parent_name = parent_dir.as_ref().and_then(|dir| {
        Path::new(dir)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
    });
    let parent_dir_posix = parent_dir.as_ref().map(|dir| to_posix_path(dir));

    patterns.iter().any(|pattern| {
        let normalized_pattern = to_posix_path(pattern);
        if minimatch(&rel, &normalized_pattern)
            || minimatch(&name, &normalized_pattern)
            || minimatch(&file_path_posix, &normalized_pattern)
        {
            return true;
        }
        if !is_skill_file {
            return false;
        }
        minimatch(parent_rel.as_deref().unwrap_or(""), &normalized_pattern)
            || minimatch(parent_name.as_deref().unwrap_or(""), &normalized_pattern)
            || minimatch(
                parent_dir_posix.as_deref().unwrap_or(""),
                &normalized_pattern,
            )
    })
}

fn normalize_exact_pattern(pattern: &str) -> String {
    let normalized = if pattern.starts_with("./") || pattern.starts_with(".\\") {
        &pattern[2..]
    } else {
        pattern
    };
    to_posix_path(normalized)
}

fn matches_any_exact_pattern(file_path: &str, patterns: &[String], base_dir: &str) -> bool {
    if patterns.is_empty() {
        return false;
    }
    let rel = to_posix_path(&node_relative_path(base_dir, file_path));
    let name = Path::new(file_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let file_path_posix = to_posix_path(file_path);
    let is_skill_file = name == "SKILL.md";
    let parent_dir = if is_skill_file {
        Path::new(file_path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
    } else {
        None
    };
    let parent_rel = parent_dir
        .as_ref()
        .map(|dir| to_posix_path(&node_relative_path(base_dir, dir)));
    let parent_dir_posix = parent_dir.as_ref().map(|dir| to_posix_path(dir));

    patterns.iter().any(|pattern| {
        let normalized = normalize_exact_pattern(pattern);
        if normalized == rel || normalized == file_path_posix {
            return true;
        }
        if !is_skill_file {
            return false;
        }
        Some(&normalized) == parent_rel.as_ref() || Some(&normalized) == parent_dir_posix.as_ref()
    })
}

fn get_override_patterns(entries: &[String]) -> Vec<String> {
    entries
        .iter()
        .filter(|pattern| is_override_pattern(pattern))
        .cloned()
        .collect()
}

fn is_enabled_by_overrides(file_path: &str, patterns: &[String], base_dir: &str) -> bool {
    let overrides = get_override_patterns(patterns);
    let excludes: Vec<String> = overrides
        .iter()
        .filter(|p| p.starts_with('!'))
        .map(|p| p[1..].to_string())
        .collect();
    let force_includes: Vec<String> = overrides
        .iter()
        .filter(|p| p.starts_with('+'))
        .map(|p| p[1..].to_string())
        .collect();
    let force_excludes: Vec<String> = overrides
        .iter()
        .filter(|p| p.starts_with('-'))
        .map(|p| p[1..].to_string())
        .collect();

    let mut enabled = true;
    if !excludes.is_empty() && matches_any_pattern(file_path, &excludes, base_dir) {
        enabled = false;
    }
    if !force_includes.is_empty() && matches_any_exact_pattern(file_path, &force_includes, base_dir)
    {
        enabled = true;
    }
    if !force_excludes.is_empty() && matches_any_exact_pattern(file_path, &force_excludes, base_dir)
    {
        enabled = false;
    }
    enabled
}

/// Apply patterns to paths and return the set of enabled paths.
/// Pattern types:
/// - Plain patterns: include matching paths
/// - `!pattern`: exclude matching paths
/// - `+path`: force-include exact path (overrides exclusions)
/// - `-path`: force-exclude exact path (overrides force-includes)
fn apply_patterns(all_paths: &[String], patterns: &[String], base_dir: &str) -> HashSet<String> {
    let mut includes: Vec<String> = Vec::new();
    let mut excludes: Vec<String> = Vec::new();
    let mut force_includes: Vec<String> = Vec::new();
    let mut force_excludes: Vec<String> = Vec::new();

    for p in patterns {
        if let Some(rest) = p.strip_prefix('+') {
            force_includes.push(rest.to_string());
        } else if let Some(rest) = p.strip_prefix('-') {
            force_excludes.push(rest.to_string());
        } else if let Some(rest) = p.strip_prefix('!') {
            excludes.push(rest.to_string());
        } else {
            includes.push(p.clone());
        }
    }

    // Step 1: Apply includes (or all if no includes)
    let mut result: Vec<String> = if includes.is_empty() {
        all_paths.to_vec()
    } else {
        all_paths
            .iter()
            .filter(|file_path| matches_any_pattern(file_path, &includes, base_dir))
            .cloned()
            .collect()
    };

    // Step 2: Apply excludes
    if !excludes.is_empty() {
        result.retain(|file_path| !matches_any_pattern(file_path, &excludes, base_dir));
    }

    // Step 3: Force-include (add back from allPaths, overriding exclusions)
    if !force_includes.is_empty() {
        for file_path in all_paths {
            if !result.contains(file_path)
                && matches_any_exact_pattern(file_path, &force_includes, base_dir)
            {
                result.push(file_path.clone());
            }
        }
    }

    // Step 4: Force-exclude (remove even if included or force-included)
    if !force_excludes.is_empty() {
        result.retain(|file_path| !matches_any_exact_pattern(file_path, &force_excludes, base_dir));
    }

    result.into_iter().collect()
}

fn apply_autoload_disabled_patterns(
    all_paths: &[String],
    patterns: &[String],
    base_dir: &str,
) -> IndexMap<String, bool> {
    let mut result = IndexMap::new();
    for pattern in patterns {
        let target = pattern
            .strip_prefix('+')
            .or_else(|| pattern.strip_prefix('-'))
            .or_else(|| pattern.strip_prefix('!'))
            .unwrap_or(pattern);
        let enabled = !pattern.starts_with('-') && !pattern.starts_with('!');
        let exact = pattern.starts_with('+') || pattern.starts_with('-');
        for file_path in all_paths {
            let matched = if exact {
                matches_any_exact_pattern(file_path, &[target.to_string()], base_dir)
            } else {
                matches_any_pattern(file_path, &[target.to_string()], base_dir)
            };
            if matched {
                result.insert(file_path.clone(), enabled);
            }
        }
    }
    result
}

// ============================================================================
// npm semver helpers (upstream npm `semver` usage)
// ============================================================================

fn is_exact_npm_version(version: Option<&str>) -> bool {
    match version {
        Some(version) => semver::Version::parse(version).is_ok(),
        None => false,
    }
}

fn get_npm_version_range(version: Option<&str>) -> Option<String> {
    version.and_then(|version| {
        semver::VersionReq::parse(version)
            .ok()
            .map(|_| version.to_string())
    })
}

fn semver_satisfies(version: &str, range: &str) -> bool {
    match (
        semver::Version::parse(version),
        semver::VersionReq::parse(range),
    ) {
        (Ok(version), Ok(range)) => range.matches(&version),
        _ => false,
    }
}

fn semver_gt(a: &str, b: &str) -> bool {
    match (semver::Version::parse(a), semver::Version::parse(b)) {
        (Ok(a), Ok(b)) => a > b,
        _ => false,
    }
}

fn semver_max_satisfying(versions: &[String], range: &str) -> Option<String> {
    let req = semver::VersionReq::parse(range).ok()?;
    versions
        .iter()
        .filter_map(|version| {
            semver::Version::parse(version)
                .ok()
                .map(|parsed| (version, parsed))
        })
        .filter(|(_, parsed)| req.matches(parsed))
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(version, _)| version.clone())
}

fn semver_latest(versions: &[String]) -> Option<String> {
    let mut parsed: Vec<(String, semver::Version)> = versions
        .iter()
        .filter_map(|version| {
            semver::Version::parse(version)
                .ok()
                .map(|parsed| (version.clone(), parsed))
        })
        .collect();
    parsed.sort_by(|a, b| b.1.cmp(&a.1));
    parsed.into_iter().next().map(|(version, _)| version)
}

fn is_offline_mode_enabled() -> bool {
    match std::env::var("PI_OFFLINE") {
        Ok(value) if !value.is_empty() => {
            value == "1" || value.to_lowercase() == "true" || value.to_lowercase() == "yes"
        }
        _ => false,
    }
}

// ============================================================================
// Command runner seam (module docs D55)
// ============================================================================

/// One process invocation (upstream `runCommand`/`runCommandCapture`/
/// `runCommandSync` arguments).
#[derive(Debug, Clone)]
pub struct CommandCall {
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub env: Vec<(String, String)>,
}

impl CommandCall {
    fn new(command: &str, args: &[String]) -> Self {
        CommandCall {
            command: command.to_string(),
            args: args.to_vec(),
            cwd: None,
            env: Vec::new(),
        }
    }

    fn with_cwd(mut self, cwd: Option<&str>) -> Self {
        self.cwd = cwd.map(str::to_string);
        self
    }

    fn display(&self) -> String {
        format!("{} {}", self.command, self.args.join(" "))
    }
}

/// Seam replacing upstream's prototype-spyable private spawn methods (module
/// docs D55).
pub trait CommandRunner: Send + Sync {
    /// Upstream `runCommand` — inherit stdio, wait for exit.
    fn run(&self, call: &CommandCall) -> Result<(), String>;
    /// Upstream `runCommandCapture` — piped output, trimmed stdout, optional
    /// timeout.
    fn run_capture(&self, call: &CommandCall, timeout_ms: Option<u64>) -> Result<String, String>;
    /// Upstream `runCommandSync` — synchronous piped execution.
    fn run_sync(&self, call: &CommandCall) -> Result<String, String>;
}

/// Real process runner.
pub struct ProcessCommandRunner;

impl ProcessCommandRunner {
    fn env_vars(call: &CommandCall) -> Vec<(String, String)> {
        // Upstream `getEnv` falls back to `/proc/self/environ` on Linux when
        // the process env is empty; Rust commands inherit the parent env by
        // default, so only the explicit overrides are layered on top.
        call.env.clone()
    }
}

impl CommandRunner for ProcessCommandRunner {
    fn run(&self, call: &CommandCall) -> Result<(), String> {
        let mut command = std::process::Command::new(&call.command);
        command.args(&call.args);
        if crate::core::output_guard::is_stdout_taken_over() {
            // Upstream `stdio: ["ignore", 2, 2]` in `spawnCommand`: raw
            // protocol output owns stdout, so the child's stdin/stdout are
            // dropped and only its stderr passes through.
            command.stdin(std::process::Stdio::null());
            command.stdout(std::process::Stdio::null());
        }
        if let Some(cwd) = &call.cwd {
            command.current_dir(cwd);
        }
        for (key, value) in Self::env_vars(call) {
            command.env(key, value);
        }
        let status = command.status().map_err(|error| error.to_string())?;
        if status.success() {
            return Ok(());
        }
        Err(format!(
            "{} failed with code {}",
            call.display(),
            exit_code_of(&status)
        ))
    }

    fn run_capture(&self, call: &CommandCall, timeout_ms: Option<u64>) -> Result<String, String> {
        let mut command = std::process::Command::new(&call.command);
        command.args(&call.args);
        command.stdin(std::process::Stdio::null());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        if let Some(cwd) = &call.cwd {
            command.current_dir(cwd);
        }
        for (key, value) in Self::env_vars(call) {
            command.env(key, value);
        }
        let mut child = command.spawn().map_err(|error| error.to_string())?;

        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();
        let read_out = std::thread::spawn(move || {
            let mut buf = String::new();
            if let Some(pipe) = stdout_pipe.as_mut() {
                let _ = pipe.read_to_string(&mut buf);
            }
            buf
        });
        let read_err = std::thread::spawn(move || {
            let mut buf = String::new();
            if let Some(pipe) = stderr_pipe.as_mut() {
                let _ = pipe.read_to_string(&mut buf);
            }
            buf
        });

        let status = match timeout_ms {
            Some(ms) => {
                let deadline = std::time::Instant::now() + Duration::from_millis(ms);
                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => break Some(status),
                        Ok(None) => {
                            if std::time::Instant::now() >= deadline {
                                let _ = child.kill();
                                let _ = child.wait();
                                return Err(format!("{} timed out after {}ms", call.display(), ms));
                            }
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => return Err(error.to_string()),
                    }
                }
            }
            None => child.wait().map(Some).map_err(|error| error.to_string())?,
        };

        let stdout = read_out.join().unwrap_or_default();
        let stderr = read_err.join().unwrap_or_default();
        match status.and_then(|status| status.code()) {
            Some(0) => Ok(stdout.trim().to_string()),
            Some(code) => {
                let detail = if stderr.is_empty() { &stdout } else { &stderr };
                Err(format!(
                    "{} failed with code {}: {}",
                    call.display(),
                    code,
                    detail
                ))
            }
            None => {
                let detail = if stderr.is_empty() { &stdout } else { &stderr };
                Err(format!(
                    "{} failed with signal unknown: {}",
                    call.display(),
                    detail
                ))
            }
        }
    }

    fn run_sync(&self, call: &CommandCall) -> Result<String, String> {
        let mut command = std::process::Command::new(&call.command);
        command.args(&call.args);
        command.stdin(std::process::Stdio::null());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        for (key, value) in Self::env_vars(call) {
            command.env(key, value);
        }
        let output = command.output().map_err(|error| error.to_string())?;
        let result = SyncOutput {
            status: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        };
        if result.status != Some(0) {
            let detail = if result.stderr.is_empty() {
                &result.stdout
            } else {
                &result.stderr
            };
            return Err(format!("Failed to run {}: {}", call.display(), detail));
        }
        Ok(if result.stdout.is_empty() {
            result.stderr.trim().to_string()
        } else {
            result.stdout.trim().to_string()
        })
    }
}

fn exit_code_of(status: &std::process::ExitStatus) -> i32 {
    status.code().unwrap_or(-1)
}

// ============================================================================
// DefaultPackageManager
// ============================================================================

/// Upstream `DefaultPackageManager`.
pub struct DefaultPackageManager {
    cwd: String,
    agent_dir: String,
    settings_manager: Arc<Mutex<SettingsManager>>,
    builtin_extensions: Vec<String>,
    global_npm_root: Mutex<Option<(String, String)>>,
    progress_callback: Mutex<Option<ProgressCallback>>,
    runner: Arc<dyn CommandRunner>,
}

impl DefaultPackageManager {
    pub fn new(options: PackageManagerOptions) -> Self {
        Self::with_runner(options, Arc::new(ProcessCommandRunner))
    }

    /// Inject a custom [`CommandRunner`] (test seam, module docs D55).
    pub fn with_runner(options: PackageManagerOptions, runner: Arc<dyn CommandRunner>) -> Self {
        DefaultPackageManager {
            cwd: resolve_path(
                &options.cwd,
                &std::env::current_dir()
                    .unwrap_or_default()
                    .to_string_lossy(),
            ),
            agent_dir: resolve_path(
                &options.agent_dir,
                &std::env::current_dir()
                    .unwrap_or_default()
                    .to_string_lossy(),
            ),
            settings_manager: options.settings_manager,
            builtin_extensions: options.builtin_extensions,
            global_npm_root: Mutex::new(None),
            progress_callback: Mutex::new(None),
            runner,
        }
    }

    fn read_settings(&self) -> (Settings, Settings) {
        let manager = self.settings_manager.lock();
        (
            manager.get_global_settings(),
            manager.get_project_settings(),
        )
    }

    fn packages_of(settings: &Settings) -> Vec<PackageSource> {
        settings
            .get("packages")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default()
    }

    fn list_of(settings: &Settings, key: &str) -> Vec<String> {
        settings
            .get(key)
            .and_then(|value| value.as_array())
            .map(|array| {
                array
                    .iter()
                    .filter_map(|entry| entry.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn set_progress_callback(&self, callback: Option<ProgressCallback>) {
        *self.progress_callback.lock() = callback;
    }

    fn emit_progress(&self, event: ProgressEvent) {
        if let Some(callback) = self.progress_callback.lock().as_ref() {
            callback(&event);
        }
    }

    pub fn add_source_to_settings(&self, source: &str, local: bool) -> bool {
        let scope = if local {
            SourceScope::Project
        } else {
            SourceScope::User
        };
        let (current_packages, source_match) = {
            let manager = self.settings_manager.lock();
            let current_settings = if scope == SourceScope::Project {
                manager.get_project_settings()
            } else {
                manager.get_global_settings()
            };
            let current_packages = Self::packages_of(&current_settings);
            let matched = current_packages
                .iter()
                .position(|existing| self.package_sources_match(existing, source, scope));
            (current_packages, matched)
        };
        let normalized_source = self.normalize_package_source_for_settings(source, scope);
        if let Some(match_index) = source_match {
            let existing = &current_packages[match_index];
            if Self::package_source_string(existing) == normalized_source {
                return false;
            }
            let mut next_packages = current_packages.clone();
            next_packages[match_index] = match existing {
                PackageSource::Simple(_) => PackageSource::Simple(normalized_source),
                PackageSource::Filtered { .. } => {
                    let mut updated = existing.clone();
                    if let PackageSource::Filtered {
                        source: replaced, ..
                    } = &mut updated
                    {
                        *replaced = normalized_source;
                    }
                    updated
                }
            };
            self.store_packages(next_packages, scope);
            return true;
        }
        let mut next_packages = current_packages;
        next_packages.push(PackageSource::Simple(normalized_source));
        self.store_packages(next_packages, scope);
        true
    }

    pub fn remove_source_from_settings(&self, source: &str, local: bool) -> bool {
        let scope = if local {
            SourceScope::Project
        } else {
            SourceScope::User
        };
        let current_packages = {
            let manager = self.settings_manager.lock();
            let current_settings = if scope == SourceScope::Project {
                manager.get_project_settings()
            } else {
                manager.get_global_settings()
            };
            Self::packages_of(&current_settings)
        };
        let next_packages: Vec<PackageSource> = current_packages
            .iter()
            .filter(|existing| !self.package_sources_match(existing, source, scope))
            .cloned()
            .collect();
        let changed = next_packages.len() != current_packages.len();
        if !changed {
            return false;
        }
        self.store_packages(next_packages, scope);
        true
    }

    fn store_packages(&self, packages: Vec<PackageSource>, scope: SourceScope) {
        let mut manager = self.settings_manager.lock();
        if scope == SourceScope::Project {
            manager.set_project_packages(&packages);
        } else {
            manager.set_packages(&packages);
        }
    }

    pub fn get_installed_path(&self, source: &str, scope: SourceScope) -> Option<String> {
        match self.parse_source(source) {
            ParsedSource::Npm(parsed) => {
                let path = self.get_npm_install_path(&parsed, scope);
                Path::new(&path).exists().then_some(path)
            }
            ParsedSource::Git(parsed) => {
                let path = self.get_git_install_path(&parsed, scope).ok()?;
                Path::new(&path).exists().then_some(path)
            }
            ParsedSource::Local(parsed) => {
                let base_dir = self.get_base_dir_for_scope(scope);
                let path = self.resolve_path_from_base(&parsed.path, &base_dir);
                Path::new(&path).exists().then_some(path)
            }
        }
    }

    async fn with_progress(
        &self,
        action: ProgressAction,
        source: &str,
        message: &str,
        operation: impl Future<Output = Result<(), String>>,
    ) -> Result<(), String> {
        self.emit_progress(ProgressEvent {
            event_type: ProgressEventType::Start,
            action,
            source: source.to_string(),
            message: Some(message.to_string()),
        });
        match operation.await {
            Ok(()) => {
                self.emit_progress(ProgressEvent {
                    event_type: ProgressEventType::Complete,
                    action,
                    source: source.to_string(),
                    message: None,
                });
                Ok(())
            }
            Err(error) => {
                self.emit_progress(ProgressEvent {
                    event_type: ProgressEventType::Error,
                    action,
                    source: source.to_string(),
                    message: Some(error.clone()),
                });
                Err(error)
            }
        }
    }

    /// Upstream `resolve`.
    pub async fn resolve(
        &self,
        on_missing: Option<&dyn Fn(&str) -> MissingSourceAction>,
    ) -> Result<ResolvedPaths, String> {
        let mut accumulator = ResourceAccumulator::default();
        let (global_settings, project_settings) = self.read_settings();

        // Collect all packages with scope (project first so cwd resources win collisions)
        let mut all_packages: Vec<(PackageSource, SourceScope)> = Vec::new();
        for pkg in Self::packages_of(&project_settings) {
            all_packages.push((pkg, SourceScope::Project));
        }
        for pkg in Self::packages_of(&global_settings) {
            all_packages.push((pkg, SourceScope::User));
        }

        // Dedupe: project scope wins over global for same package identity
        let package_sources = self.dedupe_packages(all_packages);
        self.resolve_package_sources(&package_sources, &mut accumulator, on_missing)
            .await?;

        let global_base_dir = self.agent_dir.clone();
        let project_base_dir = Path::new(&self.cwd)
            .join(CONFIG_DIR_NAME)
            .to_string_lossy()
            .into_owned();

        for resource_type in RESOURCE_TYPES {
            let target = accumulator_target(&mut accumulator, resource_type);
            let global_entries = Self::list_of(&global_settings, resource_type.key());
            let project_entries = Self::list_of(&project_settings, resource_type.key());
            let project_metadata = PathMetadata {
                source: "local".to_string(),
                scope: SourceScope::Project,
                origin: SourceOrigin::TopLevel,
                base_dir: None,
                package_root: None,
            };
            let user_metadata = PathMetadata {
                source: "local".to_string(),
                scope: SourceScope::User,
                origin: SourceOrigin::TopLevel,
                base_dir: None,
                package_root: None,
            };
            let project_target_snapshot = project_entries.clone();
            self.resolve_local_entries(
                &project_target_snapshot,
                resource_type,
                target,
                project_metadata,
                &project_base_dir,
            );
            let global_target_snapshot = global_entries.clone();
            let target = accumulator_target(&mut accumulator, resource_type);
            self.resolve_local_entries(
                &global_target_snapshot,
                resource_type,
                target,
                user_metadata,
                &global_base_dir,
            );
        }

        self.add_auto_discovered_resources(&mut accumulator, &global_settings, &project_settings);

        // Built-in extensions are enabled unless the user `extensions` setting excludes them, for example
        // with `-builtin:mcp`. A matching `+`, `-`, or `!` entry in the project setting overrides that.
        for name in &self.builtin_extensions {
            let path = format!("{BUILTIN_PATH_PREFIX}{name}");
            let project_patterns =
                get_override_patterns(&Self::list_of(&project_settings, "extensions"));
            let project_enabled = apply_autoload_disabled_patterns(
                &[path.clone()],
                &project_patterns,
                &project_base_dir,
            )
            .get(&path)
            .copied();
            let metadata = PathMetadata {
                source: "builtin".to_string(),
                scope: if project_enabled.is_some() {
                    SourceScope::Project
                } else {
                    SourceScope::User
                },
                origin: SourceOrigin::TopLevel,
                base_dir: None,
                package_root: None,
            };
            let enabled = project_enabled.unwrap_or_else(|| {
                is_enabled_by_overrides(
                    &path,
                    &Self::list_of(&global_settings, "extensions"),
                    &global_base_dir,
                )
            });
            add_resource(&mut accumulator.extensions, &path, metadata, enabled);
        }

        Ok(to_resolved_paths(accumulator))
    }

    /// Upstream `resolveExtensionSources`.
    pub async fn resolve_extension_sources(
        &self,
        sources: &[String],
        local: bool,
        temporary: bool,
    ) -> Result<ResolvedPaths, String> {
        let mut accumulator = ResourceAccumulator::default();
        let scope = if temporary {
            SourceScope::Temporary
        } else if local {
            SourceScope::Project
        } else {
            SourceScope::User
        };
        // `-e builtin:<name>` loads a built-in extension; the resource loader reports unknown names.
        for source in sources
            .iter()
            .filter(|source| source.starts_with(BUILTIN_PATH_PREFIX))
        {
            add_resource(
                &mut accumulator.extensions,
                source,
                PathMetadata {
                    source: "builtin".to_string(),
                    scope,
                    origin: SourceOrigin::TopLevel,
                    base_dir: None,
                    package_root: None,
                },
                true,
            );
        }
        let package_sources: Vec<(PackageSource, SourceScope)> = sources
            .iter()
            .filter(|source| !source.starts_with(BUILTIN_PATH_PREFIX))
            .map(|source| (PackageSource::Simple(source.clone()), scope))
            .collect();
        self.resolve_package_sources(&package_sources, &mut accumulator, None)
            .await?;
        Ok(to_resolved_paths(accumulator))
    }

    /// Upstream `listConfiguredPackages`.
    pub fn list_configured_packages(&self) -> Vec<ConfiguredPackage> {
        let (global_settings, project_settings) = self.read_settings();
        let mut configured_packages: Vec<ConfiguredPackage> = Vec::new();

        for pkg in Self::packages_of(&global_settings) {
            let source = Self::package_source_string(&pkg);
            configured_packages.push(ConfiguredPackage {
                installed_path: self.get_installed_path(&source, SourceScope::User),
                source,
                scope: SourceScope::User,
                filtered: matches!(pkg, PackageSource::Filtered { .. }),
            });
        }

        for pkg in Self::packages_of(&project_settings) {
            let source = Self::package_source_string(&pkg);
            configured_packages.push(ConfiguredPackage {
                installed_path: self.get_installed_path(&source, SourceScope::Project),
                source,
                scope: SourceScope::Project,
                filtered: matches!(pkg, PackageSource::Filtered { .. }),
            });
        }

        configured_packages
    }

    /// Upstream `install`.
    pub async fn install(&self, source: &str, local: bool) -> Result<(), String> {
        let parsed = self.parse_source(source);
        let scope = if local {
            SourceScope::Project
        } else {
            SourceScope::User
        };
        self.assert_project_trusted_for_scope(scope)?;
        self.with_progress(
            ProgressAction::Install,
            source,
            &format!("Installing {source}..."),
            async {
                match parsed {
                    ParsedSource::Npm(parsed) => self.install_npm(&parsed, scope, false).await,
                    ParsedSource::Git(parsed) => self.install_git(&parsed, scope).await,
                    ParsedSource::Local(parsed) => {
                        let resolved = self.resolve_path(&parsed.path);
                        if !Path::new(&resolved).exists() {
                            return Err(format!("Path does not exist: {resolved}"));
                        }
                        Ok(())
                    }
                }
            },
        )
        .await
    }

    /// Upstream `installAndPersist`.
    pub async fn install_and_persist(&self, source: &str, local: bool) -> Result<(), String> {
        self.install(source, local).await?;
        self.add_source_to_settings(source, local);
        Ok(())
    }

    /// Upstream `remove`.
    pub async fn remove(&self, source: &str, local: bool) -> Result<(), String> {
        let parsed = self.parse_source(source);
        let scope = if local {
            SourceScope::Project
        } else {
            SourceScope::User
        };
        self.assert_project_trusted_for_scope(scope)?;
        self.with_progress(
            ProgressAction::Remove,
            source,
            &format!("Removing {source}..."),
            async {
                match parsed {
                    ParsedSource::Npm(parsed) => self.uninstall_npm(&parsed, scope).await,
                    ParsedSource::Git(parsed) => self.remove_git(&parsed, scope),
                    ParsedSource::Local(_) => Ok(()),
                }
            },
        )
        .await
    }

    /// Upstream `removeAndPersist`.
    pub async fn remove_and_persist(&self, source: &str, local: bool) -> Result<bool, String> {
        self.remove(source, local).await?;
        Ok(self.remove_source_from_settings(source, local))
    }

    /// Upstream `update`.
    pub async fn update(&self, source: Option<&str>) -> Result<(), String> {
        let (global_settings, project_settings) = self.read_settings();
        let identity = source.map(|source| self.get_package_identity(source, None));
        let mut matched = false;
        let mut update_sources: Vec<ConfiguredUpdateSource> = Vec::new();

        for pkg in Self::packages_of(&global_settings) {
            let source_str = Self::package_source_string(&pkg);
            if let Some(identity) = &identity {
                if self.get_package_identity(&source_str, Some(SourceScope::User)) != *identity {
                    continue;
                }
            }
            matched = true;
            update_sources.push(ConfiguredUpdateSource {
                source: source_str,
                scope: SourceScope::User,
            });
        }
        for pkg in Self::packages_of(&project_settings) {
            let source_str = Self::package_source_string(&pkg);
            if let Some(identity) = &identity {
                if self.get_package_identity(&source_str, Some(SourceScope::Project)) != *identity {
                    continue;
                }
            }
            matched = true;
            update_sources.push(ConfiguredUpdateSource {
                source: source_str,
                scope: SourceScope::Project,
            });
        }

        if source.is_some() && !matched {
            let mut configured = Self::packages_of(&global_settings);
            configured.extend(Self::packages_of(&project_settings));
            return Err(self.build_no_matching_package_message(source.unwrap(), &configured));
        }

        self.update_configured_sources(update_sources).await
    }

    async fn update_configured_sources(
        &self,
        sources: Vec<ConfiguredUpdateSource>,
    ) -> Result<(), String> {
        if is_offline_mode_enabled() || sources.is_empty() {
            return Ok(());
        }

        let mut npm_candidates: Vec<NpmUpdateTarget> = Vec::new();
        let mut git_candidates: Vec<GitUpdateTarget> = Vec::new();

        for entry in sources {
            match self.parse_source(&entry.source) {
                ParsedSource::Npm(parsed) => {
                    // Pinned npm versions are fixed. Pinned git refs are configured checkout targets,
                    // so include them to reconcile an existing clone when the configured ref changes.
                    if !parsed.pinned {
                        npm_candidates.push(NpmUpdateTarget { entry, parsed });
                    }
                }
                ParsedSource::Git(parsed) => git_candidates.push(GitUpdateTarget { entry, parsed }),
                ParsedSource::Local(_) => {}
            }
        }

        let npm_check_tasks: Vec<_> = npm_candidates
            .into_iter()
            .map(|entry| {
                let this = &self;
                move || async move {
                    let should_update = this
                        .should_update_npm_source(&entry.parsed, entry.entry.scope)
                        .await;
                    (entry, should_update)
                }
            })
            .collect();
        let npm_check_results =
            run_with_concurrency(npm_check_tasks, UPDATE_CHECK_CONCURRENCY).await;
        let mut user_npm_updates: Vec<NpmUpdateTarget> = Vec::new();
        let mut project_npm_updates: Vec<NpmUpdateTarget> = Vec::new();
        for (entry, should_update) in npm_check_results {
            if !should_update {
                continue;
            }
            if entry.entry.scope == SourceScope::User {
                user_npm_updates.push(entry);
            } else {
                project_npm_updates.push(entry);
            }
        }

        let mut tasks: Vec<std::pin::Pin<Box<dyn Future<Output = Result<(), String>> + '_>>> =
            Vec::new();
        if !user_npm_updates.is_empty() {
            tasks.push(Box::pin(async {
                self.update_npm_batch(user_npm_updates, SourceScope::User)
                    .await
            }));
        }
        if !project_npm_updates.is_empty() {
            tasks.push(Box::pin(async {
                self.update_npm_batch(project_npm_updates, SourceScope::Project)
                    .await
            }));
        }
        if !git_candidates.is_empty() {
            let git_tasks: Vec<_> = git_candidates
                .into_iter()
                .map(|entry| {
                    let this = &self;
                    move || async move {
                        this.with_progress(
                            ProgressAction::Update,
                            &entry.entry.source,
                            &format!("Updating {}...", entry.entry.source),
                            this.update_git(&entry.parsed, entry.entry.scope),
                        )
                        .await
                    }
                })
                .collect();
            tasks.push(Box::pin(async move {
                let results = run_with_concurrency(git_tasks, GIT_UPDATE_CONCURRENCY).await;
                results
                    .into_iter()
                    .find(|result| result.is_err())
                    .unwrap_or(Ok(()))
            }));
        }

        let results = join_all(tasks).await;
        results
            .into_iter()
            .find(|result| result.is_err())
            .unwrap_or(Ok(()))
    }

    async fn should_update_npm_source(&self, source: &NpmSource, scope: SourceScope) -> bool {
        let Ok(installed_path) = self.get_managed_npm_install_path(source, scope) else {
            return true;
        };
        let installed_version = if Path::new(&installed_path).exists() {
            self.get_installed_npm_version(&installed_path)
        } else {
            None
        };
        let Some(installed_version) = installed_version else {
            return true;
        };

        match self
            .get_latest_npm_version(
                source
                    .version
                    .as_ref()
                    .map_or(&source.name, |_| &source.spec),
                source.range.as_deref(),
            )
            .await
        {
            Ok(target_version) => semver_gt(&target_version, &installed_version),
            // Preserve existing update behavior when version lookup fails.
            Err(_) => true,
        }
    }

    async fn update_npm_batch(
        &self,
        sources: Vec<NpmUpdateTarget>,
        scope: SourceScope,
    ) -> Result<(), String> {
        if sources.is_empty() {
            return Ok(());
        }

        let source_label = if sources.len() == 1 {
            sources[0].entry.source.clone()
        } else {
            format!("{} npm packages", scope_label(scope))
        };
        let message = if sources.len() == 1 {
            format!("Updating {}...", sources[0].entry.source)
        } else {
            format!("Updating {} npm packages...", scope_label(scope))
        };
        let specs: Vec<String> = sources
            .iter()
            .map(|entry| match &entry.parsed.version {
                Some(_) => entry.parsed.spec.clone(),
                None => format!("{}@latest", entry.parsed.name),
            })
            .collect();

        self.with_progress(ProgressAction::Update, &source_label, &message, async {
            self.install_npm_batch(&specs, scope).await
        })
        .await
    }

    async fn install_npm_batch(&self, specs: &[String], scope: SourceScope) -> Result<(), String> {
        let install_root = self.get_npm_install_root(scope, false)?;
        self.ensure_npm_project(&install_root);
        let npm_command = self.get_npm_command();
        let mut args = npm_command.1.clone();
        args.extend(self.get_npm_install_args(specs, &install_root));
        self.runner.run(&CommandCall::new(&npm_command.0, &args))?;
        Ok(())
    }

    /// Upstream `checkForAvailableUpdates`.
    pub async fn check_for_available_updates(&self) -> Result<Vec<PackageUpdate>, String> {
        if is_offline_mode_enabled() {
            return Ok(Vec::new());
        }

        let (global_settings, project_settings) = self.read_settings();
        let mut all_packages: Vec<(PackageSource, SourceScope)> = Vec::new();
        for pkg in Self::packages_of(&project_settings) {
            all_packages.push((pkg, SourceScope::Project));
        }
        for pkg in Self::packages_of(&global_settings) {
            all_packages.push((pkg, SourceScope::User));
        }

        let package_sources = self.dedupe_packages(all_packages);
        self.check_updates_inner(&package_sources).await
    }

    async fn check_updates_inner(
        &self,
        package_sources: &[(PackageSource, SourceScope)],
    ) -> Result<Vec<PackageUpdate>, String> {
        let checks: Vec<_> = package_sources
            .iter()
            .map(|(pkg, scope)| {
                let this = &self;
                let pkg = pkg.clone();
                let scope = *scope;
                move || async move {
                    let source = Self::package_source_string(&pkg);
                    let parsed = this.parse_source(&source);
                    match parsed {
                        ParsedSource::Local(_) => return None,
                        ParsedSource::Npm(parsed) => {
                            if parsed.pinned {
                                return None;
                            }
                            let installed_path = this.get_npm_install_path(&parsed, scope);
                            if !Path::new(&installed_path).exists() {
                                return None;
                            }
                            let has_update = this
                                .npm_has_available_update(&parsed, &installed_path)
                                .await;
                            if !has_update {
                                return None;
                            }
                            Some(PackageUpdate {
                                display_name: parsed.name,
                                source,
                                update_type: PackageUpdateType::Npm,
                                scope,
                            })
                        }
                        ParsedSource::Git(parsed) => {
                            if parsed.pinned {
                                return None;
                            }
                            let installed_path = this.get_git_install_path(&parsed, scope).ok()?;
                            if !Path::new(&installed_path).exists() {
                                return None;
                            }
                            let has_update = this.git_has_available_update(&installed_path).await;
                            if !has_update {
                                return None;
                            }
                            Some(PackageUpdate {
                                display_name: format!("{}/{}", parsed.host, parsed.path),
                                source,
                                update_type: PackageUpdateType::Git,
                                scope,
                            })
                        }
                    }
                }
            })
            .collect();
        let results = run_with_concurrency(checks, UPDATE_CHECK_CONCURRENCY).await;
        Ok(results.into_iter().flatten().collect())
    }

    async fn resolve_package_sources(
        &self,
        sources: &[(PackageSource, SourceScope)],
        accumulator: &mut ResourceAccumulator,
        on_missing: Option<&dyn Fn(&str) -> MissingSourceAction>,
    ) -> Result<(), String> {
        for (pkg, scope) in sources {
            let source_str = Self::package_source_string(pkg);
            let filter = PackageFilter::from_source(pkg);
            let delta_base = self.find_autoload_delta_base(pkg, *scope, sources);
            let resolved_source = delta_base
                .as_ref()
                .map_or(&source_str, |base| &base.0)
                .clone();
            let resolved_scope = delta_base.as_ref().map_or(*scope, |base| base.1);
            let parsed = self.parse_source(&resolved_source);
            let parsed_for_install = parsed.clone();
            let mut metadata = PathMetadata {
                source: source_str.clone(),
                scope: *scope,
                origin: SourceOrigin::Package,
                base_dir: None,
                package_root: None,
            };

            match parsed {
                ParsedSource::Local(local) => {
                    let base_dir = self.get_base_dir_for_scope(resolved_scope);
                    self.resolve_local_extension_source(
                        &local,
                        accumulator,
                        filter.as_ref(),
                        &mut metadata,
                        &base_dir,
                    );
                }
                ParsedSource::Npm(npm) => {
                    let mut installed_path = self.get_npm_install_path(&npm, resolved_scope);
                    let needs_install = !Path::new(&installed_path).exists()
                        || !self.installed_npm_matches_configured_version(&npm, &installed_path);
                    if needs_install {
                        if !self
                            .install_missing(
                                &parsed_for_install,
                                resolved_scope,
                                &resolved_source,
                                on_missing,
                            )
                            .await?
                        {
                            continue;
                        }
                        installed_path = self.get_npm_install_path(&npm, resolved_scope);
                    }
                    metadata.base_dir = Some(installed_path.clone());
                    metadata.package_root = Some(installed_path.clone());
                    self.collect_package_resources(
                        &installed_path,
                        accumulator,
                        filter.as_ref(),
                        &mut metadata,
                    );
                }
                ParsedSource::Git(git) => {
                    let installed_path = self.get_git_install_path(&git, resolved_scope)?;
                    if !Path::new(&installed_path).exists() {
                        if !self
                            .install_missing(
                                &parsed_for_install,
                                resolved_scope,
                                &resolved_source,
                                on_missing,
                            )
                            .await?
                        {
                            continue;
                        }
                    } else if resolved_scope == SourceScope::Temporary
                        && !git.pinned
                        && !is_offline_mode_enabled()
                    {
                        self.refresh_temporary_git_source(&git, &resolved_source)
                            .await;
                    }
                    metadata.base_dir = Some(installed_path.clone());
                    metadata.package_root = Some(installed_path.clone());
                    self.collect_package_resources(
                        &installed_path,
                        accumulator,
                        filter.as_ref(),
                        &mut metadata,
                    );
                }
            }
        }
        Ok(())
    }

    async fn install_missing(
        &self,
        parsed: &ParsedSource,
        scope: SourceScope,
        resolved_source: &str,
        on_missing: Option<&dyn Fn(&str) -> MissingSourceAction>,
    ) -> Result<bool, String> {
        if is_offline_mode_enabled() {
            return Ok(false);
        }
        let Some(on_missing) = on_missing else {
            self.install_parsed_source(parsed, scope).await?;
            return Ok(true);
        };
        match on_missing(resolved_source) {
            MissingSourceAction::Skip => Ok(false),
            MissingSourceAction::Error => Err(format!("Missing source: {resolved_source}")),
            MissingSourceAction::Install => {
                self.install_parsed_source(parsed, scope).await?;
                Ok(true)
            }
        }
    }

    fn find_autoload_delta_base(
        &self,
        pkg: &PackageSource,
        scope: SourceScope,
        sources: &[(PackageSource, SourceScope)],
    ) -> Option<(String, SourceScope)> {
        if scope != SourceScope::Project {
            return None;
        }
        let PackageSource::Filtered {
            source,
            autoload: Some(false),
            ..
        } = pkg
        else {
            return None;
        };
        let identity = self.get_package_identity(source, Some(scope));
        sources.iter().find_map(|(entry_pkg, entry_scope)| {
            if *entry_scope != SourceScope::User {
                return None;
            }
            if self.get_package_identity(
                &Self::package_source_string(entry_pkg),
                Some(SourceScope::User),
            ) == identity
            {
                Some((Self::package_source_string(entry_pkg), SourceScope::User))
            } else {
                None
            }
        })
    }

    fn resolve_local_extension_source(
        &self,
        source: &LocalSource,
        accumulator: &mut ResourceAccumulator,
        filter: Option<&PackageFilter>,
        metadata: &mut PathMetadata,
        base_dir: &str,
    ) {
        let resolved = self.resolve_path_from_base(&source.path, base_dir);
        if !Path::new(&resolved).exists() {
            return;
        }

        match std::fs::metadata(&resolved) {
            Ok(stats) if stats.is_file() => {
                metadata.base_dir = Path::new(&resolved)
                    .parent()
                    .map(|p| p.to_string_lossy().into_owned());
                add_resource(
                    &mut accumulator.extensions,
                    &resolved,
                    metadata.clone(),
                    true,
                );
            }
            Ok(stats) if stats.is_dir() => {
                metadata.base_dir = Some(resolved.clone());
                metadata.package_root = Some(resolved.clone());
                let resources =
                    self.collect_package_resources(&resolved, accumulator, filter, metadata);
                if !resources {
                    add_resource(
                        &mut accumulator.extensions,
                        &resolved,
                        metadata.clone(),
                        true,
                    );
                }
            }
            _ => {}
        }
    }

    async fn install_parsed_source(
        &self,
        parsed: &ParsedSource,
        scope: SourceScope,
    ) -> Result<(), String> {
        match parsed {
            ParsedSource::Npm(npm) => {
                self.install_npm(npm, scope, scope == SourceScope::Temporary)
                    .await
            }
            ParsedSource::Git(git) => self.install_git(git, scope).await,
            ParsedSource::Local(_) => Ok(()),
        }
    }

    fn package_source_string(pkg: &PackageSource) -> String {
        match pkg {
            PackageSource::Simple(source) => source.clone(),
            PackageSource::Filtered { source, .. } => source.clone(),
        }
    }

    fn get_source_match_key_for_input(&self, source: &str) -> String {
        match self.parse_source(source) {
            ParsedSource::Npm(parsed) => format!("npm:{}", parsed.name),
            ParsedSource::Git(parsed) => format!("git:{}/{}", parsed.host, parsed.path),
            ParsedSource::Local(parsed) => format!("local:{}", self.resolve_path(&parsed.path)),
        }
    }

    fn get_source_match_key_for_settings(&self, source: &str, scope: SourceScope) -> String {
        match self.parse_source(source) {
            ParsedSource::Npm(parsed) => format!("npm:{}", parsed.name),
            ParsedSource::Git(parsed) => format!("git:{}/{}", parsed.host, parsed.path),
            ParsedSource::Local(parsed) => {
                let base_dir = self.get_base_dir_for_scope(scope);
                format!(
                    "local:{}",
                    self.resolve_path_from_base(&parsed.path, &base_dir)
                )
            }
        }
    }

    fn build_no_matching_package_message(
        &self,
        source: &str,
        configured_packages: &[PackageSource],
    ) -> String {
        match self.find_suggested_configured_source(source, configured_packages) {
            None => format!("No matching package found for {source}"),
            Some(suggestion) => {
                format!("No matching package found for {source}. Did you mean {suggestion}?")
            }
        }
    }

    fn find_suggested_configured_source(
        &self,
        source: &str,
        configured_packages: &[PackageSource],
    ) -> Option<String> {
        let trimmed_source = source.trim();
        let mut suggestions: Vec<String> = Vec::new();

        for pkg in configured_packages {
            let source_str = Self::package_source_string(pkg);
            match self.parse_source(&source_str) {
                ParsedSource::Npm(parsed) => {
                    if trimmed_source == parsed.name || trimmed_source == parsed.spec {
                        push_unique(&mut suggestions, source_str);
                    }
                }
                ParsedSource::Git(parsed) => {
                    let shorthand = format!("{}/{}", parsed.host, parsed.path);
                    let shorthand_with_ref = parsed
                        .ref_
                        .as_ref()
                        .map(|ref_| format!("{shorthand}@{ref_}"));
                    if trimmed_source == shorthand
                        || shorthand_with_ref.as_deref() == Some(trimmed_source)
                    {
                        push_unique(&mut suggestions, source_str);
                    }
                }
                ParsedSource::Local(_) => {}
            }
        }

        suggestions.into_iter().next()
    }

    fn package_sources_match(
        &self,
        existing: &PackageSource,
        input_source: &str,
        scope: SourceScope,
    ) -> bool {
        let left =
            self.get_source_match_key_for_settings(&Self::package_source_string(existing), scope);
        let right = self.get_source_match_key_for_input(input_source);
        left == right
    }

    fn normalize_package_source_for_settings(&self, source: &str, scope: SourceScope) -> String {
        let parsed = self.parse_source(source);
        let ParsedSource::Local(parsed) = parsed else {
            return source.to_string();
        };
        let base_dir = self.get_base_dir_for_scope(scope);
        let resolved = self.resolve_path(&parsed.path);
        let rel = node_relative_path(&base_dir, &resolved);
        if rel.is_empty() {
            ".".to_string()
        } else {
            rel
        }
    }

    pub fn parse_source(&self, source: &str) -> ParsedSource {
        if let Some(spec) = source.strip_prefix("npm:") {
            let spec = spec.trim();
            let (name, version) = parse_npm_spec(spec);
            return ParsedSource::Npm(NpmSource {
                spec: spec.to_string(),
                name,
                version: version.clone(),
                range: get_npm_version_range(version.as_deref()),
                pinned: is_exact_npm_version(version.as_deref()),
            });
        }

        if is_local_path(source) {
            return ParsedSource::Local(LocalSource {
                path: source.to_string(),
            });
        }

        // Try parsing as git URL
        if let Some(git_parsed) = parse_git_url(source) {
            return ParsedSource::Git(git_parsed);
        }

        ParsedSource::Local(LocalSource {
            path: source.to_string(),
        })
    }

    fn installed_npm_matches_configured_version(
        &self,
        source: &NpmSource,
        installed_path: &str,
    ) -> bool {
        let Some(installed_version) = self.get_installed_npm_version(installed_path) else {
            return false;
        };
        match &source.range {
            Some(range) => semver_satisfies(&installed_version, range),
            None => true,
        }
    }

    async fn npm_has_available_update(&self, source: &NpmSource, installed_path: &str) -> bool {
        if is_offline_mode_enabled() {
            return false;
        }

        let Some(installed_version) = self.get_installed_npm_version(installed_path) else {
            return false;
        };

        match self
            .get_latest_npm_version(
                source
                    .version
                    .as_ref()
                    .map_or(&source.name, |_| &source.spec),
                source.range.as_deref(),
            )
            .await
        {
            Ok(target_version) => semver_gt(&target_version, &installed_version),
            Err(_) => false,
        }
    }

    fn get_installed_npm_version(&self, installed_path: &str) -> Option<String> {
        let package_json_path = Path::new(installed_path).join("package.json");
        if !package_json_path.exists() {
            return None;
        }
        let content = std::fs::read_to_string(&package_json_path).ok()?;
        let pkg: serde_json::Value = serde_json::from_str(strip_bom(&content)).ok()?;
        pkg.get("version")
            .and_then(|value| value.as_str())
            .map(str::to_string)
    }

    async fn get_latest_npm_version(
        &self,
        package_spec: &str,
        range: Option<&str>,
    ) -> Result<String, String> {
        let npm_command = self.get_npm_command();
        let mut args = npm_command.1.clone();
        args.extend([
            "view".to_string(),
            package_spec.to_string(),
            "version".to_string(),
            "--json".to_string(),
        ]);
        let mut call = CommandCall::new(&npm_command.0, &args);
        call.cwd = Some(self.cwd.clone());
        let stdout = self.runner.run_capture(&call, Some(NETWORK_TIMEOUT_MS))?;
        let raw = stdout.trim();
        if raw.is_empty() {
            return Err("Empty response from npm view".to_string());
        }
        let parsed: serde_json::Value =
            serde_json::from_str(raw).map_err(|error| error.to_string())?;
        if let Some(version) = parsed.as_str() {
            return Ok(version.to_string());
        }
        if let Some(array) = parsed.as_array() {
            let versions: Vec<String> = array
                .iter()
                .filter_map(|value| value.as_str())
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect();
            let latest = match range {
                Some(range) => semver_max_satisfying(&versions, range),
                None => semver_latest(&versions),
            };
            if let Some(latest) = latest {
                return Ok(latest);
            }
        }
        Err("Unexpected response from npm view".to_string())
    }

    async fn git_has_available_update(&self, installed_path: &str) -> bool {
        if is_offline_mode_enabled() {
            return false;
        }

        let mut head_call = CommandCall::new("git", &["rev-parse".to_string(), "HEAD".to_string()]);
        head_call.cwd = Some(installed_path.to_string());
        let Ok(local_head) = self
            .runner
            .run_capture(&head_call, Some(NETWORK_TIMEOUT_MS))
        else {
            return false;
        };
        let Ok(remote_head) = self.get_remote_git_head(installed_path).await else {
            return false;
        };
        local_head.trim() != remote_head.trim()
    }

    async fn get_remote_git_head(&self, installed_path: &str) -> Result<String, String> {
        if let Some(upstream_ref) = self.get_git_upstream_ref(installed_path) {
            let remote_head = self
                .run_git_remote_command(
                    installed_path,
                    &[
                        "ls-remote".to_string(),
                        "origin".to_string(),
                        upstream_ref.clone(),
                    ],
                )
                .await?;
            if let Some(hash) = find_head_hash(&remote_head, None) {
                return Ok(hash);
            }
        }

        let remote_head = self
            .run_git_remote_command(
                installed_path,
                &[
                    "ls-remote".to_string(),
                    "origin".to_string(),
                    "HEAD".to_string(),
                ],
            )
            .await?;
        match find_head_hash(&remote_head, Some("HEAD")) {
            Some(hash) => Ok(hash),
            None => Err("Failed to determine remote HEAD".to_string()),
        }
    }

    async fn get_local_git_update_target(
        &self,
        installed_path: &str,
    ) -> Result<(String, String, Vec<String>), String> {
        let mut abbrev_call = CommandCall::new(
            "git",
            &[
                "rev-parse".to_string(),
                "--abbrev-ref".to_string(),
                "@{upstream}".to_string(),
            ],
        );
        abbrev_call.cwd = Some(installed_path.to_string());
        match self
            .runner
            .run_capture(&abbrev_call, Some(NETWORK_TIMEOUT_MS))
        {
            Ok(upstream) if upstream.trim().starts_with("origin/") => {
                let branch = upstream.trim()["origin/".len()..].to_string();
                if branch.is_empty() {
                    return Err("Missing upstream branch name".to_string());
                }
                let mut head_call =
                    CommandCall::new("git", &["rev-parse".to_string(), "@{upstream}".to_string()]);
                head_call.cwd = Some(installed_path.to_string());
                let head = self
                    .runner
                    .run_capture(&head_call, Some(NETWORK_TIMEOUT_MS))?;
                Ok((
                    "@{upstream}".to_string(),
                    head,
                    vec![
                        "fetch".to_string(),
                        "--prune".to_string(),
                        "--no-tags".to_string(),
                        "origin".to_string(),
                        format!("+refs/heads/{branch}:refs/remotes/origin/{branch}"),
                    ],
                ))
            }
            // Upstream throws inside its try and falls through to the
            // origin/HEAD reconciliation for missing or non-origin upstreams.
            _ => {
                self.get_local_git_update_target_fallback(installed_path)
                    .await
            }
        }
    }

    async fn get_local_git_update_target_fallback(
        &self,
        installed_path: &str,
    ) -> Result<(String, String, Vec<String>), String> {
        let mut set_head_call = CommandCall::new(
            "git",
            &[
                "remote".to_string(),
                "set-head".to_string(),
                "origin".to_string(),
                "-a".to_string(),
            ],
        );
        set_head_call.cwd = Some(installed_path.to_string());
        let _ = self.runner.run(&set_head_call);
        let mut head_call =
            CommandCall::new("git", &["rev-parse".to_string(), "origin/HEAD".to_string()]);
        head_call.cwd = Some(installed_path.to_string());
        let head = self
            .runner
            .run_capture(&head_call, Some(NETWORK_TIMEOUT_MS))?;
        let mut sym_call = CommandCall::new(
            "git",
            &[
                "symbolic-ref".to_string(),
                "refs/remotes/origin/HEAD".to_string(),
            ],
        );
        sym_call.cwd = Some(installed_path.to_string());
        let origin_head_ref = self
            .runner
            .run_capture(&sym_call, Some(NETWORK_TIMEOUT_MS))
            .unwrap_or_default();
        let branch = origin_head_ref
            .trim()
            .strip_prefix("refs/remotes/origin/")
            .unwrap_or("");
        if !branch.is_empty() {
            Ok((
                "origin/HEAD".to_string(),
                head,
                vec![
                    "fetch".to_string(),
                    "--prune".to_string(),
                    "--no-tags".to_string(),
                    "origin".to_string(),
                    format!("+refs/heads/{branch}:refs/remotes/origin/{branch}"),
                ],
            ))
        } else {
            Ok((
                "origin/HEAD".to_string(),
                head,
                vec![
                    "fetch".to_string(),
                    "--prune".to_string(),
                    "--no-tags".to_string(),
                    "origin".to_string(),
                    "+HEAD:refs/remotes/origin/HEAD".to_string(),
                ],
            ))
        }
    }

    fn get_git_upstream_ref(&self, installed_path: &str) -> Option<String> {
        let mut abbrev_call = CommandCall::new(
            "git",
            &[
                "rev-parse".to_string(),
                "--abbrev-ref".to_string(),
                "@{upstream}".to_string(),
            ],
        );
        abbrev_call.cwd = Some(installed_path.to_string());
        let upstream = self
            .runner
            .run_capture(&abbrev_call, Some(NETWORK_TIMEOUT_MS))
            .ok()?;
        let trimmed = upstream.trim();
        let branch = trimmed.strip_prefix("origin/")?;
        if branch.is_empty() {
            return None;
        }
        Some(format!("refs/heads/{branch}"))
    }

    async fn run_git_remote_command(
        &self,
        installed_path: &str,
        args: &[String],
    ) -> Result<String, String> {
        let mut call = CommandCall::new("git", args);
        call.cwd = Some(installed_path.to_string());
        call.env
            .push(("GIT_TERMINAL_PROMPT".to_string(), "0".to_string()));
        self.runner.run_capture(&call, Some(NETWORK_TIMEOUT_MS))
    }

    /// Get a unique identity for a package, ignoring version/ref.
    /// Used to detect when the same package is in both global and project settings.
    /// For git packages, uses normalized host/path to ensure SSH and HTTPS URLs
    /// for the same repository are treated as identical.
    pub fn get_package_identity(&self, source: &str, scope: Option<SourceScope>) -> String {
        match self.parse_source(source) {
            ParsedSource::Npm(parsed) => format!("npm:{}", parsed.name),
            ParsedSource::Git(parsed) => format!("git:{}/{}", parsed.host, parsed.path),
            ParsedSource::Local(parsed) => match scope {
                Some(scope) => {
                    let base_dir = self.get_base_dir_for_scope(scope);
                    format!(
                        "local:{}",
                        self.resolve_path_from_base(&parsed.path, &base_dir)
                    )
                }
                None => format!("local:{}", self.resolve_path(&parsed.path)),
            },
        }
    }

    /// Dedupe packages: if same package identity appears in both global and project,
    /// keep only the project one (project wins). A project entry with autoload=false
    /// is a delta over the global entry, so both are kept (delta first).
    fn dedupe_packages(
        &self,
        packages: Vec<(PackageSource, SourceScope)>,
    ) -> Vec<(PackageSource, SourceScope)> {
        let mut result: Vec<(PackageSource, SourceScope)> = Vec::new();
        let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for entry in packages {
            let identity =
                self.get_package_identity(&Self::package_source_string(&entry.0), Some(entry.1));
            match seen.get(&identity).copied() {
                None => {
                    seen.insert(identity, result.len());
                    result.push(entry);
                }
                Some(index) => {
                    let (existing_pkg, existing_scope) = &result[index];
                    if *existing_scope == SourceScope::Project && entry.1 == SourceScope::User {
                        if matches!(
                            existing_pkg,
                            PackageSource::Filtered {
                                autoload: Some(false),
                                ..
                            }
                        ) {
                            result.push(entry);
                        }
                    } else if entry.1 == SourceScope::Project {
                        result[index] = entry;
                    }
                }
            }
        }
        result
    }

    fn assert_project_trusted_for_scope(&self, scope: SourceScope) -> Result<(), String> {
        if scope == SourceScope::Project && !self.settings_manager.lock().is_project_trusted() {
            return Err(
                "Project is not trusted; refusing to access project package storage".to_string(),
            );
        }
        Ok(())
    }

    fn get_npm_command(&self) -> (String, Vec<String>) {
        let configured_command = self.settings_manager.lock().get_npm_command();
        let Some(configured_command) = configured_command else {
            return ("npm".to_string(), Vec::new());
        };
        if configured_command.is_empty() {
            return ("npm".to_string(), Vec::new());
        }
        let (command, args) = configured_command.split_first().unwrap();
        if command.is_empty() {
            panic!("Invalid npmCommand: first array entry must be a non-empty command");
        }
        (command.clone(), args.to_vec())
    }

    pub fn get_package_manager_name(&self) -> String {
        let npm_command = self.get_npm_command();
        let normalize_command_name = |command: &str| -> String {
            Path::new(command)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| command.to_string())
                .replace(".cmd", "")
                .replace(".exe", "")
        };
        let supported_package_managers = ["npm", "pnpm", "bun"];
        let direct_command = normalize_command_name(&npm_command.0);
        if let Some(separator_index) = npm_command.1.iter().rposition(|arg| arg == "--") {
            return match npm_command.1.get(separator_index + 1) {
                Some(wrapped_command) => normalize_command_name(wrapped_command),
                None => direct_command,
            };
        }
        if supported_package_managers.contains(&direct_command.as_str()) {
            return direct_command;
        }

        let mut wrapped_package_managers: Vec<String> = Vec::new();
        for arg in &npm_command.1 {
            let normalized = normalize_command_name(arg);
            if supported_package_managers.contains(&normalized.as_str())
                && !wrapped_package_managers.contains(&normalized)
            {
                wrapped_package_managers.push(normalized);
            }
        }
        if wrapped_package_managers.len() > 1 {
            panic!(
                "Ambiguous npmCommand package managers: {}",
                wrapped_package_managers.join(", ")
            );
        }
        wrapped_package_managers
            .into_iter()
            .next()
            .unwrap_or(direct_command)
    }

    async fn run_npm_command(&self, args: &[String], cwd: Option<&str>) -> Result<(), String> {
        let npm_command = self.get_npm_command();
        let mut all_args = npm_command.1.clone();
        all_args.extend_from_slice(args);
        self.runner
            .run(&CommandCall::new(&npm_command.0, &all_args).with_cwd(cwd))
    }

    fn get_git_dependency_install_args(&self) -> Vec<String> {
        match self.get_package_manager_name().as_str() {
            "bun" => vec![
                "install".to_string(),
                "--omit=dev".to_string(),
                "--omit=peer".to_string(),
            ],
            "pnpm" => vec![
                "install".to_string(),
                "--prod".to_string(),
                "--config.auto-install-peers=false".to_string(),
                "--config.strict-peer-dependencies=false".to_string(),
                "--config.strict-dep-builds=false".to_string(),
            ],
            "npm" => vec![
                "install".to_string(),
                "--omit=dev".to_string(),
                "--legacy-peer-deps".to_string(),
            ],
            _ => vec!["install".to_string()],
        }
    }

    fn run_npm_command_sync(&self, args: &[String]) -> Result<String, String> {
        let npm_command = self.get_npm_command();
        let mut all_args = npm_command.1.clone();
        all_args.extend_from_slice(args);
        self.runner
            .run_sync(&CommandCall::new(&npm_command.0, &all_args))
    }

    fn get_npm_install_args(&self, specs: &[String], install_root: &str) -> Vec<String> {
        let package_manager_name = self.get_package_manager_name();
        // Extension packages run inside pi and resolve pi APIs through loader aliases/virtual modules.
        // Disable peer dependency resolution for managed installs (npm's --legacy-peer-deps, and
        // equivalent bun/pnpm settings) so package managers do not install or solve host-provided
        // @earendil-works/pi-* peers. Stale auto-installed pi peers can otherwise block updates.
        if package_manager_name == "bun" {
            let mut args = vec!["install".to_string()];
            args.extend_from_slice(specs);
            args.extend([
                "--cwd".to_string(),
                install_root.to_string(),
                "--omit=peer".to_string(),
            ]);
            return args;
        }
        if package_manager_name == "pnpm" {
            let mut args = vec!["install".to_string()];
            args.extend_from_slice(specs);
            args.extend([
                "--prefix".to_string(),
                install_root.to_string(),
                "--config.auto-install-peers=false".to_string(),
                "--config.strict-peer-dependencies=false".to_string(),
                "--config.strict-dep-builds=false".to_string(),
            ]);
            return args;
        }
        let mut args = vec!["install".to_string()];
        args.extend_from_slice(specs);
        args.extend([
            "--prefix".to_string(),
            install_root.to_string(),
            "--legacy-peer-deps".to_string(),
        ]);
        args
    }

    async fn install_npm(
        &self,
        source: &NpmSource,
        scope: SourceScope,
        temporary: bool,
    ) -> Result<(), String> {
        let install_root = self.get_npm_install_root(scope, temporary)?;
        self.ensure_npm_project(&install_root);
        let args = self.get_npm_install_args(&[source.spec.clone()], &install_root);
        self.run_npm_command(&args, None).await
    }

    async fn uninstall_npm(&self, source: &NpmSource, scope: SourceScope) -> Result<(), String> {
        let install_root = self.get_npm_install_root(scope, false)?;
        if !Path::new(&install_root).exists() {
            return Ok(());
        }
        let package_manager_name = self.get_package_manager_name();
        if package_manager_name == "bun" {
            return self
                .run_npm_command(
                    &[
                        "uninstall".to_string(),
                        source.name.clone(),
                        "--cwd".to_string(),
                        install_root,
                    ],
                    None,
                )
                .await;
        }
        let mut args = vec![
            "uninstall".to_string(),
            source.name.clone(),
            "--prefix".to_string(),
            install_root,
        ];
        if package_manager_name != "pnpm" {
            args.push("--legacy-peer-deps".to_string());
        }
        self.run_npm_command(&args, None).await
    }

    async fn install_git(&self, source: &GitSource, scope: SourceScope) -> Result<(), String> {
        let target_dir = self.get_git_install_path(source, scope)?;
        if Path::new(&target_dir).exists() {
            if let Some(ref_) = &source.ref_ {
                return self
                    .ensure_git_ref(
                        &target_dir,
                        &["fetch".to_string(), "origin".to_string(), ref_.clone()],
                        "FETCH_HEAD",
                    )
                    .await;
            }
            let (target_ref, _head, fetch_args) =
                self.get_local_git_update_target(&target_dir).await?;
            return self
                .ensure_git_ref(&target_dir, &fetch_args, &target_ref)
                .await;
        }
        let git_root = self.get_git_install_root(scope);
        if let Some(git_root) = &git_root {
            self.ensure_git_ignore(git_root);
        }
        if let Some(parent) = Path::new(&target_dir).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::remove_file(self.get_git_update_marker_path(&target_dir));

        let result: Result<(), String> = async {
            self.runner.run(&CommandCall::new(
                "git",
                &["clone".to_string(), source.repo.clone(), target_dir.clone()],
            ))?;
            if let Some(ref_) = &source.ref_ {
                let checkout_call =
                    CommandCall::new("git", &["checkout".to_string(), ref_.clone()])
                        .with_cwd(Some(&target_dir));
                self.runner.run(&checkout_call)?;
            }
            let package_json_path = Path::new(&target_dir).join("package.json");
            if package_json_path.exists() {
                self.run_npm_command(&self.get_git_dependency_install_args(), Some(&target_dir))
                    .await?;
            }
            Ok(())
        }
        .await;

        if let Err(error) = result {
            let _ = std::fs::remove_dir_all(&target_dir);
            self.prune_empty_git_parents(&target_dir, git_root.as_deref());
            return Err(error);
        }
        Ok(())
    }

    async fn update_git(&self, source: &GitSource, scope: SourceScope) -> Result<(), String> {
        let target_dir = self.get_git_install_path(source, scope)?;
        if !Path::new(&target_dir).exists() {
            return self.install_git(source, scope).await;
        }

        if let Some(ref_) = &source.ref_ {
            return self
                .ensure_git_ref(
                    &target_dir,
                    &["fetch".to_string(), "origin".to_string(), ref_.clone()],
                    "FETCH_HEAD",
                )
                .await;
        }

        let (target_ref, _head, fetch_args) = self.get_local_git_update_target(&target_dir).await?;
        self.ensure_git_ref(&target_dir, &fetch_args, &target_ref)
            .await
    }

    fn has_missing_git_dependencies(&self, target_dir: &str) -> bool {
        let package_json_path = Path::new(target_dir).join("package.json");
        if !package_json_path.exists() {
            return false;
        }

        let Ok(content) = std::fs::read_to_string(&package_json_path) else {
            return false;
        };
        let Ok(manifest) = serde_json::from_str::<serde_json::Value>(strip_bom(&content)) else {
            return false;
        };
        let Some(dependencies) = manifest
            .get("dependencies")
            .and_then(|value| value.as_object())
        else {
            return false;
        };

        let node_modules_dir = resolve_path("node_modules", target_dir);
        dependencies.keys().any(|name| {
            let dependency_path = resolve_path(name, &node_modules_dir);
            if !dependency_path.starts_with(&format!("{node_modules_dir}/")) {
                return false;
            }
            !Path::new(&dependency_path).exists()
        })
    }

    async fn repair_missing_git_dependencies(&self, target_dir: &str) -> Result<(), String> {
        if !self.has_missing_git_dependencies(target_dir) {
            return Ok(());
        }
        self.run_npm_command(&self.get_git_dependency_install_args(), Some(target_dir))
            .await
    }

    fn get_git_update_marker_path(&self, target_dir: &str) -> String {
        let parent = Path::new(target_dir).parent().unwrap_or(Path::new(""));
        let name = Path::new(target_dir)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        parent
            .join(format!(".{name}.pi-update-incomplete"))
            .to_string_lossy()
            .into_owned()
    }

    async fn clean_and_install_git_dependencies(
        &self,
        target_dir: &str,
        marker_path: &str,
    ) -> Result<(), String> {
        // Clean untracked files (extensions should be pristine). If this fails after
        // deleting dependencies, repair them so the existing extension still loads.
        if let Err(error) = self.runner.run(
            &CommandCall::new("git", &["clean".to_string(), "-fdx".to_string()])
                .with_cwd(Some(target_dir)),
        ) {
            let _ = self.repair_missing_git_dependencies(target_dir).await;
            return Err(error);
        }

        let package_json_path = Path::new(target_dir).join("package.json");
        if package_json_path.exists() {
            self.run_npm_command(&self.get_git_dependency_install_args(), Some(target_dir))
                .await?;
        }
        let _ = std::fs::remove_file(marker_path);
        Ok(())
    }

    async fn ensure_git_ref(
        &self,
        target_dir: &str,
        fetch_args: &[String],
        ref_: &str,
    ) -> Result<(), String> {
        // Fetch only the ref we will reset to, avoiding unrelated branch/tag noise.
        self.runner
            .run(&CommandCall::new("git", fetch_args).with_cwd(Some(target_dir)))?;

        let mut head_call = CommandCall::new("git", &["rev-parse".to_string(), "HEAD".to_string()]);
        head_call.cwd = Some(target_dir.to_string());
        let local_head = self
            .runner
            .run_capture(&head_call, Some(NETWORK_TIMEOUT_MS))?;
        let commit_ref = format!("{ref_}^{{commit}}");
        let mut target_call =
            CommandCall::new("git", &["rev-parse".to_string(), commit_ref.clone()]);
        target_call.cwd = Some(target_dir.to_string());
        let target_head = self
            .runner
            .run_capture(&target_call, Some(NETWORK_TIMEOUT_MS))?;
        let marker_path = self.get_git_update_marker_path(target_dir);
        if local_head.trim() == target_head.trim() {
            if Path::new(&marker_path).exists() {
                self.clean_and_install_git_dependencies(target_dir, &marker_path)
                    .await?;
            } else {
                self.repair_missing_git_dependencies(target_dir).await?;
            }
            return Ok(());
        }

        let _ = std::fs::write(&marker_path, "");
        self.runner.run(
            &CommandCall::new(
                "git",
                &["reset".to_string(), "--hard".to_string(), commit_ref],
            )
            .with_cwd(Some(target_dir)),
        )?;
        self.clean_and_install_git_dependencies(target_dir, &marker_path)
            .await
    }

    async fn refresh_temporary_git_source(&self, source: &GitSource, source_str: &str) {
        if is_offline_mode_enabled() {
            return;
        }
        let result = self
            .with_progress(
                ProgressAction::Pull,
                source_str,
                &format!("Refreshing {source_str}..."),
                async { self.update_git(source, SourceScope::Temporary).await },
            )
            .await;
        // Keep cached temporary checkout if refresh fails.
        let _ = result;
    }

    fn remove_git(&self, source: &GitSource, scope: SourceScope) -> Result<(), String> {
        let target_dir = self.get_git_install_path(source, scope)?;
        let _ = std::fs::remove_dir_all(&target_dir);
        let _ = std::fs::remove_file(self.get_git_update_marker_path(&target_dir));
        self.prune_empty_git_parents(&target_dir, self.get_git_install_root(scope).as_deref());
        Ok(())
    }

    fn prune_empty_git_parents(&self, target_dir: &str, install_root: Option<&str>) {
        let Some(install_root) = install_root else {
            return;
        };
        let resolved_root = resolve_path(
            install_root,
            &std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy(),
        );
        let mut current = Path::new(target_dir)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        while current.starts_with(&resolved_root) && current != resolved_root {
            if !Path::new(&current).exists() {
                current = Path::new(&current)
                    .parent()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default();
                continue;
            }
            let entries = match std::fs::read_dir(&current) {
                Ok(entries) => entries.count(),
                Err(_) => break,
            };
            if entries > 0 {
                break;
            }
            if std::fs::remove_dir_all(&current).is_err() {
                break;
            }
            current = Path::new(&current)
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
        }
    }

    fn ensure_npm_project(&self, install_root: &str) {
        if !Path::new(install_root).exists() {
            let _ = std::fs::create_dir_all(install_root);
        }
        mark_path_ignored_by_cloud_sync(install_root);
        self.ensure_git_ignore(install_root);
        let package_json_path = Path::new(install_root).join("package.json");
        if !package_json_path.exists() {
            let pkg_json = serde_json::json!({ "name": "pi-extensions", "private": true });
            let _ = std::fs::write(
                &package_json_path,
                serde_json::to_string_pretty(&pkg_json).unwrap(),
            );
        }
    }

    fn ensure_git_ignore(&self, dir: &str) {
        if !Path::new(dir).exists() {
            let _ = std::fs::create_dir_all(dir);
        }
        let ignore_path = Path::new(dir).join(".gitignore");
        if !ignore_path.exists() {
            let _ = std::fs::write(&ignore_path, "*\n!.gitignore\n");
        }
    }

    fn get_npm_install_root(&self, scope: SourceScope, temporary: bool) -> Result<String, String> {
        if temporary {
            return self.get_temporary_dir("npm", None, None);
        }
        if scope == SourceScope::Project {
            self.assert_project_trusted_for_scope(scope)?;
            return Ok(Path::new(&self.cwd)
                .join(CONFIG_DIR_NAME)
                .join("npm")
                .to_string_lossy()
                .into_owned());
        }
        Ok(Path::new(&self.agent_dir)
            .join("npm")
            .to_string_lossy()
            .into_owned())
    }

    fn get_global_npm_root(&self) -> Result<String, String> {
        let npm_command = self.get_npm_command();
        let command_key = format!("{}\0{}", npm_command.0, npm_command.1.join("\0"));
        if let Some((key, root)) = self.global_npm_root.lock().as_ref() {
            if *key == command_key {
                return Ok(root.clone());
            }
        }
        let root = if self.get_package_manager_name() == "bun" {
            let bin_dir = self.run_npm_command_sync(&[
                "pm".to_string(),
                "bin".to_string(),
                "-g".to_string(),
            ])?;
            let bin_dir = bin_dir.trim();
            Path::new(bin_dir)
                .parent()
                .map(|parent| parent.join("install").join("global").join("node_modules"))
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        } else {
            self.run_npm_command_sync(&["root".to_string(), "-g".to_string()])?
                .trim()
                .to_string()
        };
        *self.global_npm_root.lock() = Some((command_key, root.clone()));
        Ok(root)
    }

    fn get_pnpm_global_package_path(&self, package_name: &str) -> Option<String> {
        if self.get_package_manager_name() != "pnpm" {
            return None;
        }

        let output = self
            .run_npm_command_sync(&[
                "list".to_string(),
                "-g".to_string(),
                "--depth".to_string(),
                "0".to_string(),
                "--json".to_string(),
            ])
            .ok()?;
        let entries: serde_json::Value = serde_json::from_str(&output).ok()?;
        let entries = entries.as_array()?;
        for entry in entries {
            if let Some(path) = entry
                .get("dependencies")
                .and_then(|deps| deps.get(package_name))
                .and_then(|pkg| pkg.get("path"))
                .and_then(|path| path.as_str())
            {
                return Some(path.to_string());
            }
        }
        None
    }

    fn get_managed_npm_install_path(
        &self,
        source: &NpmSource,
        scope: SourceScope,
    ) -> Result<String, String> {
        if scope == SourceScope::Temporary {
            return Ok(Path::new(&self.get_temporary_dir("npm", None, None)?)
                .join("node_modules")
                .join(&source.name)
                .to_string_lossy()
                .into_owned());
        }
        if scope == SourceScope::Project {
            self.assert_project_trusted_for_scope(scope)?;
            return Ok(Path::new(&self.cwd)
                .join(CONFIG_DIR_NAME)
                .join("npm")
                .join("node_modules")
                .join(&source.name)
                .to_string_lossy()
                .into_owned());
        }
        Ok(Path::new(&self.agent_dir)
            .join("npm")
            .join("node_modules")
            .join(&source.name)
            .to_string_lossy()
            .into_owned())
    }

    fn get_legacy_global_npm_install_path(&self, source: &NpmSource) -> Option<String> {
        match self.get_pnpm_global_package_path(&source.name) {
            Some(path) => Some(path),
            None => self.get_global_npm_root().ok().map(|root| {
                Path::new(&root)
                    .join(&source.name)
                    .to_string_lossy()
                    .into_owned()
            }),
        }
    }

    pub fn get_npm_install_path(&self, source: &NpmSource, scope: SourceScope) -> String {
        let managed_path = match self.get_managed_npm_install_path(source, scope) {
            Ok(path) => path,
            Err(_) => return String::new(),
        };
        if scope != SourceScope::User || Path::new(&managed_path).exists() {
            return managed_path;
        }
        match self.get_legacy_global_npm_install_path(source) {
            Some(legacy_path) if Path::new(&legacy_path).exists() => legacy_path,
            _ => managed_path,
        }
    }

    pub fn get_git_install_path(
        &self,
        source: &GitSource,
        scope: SourceScope,
    ) -> Result<String, String> {
        if scope == SourceScope::Temporary {
            // Include the ref in the hash so each pinned ref gets its own checkout.
            return self.get_temporary_dir(
                &format!("git-{}", source.host),
                Some(&source.path),
                source.ref_.as_deref(),
            );
        }
        let install_root = self
            .get_git_install_root(scope)
            .ok_or_else(|| "Missing git install root".to_string())?;
        self.resolve_managed_path(&install_root, &[source.host.clone(), source.path.clone()])
    }

    fn get_git_install_root(&self, scope: SourceScope) -> Option<String> {
        if scope == SourceScope::Temporary {
            return None;
        }
        if scope == SourceScope::Project {
            self.assert_project_trusted_for_scope(scope).ok()?;
            return Some(
                Path::new(&self.cwd)
                    .join(CONFIG_DIR_NAME)
                    .join("git")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        Some(
            Path::new(&self.agent_dir)
                .join("git")
                .to_string_lossy()
                .into_owned(),
        )
    }

    fn get_temporary_dir(
        &self,
        prefix: &str,
        suffix: Option<&str>,
        ref_: Option<&str>,
    ) -> Result<String, String> {
        let temp_folder = get_extension_temp_folder(&self.agent_dir);
        let root = self.resolve_managed_path(&temp_folder, &[prefix.to_string()])?;
        let hash_input = format!(
            "{prefix}-{}{}",
            suffix.unwrap_or(""),
            ref_.map(|ref_| format!("@{ref_}")).unwrap_or_default()
        );
        let hash = format!("{:x}", sha2::Sha256::digest(hash_input.as_bytes()));
        let hash = &hash[..8];
        self.resolve_managed_path(&root, &[hash.to_string(), suffix.unwrap_or("").to_string()])
    }

    fn resolve_managed_path(&self, root: &str, parts: &[String]) -> Result<String, String> {
        let resolved_root = resolve_path(
            root,
            &std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy(),
        );
        let mut joined = resolved_root.clone();
        for part in parts {
            joined = Path::new(&joined).join(part).to_string_lossy().into_owned();
        }
        let resolved_path = resolve_path(&joined, &resolved_root);
        if resolved_path != resolved_root
            && !resolved_path.starts_with(&format!("{resolved_root}/"))
        {
            return Err(format!(
                "Refusing to use path outside package install root: {resolved_path}"
            ));
        }
        Ok(resolved_path)
    }

    fn get_base_dir_for_scope(&self, scope: SourceScope) -> String {
        if scope == SourceScope::Project {
            self.assert_project_trusted_for_scope(scope)
                .unwrap_or_default();
            return Path::new(&self.cwd)
                .join(CONFIG_DIR_NAME)
                .to_string_lossy()
                .into_owned();
        }
        if scope == SourceScope::User {
            return self.agent_dir.clone();
        }
        self.cwd.clone()
    }

    fn resolve_path(&self, input: &str) -> String {
        self.resolve_path_from_base(input, &self.cwd)
    }

    fn resolve_path_from_base(&self, input: &str, base_dir: &str) -> String {
        let normalized = normalize_path(
            input,
            &PathInputOptions {
                trim: true,
                home_dir: Some(get_home_dir()),
                ..Default::default()
            },
        );
        resolve_path(&normalized, base_dir)
    }

    fn collect_package_resources(
        &self,
        package_root: &str,
        accumulator: &mut ResourceAccumulator,
        filter: Option<&PackageFilter>,
        metadata: &mut PathMetadata,
    ) -> bool {
        if let Some(filter) = filter {
            for resource_type in RESOURCE_TYPES {
                let patterns = filter.patterns(resource_type);
                let target = accumulator_target(accumulator, resource_type);
                if filter.autoload == Some(false) {
                    let empty: Vec<String> = Vec::new();
                    self.apply_package_delta_filter(
                        package_root,
                        patterns.unwrap_or(&empty),
                        resource_type,
                        target,
                        metadata,
                    );
                } else if let Some(patterns) = patterns {
                    let patterns = patterns.clone();
                    let target = accumulator_target(accumulator, resource_type);
                    self.apply_package_filter(
                        package_root,
                        &patterns,
                        resource_type,
                        target,
                        metadata,
                    );
                } else {
                    self.collect_default_resources(package_root, resource_type, target, metadata);
                }
            }
            return true;
        }

        let manifest = read_pi_manifest(
            &Path::new(package_root)
                .join("package.json")
                .to_string_lossy(),
        );
        if let Some(manifest) = manifest {
            for resource_type in RESOURCE_TYPES {
                let entries = manifest_entries(&manifest, resource_type);
                let target = accumulator_target(accumulator, resource_type);
                let mut metadata = metadata.clone();
                self.add_manifest_entries(
                    entries.as_deref(),
                    package_root,
                    resource_type,
                    target,
                    &mut metadata,
                );
            }
            return true;
        }

        let mut has_any_dir = false;
        for resource_type in RESOURCE_TYPES {
            let dir = Path::new(package_root).join(resource_type.key());
            if dir.exists() {
                // Collect all files from the directory (all enabled by default)
                let files = collect_resource_files(&dir.to_string_lossy(), resource_type);
                let target = accumulator_target(accumulator, resource_type);
                for f in files {
                    add_resource(target, &f, metadata.clone(), true);
                }
                has_any_dir = true;
            }
        }
        has_any_dir
    }

    fn collect_default_resources(
        &self,
        package_root: &str,
        resource_type: ResourceType,
        target: &mut IndexMap<String, ResourceEntry>,
        metadata: &mut PathMetadata,
    ) {
        let manifest = read_pi_manifest(
            &Path::new(package_root)
                .join("package.json")
                .to_string_lossy(),
        );
        let entries = manifest
            .as_ref()
            .and_then(|manifest| manifest_entries(manifest, resource_type));
        if let Some(entries) = entries {
            let mut metadata = metadata.clone();
            self.add_manifest_entries(
                Some(&entries),
                package_root,
                resource_type,
                target,
                &mut metadata,
            );
            return;
        }
        let dir = Path::new(package_root).join(resource_type.key());
        if dir.exists() {
            // Collect all files from the directory (all enabled by default)
            let files = collect_resource_files(&dir.to_string_lossy(), resource_type);
            for f in files {
                add_resource(target, &f, metadata.clone(), true);
            }
        }
    }

    fn apply_package_filter(
        &self,
        package_root: &str,
        user_patterns: &[String],
        resource_type: ResourceType,
        target: &mut IndexMap<String, ResourceEntry>,
        metadata: &mut PathMetadata,
    ) {
        let (all_files, _enabled) = self.collect_manifest_files(package_root, resource_type);

        if user_patterns.is_empty() {
            // Empty array explicitly disables all resources of this type
            for f in &all_files {
                add_resource(target, f, metadata.clone(), false);
            }
            return;
        }

        // Apply user patterns
        let enabled_by_user = apply_patterns(&all_files, user_patterns, package_root);

        for f in &all_files {
            let enabled = enabled_by_user.contains(f);
            add_resource(target, f, metadata.clone(), enabled);
        }
    }

    fn apply_package_delta_filter(
        &self,
        package_root: &str,
        user_patterns: &[String],
        resource_type: ResourceType,
        target: &mut IndexMap<String, ResourceEntry>,
        metadata: &mut PathMetadata,
    ) {
        if user_patterns.is_empty() {
            return;
        }

        let (all_files, _enabled) = self.collect_manifest_files(package_root, resource_type);
        let enabled_by_user =
            apply_autoload_disabled_patterns(&all_files, user_patterns, package_root);
        for (file_path, enabled) in enabled_by_user {
            add_resource(target, &file_path, metadata.clone(), enabled);
        }
    }

    /// Collect all files from a package for a resource type, applying manifest patterns.
    /// Returns (allFiles, enabledByManifest) where enabledByManifest is the set of files
    /// that pass the manifest's own patterns.
    fn collect_manifest_files(
        &self,
        package_root: &str,
        resource_type: ResourceType,
    ) -> (Vec<String>, HashSet<String>) {
        let manifest = read_pi_manifest(
            &Path::new(package_root)
                .join("package.json")
                .to_string_lossy(),
        );
        let entries = manifest
            .as_ref()
            .and_then(|manifest| manifest_entries(manifest, resource_type));
        if let Some(entries) = entries.filter(|entries| !entries.is_empty()) {
            let all_files =
                self.collect_files_from_manifest_entries(&entries, package_root, resource_type);
            let manifest_patterns: Vec<String> = entries
                .iter()
                .filter(|entry| is_override_pattern(entry))
                .cloned()
                .collect();
            let (all_files, enabled_by_manifest) = if !manifest_patterns.is_empty() {
                let enabled = apply_patterns(&all_files, &manifest_patterns, package_root);
                let all: Vec<String> = all_files
                    .iter()
                    .filter(|f| enabled.contains(*f))
                    .cloned()
                    .collect();
                (all, enabled)
            } else {
                let enabled: HashSet<String> = all_files.iter().cloned().collect();
                (all_files, enabled)
            };
            return (all_files, enabled_by_manifest);
        }

        let convention_dir = Path::new(package_root).join(resource_type.key());
        if !convention_dir.exists() {
            return (Vec::new(), HashSet::new());
        }
        let all_files = collect_resource_files(&convention_dir.to_string_lossy(), resource_type);
        let enabled_by_manifest: HashSet<String> = all_files.iter().cloned().collect();
        (all_files, enabled_by_manifest)
    }

    fn add_manifest_entries(
        &self,
        entries: Option<&[String]>,
        root: &str,
        resource_type: ResourceType,
        target: &mut IndexMap<String, ResourceEntry>,
        metadata: &PathMetadata,
    ) {
        let Some(entries) = entries else { return };

        let all_files = self.collect_files_from_manifest_entries(entries, root, resource_type);
        let patterns: Vec<String> = entries
            .iter()
            .filter(|entry| is_override_pattern(entry))
            .cloned()
            .collect();
        let enabled_paths = apply_patterns(&all_files, &patterns, root);

        for f in &all_files {
            if enabled_paths.contains(f) {
                add_resource(target, f, metadata.clone(), true);
            }
        }
    }

    fn collect_files_from_manifest_entries(
        &self,
        entries: &[String],
        root: &str,
        resource_type: ResourceType,
    ) -> Vec<String> {
        let source_entries: Vec<&String> = entries
            .iter()
            .filter(|entry| !is_override_pattern(entry))
            .collect();
        let mut resolved: Vec<String> = Vec::new();
        for entry in source_entries {
            if has_glob_pattern(entry) {
                resolved.extend(expand_package_glob(entry, root));
            } else {
                // Plain `path.resolve(root, entry)` semantics — no tilde expansion.
                let joined = crate::utils::paths::join(root, entry);
                resolved.push(resolve_path(&joined, root));
            }
        }
        self.collect_files_from_paths(&resolved, resource_type)
    }

    fn resolve_local_entries(
        &self,
        entries: &[String],
        resource_type: ResourceType,
        target: &mut IndexMap<String, ResourceEntry>,
        metadata: PathMetadata,
        base_dir: &str,
    ) {
        if entries.is_empty() {
            return;
        }

        // Collect all files from plain entries (non-pattern entries)
        let (plain, patterns) = split_patterns(entries);
        let resolved_plain: Vec<String> = plain
            .iter()
            .map(|p| self.resolve_path_from_base(p, base_dir))
            .collect();
        let all_files = self.collect_files_from_paths(&resolved_plain, resource_type);

        // Determine which files are enabled based on patterns
        let enabled_paths = apply_patterns(&all_files, &patterns, base_dir);

        // Add all files with their enabled state
        for f in &all_files {
            let enabled = enabled_paths.contains(f);
            add_resource(target, f, metadata.clone(), enabled);
        }
    }

    fn add_auto_discovered_resources(
        &self,
        accumulator: &mut ResourceAccumulator,
        global_settings: &Settings,
        project_settings: &Settings,
    ) {
        let global_base_dir = self.agent_dir.clone();
        let project_base_dir = Path::new(&self.cwd)
            .join(CONFIG_DIR_NAME)
            .to_string_lossy()
            .into_owned();

        let user_metadata = PathMetadata {
            source: "auto".to_string(),
            scope: SourceScope::User,
            origin: SourceOrigin::TopLevel,
            base_dir: Some(global_base_dir.clone()),
            package_root: None,
        };
        let project_metadata = PathMetadata {
            source: "auto".to_string(),
            scope: SourceScope::Project,
            origin: SourceOrigin::TopLevel,
            base_dir: Some(project_base_dir.clone()),
            package_root: None,
        };

        let user_overrides = (
            Self::list_of(global_settings, "extensions"),
            Self::list_of(global_settings, "skills"),
            Self::list_of(global_settings, "prompts"),
            Self::list_of(global_settings, "themes"),
        );
        let project_overrides = (
            Self::list_of(project_settings, "extensions"),
            Self::list_of(project_settings, "skills"),
            Self::list_of(project_settings, "prompts"),
            Self::list_of(project_settings, "themes"),
        );

        let user_dirs = (
            Path::new(&global_base_dir)
                .join("extensions")
                .to_string_lossy()
                .into_owned(),
            Path::new(&global_base_dir)
                .join("skills")
                .to_string_lossy()
                .into_owned(),
            Path::new(&global_base_dir)
                .join("prompts")
                .to_string_lossy()
                .into_owned(),
            Path::new(&global_base_dir)
                .join("themes")
                .to_string_lossy()
                .into_owned(),
        );
        let project_dirs = (
            Path::new(&project_base_dir)
                .join("extensions")
                .to_string_lossy()
                .into_owned(),
            Path::new(&project_base_dir)
                .join("skills")
                .to_string_lossy()
                .into_owned(),
            Path::new(&project_base_dir)
                .join("prompts")
                .to_string_lossy()
                .into_owned(),
            Path::new(&project_base_dir)
                .join("themes")
                .to_string_lossy()
                .into_owned(),
        );
        let user_agents_skills_dir = Path::new(&get_home_dir())
            .join(".agents")
            .join("skills")
            .to_string_lossy()
            .into_owned();
        let project_trusted = self.settings_manager.lock().is_project_trusted();
        let project_agents_skill_dirs: Vec<String> = if project_trusted {
            collect_ancestor_agents_skill_dirs(&self.cwd)
                .into_iter()
                .filter(|dir| {
                    resolve_path(dir, &self.cwd) != resolve_path(&user_agents_skills_dir, &self.cwd)
                })
                .collect()
        } else {
            Vec::new()
        };

        // addResources helper
        let add_resources = |accumulator: &mut ResourceAccumulator,
                             resource_type: ResourceType,
                             paths: Vec<String>,
                             metadata: PathMetadata,
                             overrides: &[String],
                             base_dir: &str| {
            let target = accumulator_target(accumulator, resource_type);
            for path in paths {
                let enabled = is_enabled_by_overrides(&path, overrides, base_dir);
                add_resource(target, &path, metadata.clone(), enabled);
            }
        };

        if project_trusted {
            // Project extensions from .pi/
            add_resources(
                accumulator,
                ResourceType::Extensions,
                collect_auto_extension_entries(&project_dirs.0),
                project_metadata.clone(),
                &project_overrides.0,
                &project_base_dir,
            );

            // Project skills from .pi/
            add_resources(
                accumulator,
                ResourceType::Skills,
                collect_auto_skill_entries(&project_dirs.1, SkillDiscoveryMode::Pi),
                project_metadata.clone(),
                &project_overrides.1,
                &project_base_dir,
            );
        }

        // Project skills from .agents/ (each with its own baseDir)
        for agents_skills_dir in &project_agents_skill_dirs {
            let agents_base_dir = Path::new(agents_skills_dir)
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(); // the .agents directory
            let mut agents_metadata = project_metadata.clone();
            agents_metadata.base_dir = Some(agents_base_dir.clone());
            add_resources(
                accumulator,
                ResourceType::Skills,
                collect_auto_skill_entries(agents_skills_dir, SkillDiscoveryMode::Agents),
                agents_metadata,
                &project_overrides.1,
                &agents_base_dir,
            );
        }

        if project_trusted {
            add_resources(
                accumulator,
                ResourceType::Prompts,
                collect_auto_prompt_entries(&project_dirs.2),
                project_metadata.clone(),
                &project_overrides.2,
                &project_base_dir,
            );
            add_resources(
                accumulator,
                ResourceType::Themes,
                collect_auto_theme_entries(&project_dirs.3),
                project_metadata.clone(),
                &project_overrides.3,
                &project_base_dir,
            );
        }

        // User extensions from ~/.pi/agent/
        add_resources(
            accumulator,
            ResourceType::Extensions,
            collect_auto_extension_entries(&user_dirs.0),
            user_metadata.clone(),
            &user_overrides.0,
            &global_base_dir,
        );

        // User skills from ~/.pi/agent/
        add_resources(
            accumulator,
            ResourceType::Skills,
            collect_auto_skill_entries(&user_dirs.1, SkillDiscoveryMode::Pi),
            user_metadata.clone(),
            &user_overrides.1,
            &global_base_dir,
        );

        // User skills from ~/.agents/ (with its own baseDir)
        let user_agents_base_dir = Path::new(&user_agents_skills_dir)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut user_agents_metadata = user_metadata.clone();
        user_agents_metadata.base_dir = Some(user_agents_base_dir.clone());
        add_resources(
            accumulator,
            ResourceType::Skills,
            collect_auto_skill_entries(&user_agents_skills_dir, SkillDiscoveryMode::Agents),
            user_agents_metadata,
            &user_overrides.1,
            &user_agents_base_dir,
        );

        add_resources(
            accumulator,
            ResourceType::Prompts,
            collect_auto_prompt_entries(&user_dirs.2),
            user_metadata.clone(),
            &user_overrides.2,
            &global_base_dir,
        );
        add_resources(
            accumulator,
            ResourceType::Themes,
            collect_auto_theme_entries(&user_dirs.3),
            user_metadata.clone(),
            &user_overrides.3,
            &global_base_dir,
        );
    }

    fn collect_files_from_paths(
        &self,
        paths: &[String],
        resource_type: ResourceType,
    ) -> Vec<String> {
        let mut files: Vec<String> = Vec::new();
        for p in paths {
            if !Path::new(p).exists() {
                continue;
            }

            let Ok(stats) = std::fs::metadata(p) else {
                continue;
            };
            if stats.is_file() {
                files.push(p.clone());
            } else if stats.is_dir() {
                files.extend(collect_resource_files(p, resource_type));
            }
        }
        files
    }
}

fn scope_label(scope: SourceScope) -> &'static str {
    match scope {
        SourceScope::User => "user",
        SourceScope::Project => "project",
        SourceScope::Temporary => "temporary",
    }
}

fn push_unique(list: &mut Vec<String>, value: String) {
    if !list.contains(&value) {
        list.push(value);
    }
}

/// Upstream `parseNpmSpec` — `/^(@?[^@]+(?:\/[^@]+)?)(?:@(.+))?$/`.
pub fn parse_npm_spec(spec: &str) -> (String, Option<String>) {
    if spec.is_empty() {
        return (spec.to_string(), None);
    }
    if let Some(sep) = spec[1..].find('@') {
        let sep = sep + 1;
        if sep + 1 <= spec.len() && sep < spec.len() {
            return (spec[..sep].to_string(), Some(spec[sep + 1..].to_string()));
        }
    }
    (spec.to_string(), None)
}

/// Find a 40-hex-char commit hash at a line start in `ls-remote` output
/// (upstream `/^([0-9a-f]{40})\s+/m` / `... HEAD$/m`).
fn find_head_hash(output: &str, suffix: Option<&str>) -> Option<String> {
    for line in output.lines() {
        if line.len() < 40 {
            continue;
        }
        let (hash, rest) = line.split_at(40);
        if !hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            continue;
        }
        match suffix {
            Some(suffix) if rest.trim_end() == format!(" {suffix}") => {
                return Some(hash.to_string())
            }
            Some(_) => continue,
            None if rest.starts_with(' ') || rest.starts_with('\t') => {
                return Some(hash.to_string())
            }
            None => continue,
        }
    }
    None
}

fn accumulator_target(
    accumulator: &mut ResourceAccumulator,
    resource_type: ResourceType,
) -> &mut IndexMap<String, ResourceEntry> {
    match resource_type {
        ResourceType::Extensions => &mut accumulator.extensions,
        ResourceType::Skills => &mut accumulator.skills,
        ResourceType::Prompts => &mut accumulator.prompts,
        ResourceType::Themes => &mut accumulator.themes,
    }
}

fn manifest_entries(manifest: &PiManifest, resource_type: ResourceType) -> Option<Vec<String>> {
    match resource_type {
        ResourceType::Extensions => manifest.extensions.clone(),
        ResourceType::Skills => manifest.skills.clone(),
        ResourceType::Prompts => manifest.prompts.clone(),
        ResourceType::Themes => manifest.themes.clone(),
    }
}

fn add_resource(
    map: &mut IndexMap<String, ResourceEntry>,
    path: &str,
    metadata: PathMetadata,
    enabled: bool,
) {
    if path.is_empty() {
        return;
    }
    map.entry(path.to_string())
        .or_insert(ResourceEntry { metadata, enabled });
}

fn to_resolved_paths(accumulator: ResourceAccumulator) -> ResolvedPaths {
    let map_to_resolved = |entries: IndexMap<String, ResourceEntry>| -> Vec<ResolvedResource> {
        let mut resolved: Vec<ResolvedResource> = entries
            .into_iter()
            .map(|(path, entry)| ResolvedResource {
                path,
                enabled: entry.enabled,
                metadata: entry.metadata,
            })
            .collect();
        resolved.sort_by_key(|entry| resource_precedence_rank(&entry.metadata));

        let mut seen: HashSet<String> = HashSet::new();
        resolved
            .into_iter()
            .filter(|entry| {
                let canonical_path = canonicalize_path(&entry.path);
                seen.insert(canonical_path)
            })
            .collect()
    };

    ResolvedPaths {
        extensions: map_to_resolved(accumulator.extensions),
        skills: map_to_resolved(accumulator.skills),
        prompts: map_to_resolved(accumulator.prompts),
        themes: map_to_resolved(accumulator.themes),
    }
}

/// Ordered bounded-concurrency runner (upstream `runWithConcurrency`).
async fn run_with_concurrency<T, F, Fut>(tasks: Vec<F>, limit: usize) -> Vec<T>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    if tasks.is_empty() {
        return Vec::new();
    }
    let semaphore = Arc::new(tokio::sync::Semaphore::new(limit.max(1)));
    let futures = tasks.into_iter().map(|task| {
        let semaphore = semaphore.clone();
        async move {
            let _permit = semaphore.acquire().await;
            task().await
        }
    });
    join_all(futures).await
}

#[cfg(test)]
#[path = "package_manager_tests.rs"]
mod package_manager_tests;
