//! Port of `packages/coding-agent/src/core/resource-loader.ts` — discovers and
//! loads the coding agent's resources: extensions, skills, prompt templates,
//! themes, AGENTS.md context files, and system-prompt files.
//!
//! Deviations:
//! - **D58 — reload is synchronous.** Upstream `reload()` is async because
//!   settings/package resolution can spawn processes and the extension loader
//!   awaits module compilation. The Rust port drives the same passes
//!   (pre-trust extension load, project-trust resolution, package resolution,
//!   resource loading) synchronously; package installation blocks the caller
//!   like upstream's awaited promise.
//! - **D58a — `file://` resource URLs.** Upstream normalizes resource paths
//!   through `normalizePath`, whose `fileURLToPath` step is a documented JS
//!   seam in `utils::paths` (returned unchanged). `resolve_resource_path`
//!   expands `file://` URLs itself so extension-supplied file URLs load.
//! - **D58b — warnings shape.** Upstream `LoadExtensionsResult.warnings`
//!   entries are `{path, warning}`; the Rust `LoadRecord` carries the text in
//!   its `error` field. `None` warnings are treated as empty.
//! - **D58c — unparseable extension manifests panic.** Upstream lets the
//!   manifest `SyntaxError` reject `reload()`; the port's `reload` returns
//!   `()`, so the parse failure propagates as a panic from
//!   `collect_extension_package_warnings`.
//!
//! Upstream cases that construct a full `AgentSession` defer with that
//! slice; every other case — including the extension-JS-dependent suites,
//! which execute real `.ts` fixtures through the embedded JS host since
//! D107 — is ported in `resource_loader_tests`.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::config::CONFIG_DIR_NAME;
use crate::core::diagnostics::{ResourceCollision, ResourceDiagnostic, ResourceType};
use crate::core::event_bus::create_event_bus;
use crate::core::extensions::loader::{
    clear_extension_cache, create_extension_runtime, load_extension_from_factory,
    load_extensions_cached,
};
use crate::core::extensions::types::{
    Extension, ExtensionFactory, ExtensionRuntime, InlineExtension, LoadExtensionsResult,
    LoadRecord,
};
use crate::core::footer_data_provider::find_git_paths;
use crate::core::package_manager::{
    DefaultPackageManager, PackageManagerOptions, ResolvedPaths, ResolvedResource,
};
use crate::core::prompt_templates::{
    load_prompt_templates, LoadPromptTemplatesOptions, PromptTemplate,
};
use crate::core::settings_manager::SettingsManager;
use crate::core::skills::{load_skills, LoadSkillsOptions, Skill};
use crate::core::source_info::{
    create_source_info, create_synthetic_source_info, get_synthetic_path_source, is_synthetic_path,
    PathMetadata, SourceInfo, SourceOrigin, SourceScope, BUILTIN_PATH_PREFIX,
};
use crate::core::timings::reset_timings;
use crate::theme::load_theme_from_path;
use crate::theme::Theme;
use crate::utils::paths::{canonicalize_path, is_local_path, resolve_path, resolve_path_cwd};
use crate::utils::text::strip_bom;
use crate::utils::tui_compat::TerminalColorMode;

/// Run a loader/package-manager future to completion on a throwaway
/// current-thread runtime (resource-loader drives async passes from a
/// synchronous `reload`).
fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(future)
}

/// A loaded AGENTS.md-style context file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextFile {
    pub path: String,
    pub content: String,
}

/// Extra resource paths supplied by an extension
/// (upstream `ResourceExtensionPaths`).
#[derive(Debug, Clone, Default)]
pub struct ResourceExtensionPaths {
    pub skill_paths: Vec<ResourcePathEntry>,
    pub prompt_paths: Vec<ResourcePathEntry>,
    pub theme_paths: Vec<ResourcePathEntry>,
}

/// A path plus the metadata of the extension that supplied it.
#[derive(Debug, Clone)]
pub struct ResourcePathEntry {
    pub path: String,
    pub metadata: PathMetadata,
}

/// Options for [`DefaultResourceLoader::reload`]
/// (upstream `ResourceLoaderReloadOptions`).
pub struct ResourceLoaderReloadOptions {
    /// Resolves project trust from the pre-trust extension set.
    pub resolve_project_trust: Option<Box<dyn FnOnce(&LoadExtensionsResult) -> bool + Send>>,
}

impl Default for ResourceLoaderReloadOptions {
    fn default() -> Self {
        Self {
            resolve_project_trust: None,
        }
    }
}

/// Host-provided extension packages that must be peer dependencies, never
/// regular dependencies of an extension package.
const HOST_PROVIDED_EXTENSION_PACKAGES: &[&str] = &[
    "@earendil-works/pi-agent-core",
    "@earendil-works/pi-ai",
    "@earendil-works/pi-coding-agent",
    "@earendil-works/pi-tui",
    "@mariozechner/pi-agent-core",
    "@mariozechner/pi-ai",
    "@mariozechner/pi-coding-agent",
    "@mariozechner/pi-tui",
    "@sinclair/typebox",
    "typebox",
];

type ExtensionsOverride = Box<dyn Fn(LoadExtensionsResult) -> LoadExtensionsResult + Send + Sync>;
type SkillsOverride = Box<dyn Fn(SkillsResult) -> SkillsResult + Send + Sync>;
type PromptsOverride = Box<dyn Fn(PromptsResult) -> PromptsResult + Send + Sync>;
type ThemesOverride = Box<dyn Fn(ThemesResult) -> ThemesResult + Send + Sync>;
type AgentsFilesOverride = Box<dyn Fn(Vec<ContextFile>) -> Vec<ContextFile> + Send + Sync>;
type SystemPromptOverride = Box<dyn Fn(Option<String>) -> Option<String> + Send + Sync>;
type AppendSystemPromptOverride = Box<dyn Fn(Vec<String>) -> Vec<String> + Send + Sync>;

pub type SkillsResult = SkillsPromptResult<Skill>;
pub type PromptsResult = SkillsPromptResult<PromptTemplate>;

/// `{ skills|prompts, diagnostics }` payload shape shared by the skill and
/// prompt loaders.
#[derive(Debug, Clone, Default)]
pub struct SkillsPromptResult<T> {
    pub items: Vec<T>,
    pub diagnostics: Vec<ResourceDiagnostic>,
}

pub type ThemesResult = SkillsPromptResult<Theme>;

/// Warn when an extension package's `package.json` declares host-provided
/// packages as regular `dependencies`
/// (upstream `collectExtensionPackageWarnings`).
fn collect_extension_package_warnings(
    extension_paths: &[String],
    metadata_by_path: &HashMap<String, PathMetadata>,
) -> Vec<LoadRecord> {
    let mut warnings = Vec::new();
    let mut package_roots: HashSet<String> = HashSet::new();
    for path in extension_paths {
        if let Some(metadata) = metadata_by_path.get(path) {
            if let Some(root) = &metadata.package_root {
                package_roots.insert(root.clone());
            }
        }
    }
    for package_root in package_roots {
        let package_json_path = Path::new(&package_root).join("package.json");
        if !package_json_path.exists() {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(&package_json_path) else {
            continue;
        };
        let Ok(manifest) = serde_json::from_str::<serde_json::Value>(strip_bom(&raw)) else {
            // Upstream lets the SyntaxError propagate out of reload.
            panic!("{}", invalid_manifest_message(&package_json_path));
        };
        let Some(deps) = manifest.get("dependencies").filter(|d| d.is_object()) else {
            continue;
        };
        let mut host_dependencies: Vec<String> = deps
            .as_object()
            .unwrap()
            .keys()
            .filter(|name| HOST_PROVIDED_EXTENSION_PACKAGES.contains(&name.as_str()))
            .cloned()
            .collect();
        host_dependencies.sort();
        if host_dependencies.is_empty() {
            continue;
        }
        warnings.push(LoadRecord {
			path: package_json_path.to_string_lossy().to_string(),
			error: format!(
				"Host-provided extension packages must be declared in peerDependencies with a \"*\" range, not \
				 dependencies: {}. Installed copies can bypass the extension loader and create duplicate runtime \
				 modules.",
				host_dependencies.join(", ")
			),
		});
    }
    warnings
}

fn invalid_manifest_message(path: &Path) -> String {
    format!("Failed to parse {}", path.display())
}

/// Merge warning records, deduplicated by path, later entries winning
/// (upstream `mergeExtensionWarnings`).
fn merge_extension_warnings(result: &mut LoadExtensionsResult, warnings: Vec<LoadRecord>) {
    let mut merged: HashMap<String, LoadRecord> = HashMap::new();
    for record in result.warnings.take().unwrap_or_default() {
        merged.insert(record.path.clone(), record);
    }
    for record in warnings {
        merged.insert(record.path.clone(), record);
    }
    let mut records: Vec<LoadRecord> = merged.into_values().collect();
    records.sort_by(|a, b| a.path.cmp(&b.path));
    result.warnings = Some(records);
}

/// A built-in extension declaration (`builtin:<name>` factory).
pub struct BuiltinExtension {
    pub name: String,
    pub factory: ExtensionFactory,
    pub replaceable: bool,
}

/// Split inline extension inputs into plain factories and named built-ins
/// (upstream `isBuiltinExtension` filtering in the constructor).
fn split_inline_extensions(
    inputs: Vec<InlineExtension>,
) -> (Vec<InlineExtension>, Vec<BuiltinExtension>) {
    let mut factories = Vec::new();
    let mut builtins = Vec::new();
    for input in inputs {
        match input {
            InlineExtension::Factory(factory) => factories.push(InlineExtension::Factory(factory)),
            InlineExtension::Named {
                name,
                factory,
                hidden: _,
                replaceable,
                builtin,
            } => {
                if builtin.unwrap_or(false) {
                    builtins.push(BuiltinExtension {
                        name,
                        factory,
                        replaceable: replaceable.unwrap_or(false),
                    });
                } else {
                    factories.push(InlineExtension::Named {
                        name,
                        factory,
                        hidden: None,
                        replaceable,
                        builtin: None,
                    });
                }
            }
        }
    }
    (factories, builtins)
}

/// Leave out replaceable extensions that share a tool, command, or flag name
/// with another extension (upstream `omitReplacedExtensions`).
fn omit_replaced_extensions(
    extensions: Vec<Extension>,
    warnings: &mut Vec<LoadRecord>,
) -> Vec<Extension> {
    let names = |extension: &Extension| -> Vec<String> {
        let mut result: Vec<String> = extension
            .tools
            .keys()
            .map(|name| format!("tool:{name}"))
            .collect();
        result.extend(
            extension
                .commands
                .keys()
                .map(|name| format!("command:{name}")),
        );
        result.extend(extension.flags.keys().map(|name| format!("flag:{name}")));
        result
    };
    // Name -> path of the non-replaceable extension that registered it.
    let mut taken: HashMap<String, String> = HashMap::new();
    for extension in &extensions {
        if extension.replaceable.unwrap_or(false) {
            continue;
        }
        for name in names(extension) {
            taken.insert(name, extension.path.clone());
        }
    }
    let mut kept: Vec<Extension> = Vec::new();
    for extension in extensions {
        if !extension.replaceable.unwrap_or(false) {
            kept.push(extension);
            continue;
        }
        let replacement = names(&extension)
            .into_iter()
            .find_map(|name| taken.get(&name).cloned().map(|path| (name, path)));
        let Some((name, replacement_path)) = replacement else {
            kept.push(extension);
            continue;
        };
        if let Some(builtin_name) = extension.path.strip_prefix(BUILTIN_PATH_PREFIX) {
            let (kind, raw_name) = name.split_once(':').unwrap_or((name.as_str(), ""));
            let registered_name = match kind {
                "command" => format!("/{raw_name}"),
                "flag" => format!("--{raw_name}"),
                _ => raw_name.to_string(),
            };
            warnings.push(LoadRecord {
                path: extension.path.clone(),
                error: format!(
					"Extension {replacement_path} registers {kind} `{registered_name}`, so built-in extension \
					 `{builtin_name}` was not loaded. To use `{builtin_name}`, run `pi config` and make sure it \
					 is enabled under Built-in extensions, then disable or remove the existing extension. We \
					 recommend only having one or the other loaded at a time."
				),
            });
        }
    }
    kept
}

/// Read a prompt-input string: existing file paths are read from disk,
/// anything else is literal text (upstream `resolvePromptInput`).
fn resolve_prompt_input(input: &str, description: &str) -> Option<String> {
    if input.is_empty() {
        return None;
    }
    if Path::new(input).exists() {
        return match std::fs::read_to_string(input) {
            Ok(content) => Some(strip_bom(&content).to_string()),
            Err(error) => {
                eprintln!("Warning: Could not read {description} file {input}: {error}");
                Some(input.to_string())
            }
        };
    }
    Some(input.to_string())
}

/// Load the context file of a single directory, if any
/// (upstream `loadContextFileFromDir`).
fn load_context_file_from_dir(dir: &str) -> Option<ContextFile> {
    const CANDIDATES: &[&str] = &[
        "AGENTS.override.md",
        "AGENTS.md",
        "AGENTS.MD",
        "CLAUDE.md",
        "CLAUDE.MD",
    ];
    for filename in CANDIDATES {
        let file_path = Path::new(dir).join(filename);
        if !file_path.exists() {
            continue;
        }
        match std::fs::metadata(&file_path) {
            Ok(metadata) if metadata.is_file() => match std::fs::read_to_string(&file_path) {
                Ok(content) => {
                    return Some(ContextFile {
                        path: file_path.to_string_lossy().to_string(),
                        content: strip_bom(&content).to_string(),
                    })
                }
                Err(error) => eprintln!("Warning: Could not read {}: {error}", file_path.display()),
            },
            Ok(_) => continue,
            Err(error) => eprintln!("Warning: Could not read {}: {error}", file_path.display()),
        }
    }
    None
}

/// The main repo's context file that a nested linked worktree's own copy
/// shadows (upstream `findShadowedContextFile`); returned canonicalized.
fn find_shadowed_context_file(cwd: &str) -> Option<String> {
    let git_paths = find_git_paths(cwd)?;
    let common_git_dir = canonicalize_path(&git_paths.common_git_dir);
    let worktree_root = canonicalize_path(&git_paths.repo_dir);
    let main_repo_root = Path::new(&common_git_dir)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let prefix = format!("{}{}", main_repo_root.display(), std::path::MAIN_SEPARATOR);
    if !worktree_root.starts_with(&prefix) {
        return None;
    }
    if canonicalize_path(&main_repo_root.join(".git").to_string_lossy()) != common_git_dir {
        return None;
    }
    let worktree_context_file = load_context_file_from_dir(&worktree_root)?;
    Some(
        main_repo_root
            .join(Path::new(&worktree_context_file.path).file_name()?)
            .to_string_lossy()
            .to_string(),
    )
}

/// Discover AGENTS.md/CLAUDE.md context files from the agent dir and every
/// ancestor of cwd (upstream `loadProjectContextFiles`).
pub fn load_project_context_files(cwd: &str, agent_dir: &str) -> Vec<ContextFile> {
    let resolved_cwd = resolve_path_cwd(cwd);
    let resolved_agent_dir = resolve_path_cwd(agent_dir);

    let mut context_files: Vec<ContextFile> = Vec::new();
    let mut seen_paths: HashSet<String> = HashSet::new();

    if let Some(global_context) = load_context_file_from_dir(&resolved_agent_dir) {
        seen_paths.insert(global_context.path.clone());
        context_files.push(global_context);
    }

    let mut ancestor_context_files: Vec<ContextFile> = Vec::new();
    let shadowed_context_file = find_shadowed_context_file(&resolved_cwd);
    let mut current_dir = resolved_cwd.clone();
    loop {
        let context_file = load_context_file_from_dir(&current_dir);
        let is_shadowed = shadowed_context_file.as_ref().is_some_and(|shadowed| {
            context_file
                .as_ref()
                .is_some_and(|file| canonicalize_path(&file.path) == *shadowed)
        });
        if let Some(file) = context_file {
            if !is_shadowed && !seen_paths.contains(&file.path) {
                seen_paths.insert(file.path.clone());
                ancestor_context_files.insert(0, file);
            }
        }
        let Some(parent_dir) = Path::new(&current_dir).parent() else {
            break;
        };
        let parent_dir = parent_dir.to_string_lossy().to_string();
        if parent_dir == current_dir {
            break;
        }
        current_dir = parent_dir;
    }

    context_files.extend(ancestor_context_files);
    context_files
}

/// Options for [`DefaultResourceLoader::new`] (upstream
/// `DefaultResourceLoaderOptions`).
pub struct DefaultResourceLoaderOptions {
    pub cwd: String,
    pub agent_dir: String,
    pub settings_manager: Option<SettingsManager>,
    pub event_bus: Option<crate::core::event_bus::EventBusController>,
    pub additional_extension_paths: Vec<String>,
    pub additional_skill_paths: Vec<String>,
    pub additional_prompt_template_paths: Vec<String>,
    pub additional_theme_paths: Vec<String>,
    pub extension_factories: Vec<InlineExtension>,
    pub no_extensions: bool,
    /// Built-in extensions not to load, by name (such as `mcp`), even when
    /// settings or `-e` enable them.
    pub disabled_builtin_extensions: Vec<String>,
    pub no_skills: bool,
    pub no_prompt_templates: bool,
    pub no_themes: bool,
    pub no_context_files: bool,
    pub system_prompt: Option<String>,
    pub append_system_prompt: Option<Vec<String>>,
    pub extensions_override: Option<ExtensionsOverride>,
    pub skills_override: Option<SkillsOverride>,
    pub prompts_override: Option<PromptsOverride>,
    pub themes_override: Option<ThemesOverride>,
    pub agents_files_override: Option<AgentsFilesOverride>,
    pub system_prompt_override: Option<SystemPromptOverride>,
    pub append_system_prompt_override: Option<AppendSystemPromptOverride>,
}

impl Default for DefaultResourceLoaderOptions {
    fn default() -> Self {
        Self {
            cwd: ".".to_string(),
            agent_dir: ".".to_string(),
            settings_manager: None,
            event_bus: None,
            additional_extension_paths: Vec::new(),
            additional_skill_paths: Vec::new(),
            additional_prompt_template_paths: Vec::new(),
            additional_theme_paths: Vec::new(),
            extension_factories: Vec::new(),
            no_extensions: false,
            disabled_builtin_extensions: Vec::new(),
            no_skills: false,
            no_prompt_templates: false,
            no_themes: false,
            no_context_files: false,
            system_prompt: None,
            append_system_prompt: None,
            extensions_override: None,
            skills_override: None,
            prompts_override: None,
            themes_override: None,
            agents_files_override: None,
            system_prompt_override: None,
            append_system_prompt_override: None,
        }
    }
}

/// Default resource loader (upstream `DefaultResourceLoader`).
pub struct DefaultResourceLoader {
    cwd: String,
    agent_dir: String,
    settings_manager: Arc<parking_lot::Mutex<SettingsManager>>,
    event_bus: crate::core::event_bus::EventBusController,
    package_manager: DefaultPackageManager,
    additional_extension_paths: Vec<String>,
    additional_skill_paths: Vec<String>,
    additional_prompt_template_paths: Vec<String>,
    additional_theme_paths: Vec<String>,
    extension_factories: Vec<InlineExtension>,
    builtin_extensions: HashMap<String, BuiltinExtension>,
    no_extensions: bool,
    disabled_builtin_extensions: std::collections::HashSet<String>,
    no_skills: bool,
    no_prompt_templates: bool,
    no_themes: bool,
    no_context_files: bool,
    system_prompt_source: Option<String>,
    append_system_prompt_source: Option<Vec<String>>,
    extensions_override: Option<ExtensionsOverride>,
    skills_override: Option<SkillsOverride>,
    prompts_override: Option<PromptsOverride>,
    themes_override: Option<ThemesOverride>,
    agents_files_override: Option<AgentsFilesOverride>,
    system_prompt_override: Option<SystemPromptOverride>,
    append_system_prompt_override: Option<AppendSystemPromptOverride>,

    extensions_result: LoadExtensionsResult,
    skills: Vec<Skill>,
    skill_diagnostics: Vec<ResourceDiagnostic>,
    prompts: Vec<PromptTemplate>,
    prompt_diagnostics: Vec<ResourceDiagnostic>,
    themes: Vec<Theme>,
    theme_diagnostics: Vec<ResourceDiagnostic>,
    agents_files: Vec<ContextFile>,
    system_prompt: Option<String>,
    system_prompt_source_path: Option<String>,
    append_system_prompt: Vec<String>,
    append_system_prompt_source_paths: Vec<String>,
    last_skill_paths: Vec<String>,
    extension_skill_source_infos: HashMap<String, SourceInfo>,
    extension_prompt_source_infos: HashMap<String, SourceInfo>,
    extension_theme_source_infos: HashMap<String, SourceInfo>,
    resource_metadata_by_path: HashMap<String, PathMetadata>,
    last_prompt_paths: Vec<String>,
    last_theme_paths: Vec<String>,
    loaded: bool,
}

impl DefaultResourceLoader {
    pub fn new(mut options: DefaultResourceLoaderOptions) -> Self {
        let cwd = resolve_path_cwd(&options.cwd);
        let agent_dir = resolve_path_cwd(&options.agent_dir);
        let settings_manager = Arc::new(parking_lot::Mutex::new(
            options
                .settings_manager
                .take()
                .unwrap_or_else(|| SettingsManager::create(&cwd, &agent_dir, Default::default())),
        ));
        let event_bus = options.event_bus.take().unwrap_or_else(create_event_bus);
        let (factories, builtins) =
            split_inline_extensions(std::mem::take(&mut options.extension_factories));
        let builtin_names: Vec<String> = builtins.iter().map(|b| b.name.clone()).collect();
        let package_manager = DefaultPackageManager::new(PackageManagerOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            settings_manager: Arc::clone(&settings_manager),
            builtin_extensions: builtin_names,
        });
        Self {
            additional_extension_paths: std::mem::take(&mut options.additional_extension_paths),
            additional_skill_paths: std::mem::take(&mut options.additional_skill_paths),
            additional_prompt_template_paths: std::mem::take(
                &mut options.additional_prompt_template_paths,
            ),
            additional_theme_paths: std::mem::take(&mut options.additional_theme_paths),
            system_prompt_source: options.system_prompt.take(),
            append_system_prompt_source: options.append_system_prompt.take(),
            extensions_override: options.extensions_override.take(),
            skills_override: options.skills_override.take(),
            prompts_override: options.prompts_override.take(),
            themes_override: options.themes_override.take(),
            agents_files_override: options.agents_files_override.take(),
            system_prompt_override: options.system_prompt_override.take(),
            append_system_prompt_override: options.append_system_prompt_override.take(),
            cwd,
            agent_dir,
            settings_manager,
            event_bus,
            package_manager,
            extension_factories: factories,
            builtin_extensions: builtins.into_iter().map(|b| (b.name.clone(), b)).collect(),
            no_extensions: options.no_extensions,
            disabled_builtin_extensions: options.disabled_builtin_extensions.into_iter().collect(),
            no_skills: options.no_skills,
            no_prompt_templates: options.no_prompt_templates,
            no_themes: options.no_themes,
            no_context_files: options.no_context_files,
            extensions_result: LoadExtensionsResult {
                extensions: Vec::new(),
                errors: Vec::new(),
                warnings: None,
                runtime: create_extension_runtime(),
            },
            skills: Vec::new(),
            skill_diagnostics: Vec::new(),
            prompts: Vec::new(),
            prompt_diagnostics: Vec::new(),
            themes: Vec::new(),
            theme_diagnostics: Vec::new(),
            agents_files: Vec::new(),
            system_prompt: None,
            system_prompt_source_path: None,
            append_system_prompt: Vec::new(),
            append_system_prompt_source_paths: Vec::new(),
            last_skill_paths: Vec::new(),
            extension_skill_source_infos: HashMap::new(),
            extension_prompt_source_infos: HashMap::new(),
            extension_theme_source_infos: HashMap::new(),
            resource_metadata_by_path: HashMap::new(),
            last_prompt_paths: Vec::new(),
            last_theme_paths: Vec::new(),
            loaded: false,
        }
    }

    pub fn get_extensions(&self) -> &LoadExtensionsResult {
        &self.extensions_result
    }

    /// Shared settings-manager handle (upstream: callers pass the same
    /// `SettingsManager` object around; the loader owns the shared instance,
    /// so sdk/services consumers take the handle from here).
    pub fn settings_manager(&self) -> Arc<parking_lot::Mutex<SettingsManager>> {
        Arc::clone(&self.settings_manager)
    }

    pub fn get_skills(&self) -> SkillsResult {
        SkillsResult {
            items: self.skills.clone(),
            diagnostics: self.skill_diagnostics.clone(),
        }
    }

    pub fn get_prompts(&self) -> PromptsResult {
        PromptsResult {
            items: self.prompts.clone(),
            diagnostics: self.prompt_diagnostics.clone(),
        }
    }

    pub fn get_themes(&self) -> ThemesResult {
        ThemesResult {
            items: self.themes.clone(),
            diagnostics: self.theme_diagnostics.clone(),
        }
    }

    pub fn get_agents_files(&self) -> Vec<ContextFile> {
        self.agents_files.clone()
    }

    pub fn get_system_prompt(&self) -> Option<String> {
        self.system_prompt.clone()
    }

    pub fn get_system_prompt_source(&self) -> Option<ContextFile> {
        self.system_prompt_source_path
            .as_ref()
            .map(|path| ContextFile {
                path: path.clone(),
                content: String::new(),
            })
    }

    pub fn get_append_system_prompt(&self) -> Vec<String> {
        self.append_system_prompt.clone()
    }

    pub fn get_append_system_prompt_sources(&self) -> Vec<ContextFile> {
        self.append_system_prompt_source_paths
            .iter()
            .map(|path| ContextFile {
                path: path.clone(),
                content: String::new(),
            })
            .collect()
    }

    /// Add extension-supplied resource paths and reload the affected resource
    /// kinds (upstream `extendResources`).
    pub fn extend_resources(&mut self, paths: ResourceExtensionPaths) {
        let skill_paths = self.normalize_extension_paths(paths.skill_paths);
        let prompt_paths = self.normalize_extension_paths(paths.prompt_paths);
        let theme_paths = self.normalize_extension_paths(paths.theme_paths);

        for entry in &skill_paths {
            self.extension_skill_source_infos.insert(
                entry.path.clone(),
                create_source_info(&entry.path, &entry.metadata),
            );
        }
        for entry in &prompt_paths {
            self.extension_prompt_source_infos.insert(
                entry.path.clone(),
                create_source_info(&entry.path, &entry.metadata),
            );
        }
        for entry in &theme_paths {
            self.extension_theme_source_infos.insert(
                entry.path.clone(),
                create_source_info(&entry.path, &entry.metadata),
            );
        }

        if !skill_paths.is_empty() {
            let paths: Vec<String> = skill_paths.into_iter().map(|entry| entry.path).collect();
            self.last_skill_paths = self.merge_paths(&self.last_skill_paths.clone(), &paths);
            let metadata = self.resource_metadata_by_path.clone();
            self.update_skills_from_paths(&self.last_skill_paths.clone(), Some(&metadata));
        }

        if !prompt_paths.is_empty() {
            let paths: Vec<String> = prompt_paths.into_iter().map(|entry| entry.path).collect();
            self.last_prompt_paths = self.merge_paths(&self.last_prompt_paths.clone(), &paths);
            let metadata = self.resource_metadata_by_path.clone();
            self.update_prompts_from_paths(&self.last_prompt_paths.clone(), Some(&metadata));
        }

        if !theme_paths.is_empty() {
            let paths: Vec<String> = theme_paths.into_iter().map(|entry| entry.path).collect();
            self.last_theme_paths = self.merge_paths(&self.last_theme_paths.clone(), &paths);
            let metadata = self.resource_metadata_by_path.clone();
            self.update_themes_from_paths(&self.last_theme_paths.clone(), Some(&metadata));
        }
    }

    /// Load extensions with project trust forced off — the bootstrap pass used
    /// to resolve trust (upstream `loadProjectTrustExtensions`).
    pub fn load_project_trust_extensions(&mut self) -> LoadExtensionsResult {
        // Force untrusted project settings for the bootstrap pass. This keeps
        // project-local extensions/packages out while still loading
        // user/global and temporary CLI extensions.
        self.settings_manager.lock().set_project_trusted(false);
        self.settings_manager.lock().reload();
        self.load_current_extension_set(true)
    }

    /// Full reload of every resource kind (upstream `reload`).
    pub fn reload(&mut self, options: ResourceLoaderReloadOptions) {
        reset_timings("extensions");

        if self.loaded {
            // Serialize against the loader module's cache-reading tests (the
            // gate is uncontended in production; tests use it to keep the
            // global-cache assertions deterministic).
            #[cfg(test)]
            let _cache_gate = crate::core::extensions::loader::cache_test_gate();
            clear_extension_cache();
        }

        let mut pre_trust_extensions: Option<LoadExtensionsResult> = None;
        if let Some(resolve_project_trust) = options.resolve_project_trust {
            let pre_trust = self.load_project_trust_extensions();
            let project_trusted = resolve_project_trust(&pre_trust);
            self.settings_manager
                .lock()
                .set_project_trusted(project_trusted);
            pre_trust_extensions = Some(pre_trust);
        }

        // reload() preserves SettingsManager.projectTrusted and reloads
        // settings for that trust state.
        self.settings_manager.lock().reload();
        let resolved_paths: ResolvedPaths =
            block_on(self.package_manager.resolve(None)).expect("package resolution failed");
        let cli_extension_paths = block_on(self.package_manager.resolve_extension_sources(
            &self.additional_extension_paths.clone(),
            false,
            true,
        ))
        .expect("CLI extension source resolution failed");
        // Kept on the instance so post-reload passes (extendResources) can
        // still resolve package metadata.
        self.resource_metadata_by_path = HashMap::new();

        self.extension_skill_source_infos = HashMap::new();
        self.extension_prompt_source_infos = HashMap::new();
        self.extension_theme_source_infos = HashMap::new();

        let mut metadata_by_path = HashMap::new();
        let mut store_metadata =
            |resources: &[ResolvedResource], map: &mut HashMap<String, PathMetadata>| {
                for r in resources {
                    map.entry(r.path.clone())
                        .or_insert_with(|| r.metadata.clone());
                }
            };
        store_metadata(&resolved_paths.extensions, &mut metadata_by_path);
        store_metadata(&resolved_paths.skills, &mut metadata_by_path);
        store_metadata(&resolved_paths.prompts, &mut metadata_by_path);
        store_metadata(&resolved_paths.themes, &mut metadata_by_path);

        let enabled = |resources: &[ResolvedResource]| -> Vec<ResolvedResource> {
            resources.iter().filter(|r| r.enabled).cloned().collect()
        };
        let enabled_extension_paths: Vec<String> = enabled(&resolved_paths.extensions)
            .into_iter()
            .map(|r| r.path)
            .collect();
        let enabled_skill_resources = enabled(&resolved_paths.skills);
        let enabled_prompt_paths: Vec<String> = enabled(&resolved_paths.prompts)
            .into_iter()
            .map(|r| r.path)
            .collect();
        let enabled_theme_paths: Vec<String> = enabled(&resolved_paths.themes)
            .into_iter()
            .map(|r| r.path)
            .collect();

        let enabled_skills: Vec<String> = enabled_skill_resources
            .iter()
            .map(|resource| self.map_skill_path(resource, &mut metadata_by_path))
            .collect();

        // Add CLI paths metadata.
        for r in cli_extension_paths
            .extensions
            .iter()
            .chain(cli_extension_paths.skills.iter())
        {
            metadata_by_path
                .entry(r.path.clone())
                .or_insert_with(|| PathMetadata {
                    source: "cli".to_string(),
                    scope: SourceScope::Temporary,
                    origin: SourceOrigin::TopLevel,
                    base_dir: None,
                    package_root: None,
                });
        }

        let cli_enabled_extensions: Vec<String> = cli_extension_paths
            .extensions
            .iter()
            .filter(|r| r.enabled)
            .map(|r| r.path.clone())
            .collect();
        let cli_enabled_skills: Vec<String> = cli_extension_paths
            .skills
            .iter()
            .filter(|r| r.enabled)
            .map(|r| r.path.clone())
            .collect();
        let cli_enabled_prompts: Vec<String> = cli_extension_paths
            .prompts
            .iter()
            .filter(|r| r.enabled)
            .map(|r| r.path.clone())
            .collect();
        let cli_enabled_themes: Vec<String> = cli_extension_paths
            .themes
            .iter()
            .filter(|r| r.enabled)
            .map(|r| r.path.clone())
            .collect();

        let extension_paths = if self.no_extensions {
            cli_enabled_extensions
        } else {
            self.merge_paths(&cli_enabled_extensions, &enabled_extension_paths)
        }
        .into_iter()
        .filter(|path| {
            !path.starts_with(BUILTIN_PATH_PREFIX)
                || !self
                    .disabled_builtin_extensions
                    .contains(&path[BUILTIN_PATH_PREFIX.len()..])
        })
        .collect::<Vec<String>>();

        let package_warnings =
            collect_extension_package_warnings(&extension_paths, &metadata_by_path);
        let mut extensions_result =
            self.load_final_extension_set(&extension_paths, pre_trust_extensions);
        merge_extension_warnings(&mut extensions_result, package_warnings);
        for p in &self.additional_extension_paths {
            if is_local_path(p) {
                let resolved = self.resolve_resource_path(p);
                if !Path::new(&resolved).exists() {
                    extensions_result.errors.push(LoadRecord {
                        path: resolved.clone(),
                        error: format!("Extension path does not exist: {resolved}"),
                    });
                }
            }
        }
        self.extensions_result = match self.extensions_override.as_deref() {
            Some(over) => over(extensions_result),
            None => extensions_result,
        };
        let mut extensions = std::mem::take(&mut self.extensions_result.extensions);
        self.apply_extension_source_info(&mut extensions, &metadata_by_path);
        self.extensions_result.extensions = extensions;

        let skill_paths = if self.no_skills {
            self.merge_paths(&cli_enabled_skills, &self.additional_skill_paths.clone())
        } else {
            let mut combined = cli_enabled_skills.clone();
            combined.extend(enabled_skills.clone());
            self.merge_paths(&combined, &self.additional_skill_paths.clone())
        };

        self.last_skill_paths = skill_paths;
        self.resource_metadata_by_path = metadata_by_path.clone();
        let metadata = metadata_by_path.clone();
        self.update_skills_from_paths(&self.last_skill_paths.clone(), Some(&metadata));
        for p in &self.additional_skill_paths {
            if is_local_path(p) {
                let resolved = self.resolve_resource_path(p);
                if !Path::new(&resolved).exists()
                    && !self
                        .skill_diagnostics
                        .iter()
                        .any(|d| d.path.as_deref() == Some(resolved.as_str()))
                {
                    self.skill_diagnostics.push(ResourceDiagnostic {
                        r#type: crate::core::diagnostics::DiagnosticType::Error,
                        message: "Skill path does not exist".to_string(),
                        path: Some(resolved),
                        collision: None,
                    });
                }
            }
        }

        let prompt_paths = if self.no_prompt_templates {
            self.merge_paths(
                &cli_enabled_prompts,
                &self.additional_prompt_template_paths.clone(),
            )
        } else {
            let mut combined = cli_enabled_prompts.clone();
            combined.extend(enabled_prompt_paths.clone());
            self.merge_paths(&combined, &self.additional_prompt_template_paths.clone())
        };

        self.last_prompt_paths = prompt_paths;
        self.update_prompts_from_paths(&self.last_prompt_paths.clone(), Some(&metadata));
        for p in &self.additional_prompt_template_paths {
            if is_local_path(p) {
                let resolved = self.resolve_resource_path(p);
                if !Path::new(&resolved).exists()
                    && !self
                        .prompt_diagnostics
                        .iter()
                        .any(|d| d.path.as_deref() == Some(resolved.as_str()))
                {
                    self.prompt_diagnostics.push(ResourceDiagnostic {
                        r#type: crate::core::diagnostics::DiagnosticType::Error,
                        message: "Prompt template path does not exist".to_string(),
                        path: Some(resolved),
                        collision: None,
                    });
                }
            }
        }

        let theme_paths = if self.no_themes {
            self.merge_paths(&cli_enabled_themes, &self.additional_theme_paths.clone())
        } else {
            let mut combined = cli_enabled_themes.clone();
            combined.extend(enabled_theme_paths.clone());
            self.merge_paths(&combined, &self.additional_theme_paths.clone())
        };

        self.last_theme_paths = theme_paths;
        self.update_themes_from_paths(&self.last_theme_paths.clone(), Some(&metadata));
        for p in &self.additional_theme_paths {
            let resolved = self.resolve_resource_path(p);
            if !Path::new(&resolved).exists()
                && !self
                    .theme_diagnostics
                    .iter()
                    .any(|d| d.path.as_deref() == Some(resolved.as_str()))
            {
                self.theme_diagnostics.push(ResourceDiagnostic {
                    r#type: crate::core::diagnostics::DiagnosticType::Error,
                    message: "Theme path does not exist".to_string(),
                    path: Some(resolved),
                    collision: None,
                });
            }
        }

        let agents_files = if self.no_context_files {
            Vec::new()
        } else {
            load_project_context_files(&self.cwd, &self.agent_dir)
        };
        self.agents_files = match self.agents_files_override.as_deref() {
            Some(over) => over(agents_files),
            None => agents_files,
        };

        let system_prompt_source = self
            .system_prompt_source
            .clone()
            .or_else(|| self.discover_system_prompt_file());
        let base_system_prompt = system_prompt_source
            .as_deref()
            .and_then(|s| resolve_prompt_input(s, "system prompt"));
        self.system_prompt = match self.system_prompt_override.as_deref() {
            Some(over) => over(base_system_prompt),
            None => base_system_prompt,
        };
        self.system_prompt_source_path = system_prompt_source
            .as_deref()
            .filter(|s| Path::new(s).exists())
            .map(resolve_path_cwd);

        let append_sources = match self.append_system_prompt_source.clone() {
            Some(sources) => sources,
            None => self
                .discover_append_system_prompt_file()
                .into_iter()
                .collect(),
        };
        let base_append: Vec<String> = append_sources
            .iter()
            .filter_map(|s| resolve_prompt_input(s, "append system prompt"))
            .collect();
        self.append_system_prompt = match self.append_system_prompt_override.as_deref() {
            Some(over) => over(base_append),
            None => base_append,
        };
        self.append_system_prompt_source_paths = append_sources
            .iter()
            .filter(|source| Path::new(source).exists())
            .map(|source| resolve_path_cwd(source))
            .collect();
        self.loaded = true;
    }

    /// Extension set for the trust-resolution bootstrap pass
    /// (upstream `loadCurrentExtensionSet`).
    fn load_current_extension_set(
        &mut self,
        include_inline_factories: bool,
    ) -> LoadExtensionsResult {
        let resolved_paths: ResolvedPaths =
            block_on(self.package_manager.resolve(None)).expect("package resolution failed");
        let cli_extension_paths = block_on(self.package_manager.resolve_extension_sources(
            &self.additional_extension_paths.clone(),
            false,
            true,
        ))
        .expect("CLI extension source resolution failed");
        let enabled_extensions: Vec<String> = resolved_paths
            .extensions
            .iter()
            .filter(|r| r.enabled)
            .map(|r| r.path.clone())
            .collect();
        let cli_enabled_extensions: Vec<String> = cli_extension_paths
            .extensions
            .iter()
            .filter(|r| r.enabled)
            .map(|r| r.path.clone())
            .collect();
        // Built-in extensions wait for the final pass: project settings can
        // disable them, and a loaded extension cannot be unloaded.
        let merged = if self.no_extensions {
            cli_enabled_extensions
        } else {
            self.merge_paths(&cli_enabled_extensions, &enabled_extensions)
        };
        let extension_paths: Vec<String> = merged
            .into_iter()
            .filter(|path| !path.starts_with(BUILTIN_PATH_PREFIX))
            .collect();
        let mut metadata_by_path = HashMap::new();
        for resource in resolved_paths
            .extensions
            .iter()
            .chain(cli_extension_paths.extensions.iter())
        {
            metadata_by_path.insert(resource.path.clone(), resource.metadata.clone());
        }
        let package_warnings =
            collect_extension_package_warnings(&extension_paths, &metadata_by_path);
        let mut extensions_result = block_on(load_extensions_cached(
            &extension_paths,
            &self.cwd,
            Arc::new(self.event_bus.clone()) as Arc<dyn crate::core::extensions::types::EventBus>,
            None,
        ));
        merge_extension_warnings(&mut extensions_result, package_warnings);
        if !include_inline_factories {
            return extensions_result;
        }

        let inline_extensions = self.load_extension_factories(&extensions_result.runtime);
        extensions_result
            .extensions
            .extend(inline_extensions.extensions);
        extensions_result.errors.extend(inline_extensions.errors);
        let mut replacement_warnings: Vec<LoadRecord> = Vec::new();
        let extensions = omit_replaced_extensions(
            std::mem::take(&mut extensions_result.extensions),
            &mut replacement_warnings,
        );
        extensions_result.extensions = extensions;
        merge_extension_warnings(&mut extensions_result, replacement_warnings);
        extensions_result
    }

    fn resolve_extension_load_path(&self, path: &str) -> String {
        if is_synthetic_path(path) {
            path.to_string()
        } else if let Some(expanded) = file_url_to_path(path) {
            expanded
        } else {
            resolve_path(path, &self.cwd)
        }
    }

    /// Load extension paths: files from disk and `builtin:<name>` paths from
    /// the built-in extensions (upstream `loadExtensionPaths`).
    fn load_extension_paths(
        &mut self,
        paths: &[String],
        runtime: Option<&ExtensionRuntime>,
    ) -> LoadExtensionsResult {
        let is_builtin_path = |path: &str| path.starts_with(BUILTIN_PATH_PREFIX);
        let file_paths: Vec<String> = paths
            .iter()
            .filter(|p| !is_builtin_path(p))
            .cloned()
            .collect();
        let mut result = block_on(load_extensions_cached(
            &file_paths,
            &self.cwd,
            Arc::new(self.event_bus.clone()) as Arc<dyn crate::core::extensions::types::EventBus>,
            runtime.cloned(),
        ));
        for path in paths.iter().filter(|p| is_builtin_path(p)) {
            let name = path.strip_prefix(BUILTIN_PATH_PREFIX).unwrap();
            let Some(builtin) = self.builtin_extensions.get(name) else {
                result.errors.push(LoadRecord {
                    path: path.clone(),
                    error: format!("Unknown built-in extension: {path}"),
                });
                continue;
            };
            let factory = Arc::clone(&builtin.factory);
            let replaceable = builtin.replaceable;
            let cwd = self.cwd.clone();
            match block_on(load_extension_from_factory(
                factory,
                &cwd,
                Arc::new(self.event_bus.clone())
                    as Arc<dyn crate::core::extensions::types::EventBus>,
                runtime.cloned(),
                path,
            )) {
                Ok(mut extension) => {
                    extension.hidden = Some(true);
                    extension.replaceable = Some(replaceable);
                    result.extensions.push(extension);
                }
                Err(error) => result.errors.push(LoadRecord {
                    path: path.clone(),
                    error,
                }),
            }
        }
        result
    }

    /// Final extension pass: reuse pre-trust loads, load the rest, then order
    /// by configured path and apply replaceable/conflict rules
    /// (upstream `loadFinalExtensionSet`).
    fn load_final_extension_set(
        &mut self,
        extension_paths: &[String],
        pre_trust_extensions: Option<LoadExtensionsResult>,
    ) -> LoadExtensionsResult {
        // Without a pre-trust pass nothing is preloaded, and inline extensions
        // load here.
        let preloaded = pre_trust_extensions
            .as_ref()
            .map(|r| r.extensions.clone())
            .unwrap_or_default();
        let mut preloaded_by_path: HashMap<String, Extension> = HashMap::new();
        for extension in &preloaded {
            if !extension.path.starts_with("<inline:") {
                preloaded_by_path.insert(extension.resolved_path.clone(), extension.clone());
            }
        }
        let failed_preload_paths: HashSet<String> = pre_trust_extensions
            .as_ref()
            .map(|r| {
                r.errors
                    .iter()
                    .map(|e| self.resolve_extension_load_path(&e.path))
                    .collect()
            })
            .unwrap_or_default();
        let remaining_paths: Vec<String> = extension_paths
            .iter()
            .filter(|path| {
                let resolved_path = self.resolve_extension_load_path(path);
                !preloaded_by_path.contains_key(&resolved_path)
                    && !failed_preload_paths.contains(&resolved_path)
            })
            .cloned()
            .collect();
        let remaining_extensions = self.load_extension_paths(
            &remaining_paths,
            pre_trust_extensions.as_ref().map(|r| &r.runtime),
        );
        let mut loaded_by_path = preloaded_by_path;
        for extension in &remaining_extensions.extensions {
            loaded_by_path.insert(extension.resolved_path.clone(), extension.clone());
        }

        let inline_extensions = match &pre_trust_extensions {
            Some(pre) => LoadExtensionsResult {
                extensions: pre
                    .extensions
                    .iter()
                    .filter(|e| e.path.starts_with("<inline:"))
                    .cloned()
                    .collect(),
                errors: Vec::new(),
                warnings: None,
                runtime: create_extension_runtime(),
            },
            None => self.load_extension_factories(&remaining_extensions.runtime),
        };
        let mut ordered_extensions: Vec<Extension> = extension_paths
            .iter()
            .filter_map(|path| {
                loaded_by_path
                    .get(&self.resolve_extension_load_path(path))
                    .cloned()
            })
            .collect();
        ordered_extensions.extend(inline_extensions.extensions);

        let mut replacement_warnings: Vec<LoadRecord> = Vec::new();
        let extensions = omit_replaced_extensions(ordered_extensions, &mut replacement_warnings);
        let mut extensions_result = LoadExtensionsResult {
            extensions,
            errors: pre_trust_extensions
                .as_ref()
                .map(|r| r.errors.clone())
                .unwrap_or_default()
                .into_iter()
                .chain(remaining_extensions.errors.iter().cloned())
                .chain(inline_extensions.errors.iter().cloned())
                .collect(),
            warnings: Some(
                pre_trust_extensions
                    .as_ref()
                    .and_then(|r| r.warnings.clone())
                    .unwrap_or_default()
                    .into_iter()
                    .chain(remaining_extensions.warnings.clone().unwrap_or_default())
                    .collect(),
            ),
            runtime: remaining_extensions.runtime,
        };
        merge_extension_warnings(&mut extensions_result, replacement_warnings);
        self.add_extension_conflict_diagnostics(&mut extensions_result);
        extensions_result
    }

    /// Detect extension conflicts (tools/flags with the same name from
    /// different extensions) and report them as error records
    /// (upstream `addExtensionConflictDiagnostics` + `detectExtensionConflicts`).
    fn add_extension_conflict_diagnostics(&mut self, extensions_result: &mut LoadExtensionsResult) {
        let mut tool_owners: HashMap<String, String> = HashMap::new();
        let mut flag_owners: HashMap<String, String> = HashMap::new();
        let mut conflicts: Vec<(String, String)> = Vec::new();
        for ext in &extensions_result.extensions {
            for tool_name in ext.tools.keys() {
                match tool_owners.get(tool_name) {
                    Some(existing) if existing != &ext.path => {
                        conflicts.push((
                            ext.path.clone(),
                            format!("Tool \"{tool_name}\" conflicts with {existing}"),
                        ));
                    }
                    _ => {
                        tool_owners.insert(tool_name.clone(), ext.path.clone());
                    }
                }
            }
            for flag_name in ext.flags.keys() {
                match flag_owners.get(flag_name) {
                    Some(existing) if existing != &ext.path => {
                        conflicts.push((
                            ext.path.clone(),
                            format!("Flag \"--{flag_name}\" conflicts with {existing}"),
                        ));
                    }
                    _ => {
                        flag_owners.insert(flag_name.clone(), ext.path.clone());
                    }
                }
            }
        }
        for (path, message) in conflicts {
            extensions_result.errors.push(LoadRecord {
                path,
                error: message,
            });
        }
    }

    /// Resolve a skill resource to its SKILL.md when auto-discovered as a
    /// directory of an auto/package source (upstream `mapSkillPath`).
    fn map_skill_path(
        &self,
        resource: &ResolvedResource,
        metadata_by_path: &mut HashMap<String, PathMetadata>,
    ) -> String {
        if resource.metadata.source != "auto" && resource.metadata.origin != SourceOrigin::Package {
            return resource.path.clone();
        }
        let Ok(stats) = std::fs::metadata(&resource.path) else {
            return resource.path.clone();
        };
        if !stats.is_dir() {
            return resource.path.clone();
        }
        let skill_file = Path::new(&resource.path).join("SKILL.md");
        if skill_file.exists() {
            metadata_by_path
                .entry(skill_file.to_string_lossy().to_string())
                .or_insert_with(|| resource.metadata.clone());
            return skill_file.to_string_lossy().to_string();
        }
        resource.path.clone()
    }

    fn normalize_extension_paths(&self, entries: Vec<ResourcePathEntry>) -> Vec<ResourcePathEntry> {
        entries
            .into_iter()
            .map(|entry| ResourcePathEntry {
                metadata: match &entry.metadata.base_dir {
                    Some(base_dir) => {
                        let mut metadata = entry.metadata.clone();
                        metadata.base_dir = Some(self.resolve_resource_path(base_dir));
                        metadata
                    }
                    None => entry.metadata,
                },
                path: self.resolve_resource_path(&entry.path),
            })
            .collect()
    }

    fn update_skills_from_paths(
        &mut self,
        skill_paths: &[String],
        metadata_by_path: Option<&HashMap<String, PathMetadata>>,
    ) {
        let skills_result = if self.no_skills && skill_paths.is_empty() {
            SkillsResult {
                items: Vec::new(),
                diagnostics: Vec::new(),
            }
        } else {
            let loaded = load_skills(LoadSkillsOptions {
                cwd: &self.cwd,
                agent_dir: Some(&self.agent_dir),
                skill_paths,
                include_defaults: false,
            });
            SkillsResult {
                items: loaded.skills,
                diagnostics: loaded.diagnostics,
            }
        };
        let resolved_skills = match self.skills_override.as_deref() {
            Some(over) => over(skills_result),
            None => skills_result,
        };
        self.skills = resolved_skills
            .items
            .into_iter()
            .map(|mut skill| {
                let source_info = self
                    .find_source_info_for_path(
                        &skill.file_path,
                        Some(&self.extension_skill_source_infos),
                        metadata_by_path,
                    )
                    .or(Some(skill.source_info.clone()))
                    .or_else(|| Some(self.get_default_source_info_for_path(&skill.file_path)));
                if let Some(info) = source_info {
                    skill.source_info = info;
                }
                skill
            })
            .collect();
        self.skill_diagnostics = resolved_skills.diagnostics;
    }

    fn update_prompts_from_paths(
        &mut self,
        prompt_paths: &[String],
        metadata_by_path: Option<&HashMap<String, PathMetadata>>,
    ) {
        let prompts_result = if self.no_prompt_templates && prompt_paths.is_empty() {
            PromptsResult {
                items: Vec::new(),
                diagnostics: Vec::new(),
            }
        } else {
            let loaded = load_prompt_templates(&LoadPromptTemplatesOptions {
                cwd: self.cwd.clone(),
                agent_dir: self.agent_dir.clone(),
                prompt_paths: prompt_paths.to_vec(),
                include_defaults: false,
            });
            let deduped = Self::dedupe_prompts(loaded.templates);
            PromptsResult {
                items: deduped.0,
                diagnostics: loaded.diagnostics.into_iter().chain(deduped.1).collect(),
            }
        };
        let resolved_prompts = match self.prompts_override.as_deref() {
            Some(over) => over(prompts_result),
            None => prompts_result,
        };
        self.prompts = resolved_prompts
            .items
            .into_iter()
            .map(|mut prompt| {
                let source_info = self
                    .find_source_info_for_path(
                        &prompt.file_path,
                        Some(&self.extension_prompt_source_infos),
                        metadata_by_path,
                    )
                    .or(Some(prompt.source_info.clone()))
                    .or_else(|| Some(self.get_default_source_info_for_path(&prompt.file_path)));
                if let Some(info) = source_info {
                    prompt.source_info = info;
                }
                prompt
            })
            .collect();
        self.prompt_diagnostics = resolved_prompts.diagnostics;
    }

    fn update_themes_from_paths(
        &mut self,
        theme_paths: &[String],
        metadata_by_path: Option<&HashMap<String, PathMetadata>>,
    ) {
        let themes_result = if self.no_themes && theme_paths.is_empty() {
            ThemesResult {
                items: Vec::new(),
                diagnostics: Vec::new(),
            }
        } else {
            // Theme construction only needs trueColor, so skip the unrelated
            // tmux hyperlink probe.
            let overrides = self
                .settings_manager
                .lock()
                .get_terminal_capability_overrides();
            let mut capabilities = crate::utils::tui_compat::detect_capabilities();
            capabilities.hyperlinks = false;
            if let Some(true_color) = overrides.true_color {
                capabilities.true_color = true_color;
            }
            let color_mode = crate::utils::tui_compat::get_terminal_color_mode(capabilities);
            let loaded = self.load_themes(theme_paths, false, color_mode);
            let deduped = Self::dedupe_themes(loaded.items);
            ThemesResult {
                items: deduped.0,
                diagnostics: loaded.diagnostics.into_iter().chain(deduped.1).collect(),
            }
        };
        let resolved_themes = match self.themes_override.as_deref() {
            Some(over) => over(themes_result),
            None => themes_result,
        };
        self.themes = resolved_themes
            .items
            .into_iter()
            .map(|mut theme| {
                if let Some(source_path) = theme.source_path.clone() {
                    theme.source_info = self
                        .find_source_info_for_path(
                            &source_path,
                            Some(&self.extension_theme_source_infos),
                            metadata_by_path,
                        )
                        .or(theme.source_info.take())
                        .or_else(|| Some(self.get_default_source_info_for_path(&source_path)));
                }
                theme
            })
            .collect();
        self.theme_diagnostics = resolved_themes.diagnostics;
    }

    fn apply_extension_source_info(
        &self,
        extensions: &mut [Extension],
        metadata_by_path: &HashMap<String, PathMetadata>,
    ) {
        for extension in extensions.iter_mut() {
            let source_info = self
                .find_source_info_for_path(&extension.path, None, Some(metadata_by_path))
                .unwrap_or_else(|| self.get_default_source_info_for_path(&extension.path));
            for command in extension.commands.values_mut() {
                command.source_info = source_info.clone();
            }
            for tool in extension.tools.values_mut() {
                tool.source_info = source_info.clone();
            }
        }
    }

    fn find_source_info_for_path(
        &self,
        resource_path: &str,
        extra_source_infos: Option<&HashMap<String, SourceInfo>>,
        metadata_by_path: Option<&HashMap<String, PathMetadata>>,
    ) -> Option<SourceInfo> {
        if resource_path.is_empty() {
            return None;
        }

        if is_synthetic_path(resource_path) {
            return Some(self.get_default_source_info_for_path(resource_path));
        }

        let normalized_resource_path = path_resolve(resource_path);
        if let Some(extra_source_infos) = extra_source_infos {
            for (source_path, source_info) in extra_source_infos {
                let normalized_source_path = path_resolve(source_path);
                if normalized_resource_path == normalized_source_path
                    || normalized_resource_path.starts_with(&format!(
                        "{normalized_source_path}{}",
                        std::path::MAIN_SEPARATOR
                    ))
                {
                    let mut info = source_info.clone();
                    info.path = resource_path.to_string();
                    return Some(info);
                }
            }
        }

        if let Some(metadata_by_path) = metadata_by_path {
            if let Some(exact) = metadata_by_path
                .get(&normalized_resource_path)
                .or_else(|| metadata_by_path.get(resource_path))
            {
                return Some(create_source_info(resource_path, exact));
            }

            for (source_path, metadata) in metadata_by_path {
                let normalized_source_path = path_resolve(source_path);
                if normalized_resource_path == normalized_source_path
                    || normalized_resource_path.starts_with(&format!(
                        "{normalized_source_path}{}",
                        std::path::MAIN_SEPARATOR
                    ))
                {
                    return Some(create_source_info(resource_path, metadata));
                }
            }
        }

        None
    }

    fn get_default_source_info_for_path(&self, file_path: &str) -> SourceInfo {
        if let Some(synthetic_source) = get_synthetic_path_source(file_path) {
            return SourceInfo {
                path: file_path.to_string(),
                source: synthetic_source,
                scope: SourceScope::Temporary,
                origin: SourceOrigin::TopLevel,
                base_dir: None,
            };
        }

        let normalized_path = path_resolve(file_path);
        let join_all = |root: &str, names: &[&str]| -> Vec<String> {
            names
                .iter()
                .map(|name| Path::new(root).join(name).to_string_lossy().to_string())
                .collect()
        };
        let agent_roots = join_all(
            &self.agent_dir,
            &["skills", "prompts", "themes", "extensions"],
        );
        let project_roots = join_all(
            &self.cwd,
            &[
                &format!("{CONFIG_DIR_NAME}/skills"),
                &format!("{CONFIG_DIR_NAME}/prompts"),
                &format!("{CONFIG_DIR_NAME}/themes"),
                &format!("{CONFIG_DIR_NAME}/extensions"),
            ],
        );

        for root in &agent_roots {
            if self.is_under_path(&normalized_path, root) {
                return SourceInfo {
                    path: file_path.to_string(),
                    source: "local".to_string(),
                    scope: SourceScope::User,
                    origin: SourceOrigin::TopLevel,
                    base_dir: Some(root.clone()),
                };
            }
        }

        for root in &project_roots {
            if self.is_under_path(&normalized_path, root) {
                return SourceInfo {
                    path: file_path.to_string(),
                    source: "local".to_string(),
                    scope: SourceScope::Project,
                    origin: SourceOrigin::TopLevel,
                    base_dir: Some(root.clone()),
                };
            }
        }

        let base_dir = match std::fs::metadata(&normalized_path) {
            Ok(stats) if stats.is_dir() => normalized_path.clone(),
            _ => Path::new(&normalized_path)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or(normalized_path.clone()),
        };
        SourceInfo {
            path: file_path.to_string(),
            source: "local".to_string(),
            scope: SourceScope::Temporary,
            origin: SourceOrigin::TopLevel,
            base_dir: Some(base_dir),
        }
    }

    fn merge_paths(&self, primary: &[String], additional: &[String]) -> Vec<String> {
        let mut merged: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for p in primary.iter().chain(additional.iter()) {
            let resolved = self.resolve_resource_path(p);
            let canonical_path = canonicalize_path(&resolved);
            if !seen.insert(canonical_path) {
                continue;
            }
            merged.push(resolved);
        }
        merged
    }

    fn resolve_resource_path(&self, p: &str) -> String {
        if is_synthetic_path(p) {
            p.to_string()
        } else if let Some(expanded) = file_url_to_path(p) {
            expanded
        } else {
            resolve_path(p, &self.cwd)
        }
    }

    fn load_themes(
        &self,
        paths: &[String],
        include_defaults: bool,
        color_mode: TerminalColorMode,
    ) -> ThemesResult {
        let mut themes: Vec<Theme> = Vec::new();
        let mut diagnostics: Vec<ResourceDiagnostic> = Vec::new();
        if include_defaults {
            let default_dirs = [
                Path::new(&self.agent_dir).join("themes"),
                Path::new(&self.cwd).join(CONFIG_DIR_NAME).join("themes"),
            ];
            for dir in &default_dirs {
                load_themes_from_dir(
                    &dir.to_string_lossy(),
                    &mut themes,
                    &mut diagnostics,
                    color_mode,
                );
            }
        }

        for p in paths {
            let resolved = self.resolve_resource_path(p);
            if !Path::new(&resolved).exists() {
                diagnostics.push(ResourceDiagnostic {
                    r#type: crate::core::diagnostics::DiagnosticType::Warning,
                    message: "theme path does not exist".to_string(),
                    path: Some(resolved),
                    collision: None,
                });
                continue;
            }

            match std::fs::metadata(&resolved) {
                Ok(stats) if stats.is_dir() => {
                    load_themes_from_dir(&resolved, &mut themes, &mut diagnostics, color_mode)
                }
                Ok(stats) if stats.is_file() && resolved.ends_with(".json") => {
                    load_theme_from_file(&resolved, &mut themes, &mut diagnostics, color_mode)
                }
                Ok(_) => diagnostics.push(ResourceDiagnostic {
                    r#type: crate::core::diagnostics::DiagnosticType::Warning,
                    message: "theme path is not a json file".to_string(),
                    path: Some(resolved),
                    collision: None,
                }),
                Err(error) => diagnostics.push(ResourceDiagnostic {
                    r#type: crate::core::diagnostics::DiagnosticType::Warning,
                    message: error.to_string(),
                    path: Some(resolved),
                    collision: None,
                }),
            }
        }

        ThemesResult {
            items: themes,
            diagnostics,
        }
    }

    fn load_extension_factories(&self, runtime: &ExtensionRuntime) -> LoadExtensionsResult {
        let mut extensions: Vec<Extension> = Vec::new();
        let mut errors: Vec<LoadRecord> = Vec::new();

        for (index, input) in self.extension_factories.iter().enumerate() {
            let (is_named, factory, extension_path, hidden, replaceable) = match input {
                InlineExtension::Factory(factory) => (
                    false,
                    Arc::clone(factory),
                    format!("<inline:{}>", index + 1),
                    false,
                    false,
                ),
                InlineExtension::Named {
                    name,
                    factory,
                    hidden,
                    replaceable,
                    builtin: _,
                } => (
                    true,
                    Arc::clone(factory),
                    format!("<inline:{name}>"),
                    hidden.unwrap_or(false),
                    replaceable.unwrap_or(false),
                ),
            };
            match block_on(load_extension_from_factory(
                factory,
                &self.cwd,
                Arc::new(self.event_bus.clone())
                    as Arc<dyn crate::core::extensions::types::EventBus>,
                Some(runtime.clone()),
                &extension_path,
            )) {
                Ok(mut extension) => {
                    extension.hidden = Some(is_named && hidden);
                    extension.replaceable = Some(is_named && replaceable);
                    extensions.push(extension);
                }
                Err(error) => errors.push(LoadRecord {
                    path: extension_path,
                    error,
                }),
            }
        }

        LoadExtensionsResult {
            extensions,
            errors,
            warnings: None,
            runtime: create_extension_runtime(),
        }
    }

    fn dedupe_prompts(
        prompts: Vec<PromptTemplate>,
    ) -> (Vec<PromptTemplate>, Vec<ResourceDiagnostic>) {
        let mut seen: HashMap<String, PromptTemplate> = HashMap::new();
        let mut diagnostics: Vec<ResourceDiagnostic> = Vec::new();

        for prompt in prompts {
            if let Some(existing) = seen.get(&prompt.name) {
                diagnostics.push(ResourceDiagnostic {
                    r#type: crate::core::diagnostics::DiagnosticType::Collision,
                    message: format!("name \"/{}\" collision", prompt.name),
                    path: Some(prompt.file_path.clone()),
                    collision: Some(ResourceCollision {
                        resource_type: ResourceType::Prompt,
                        name: prompt.name.clone(),
                        winner_path: existing.file_path.clone(),
                        loser_path: prompt.file_path.clone(),
                        winner_source: None,
                        loser_source: None,
                    }),
                });
            } else {
                seen.insert(prompt.name.clone(), prompt);
            }
        }

        (seen.into_values().collect::<Vec<_>>(), diagnostics)
    }

    fn dedupe_themes(themes: Vec<Theme>) -> (Vec<Theme>, Vec<ResourceDiagnostic>) {
        let mut seen: HashMap<String, Theme> = HashMap::new();
        let mut diagnostics: Vec<ResourceDiagnostic> = Vec::new();

        for theme in themes {
            let name = theme.name.clone().unwrap_or_else(|| "unnamed".to_string());
            if let Some(existing) = seen.get(&name) {
                diagnostics.push(ResourceDiagnostic {
                    r#type: crate::core::diagnostics::DiagnosticType::Collision,
                    message: format!("name \"{name}\" collision"),
                    path: theme.source_path.clone(),
                    collision: Some(ResourceCollision {
                        resource_type: ResourceType::Theme,
                        name: name.clone(),
                        winner_path: existing
                            .source_path
                            .clone()
                            .unwrap_or_else(|| "<builtin>".to_string()),
                        loser_path: theme
                            .source_path
                            .clone()
                            .unwrap_or_else(|| "<builtin>".to_string()),
                        winner_source: None,
                        loser_source: None,
                    }),
                });
            } else {
                seen.insert(name, theme);
            }
        }

        (seen.into_values().collect(), diagnostics)
    }

    fn discover_system_prompt_file(&self) -> Option<String> {
        let project_path = Path::new(&self.cwd).join(CONFIG_DIR_NAME).join("SYSTEM.md");
        if self.settings_manager.lock().is_project_trusted() && project_path.exists() {
            return Some(project_path.to_string_lossy().to_string());
        }

        let global_path = Path::new(&self.agent_dir).join("SYSTEM.md");
        if global_path.exists() {
            return Some(global_path.to_string_lossy().to_string());
        }

        None
    }

    fn discover_append_system_prompt_file(&self) -> Option<String> {
        let project_path = Path::new(&self.cwd)
            .join(CONFIG_DIR_NAME)
            .join("APPEND_SYSTEM.md");
        if self.settings_manager.lock().is_project_trusted() && project_path.exists() {
            return Some(project_path.to_string_lossy().to_string());
        }

        let global_path = Path::new(&self.agent_dir).join("APPEND_SYSTEM.md");
        if global_path.exists() {
            return Some(global_path.to_string_lossy().to_string());
        }

        None
    }

    fn is_under_path(&self, target: &str, root: &str) -> bool {
        let normalized_root = path_resolve(root);
        if target == normalized_root {
            return true;
        }
        let prefix = format!("{normalized_root}{}", std::path::MAIN_SEPARATOR);
        target.starts_with(&prefix)
    }
}

/// Load every `.json` theme file in a directory
/// (upstream `loadThemesFromDir`, module-level helper).
fn load_themes_from_dir(
    dir: &str,
    themes: &mut Vec<Theme>,
    diagnostics: &mut Vec<ResourceDiagnostic>,
    color_mode: TerminalColorMode,
) {
    let dir_path = PathBuf::from(dir);
    if !dir_path.exists() {
        return;
    }

    let Ok(entries) = std::fs::read_dir(&dir_path) else {
        diagnostics.push(ResourceDiagnostic {
            r#type: crate::core::diagnostics::DiagnosticType::Warning,
            message: "failed to read theme directory".to_string(),
            path: Some(dir.to_string()),
            collision: None,
        });
        return;
    };
    for entry in entries.flatten() {
        let entry_path = entry.path();
        let mut is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
        if !is_file && entry.file_type().map(|t| t.is_symlink()).unwrap_or(false) {
            is_file = std::fs::metadata(&entry_path)
                .map(|m| m.is_file())
                .unwrap_or(false);
            if !is_file {
                continue;
            }
        }
        if !is_file {
            continue;
        }
        if !entry.file_name().to_string_lossy().ends_with(".json") {
            continue;
        }
        load_theme_from_file(
            &entry_path.to_string_lossy(),
            themes,
            diagnostics,
            color_mode,
        );
    }
}

/// Load one theme JSON file, recording warnings on failure
/// (upstream `loadThemeFromFile`).
fn load_theme_from_file(
    file_path: &str,
    themes: &mut Vec<Theme>,
    diagnostics: &mut Vec<ResourceDiagnostic>,
    color_mode: TerminalColorMode,
) {
    match load_theme_from_path(file_path, Some(color_mode)) {
        Ok(theme) => themes.push(theme),
        Err(error) => diagnostics.push(ResourceDiagnostic {
            r#type: crate::core::diagnostics::DiagnosticType::Warning,
            message: error,
            path: Some(file_path.to_string()),
            collision: None,
        }),
    }
}

/// `path.resolve` equivalent (absolute-ize against the process cwd).
fn path_resolve(path: &str) -> String {
    if Path::new(path).is_absolute() {
        path.to_string()
    } else {
        Path::new(&resolve_path_cwd("."))
            .join(path)
            .to_string_lossy()
            .to_string()
    }
}

/// `fileURLToPath` equivalent for `file://` URLs supplied as resource paths
/// (D58a): expands `file:///abs/path` (and `file://localhost/...`) with
/// percent-decoding; returns `None` for anything else.
fn file_url_to_path(url: &str) -> Option<String> {
    let rest = url.strip_prefix("file://")?;
    let rest = rest.strip_prefix("localhost").unwrap_or(rest);
    if !rest.starts_with('/') {
        return None;
    }
    let mut out = String::with_capacity(rest.len());
    let bytes = rest.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = &rest[index + 1..index + 3];
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                out.push(byte as char);
                index += 3;
                continue;
            }
        }
        out.push(rest[index..].chars().next().unwrap());
        index += rest[index..].chars().next().unwrap().len_utf8();
    }
    Some(out)
}

// ============================================================================
// Tests — 1:1 port of test/resource-loader.test.ts and
// test/resource-loader-theme.test.ts cases that do not execute extension
// JS/TS modules or construct an AgentSession/ExtensionRunner.
// ============================================================================

// ============================================================================
// Tests — 1:1 port of test/resource-loader.test.ts and
// test/resource-loader-theme.test.ts cases that do not execute extension
// JS/TS modules or construct an AgentSession/ExtensionRunner. Deferred cases
// are listed at the bottom of this module.
// ============================================================================

#[cfg(test)]
mod resource_loader_tests {
    use super::*;
    use crate::core::diagnostics::DiagnosticType;
    use crate::core::extensions::types::{CommandHandler, RegisteredCommandOptions};
    use crate::core::settings_manager::{Settings, SettingsManagerCreateOptions};
    use parking_lot::Mutex;

    /// Temp workspace cleaned up on drop (`mkdtempSync` + `rmSync` upstream).
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            // Nanosecond timestamps alone collide when two tests construct
            // workspaces in the same tick under parallel load; the process-wide
            // counter makes names unique.
            static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let unique = format!(
                "rl-{tag}-{}-{}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                std::process::id(),
                SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            );
            let path = std::env::temp_dir().join(unique);
            std::fs::create_dir_all(&path).unwrap();
            TempDir(path)
        }

        fn join(&self, rest: &str) -> PathBuf {
            self.0.join(rest)
        }

        fn str(&self, rest: &str) -> String {
            self.0.join(rest).to_string_lossy().to_string()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_file(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    /// The upstream tests read `src/modes/interactive/theme/dark.json`; the
    /// Rust port embeds it in the theme module (D56c).
    fn dark_theme_json() -> serde_json::Value {
        serde_json::from_str(include_str!("../theme/dark.json")).unwrap()
    }

    fn named_theme_fixture(name: &str) -> String {
        let mut theme = dark_theme_json();
        theme["name"] = serde_json::Value::String(name.to_string());
        serde_json::to_string_pretty(&theme).unwrap()
    }

    struct Workspace {
        _root: TempDir,
        agent_dir: String,
        cwd: String,
    }

    impl Workspace {
        fn new() -> Self {
            let root = TempDir::new("ws");
            let agent_dir = root.str("agent");
            let cwd = root.str("project");
            std::fs::create_dir_all(&agent_dir).unwrap();
            std::fs::create_dir_all(&cwd).unwrap();
            Workspace {
                _root: root,
                agent_dir,
                cwd,
            }
        }

        fn str(&self, rest: &str) -> String {
            self._root.str(rest)
        }

        fn loader(
            &self,
            configure: impl FnOnce(&mut DefaultResourceLoaderOptions),
        ) -> DefaultResourceLoader {
            let mut options = DefaultResourceLoaderOptions {
                cwd: self.cwd.clone(),
                agent_dir: self.agent_dir.clone(),
                ..Default::default()
            };
            configure(&mut options);
            DefaultResourceLoader::new(options)
        }
    }

    fn no_pause() -> RegisteredCommandOptions {
        RegisteredCommandOptions {
            description: Some("noop".to_string()),
            get_argument_completions: None,
            handler: Arc::new(|_args: String, _ctx| {
                Box::pin(async {}) as std::pin::Pin<Box<dyn Future<Output = ()> + Send>>
            }) as CommandHandler,
        }
    }

    // -- reload -----------------------------------------------------------

    #[test]
    fn initializes_with_empty_results_before_reload() {
        let ws = Workspace::new();
        let loader = ws.loader(|_| {});
        assert!(loader.get_extensions().extensions.is_empty());
        assert!(loader.get_skills().items.is_empty());
        assert!(loader.get_prompts().items.is_empty());
        assert!(loader.get_themes().items.is_empty());
    }

    #[test]
    fn discovers_skills_from_agent_dir() {
        let ws = Workspace::new();
        write_file(
            &ws._root.join("agent/skills/test-skill.md"),
            "---\nname: test-skill\ndescription: A test skill\n---\nSkill content here.",
        );
        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());
        assert!(loader
            .get_skills()
            .items
            .iter()
            .any(|s| s.name == "test-skill"));
    }

    #[test]
    fn ignores_extra_markdown_files_in_auto_discovered_skill_dirs() {
        let ws = Workspace::new();
        write_file(
            &ws._root
                .join("agent/skills/pi-skills/browser-tools/SKILL.md"),
            "---\nname: browser-tools\ndescription: Browser tools\n---\nSkill content here.",
        );
        write_file(
            &ws._root
                .join("agent/skills/pi-skills/browser-tools/EFFICIENCY.md"),
            "No frontmatter here",
        );
        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());
        let skills = loader.get_skills();
        assert!(skills.items.iter().any(|s| s.name == "browser-tools"));
        assert!(!skills.diagnostics.iter().any(|d| d
            .path
            .as_deref()
            .is_some_and(|p| p.ends_with("EFFICIENCY.md"))));
    }

    #[test]
    fn discovers_prompts_from_agent_dir() {
        let ws = Workspace::new();
        write_file(
            &ws._root.join("agent/prompts/test-prompt.md"),
            "---\ndescription: A test prompt\n---\nPrompt content.",
        );
        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());
        assert!(loader
            .get_prompts()
            .items
            .iter()
            .any(|p| p.name == "test-prompt"));
    }

    /// Regression test for #9354. Deviation D57b/D10: the crate's minimal
    /// YAML frontmatter parser accepts the upstream fixture
    /// (`description: Broken: unquoted colon`) as a plain scalar, so the
    /// invalid frontmatter here is a non-mapping document, which both
    /// parsers reject.
    #[test]
    fn reports_invalid_prompt_frontmatter_while_loading_valid_siblings() {
        let ws = Workspace::new();
        let invalid_prompt_path = ws._root.join("agent/prompts/invalid.md");
        write_file(
            &invalid_prompt_path,
            "---\n- not\n- a mapping\n---\nDo something.\n",
        );
        write_file(
            &ws._root.join("agent/prompts/valid.md"),
            "Valid prompt content.",
        );

        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());

        let prompts = loader.get_prompts();
        let names: Vec<String> = prompts.items.iter().map(|p| p.name.clone()).collect();
        assert_eq!(names, vec!["valid".to_string()]);
        assert_eq!(prompts.diagnostics.len(), 1);
        let diagnostic = &prompts.diagnostics[0];
        assert_eq!(diagnostic.r#type, DiagnosticType::Warning);
        assert_eq!(
            diagnostic.path.as_deref(),
            Some(invalid_prompt_path.to_string_lossy().as_ref())
        );
    }

    #[test]
    fn prefers_project_resources_over_user_on_name_collisions() {
        let ws = Workspace::new();
        let user_prompt_path = ws._root.join("agent/prompts/commit.md");
        let project_prompt_path = ws._root.join("project/.pi/prompts/commit.md");
        write_file(&user_prompt_path, "User prompt");
        write_file(&project_prompt_path, "Project prompt");

        let user_skill_path = ws._root.join("agent/skills/collision-skill/SKILL.md");
        let project_skill_path = ws._root.join("project/.pi/skills/collision-skill/SKILL.md");
        write_file(
            &user_skill_path,
            "---\nname: collision-skill\ndescription: user\n---\nUser skill",
        );
        write_file(
            &project_skill_path,
            "---\nname: collision-skill\ndescription: project\n---\nProject skill",
        );

        let user_theme_path = ws._root.join("agent/themes/collision.json");
        let project_theme_path = ws._root.join("project/.pi/themes/collision.json");
        write_file(&user_theme_path, &named_theme_fixture("collision-theme"));
        write_file(&project_theme_path, &named_theme_fixture("collision-theme"));

        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());

        assert_eq!(
            loader
                .get_prompts()
                .items
                .iter()
                .find(|p| p.name == "commit")
                .unwrap()
                .file_path,
            project_prompt_path.to_string_lossy()
        );
        assert_eq!(
            loader
                .get_skills()
                .items
                .iter()
                .find(|s| s.name == "collision-skill")
                .unwrap()
                .file_path,
            project_skill_path.to_string_lossy()
        );
        assert_eq!(
            loader
                .get_themes()
                .items
                .iter()
                .find(|t| t.name.as_deref() == Some("collision-theme"))
                .unwrap()
                .source_path
                .as_deref(),
            Some(project_theme_path.to_string_lossy().as_ref())
        );
    }

    #[test]
    fn honors_overrides_for_auto_discovered_resources() {
        let ws = Workspace::new();
        let settings_manager =
            SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default());
        let mut settings_manager = settings_manager;
        settings_manager.set_extension_paths(&["-extensions/disabled.ts".to_string()]);
        settings_manager.set_skill_paths(&["-skills/skip-skill".to_string()]);
        settings_manager.set_prompt_template_paths(&["-prompts/skip.md".to_string()]);
        settings_manager.set_theme_paths(&["-themes/skip.json".to_string()]);

        write_file(
            &ws._root.join("agent/extensions/disabled.ts"),
            "export default function() {}",
        );
        write_file(
            &ws._root.join("agent/skills/skip-skill/SKILL.md"),
            "---\nname: skip-skill\ndescription: Skip me\n---\nContent",
        );
        write_file(&ws._root.join("agent/prompts/skip.md"), "Skip prompt");
        write_file(&ws._root.join("agent/themes/skip.json"), "{}");

        let mut loader = ws.loader(|options| options.settings_manager = Some(settings_manager));
        loader.reload(Default::default());

        let extensions = loader.get_extensions();
        assert!(!extensions
            .extensions
            .iter()
            .any(|e| e.path.ends_with("disabled.ts")));
        assert!(!loader
            .get_skills()
            .items
            .iter()
            .any(|s| s.name == "skip-skill"));
        assert!(!loader.get_prompts().items.iter().any(|p| p.name == "skip"));
        assert!(!loader.get_themes().items.iter().any(|t| t
            .source_path
            .as_deref()
            .is_some_and(|p| p.ends_with("skip.json"))));
    }

    // -- context files ----------------------------------------------------

    #[test]
    fn discovers_agents_md_context_files() {
        let ws = Workspace::new();
        write_file(
            &ws._root.join("project/AGENTS.md"),
            "# Project Guidelines\n\nBe helpful.",
        );
        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());
        assert!(loader
            .get_agents_files()
            .iter()
            .any(|f| f.path.contains("AGENTS.md")));
    }

    #[test]
    fn prefers_agents_override_md_within_each_directory_while_preserving_ancestor_layering() {
        let ws = Workspace::new();
        let nested_cwd = ws.str("project/service");
        std::fs::create_dir_all(&nested_cwd).unwrap();
        write_file(&ws._root.join("agent/AGENTS.md"), "global instructions");
        write_file(
            &ws._root.join("agent/AGENTS.override.md"),
            "global override",
        );
        write_file(&ws._root.join("project/AGENTS.md"), "project instructions");
        write_file(
            &ws._root.join("project/service/AGENTS.md"),
            "service instructions",
        );
        write_file(
            &ws._root.join("project/service/AGENTS.override.md"),
            "service override",
        );

        let mut loader = ws.loader(|options| options.cwd = nested_cwd);
        loader.reload(Default::default());

        assert_eq!(
            loader.get_agents_files(),
            vec![
                ContextFile {
                    path: ws.str("agent/AGENTS.override.md"),
                    content: "global override".to_string()
                },
                ContextFile {
                    path: ws.str("project/AGENTS.md"),
                    content: "project instructions".to_string()
                },
                ContextFile {
                    path: ws.str("project/service/AGENTS.override.md"),
                    content: "service override".to_string()
                },
            ]
        );
    }

    #[test]
    fn ignores_context_file_candidates_that_are_directories() {
        let ws = Workspace::new();
        std::fs::create_dir_all(ws._root.join("project/AGENTS.override.md")).unwrap();
        std::fs::create_dir_all(ws._root.join("project/AGENTS.md")).unwrap();
        write_file(&ws._root.join("project/CLAUDE.md"), "Fallback instructions");

        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());

        assert!(loader.get_agents_files().contains(&ContextFile {
            path: ws.str("project/CLAUDE.md"),
            content: "Fallback instructions".to_string(),
        }));
    }

    #[test]
    fn skips_context_file_discovery_when_no_context_files_is_true() {
        let ws = Workspace::new();
        write_file(
            &ws._root.join("project/AGENTS.override.md"),
            "# Override Guidelines\n\nBe helpful.",
        );
        write_file(
            &ws._root.join("project/AGENTS.md"),
            "# Project Guidelines\n\nBe helpful.",
        );
        write_file(
            &ws._root.join("project/CLAUDE.md"),
            "# Claude Guidelines\n\nBe helpful.",
        );

        let mut loader = ws.loader(|options| options.no_context_files = true);
        loader.reload(Default::default());
        assert!(loader.get_agents_files().is_empty());
    }

    // -- system prompts ---------------------------------------------------

    #[test]
    fn discovers_system_md_from_cwd_pi() {
        let ws = Workspace::new();
        write_file(
            &ws._root.join("project/.pi/SYSTEM.md"),
            "You are a helpful assistant.",
        );
        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());
        assert_eq!(
            loader.get_system_prompt().as_deref(),
            Some("You are a helpful assistant.")
        );
    }

    #[test]
    fn skips_project_resources_that_require_trust_when_project_is_not_trusted() {
        let ws = Workspace::new();
        write_file(
            &ws._root.join("project/.pi/SYSTEM.md"),
            "Project system prompt.",
        );
        write_file(&ws._root.join("agent/SYSTEM.md"), "Global system prompt.");
        write_file(&ws._root.join("agent/AGENTS.md"), "Global instructions");
        write_file(&ws._root.join("project/AGENTS.md"), "Project instructions");
        write_file(
            &ws._root.join("project/.pi/extensions/project.ts"),
            "throw new Error(\"x\");",
        );
        write_file(
            &ws._root.join("project/.pi/skills/project-skill/SKILL.md"),
            "---\nname: project-skill\ndescription: Project skill\n---\nProject skill content",
        );
        write_file(
            &ws._root.join("project/.pi/prompts/project.md"),
            "Project prompt",
        );
        write_file(
            &ws._root.join("project/.pi/themes/project.json"),
            &named_theme_fixture("project-theme"),
        );
        let settings_manager = SettingsManager::create(
            &ws.cwd,
            &ws.agent_dir,
            SettingsManagerCreateOptions {
                project_trusted: Some(false),
            },
        );

        let mut loader = ws.loader(|options| options.settings_manager = Some(settings_manager));
        loader.reload(Default::default());

        assert_eq!(
            loader.get_system_prompt().as_deref(),
            Some("Global system prompt.")
        );
        assert!(loader
            .get_agents_files()
            .iter()
            .any(|file| file.path == ws.str("agent/AGENTS.md")));
        assert!(loader
            .get_agents_files()
            .iter()
            .any(|file| file.path == ws.str("project/AGENTS.md")));
        let extensions = loader.get_extensions();
        assert!(extensions.extensions.is_empty());
        assert!(extensions.errors.is_empty());
        assert!(!loader
            .get_skills()
            .items
            .iter()
            .any(|s| s.name == "project-skill"));
        assert!(!loader
            .get_prompts()
            .items
            .iter()
            .any(|p| p.name == "project"));
        assert!(!loader
            .get_themes()
            .items
            .iter()
            .any(|t| t.name.as_deref() == Some("project-theme")));
    }

    #[test]
    fn discovers_append_system_md() {
        let ws = Workspace::new();
        write_file(
            &ws._root.join("project/.pi/APPEND_SYSTEM.md"),
            "Additional instructions.",
        );
        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());
        assert!(loader
            .get_append_system_prompt()
            .iter()
            .any(|prompt| prompt == "Additional instructions."));
    }

    // -- system prompt sources ---------------------------------------------

    #[test]
    fn exposes_discovered_project_system_md_as_the_system_prompt_source() {
        let ws = Workspace::new();
        let system_prompt_path = ws.str("project/.pi/SYSTEM.md");
        write_file(
            &ws._root.join("project/.pi/SYSTEM.md"),
            "Project system prompt.",
        );
        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());
        assert_eq!(
            loader.get_system_prompt().as_deref(),
            Some("Project system prompt.")
        );
        assert_eq!(
            loader.get_system_prompt_source().map(|source| source.path),
            Some(system_prompt_path)
        );
    }

    #[test]
    fn exposes_discovered_global_system_md_as_the_system_prompt_source() {
        let ws = Workspace::new();
        let system_prompt_path = ws.str("agent/SYSTEM.md");
        write_file(&ws._root.join("agent/SYSTEM.md"), "Global system prompt.");
        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());
        assert_eq!(
            loader.get_system_prompt().as_deref(),
            Some("Global system prompt.")
        );
        assert_eq!(
            loader.get_system_prompt_source().map(|source| source.path),
            Some(system_prompt_path)
        );
    }

    #[test]
    fn does_not_expose_literal_system_prompt_text_as_a_source() {
        let ws = Workspace::new();
        let mut loader = ws.loader(|options| {
            options.system_prompt = Some("Literal system prompt.".to_string());
        });
        loader.reload(Default::default());
        assert_eq!(
            loader.get_system_prompt().as_deref(),
            Some("Literal system prompt.")
        );
        assert!(loader.get_system_prompt_source().is_none());
    }

    #[test]
    fn exposes_file_backed_system_prompt_options_as_a_source() {
        let ws = Workspace::new();
        let system_prompt_path = ws.str("custom-system.md");
        write_file(&ws._root.join("custom-system.md"), "Custom system prompt.");
        let mut loader =
            ws.loader(|options| options.system_prompt = Some(system_prompt_path.clone()));
        loader.reload(Default::default());
        assert_eq!(
            loader.get_system_prompt().as_deref(),
            Some("Custom system prompt.")
        );
        assert_eq!(
            loader.get_system_prompt_source().map(|source| source.path),
            Some(system_prompt_path)
        );
    }

    #[test]
    fn exposes_discovered_append_system_md_as_an_append_system_prompt_source() {
        let ws = Workspace::new();
        let append_system_prompt_path = ws.str("project/.pi/APPEND_SYSTEM.md");
        write_file(
            &ws._root.join("project/.pi/APPEND_SYSTEM.md"),
            "Project append prompt.",
        );
        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());
        assert_eq!(
            loader.get_append_system_prompt(),
            vec!["Project append prompt.".to_string()]
        );
        assert_eq!(
            loader
                .get_append_system_prompt_sources()
                .into_iter()
                .map(|source| source.path)
                .collect::<Vec<_>>(),
            vec![append_system_prompt_path]
        );
    }

    #[test]
    fn does_not_expose_literal_append_system_prompt_text_as_a_source() {
        let ws = Workspace::new();
        let mut loader = ws.loader(|options| {
            options.append_system_prompt = Some(vec!["Literal append prompt.".to_string()]);
        });
        loader.reload(Default::default());
        assert_eq!(
            loader.get_append_system_prompt(),
            vec!["Literal append prompt.".to_string()]
        );
        assert!(loader.get_append_system_prompt_sources().is_empty());
    }

    #[test]
    fn only_exposes_file_backed_append_system_prompt_options_as_sources() {
        let ws = Workspace::new();
        let append_system_prompt_path = ws.str("custom-append.md");
        write_file(&ws._root.join("custom-append.md"), "Custom append prompt.");
        let mut loader = ws.loader(|options| {
            options.append_system_prompt = Some(vec![
                append_system_prompt_path.clone(),
                "Literal append prompt.".to_string(),
            ]);
        });
        loader.reload(Default::default());
        assert_eq!(
            loader.get_append_system_prompt(),
            vec![
                "Custom append prompt.".to_string(),
                "Literal append prompt.".to_string()
            ]
        );
        assert_eq!(
            loader
                .get_append_system_prompt_sources()
                .into_iter()
                .map(|source| source.path)
                .collect::<Vec<_>>(),
            vec![append_system_prompt_path]
        );
    }

    // -- extendResources ---------------------------------------------------

    #[test]
    fn loads_skills_and_prompts_with_extension_metadata() {
        let ws = Workspace::new();
        let extra_skill_dir = ws.str("extra-skills/extra-skill");
        let skill_path = ws.str("extra-skills/extra-skill/SKILL.md");
        write_file(
            &ws._root.join("extra-skills/extra-skill/SKILL.md"),
            "---\nname: extra-skill\ndescription: Extra skill\n---\nExtra content",
        );
        let extra_prompt_dir = ws.str("extra-prompts");
        let prompt_path = ws.str("extra-prompts/extra.md");
        write_file(
            &ws._root.join("extra-prompts/extra.md"),
            "---\ndescription: Extra prompt\n---\nExtra prompt content",
        );

        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());

        loader.extend_resources(ResourceExtensionPaths {
            skill_paths: vec![ResourcePathEntry {
                path: extra_skill_dir.clone(),
                metadata: PathMetadata {
                    source: "extension:extra".to_string(),
                    scope: SourceScope::Temporary,
                    origin: SourceOrigin::TopLevel,
                    base_dir: Some(extra_skill_dir.clone()),
                    package_root: None,
                },
            }],
            prompt_paths: vec![ResourcePathEntry {
                path: prompt_path.clone(),
                metadata: PathMetadata {
                    source: "extension:extra".to_string(),
                    scope: SourceScope::Temporary,
                    origin: SourceOrigin::TopLevel,
                    base_dir: Some(extra_prompt_dir.clone()),
                    package_root: None,
                },
            }],
            theme_paths: Vec::new(),
        });

        let loaded_skill = loader
            .get_skills()
            .items
            .into_iter()
            .find(|skill| skill.name == "extra-skill")
            .expect("extra-skill loaded");
        assert_eq!(loaded_skill.source_info.source, "extension:extra");
        assert_eq!(loaded_skill.source_info.path, skill_path);

        let loaded_prompt = loader
            .get_prompts()
            .items
            .into_iter()
            .find(|prompt| prompt.name == "extra")
            .expect("extra prompt loaded");
        assert_eq!(loaded_prompt.source_info.source, "extension:extra");
        assert_eq!(loaded_prompt.source_info.path, prompt_path);
    }

    #[test]
    fn loads_extension_resources_returned_as_file_urls() {
        let ws = Workspace::new();
        let extra_skill_dir = ws.str("extra skills/file-url-skill");
        let skill_path = ws.str("extra skills/file-url-skill/SKILL.md");
        write_file(
            &ws._root.join("extra skills/file-url-skill/SKILL.md"),
            "---\nname: file-url-skill\ndescription: File URL skill\n---\nExtra content",
        );

        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());

        let file_url = format!("file://{}", extra_skill_dir);
        loader.extend_resources(ResourceExtensionPaths {
            skill_paths: vec![ResourcePathEntry {
                path: file_url,
                metadata: PathMetadata {
                    source: "extension:file-url".to_string(),
                    scope: SourceScope::Temporary,
                    origin: SourceOrigin::TopLevel,
                    base_dir: Some(extra_skill_dir),
                    package_root: None,
                },
            }],
            prompt_paths: Vec::new(),
            theme_paths: Vec::new(),
        });

        let skills = loader.get_skills();
        // Deviation: the user's real ~/.agents skills are auto-discovered and
        // may carry unrelated diagnostics; scope the emptiness check to the
        // extension-supplied resource.
        assert!(skills.diagnostics.iter().all(|d| !d
            .path
            .as_deref()
            .is_some_and(|p| p.contains("file-url-skill"))));
        let loaded_skill = skills
            .items
            .into_iter()
            .find(|skill| skill.name == "file-url-skill")
            .expect("file-url-skill loaded");
        assert_eq!(loaded_skill.file_path, skill_path);
        assert_eq!(loaded_skill.source_info.source, "extension:file-url");
    }

    /// Regression: extension discovery used to drop package scope/source,
    /// collapsing every autocomplete source tag to [t]. See issue #6968.
    #[test]
    fn keeps_package_metadata_for_skills_prompts_and_themes() {
        let ws = Workspace::new();
        let package_root = ws._root.join("agent/npm/node_modules/metadata-pkg");
        write_file(
            &package_root.join("package.json"),
            &serde_json::to_string(
                &serde_json::json!({"name": "metadata-pkg", "version": "1.0.0"}),
            )
            .unwrap(),
        );
        write_file(
            &package_root.join("skills/package-skill/SKILL.md"),
            "---\nname: package-skill\ndescription: Package skill\n---\nPackage skill content",
        );
        write_file(
            &package_root.join("prompts/package-prompt.md"),
            "---\ndescription: Package prompt\n---\nPackage prompt content",
        );
        let mut package_theme = dark_theme_json();
        package_theme["name"] = serde_json::Value::String("package-theme".to_string());
        write_file(
            &package_root.join("themes/package-theme.json"),
            &serde_json::to_string(&package_theme).unwrap(),
        );

        let extension_resource_dir = ws._root.join("extension-resources");
        let extension_skill_dir = extension_resource_dir.join("extension-skill");
        let extension_prompts_dir = extension_resource_dir.join("prompts");
        let extension_themes_dir = extension_resource_dir.join("themes");
        write_file(
			&extension_skill_dir.join("SKILL.md"),
			"---\nname: extension-skill\ndescription: Extension skill\n---\nExtension skill content",
		);
        write_file(
            &extension_prompts_dir.join("extension-prompt.md"),
            "---\ndescription: Extension prompt\n---\nExtension prompt content",
        );
        let mut extension_theme = dark_theme_json();
        extension_theme["name"] = serde_json::Value::String("extension-theme".to_string());
        write_file(
            &extension_themes_dir.join("extension.json"),
            &serde_json::to_string(&extension_theme).unwrap(),
        );

        let settings_manager = SettingsManager::in_memory(
            &Settings::from_value(serde_json::json!({"packages": ["npm:metadata-pkg"]})),
            SettingsManagerCreateOptions::default(),
        );
        let mut loader = ws.loader(|options| options.settings_manager = Some(settings_manager));
        loader.reload(Default::default());

        let extension_metadata = |base_dir: Option<String>| PathMetadata {
            source: "extension:discovery".to_string(),
            scope: SourceScope::Temporary,
            origin: SourceOrigin::TopLevel,
            base_dir,
            package_root: None,
        };
        loader.extend_resources(ResourceExtensionPaths {
            skill_paths: vec![ResourcePathEntry {
                path: extension_skill_dir.to_string_lossy().to_string(),
                metadata: extension_metadata(None),
            }],
            prompt_paths: vec![ResourcePathEntry {
                path: extension_prompts_dir.to_string_lossy().to_string(),
                metadata: extension_metadata(None),
            }],
            theme_paths: vec![ResourcePathEntry {
                path: extension_themes_dir.to_string_lossy().to_string(),
                metadata: extension_metadata(None),
            }],
        });

        let skills = loader.get_skills();
        let package_skill = skills
            .items
            .iter()
            .find(|s| s.name == "package-skill")
            .unwrap();
        assert_eq!(package_skill.source_info.source, "npm:metadata-pkg");
        assert_eq!(package_skill.source_info.scope, SourceScope::User);
        assert_eq!(package_skill.source_info.origin, SourceOrigin::Package);

        let prompts = loader.get_prompts();
        let package_prompt = prompts
            .items
            .iter()
            .find(|p| p.name == "package-prompt")
            .unwrap();
        assert_eq!(package_prompt.source_info.source, "npm:metadata-pkg");
        assert_eq!(package_prompt.source_info.scope, SourceScope::User);
        assert_eq!(package_prompt.source_info.origin, SourceOrigin::Package);

        let themes = loader.get_themes();
        let package_theme_loaded = themes
            .items
            .iter()
            .find(|t| t.name.as_deref() == Some("package-theme"))
            .unwrap();
        assert_eq!(
            package_theme_loaded.source_info.as_ref().unwrap().source,
            "npm:metadata-pkg"
        );
        assert_eq!(
            package_theme_loaded.source_info.as_ref().unwrap().scope,
            SourceScope::User
        );
        assert_eq!(
            package_theme_loaded.source_info.as_ref().unwrap().origin,
            SourceOrigin::Package
        );

        let extension_skill = skills
            .items
            .iter()
            .find(|s| s.name == "extension-skill")
            .unwrap();
        assert_eq!(extension_skill.source_info.source, "extension:discovery");
        assert_eq!(extension_skill.source_info.scope, SourceScope::Temporary);
        assert_eq!(extension_skill.source_info.origin, SourceOrigin::TopLevel);

        let extension_prompt = prompts
            .items
            .iter()
            .find(|p| p.name == "extension-prompt")
            .unwrap();
        assert_eq!(extension_prompt.source_info.source, "extension:discovery");

        let extension_theme_loaded = themes
            .items
            .iter()
            .find(|t| t.name.as_deref() == Some("extension-theme"))
            .unwrap();
        assert_eq!(
            extension_theme_loaded.source_info.as_ref().unwrap().source,
            "extension:discovery"
        );
    }

    // -- noSkills option ---------------------------------------------------

    #[test]
    fn skips_skill_discovery_when_no_skills_is_true() {
        let ws = Workspace::new();
        write_file(
            &ws._root.join("agent/skills/test-skill.md"),
            "---\nname: test-skill\ndescription: A test skill\n---\nContent",
        );
        let mut loader = ws.loader(|options| options.no_skills = true);
        loader.reload(Default::default());
        assert!(loader.get_skills().items.is_empty());
    }

    #[test]
    fn still_loads_additional_skill_paths_when_no_skills_is_true() {
        let ws = Workspace::new();
        let custom_skill_dir = ws.str("custom-skills");
        write_file(
            &ws._root.join("custom-skills/custom.md"),
            "---\nname: custom\ndescription: Custom skill\n---\nContent",
        );
        let mut loader = ws.loader(|options| {
            options.no_skills = true;
            options.additional_skill_paths = vec![custom_skill_dir];
        });
        loader.reload(Default::default());
        assert!(loader.get_skills().items.iter().any(|s| s.name == "custom"));
    }

    // -- override functions ------------------------------------------------

    #[test]
    fn applies_skills_override() {
        let ws = Workspace::new();
        let injected_skill = Skill {
            name: "injected".to_string(),
            description: "Injected skill".to_string(),
            file_path: "/fake/path".to_string(),
            base_dir: "/fake".to_string(),
            source_info: create_synthetic_source_info("/fake/path", "custom", None, None, None),
            disable_model_invocation: false,
        };
        let mut loader = ws.loader(|options| {
            options.skills_override = Some(Box::new(move |_| SkillsResult {
                items: vec![injected_skill.clone()],
                diagnostics: Vec::new(),
            }));
        });
        loader.reload(Default::default());
        let skills = loader.get_skills();
        assert_eq!(skills.items.len(), 1);
        assert_eq!(skills.items[0].name, "injected");
    }

    #[test]
    fn applies_system_prompt_override() {
        let ws = Workspace::new();
        let mut loader = ws.loader(|options| {
            options.system_prompt_override =
                Some(Box::new(|_| Some("Custom system prompt".to_string())));
        });
        loader.reload(Default::default());
        assert_eq!(
            loader.get_system_prompt().as_deref(),
            Some("Custom system prompt")
        );
    }

    // -- extension conflict detection (builtin/inline factory cases) -------

    #[test]
    fn skips_builtin_extensions_disabled_in_settings() {
        let ws = Workspace::new();
        write_file(
            &ws._root.join("agent/settings.json"),
            &serde_json::to_string(&serde_json::json!({"extensions": ["-builtin:mcp"]})).unwrap(),
        );
        let loaded: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let make_factory = |name: &'static str, loaded: Arc<Mutex<Vec<String>>>| {
            Arc::new(
                move |_api: crate::core::extensions::types::ExtensionApiHandle| {
                    loaded.lock().push(name.to_string());
                    Box::pin(async { Ok(()) })
                        as std::pin::Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
                },
            ) as crate::core::extensions::types::ExtensionFactory
        };
        let mut loader = ws.loader(|options| {
            options.extension_factories = vec![
                InlineExtension::Named {
                    name: "mcp".to_string(),
                    factory: make_factory("mcp", Arc::clone(&loaded)),
                    hidden: None,
                    replaceable: None,
                    builtin: Some(true),
                },
                InlineExtension::Named {
                    name: "llama".to_string(),
                    factory: make_factory("llama", Arc::clone(&loaded)),
                    hidden: None,
                    replaceable: None,
                    builtin: Some(true),
                },
            ];
        });
        loader.reload(Default::default());

        let extensions = loader.get_extensions();
        let paths: Vec<String> = extensions
            .extensions
            .iter()
            .map(|e| e.path.clone())
            .collect();
        assert_eq!(paths, vec!["builtin:llama".to_string()]);
        let first = &extensions.extensions[0];
        assert_eq!(first.source_info.path, "builtin:llama");
        assert_eq!(first.source_info.source, "builtin");
        assert_eq!(first.hidden, Some(true));
        assert_eq!(*loaded.lock(), vec!["llama".to_string()]);
    }

    #[test]
    fn skips_disabled_builtin_extensions_even_when_settings_or_cli_enable_them() {
        let ws = Workspace::new();
        write_file(
            &ws._root.join("agent/settings.json"),
            &serde_json::to_string(&serde_json::json!({"extensions": ["+builtin:mcp"]})).unwrap(),
        );
        let loaded: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let make_factory = |name: &'static str, loaded: Arc<Mutex<Vec<String>>>| {
            Arc::new(
                move |_api: crate::core::extensions::types::ExtensionApiHandle| {
                    loaded.lock().push(name.to_string());
                    Box::pin(async { Ok(()) })
                        as std::pin::Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
                },
            ) as crate::core::extensions::types::ExtensionFactory
        };
        let mut loader = ws.loader(|options| {
            options.disabled_builtin_extensions = vec!["mcp".to_string()];
            options.additional_extension_paths = vec!["builtin:mcp".to_string()];
            options.extension_factories = vec![
                InlineExtension::Named {
                    name: "mcp".to_string(),
                    factory: make_factory("mcp", Arc::clone(&loaded)),
                    hidden: None,
                    replaceable: None,
                    builtin: Some(true),
                },
                InlineExtension::Named {
                    name: "llama".to_string(),
                    factory: make_factory("llama", Arc::clone(&loaded)),
                    hidden: None,
                    replaceable: None,
                    builtin: Some(true),
                },
            ];
        });
        loader.reload(Default::default());

        let extensions = loader.get_extensions();
        let paths: Vec<String> = extensions
            .extensions
            .iter()
            .map(|extension| extension.path.clone())
            .collect();
        assert_eq!(paths, vec!["builtin:llama".to_string()]);
        assert_eq!(extensions.errors, Vec::new());
        assert_eq!(*loaded.lock(), vec!["llama".to_string()]);
    }

    #[test]
    fn disables_builtin_extensions_with_no_extensions_unless_loaded_via_cli() {
        let ws = Workspace::new();
        let loaded: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let make_factory = |name: &'static str, loaded: Arc<Mutex<Vec<String>>>| {
            Arc::new(
                move |_api: crate::core::extensions::types::ExtensionApiHandle| {
                    loaded.lock().push(name.to_string());
                    Box::pin(async { Ok(()) })
                        as std::pin::Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
                },
            ) as crate::core::extensions::types::ExtensionFactory
        };
        let mut loader = ws.loader(|options| {
            options.no_extensions = true;
            options.additional_extension_paths =
                vec!["builtin:mcp".to_string(), "builtin:missing".to_string()];
            options.extension_factories = vec![
                InlineExtension::Named {
                    name: "mcp".to_string(),
                    factory: make_factory("mcp", Arc::clone(&loaded)),
                    hidden: None,
                    replaceable: None,
                    builtin: Some(true),
                },
                InlineExtension::Named {
                    name: "llama".to_string(),
                    factory: make_factory("llama", Arc::clone(&loaded)),
                    hidden: None,
                    replaceable: None,
                    builtin: Some(true),
                },
            ];
        });
        loader.reload(Default::default());

        let extensions = loader.get_extensions();
        let paths: Vec<String> = extensions
            .extensions
            .iter()
            .map(|e| e.path.clone())
            .collect();
        assert_eq!(paths, vec!["builtin:mcp".to_string()]);
        assert_eq!(
            extensions.errors,
            vec![LoadRecord {
                path: "builtin:missing".to_string(),
                error: "Unknown built-in extension: builtin:missing".to_string()
            }]
        );
        assert_eq!(*loaded.lock(), vec!["mcp".to_string()]);
    }

    #[test]
    fn applies_project_builtin_extension_overrides_after_trust_resolves() {
        let ws = Workspace::new();
        write_file(
            &ws._root.join("agent/settings.json"),
            &serde_json::to_string(&serde_json::json!({"extensions": ["-builtin:mcp"]})).unwrap(),
        );
        write_file(
            &ws._root.join("project/.pi/settings.json"),
            &serde_json::to_string(&serde_json::json!({
                "extensions": ["+builtin:mcp", "-builtin:llama"]
            }))
            .unwrap(),
        );
        let loaded: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let make_factory = |name: &'static str, loaded: Arc<Mutex<Vec<String>>>| {
            Arc::new(
                move |_api: crate::core::extensions::types::ExtensionApiHandle| {
                    loaded.lock().push(name.to_string());
                    Box::pin(async { Ok(()) })
                        as std::pin::Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
                },
            ) as crate::core::extensions::types::ExtensionFactory
        };
        let pre_trust_paths: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let pre_trust_capture = Arc::clone(&pre_trust_paths);
        let mut loader = ws.loader(|options| {
            options.extension_factories = vec![
                InlineExtension::Named {
                    name: "mcp".to_string(),
                    factory: make_factory("mcp", Arc::clone(&loaded)),
                    hidden: None,
                    replaceable: None,
                    builtin: Some(true),
                },
                InlineExtension::Named {
                    name: "plain".to_string(),
                    factory: make_factory("plain", Arc::clone(&loaded)),
                    hidden: None,
                    replaceable: None,
                    builtin: None,
                },
                InlineExtension::Named {
                    name: "llama".to_string(),
                    factory: make_factory("llama", Arc::clone(&loaded)),
                    hidden: None,
                    replaceable: None,
                    builtin: Some(true),
                },
            ];
        });
        loader.reload(ResourceLoaderReloadOptions {
            resolve_project_trust: Some(Box::new(
                move |extensions_result: &LoadExtensionsResult| {
                    // Built-in extensions wait until project settings are known.
                    let paths: Vec<String> = extensions_result
                        .extensions
                        .iter()
                        .map(|e| e.path.clone())
                        .collect();
                    *pre_trust_capture.lock() = paths;
                    true
                },
            )),
        });

        assert_eq!(*pre_trust_paths.lock(), vec!["<inline:plain>".to_string()]);
        let extensions = loader.get_extensions();
        let paths: Vec<String> = extensions
            .extensions
            .iter()
            .map(|e| e.path.clone())
            .collect();
        assert_eq!(
            paths,
            vec!["builtin:mcp".to_string(), "<inline:plain>".to_string()]
        );
        assert_eq!(*loaded.lock(), vec!["plain".to_string(), "mcp".to_string()]);
    }

    // -- loadProjectContextFiles: nested worktree dedup ---------------------

    /// Builds a linked-worktree skeleton (no git binary needed).
    fn link_worktree(main_dir: &Path, worktree_dir: &Path, name: &str) {
        let git_dir = main_dir.join(".git").join("worktrees").join(name);
        std::fs::create_dir_all(&git_dir).unwrap();
        write_file(
            &main_dir.join(".git").join("HEAD"),
            "ref: refs/heads/main\n",
        );
        write_file(&git_dir.join("HEAD"), "ref: refs/heads/feat\n");
        write_file(&git_dir.join("commondir"), "../..");
        write_file(
            &worktree_dir.join(".git"),
            &format!("gitdir: {}\n", git_dir.to_string_lossy()),
        );
    }

    struct NestedWorktree {
        outer: PathBuf,
        main: PathBuf,
        worktree: PathBuf,
        worktree_src: PathBuf,
    }

    fn setup_nested_worktree(root: &TempDir) -> NestedWorktree {
        let outer = root.join("outer");
        let main = outer.join("main");
        let worktree = main.join("worktrees").join("feat");
        let worktree_src = worktree.join("src");
        std::fs::create_dir_all(&worktree_src).unwrap();
        link_worktree(&main, &worktree, "feat");
        NestedWorktree {
            outer,
            main,
            worktree,
            worktree_src,
        }
    }

    fn context_contents(cwd: &Path, agent_dir: &str) -> Vec<String> {
        load_project_context_files(&cwd.to_string_lossy(), agent_dir)
            .into_iter()
            .map(|file| file.content)
            .collect()
    }

    #[test]
    fn skips_main_repo_duplicate_when_worktree_root_has_its_own_context() {
        let root = TempDir::new("wt1");
        let agent_dir = root.str("agent");
        let setup = setup_nested_worktree(&root);
        write_file(&setup.main.join("AGENTS.md"), "main repo instructions");
        write_file(&setup.worktree.join("AGENTS.md"), "worktree instructions");
        assert_eq!(
            context_contents(&setup.worktree_src, &agent_dir),
            vec!["worktree instructions".to_string()]
        );
    }

    #[test]
    fn still_inherits_main_repo_context_when_worktree_root_has_none() {
        let root = TempDir::new("wt2");
        let agent_dir = root.str("agent");
        let setup = setup_nested_worktree(&root);
        write_file(&setup.main.join("AGENTS.md"), "main repo instructions");
        assert_eq!(
            context_contents(&setup.worktree_src, &agent_dir),
            vec!["main repo instructions".to_string()]
        );
    }

    #[test]
    fn only_skips_same_filename_not_differently_named_context_file() {
        let root = TempDir::new("wt3");
        let agent_dir = root.str("agent");
        let setup = setup_nested_worktree(&root);
        write_file(&setup.main.join("CLAUDE.md"), "main repo instructions");
        write_file(&setup.worktree.join("AGENTS.md"), "worktree instructions");
        assert_eq!(
            context_contents(&setup.worktree_src, &agent_dir),
            vec![
                "main repo instructions".to_string(),
                "worktree instructions".to_string()
            ]
        );
    }

    #[test]
    fn does_not_skip_container_context_in_bare_layout() {
        let root = TempDir::new("wt4");
        let agent_dir = root.str("agent");
        let proj = root.join("proj");
        let bare = proj.join(".bare");
        let worktree = proj.join("main");
        let worktree_git_dir = bare.join("worktrees").join("main");
        std::fs::create_dir_all(&worktree_git_dir).unwrap();
        std::fs::create_dir_all(&worktree).unwrap();
        write_file(&bare.join("HEAD"), "ref: refs/heads/main\n");
        write_file(&worktree_git_dir.join("HEAD"), "ref: refs/heads/main\n");
        write_file(&worktree_git_dir.join("commondir"), "../..");
        write_file(
            &worktree.join(".git"),
            &format!("gitdir: {}\n", worktree_git_dir.to_string_lossy()),
        );
        write_file(&proj.join("AGENTS.md"), "container instructions");
        write_file(&worktree.join("AGENTS.md"), "worktree instructions");
        assert_eq!(
            context_contents(&worktree, &agent_dir),
            vec![
                "container instructions".to_string(),
                "worktree instructions".to_string()
            ]
        );
    }

    #[test]
    fn keeps_loading_ancestors_above_the_main_repo() {
        let root = TempDir::new("wt5");
        let agent_dir = root.str("agent");
        let setup = setup_nested_worktree(&root);
        write_file(&setup.outer.join("AGENTS.md"), "outer instructions");
        write_file(&setup.main.join("AGENTS.md"), "main repo instructions");
        write_file(&setup.worktree.join("AGENTS.md"), "worktree instructions");
        assert_eq!(
            context_contents(&setup.worktree_src, &agent_dir),
            vec![
                "outer instructions".to_string(),
                "worktree instructions".to_string()
            ]
        );
    }

    #[test]
    fn does_not_skip_anything_for_a_sibling_worktree() {
        let root = TempDir::new("wt6");
        let agent_dir = root.str("agent");
        let outer = root.join("outer");
        let main = outer.join("main");
        let sib = outer.join("sib-feat");
        let sib_src = sib.join("src");
        std::fs::create_dir_all(&sib_src).unwrap();
        std::fs::create_dir_all(&main).unwrap();
        write_file(&outer.join("AGENTS.md"), "outer instructions");
        write_file(&sib.join("AGENTS.md"), "sibling worktree instructions");
        link_worktree(&main, &sib, "sib");
        assert_eq!(
            context_contents(&sib_src, &agent_dir),
            vec![
                "outer instructions".to_string(),
                "sibling worktree instructions".to_string()
            ]
        );
    }

    #[test]
    fn does_not_skip_superproject_context_from_inside_a_submodule() {
        let root = TempDir::new("wt7");
        let agent_dir = root.str("agent");
        let sup = root.join("super");
        let sub = sup.join("vendor").join("lib");
        let sub_src = sub.join("src");
        std::fs::create_dir_all(&sub_src).unwrap();
        write_file(&sup.join("AGENTS.md"), "superproject instructions");
        write_file(&sub.join("AGENTS.md"), "submodule instructions");
        let sub_git_dir = sup.join(".git").join("modules").join("vendor").join("lib");
        std::fs::create_dir_all(&sub_git_dir).unwrap();
        write_file(&sub_git_dir.join("HEAD"), "ref: refs/heads/main\n");
        write_file(
            &sub.join(".git"),
            &format!("gitdir: {}\n", sub_git_dir.to_string_lossy()),
        );
        assert_eq!(
            context_contents(&sub_src, &agent_dir),
            vec![
                "superproject instructions".to_string(),
                "submodule instructions".to_string()
            ]
        );
    }

    #[test]
    fn keeps_climbing_past_an_ordinary_repo_root() {
        let root = TempDir::new("wt8");
        let agent_dir = root.str("agent");
        let outer = root.join("outer");
        let repo = outer.join("repo");
        let leaf = repo.join("src");
        std::fs::create_dir_all(&leaf).unwrap();
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        write_file(&repo.join(".git").join("HEAD"), "ref: refs/heads/main\n");
        write_file(&outer.join("AGENTS.md"), "outer instructions");
        write_file(&repo.join("AGENTS.md"), "repo instructions");
        write_file(&leaf.join("AGENTS.md"), "leaf instructions");
        assert_eq!(
            context_contents(&leaf, &agent_dir),
            vec![
                "outer instructions".to_string(),
                "repo instructions".to_string(),
                "leaf instructions".to_string()
            ]
        );
    }

    #[test]
    fn climbs_normally_when_the_gitdir_target_does_not_exist() {
        let root = TempDir::new("wt9");
        let agent_dir = root.str("agent");
        let repo = root.join("corrupt");
        let src = repo.join("src");
        std::fs::create_dir_all(&src).unwrap();
        write_file(
            &repo.join(".git"),
            "gitdir: /nonexistent/path/worktrees/feat\n",
        );
        write_file(&repo.join("AGENTS.md"), "repo instructions");
        write_file(&src.join("AGENTS.md"), "src instructions");
        assert_eq!(
            context_contents(&src, &agent_dir),
            vec![
                "repo instructions".to_string(),
                "src instructions".to_string()
            ]
        );
    }

    // -- theme color mode (resource-loader-theme.test.ts) -------------------

    fn write_capability_theme(root: &TempDir) -> String {
        let mut theme = dark_theme_json();
        theme["name"] = serde_json::Value::String("capability-test".to_string());
        theme["colors"]["userMessageBg"] = serde_json::Value::String("#3c3544".to_string());
        let theme_path = root.join("capability-test.json");
        write_file(&theme_path, &serde_json::to_string(&theme).unwrap());
        theme_path.to_string_lossy().to_string()
    }

    fn loaded_capability_theme_bg(theme_path: &str, settings: Settings) -> String {
        let settings_manager =
            SettingsManager::in_memory(&settings, SettingsManagerCreateOptions::default());
        let options = DefaultResourceLoaderOptions {
            cwd: "/nonexistent-cwd-rl".to_string(),
            agent_dir: "/nonexistent-agent-rl".to_string(),
            settings_manager: Some(settings_manager),
            additional_theme_paths: vec![theme_path.to_string()],
            no_extensions: true,
            no_skills: true,
            no_prompt_templates: true,
            no_context_files: true,
            ..Default::default()
        };
        let mut loader = DefaultResourceLoader::new(options);
        loader.reload(Default::default());
        let theme = loader
            .get_themes()
            .items
            .into_iter()
            .find(|theme| theme.name.as_deref() == Some("capability-test"))
            .expect("capability-test theme loaded");
        theme.bg("userMessageBg", "x").unwrap()
    }

    /// Regression test for #9973: the `terminal.trueColor` setting wins over
    /// the `PI_TRUE_COLOR` environment override.
    #[test]
    fn uses_true_color_setting_over_256_color_environment() {
        let _env_guard = crate::utils::test_env_lock();
        let root = TempDir::new("theme-tc");
        let theme_path = write_capability_theme(&root);
        crate::utils::tui_compat::set_capability_overrides(Default::default());
        crate::utils::tui_compat::reset_capabilities_cache();
        std::env::set_var("PI_TRUE_COLOR", "0");
        let bg = loaded_capability_theme_bg(
            &theme_path,
            Settings::from_value(serde_json::json!({"terminal": {"trueColor": true}})),
        );
        std::env::remove_var("PI_TRUE_COLOR");
        crate::utils::tui_compat::reset_capabilities_cache();
        assert_eq!(bg, "\x1b[48;2;60;53;68mx\x1b[49m");
    }

    #[test]
    fn uses_256_color_mode_when_setting_is_false_despite_truecolor_environment() {
        let _env_guard = crate::utils::test_env_lock();
        let root = TempDir::new("theme-256");
        let theme_path = write_capability_theme(&root);
        crate::utils::tui_compat::set_capability_overrides(Default::default());
        crate::utils::tui_compat::reset_capabilities_cache();
        std::env::set_var("PI_TRUE_COLOR", "1");
        let bg = loaded_capability_theme_bg(
            &theme_path,
            Settings::from_value(serde_json::json!({"terminal": {"trueColor": false}})),
        );
        std::env::remove_var("PI_TRUE_COLOR");
        crate::utils::tui_compat::reset_capabilities_cache();
        assert_eq!(bg, "\x1b[48;5;59mx\x1b[49m");
    }

    #[test]
    fn returns_to_automatic_detection_after_an_explicit_setting_is_removed() {
        let _env_guard = crate::utils::test_env_lock();
        let ws = Workspace::new();
        let theme_path = write_capability_theme(&ws._root);
        std::env::set_var("PI_TRUE_COLOR", "1");
        crate::utils::tui_compat::set_capability_overrides(Default::default());
        crate::utils::tui_compat::reset_capabilities_cache();
        let settings_path = ws._root.join("agent/settings.json");
        write_file(
            &settings_path,
            &serde_json::to_string(&serde_json::json!({"terminal": {"trueColor": false}})).unwrap(),
        );
        let settings_manager = SettingsManager::create(&ws.cwd, &ws.agent_dir, Default::default());
        let options = DefaultResourceLoaderOptions {
            cwd: ws.cwd.clone(),
            agent_dir: ws.agent_dir.clone(),
            settings_manager: Some(settings_manager),
            additional_theme_paths: vec![theme_path.clone()],
            no_extensions: true,
            no_skills: true,
            no_prompt_templates: true,
            no_context_files: true,
            ..Default::default()
        };
        let mut loader = DefaultResourceLoader::new(options);
        loader.reload(Default::default());
        {
            let themes = loader.get_themes();
            let theme = themes
                .items
                .iter()
                .find(|theme| theme.name.as_deref() == Some("capability-test"))
                .unwrap();
            assert_eq!(
                theme.bg("userMessageBg", "x").unwrap(),
                "\x1b[48;5;59mx\x1b[49m"
            );
        }

        write_file(&settings_path, "{}");
        loader.reload(Default::default());
        std::env::remove_var("PI_TRUE_COLOR");
        crate::utils::tui_compat::reset_capabilities_cache();
        let themes = loader.get_themes();
        let theme = themes
            .items
            .iter()
            .find(|theme| theme.name.as_deref() == Some("capability-test"))
            .unwrap();
        assert_eq!(
            theme.bg("userMessageBg", "x").unwrap(),
            "\x1b[48;2;60;53;68mx\x1b[49m"
        );
    }

    // Upstream cases that execute `.ts` extension modules through the JS
    // runtime — ported below now that the loader bridges to the embedded
    // host (D107). Cases constructing a full `AgentSession` defer with that
    // port.

    fn write_extension(path: &Path, name: &str, body: &str) -> String {
        write_file(&path.join(name), body);
        path.join(name).to_string_lossy().to_string()
    }

    /// Runner over a loaded extension set plus its runtime (upstream
    /// `new ExtensionRunner(extensionsResult.extensions,
    /// extensionsResult.runtime, cwd, ...)`).
    async fn runner_for(
        ws: &Workspace,
        loader: &DefaultResourceLoader,
    ) -> crate::core::extensions::runner::ExtensionRunner {
        let extensions_result = loader.get_extensions().clone();
        let session_manager = Arc::new(Mutex::new(
            crate::core::session_manager::SessionManager::in_memory(None, None, None).unwrap(),
        ));
        let model_runtime = crate::core::model_runtime::ModelRuntime::create(Default::default())
            .await
            .unwrap();
        let model_registry = Arc::new(crate::core::model_registry::ModelRegistry::new(
            model_runtime,
        ));
        crate::core::extensions::runner::ExtensionRunner::new(
            extensions_result.extensions,
            extensions_result.runtime,
            &ws.cwd,
            session_manager,
            model_registry,
        )
    }

    #[test]
    fn does_not_treat_a_project_manifest_as_the_owner_of_a_project_extension() {
        let ws = Workspace::new();
        write_file(
			&ws._root.join("project/package.json"),
			&serde_json::to_string(
				&serde_json::json!({ "dependencies": { "@earendil-works/pi-coding-agent": "1.0.0" } }),
			)
			.unwrap(),
		);
        write_extension(
            &ws._root.join("project/.pi/extensions"),
            "project-extension.ts",
            "export default function() {}",
        );

        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());

        let extensions = loader.get_extensions();
        assert_eq!(extensions.extensions.len(), 1);
        assert!(extensions.warnings.clone().unwrap_or_default().is_empty());
    }

    #[test]
    fn warns_about_host_dependencies_in_an_extension_package_manifest() {
        // Regression for #9863.
        let ws = Workspace::new();
        let package_root = ws.str("extension-package");
        write_file(
			&Path::new(&package_root).join("package.json"),
			&serde_json::to_string(
				&serde_json::json!({ "dependencies": { "@earendil-works/pi-coding-agent": "1.0.0" } }),
			)
			.unwrap(),
		);
        write_extension(
            &Path::new(&package_root).join("extensions"),
            "package-extension.ts",
            "export default function() {}",
        );

        let settings_manager = SettingsManager::in_memory(
            &Settings::from_value(serde_json::json!({ "packages": [package_root] })),
            SettingsManagerCreateOptions::default(),
        );
        let mut loader = ws.loader(|options| {
            options.settings_manager = Some(settings_manager);
        });
        loader.reload(Default::default());

        let extensions = loader.get_extensions();
        assert_eq!(extensions.extensions.len(), 1);
        assert_eq!(
			extensions.warnings.clone().unwrap_or_default(),
			vec![LoadRecord {
				path: Path::new(&package_root).join("package.json").to_string_lossy().to_string(),
				error: "Host-provided extension packages must be declared in peerDependencies with a \"*\" range, \
					not dependencies: @earendil-works/pi-coding-agent. Installed copies can bypass the extension \
					loader and create duplicate runtime modules."
					.to_string(),
			}]
		);
    }

    #[test]
    #[should_panic(expected = "Failed to parse")]
    fn fails_when_an_extension_package_manifest_cannot_be_parsed() {
        let ws = Workspace::new();
        let package_root = ws.str("invalid-extension-package");
        write_file(&Path::new(&package_root).join("package.json"), "{");
        write_extension(
            &Path::new(&package_root).join("extensions"),
            "package-extension.ts",
            "export default function() {}",
        );

        let settings_manager = SettingsManager::in_memory(
            &Settings::from_value(serde_json::json!({ "packages": [package_root] })),
            SettingsManagerCreateOptions::default(),
        );
        let mut loader = ws.loader(|options| {
            options.settings_manager = Some(settings_manager);
        });
        // Upstream `reload()` rejects with the manifest SyntaxError; the port
        // lets the parse failure propagate as a panic (D58 note).
        loader.reload(Default::default());
    }

    #[test]
    fn loads_symlinked_user_and_project_extensions_once() {
        let ws = Workspace::new();
        let shared_ext_dir = ws.str("shared-extensions");
        std::fs::create_dir_all(&shared_ext_dir).unwrap();
        write_extension(
			&Path::new(&shared_ext_dir),
			"shared.ts",
			"export default function(pi) {\n\tpi.registerCommand(\"shared\", {\n\t\tdescription: \"shared command\",\n\t\thandler: async () => {},\n\t});\n}\n",
		);

        std::fs::create_dir_all(&ws.agent_dir).unwrap();
        std::fs::create_dir_all(ws.str("project/.pi")).unwrap();
        std::os::unix::fs::symlink(&shared_ext_dir, ws.str("agent/extensions")).unwrap();
        std::os::unix::fs::symlink(&shared_ext_dir, ws.str("project/.pi/extensions")).unwrap();

        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());

        let extensions = loader.get_extensions();
        assert_eq!(extensions.extensions.len(), 1);
        assert!(extensions.errors.is_empty());

        // mergePaths processes project paths before user paths, so the project
        // alias is the canonical survivor.
        assert_eq!(
            extensions.extensions[0].path,
            ws.str("project/.pi/extensions/shared.ts")
        );
    }

    #[test]
    fn loads_user_extensions_before_trust_and_reuses_them_after_trust_resolves() {
        let ws = Workspace::new();
        let user_ext_dir = ws.str("agent/extensions");
        let project_ext_dir = ws.str("project/.pi/extensions");
        std::fs::create_dir_all(&user_ext_dir).unwrap();
        std::fs::create_dir_all(&project_ext_dir).unwrap();
        // Unique per run: the embedded host shares one JS globalThis across
        // tests (upstream seeds its counter key the same way).
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let load_count_key = format!(
            "__piTrustPreloadCount_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let counter_literal = serde_json::to_string(&load_count_key).unwrap();

        let user_path = write_extension(
            &Path::new(&user_ext_dir),
            "user.ts",
            &format!(
                "globalThis[{counter_literal}] = (globalThis[{counter_literal}] ?? 0) + 1;\n\
				 export default function(pi) {{\n\
				 \tpi.on(\"project_trust\", () => ({{ trusted: \"yes\" }}));\n\
				 \tpi.registerCommand(\"user-trust\", {{\n\
				 \t\tdescription: \"user trust\",\n\
				 \t\thandler: async () => {{}},\n\
				 \t}});\n\
				 }}\n"
            ),
        );
        write_extension(
			&Path::new(&project_ext_dir),
			"project.ts",
			"export default function(pi) {\n\tpi.registerCommand(\"project-trusted\", {\n\t\tdescription: \"project trusted\",\n\t\thandler: async () => {},\n\t});\n}\n",
		);

        let pre_trust_paths: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let pre_trust_capture = Arc::clone(&pre_trust_paths);
        let user_path_for_closure = user_path.clone();
        let mut loader = ws.loader(|_| {});
        loader.reload(ResourceLoaderReloadOptions {
            resolve_project_trust: Some(Box::new(
                move |extensions_result: &LoadExtensionsResult| {
                    let paths: Vec<String> = extensions_result
                        .extensions
                        .iter()
                        .map(|extension| extension.path.clone())
                        .collect();
                    assert_eq!(paths, vec![user_path_for_closure.clone()]);
                    *pre_trust_capture.lock() = paths;
                    true
                },
            )),
        });

        let extensions = loader.get_extensions();
        let paths: Vec<String> = extensions
            .extensions
            .iter()
            .map(|extension| extension.path.clone())
            .collect();
        assert_eq!(
            paths,
            vec![
                ws.str("project/.pi/extensions/project.ts"),
                user_path.clone()
            ]
        );

        // Upstream asserts the module-level `globalThis` counter is 1 — the
        // module must not re-import after trust resolves. The embedded host
        // exposes no eval seam, so the bridge's own import counter stands in
        // (same observable).
        assert_eq!(
            crate::core::extensions::loader::bridge_import_count(&user_path),
            1,
            "user extension must load exactly once"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn keeps_both_extensions_loaded_when_command_names_collide() {
        let ws = Workspace::new();
        let user_ext_dir = ws.str("agent/extensions");
        let project_ext_dir = ws.str("project/.pi/extensions");
        std::fs::create_dir_all(&user_ext_dir).unwrap();
        std::fs::create_dir_all(&project_ext_dir).unwrap();

        write_extension(
			&Path::new(&project_ext_dir),
			"project.ts",
			"export default function(pi) {\n\tpi.registerCommand(\"deploy\", {\n\t\tdescription: \"project deploy\",\n\t\thandler: async () => {},\n\t});\n\tpi.registerCommand(\"project-only\", {\n\t\tdescription: \"project only\",\n\t\thandler: async () => {},\n\t});\n}\n",
		);
        write_extension(
			&Path::new(&user_ext_dir),
			"user.ts",
			"export default function(pi) {\n\tpi.registerCommand(\"deploy\", {\n\t\tdescription: \"user deploy\",\n\t\thandler: async () => {},\n\t});\n\tpi.registerCommand(\"user-only\", {\n\t\tdescription: \"user only\",\n\t\thandler: async () => {},\n\t});\n}\n",
		);

        let cwd = ws.cwd.clone();
        let agent_dir = ws.agent_dir.clone();
        let loader = tokio::task::spawn_blocking(move || {
            let mut loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
                cwd,
                agent_dir,
                ..Default::default()
            });
            loader.reload(Default::default());
            loader
        })
        .await
        .unwrap();

        let extensions = loader.get_extensions();
        assert_eq!(extensions.extensions.len(), 2);
        assert!(!extensions
            .errors
            .iter()
            .any(|error| error.error.contains("Command \"/deploy\" conflicts")));

        let runner = runner_for(&ws, &loader).await;
        assert_eq!(
            runner
                .get_command("deploy:1")
                .unwrap()
                .description
                .as_deref(),
            Some("project deploy")
        );
        assert_eq!(
            runner
                .get_command("deploy:2")
                .unwrap()
                .description
                .as_deref(),
            Some("user deploy")
        );
        assert_eq!(
            runner
                .get_command("project-only")
                .unwrap()
                .description
                .as_deref(),
            Some("project only")
        );
        assert_eq!(
            runner
                .get_command("user-only")
                .unwrap()
                .description
                .as_deref(),
            Some("user only")
        );

        let commands = runner.get_registered_commands();
        let invocation_names: Vec<String> = commands
            .iter()
            .map(|command| command.invocation_name.clone())
            .collect();
        assert_eq!(
            invocation_names,
            vec!["deploy:1", "project-only", "deploy:2", "user-only"]
        );
    }

    #[test]
    fn detects_tool_conflicts_between_extensions() {
        let ws = Workspace::new();
        let ext1_dir = ws.str("agent/extensions/ext1");
        let ext2_dir = ws.str("agent/extensions/ext2");
        std::fs::create_dir_all(&ext1_dir).unwrap();
        std::fs::create_dir_all(&ext2_dir).unwrap();

        write_extension(
			&Path::new(&ext1_dir),
			"index.ts",
			"import type { ExtensionAPI } from \"@earendil-works/pi-coding-agent\";\nimport { Type } from \"typebox\";\nexport default function(pi: ExtensionAPI) {\n  pi.registerTool({\n    name: \"duplicate-tool\",\n    description: \"First\",\n    parameters: Type.Object({}),\n    execute: async () => ({ result: \"1\" }),\n  });\n}\n",
		);
        write_extension(
			&Path::new(&ext2_dir),
			"index.ts",
			"import type { ExtensionAPI } from \"@earendil-works/pi-coding-agent\";\nimport { Type } from \"typebox\";\nexport default function(pi: ExtensionAPI) {\n  pi.registerTool({\n    name: \"duplicate-tool\",\n    description: \"Second\",\n    parameters: Type.Object({}),\n    execute: async () => ({ result: \"2\" }),\n  });\n}\n",
		);

        let mut loader = ws.loader(|_| {});
        loader.reload(Default::default());

        let errors = loader.get_extensions().errors.clone();
        assert!(
            errors
                .iter()
                .any(|error| error.error.contains("duplicate-tool")
                    && error.error.contains("conflicts")),
            "expected a duplicate-tool conflict, got {errors:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prefers_explicit_cli_extensions_over_discovered_extensions_when_commands_and_tools_conflict(
    ) {
        let ws = Workspace::new();
        let global_ext_dir = ws.str("agent/extensions");
        std::fs::create_dir_all(&global_ext_dir).unwrap();
        let explicit_ext_path = ws.str("explicit-extension.ts");

        write_extension(
			&Path::new(&global_ext_dir),
			"global.ts",
			"import type { ExtensionAPI } from \"@earendil-works/pi-coding-agent\";\nimport { Type } from \"typebox\";\nexport default function(pi: ExtensionAPI) {\n  pi.registerTool({\n    name: \"duplicate-tool\",\n    description: \"global tool\",\n    parameters: Type.Object({}),\n    execute: async () => ({ result: \"global\" }),\n  });\n  pi.registerCommand(\"deploy\", {\n    description: \"global command\",\n    handler: async () => {},\n  });\n}\n",
		);
        write_extension(
			&ws._root.0,
			"explicit-extension.ts",
			"import type { ExtensionAPI } from \"@earendil-works/pi-coding-agent\";\nimport { Type } from \"typebox\";\nexport default function(pi: ExtensionAPI) {\n  pi.registerTool({\n    name: \"duplicate-tool\",\n    description: \"explicit tool\",\n    parameters: Type.Object({}),\n    execute: async () => ({ result: \"explicit\" }),\n  });\n  pi.registerCommand(\"deploy\", {\n    description: \"explicit command\",\n    handler: async () => {},\n  });\n}\n",
		);

        let cwd = ws.cwd.clone();
        let agent_dir = ws.agent_dir.clone();
        let explicit_for_options = explicit_ext_path.clone();
        let loader = tokio::task::spawn_blocking(move || {
            let mut loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
                cwd,
                agent_dir,
                additional_extension_paths: vec![explicit_for_options],
                ..Default::default()
            });
            loader.reload(Default::default());
            loader
        })
        .await
        .unwrap();

        let extensions = loader.get_extensions();
        assert_eq!(
            extensions.extensions[0].path,
            ws.str("explicit-extension.ts")
        );

        let runner = runner_for(&ws, &loader).await;
        assert_eq!(
            runner
                .get_command("deploy:1")
                .unwrap()
                .description
                .as_deref(),
            Some("explicit command")
        );
        assert_eq!(
            runner
                .get_command("deploy:2")
                .unwrap()
                .description
                .as_deref(),
            Some("global command")
        );
        assert_eq!(
            runner
                .get_tool_definition("duplicate-tool")
                .unwrap()
                .description,
            "explicit tool"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn leaves_out_replaceable_extensions_whose_names_another_extension_registers() {
        // A third-party MCP extension registering /mcp replaces the built-in
        // one instead of both running.
        let ws = Workspace::new();
        let global_ext_dir = ws.str("agent/extensions");
        std::fs::create_dir_all(&global_ext_dir).unwrap();
        write_extension(
			&Path::new(&global_ext_dir),
			"other-mcp.ts",
			"import type { ExtensionAPI } from \"@earendil-works/pi-coding-agent\";\nexport default function(pi: ExtensionAPI) {\n  pi.registerCommand(\"mcp\", { description: \"other mcp\", handler: async () => {} });\n}\n",
		);

        let cwd = ws.cwd.clone();
        let agent_dir = ws.agent_dir.clone();
        let extension_factories = vec![
            InlineExtension::Named {
                name: "mcp".to_string(),
                factory: Arc::new(|pi: crate::core::extensions::types::ExtensionApiHandle| {
                    Box::pin(async move {
                        pi.register_command(
                            "mcp",
                            RegisteredCommandOptions {
                                description: Some("built-in mcp".to_string()),
                                ..no_pause()
                            },
                        );
                        Ok(())
                    })
                        as std::pin::Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
                }),
                hidden: None,
                replaceable: Some(true),
                builtin: None,
            },
            InlineExtension::Named {
                name: "llama".to_string(),
                factory: Arc::new(|pi: crate::core::extensions::types::ExtensionApiHandle| {
                    Box::pin(async move {
                        pi.register_command(
                            "llama",
                            RegisteredCommandOptions {
                                description: Some("built-in llama".to_string()),
                                ..no_pause()
                            },
                        );
                        Ok(())
                    })
                        as std::pin::Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
                }),
                hidden: None,
                replaceable: Some(true),
                builtin: None,
            },
        ];
        let loader = tokio::task::spawn_blocking(move || {
            let mut loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
                cwd,
                agent_dir,
                extension_factories,
                ..Default::default()
            });
            loader.reload(Default::default());
            loader
        })
        .await
        .unwrap();

        let extensions = loader.get_extensions();
        let paths: Vec<String> = extensions
            .extensions
            .iter()
            .map(|extension| extension.path.clone())
            .collect();
        assert_eq!(
            paths,
            vec![
                ws.str("agent/extensions/other-mcp.ts"),
                "<inline:llama>".to_string()
            ]
        );
        assert!(extensions.errors.is_empty());

        let runner = runner_for(&ws, &loader).await;
        assert_eq!(
            runner.get_command("mcp").unwrap().description.as_deref(),
            Some("other mcp")
        );
        assert_eq!(
            runner.get_command("llama").unwrap().description.as_deref(),
            Some("built-in llama")
        );
    }

    #[test]
    fn loads_builtin_extensions_after_file_extensions_with_and_without_trust_resolution() {
        let ws = Workspace::new();
        let user_ext_dir = ws.str("agent/extensions");
        std::fs::create_dir_all(&user_ext_dir).unwrap();
        write_extension(
            &Path::new(&user_ext_dir),
            "user.ts",
            "export default function() {}",
        );
        std::fs::create_dir_all(ws.str("project/.pi")).unwrap();
        // A project override gives the built-in project scope, which must not
        // move it ahead.
        write_file(
            &ws._root.join("project/.pi/settings.json"),
            &serde_json::to_string(&serde_json::json!({ "extensions": ["+builtin:mcp"] })).unwrap(),
        );
        let mut loader = ws.loader(|options| {
            options.extension_factories = vec![InlineExtension::Named {
                name: "mcp".to_string(),
                factory: Arc::new(|_pi: crate::core::extensions::types::ExtensionApiHandle| {
                    Box::pin(async { Ok(()) })
                        as std::pin::Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
                }),
                hidden: None,
                replaceable: None,
                builtin: Some(true),
            }];
        });
        let expected = vec![
            ws.str("agent/extensions/user.ts"),
            "builtin:mcp".to_string(),
        ];

        loader.reload(ResourceLoaderReloadOptions {
            resolve_project_trust: Some(Box::new(|_extensions_result: &LoadExtensionsResult| true)),
        });
        let paths: Vec<String> = loader
            .get_extensions()
            .extensions
            .iter()
            .map(|extension| extension.path.clone())
            .collect();
        assert_eq!(paths, expected);
        loader.reload(Default::default());
        let paths: Vec<String> = loader
            .get_extensions()
            .extensions
            .iter()
            .map(|extension| extension.path.clone())
            .collect();
        assert_eq!(paths, expected);
    }
}
