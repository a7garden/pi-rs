//! Port of `packages/coding-agent/src/config.ts` — package/app constants,
//! user-config paths, install-method detection, self-update commands,
//! package-asset paths and install-change detection.
//!
//! Upstream derives `APP_NAME`/`CONFIG_DIR_NAME` from `package.json`
//! `piConfig` overrides at runtime; the Rust port pins the defaults (`pi`,
//! `.pi`). JS runtime detection (Bun binary / bundled Node) has no Rust
//! counterpart and becomes explicit process flags (deviation D18); module
//! `import.meta`/`__dirname` asset resolution starts from the running
//! executable's directory.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::utils::paths::{home_dir, join, normalize_path, resolve_path_cwd, PathInputOptions};
use serde_json::Value;

/// Upstream `APP_NAME` (`pkg.piConfig?.name || "pi"`).
pub const APP_NAME: &str = "pi";

/// Upstream `APP_TITLE` (`piConfig ? APP_NAME : "π"`).
pub const APP_TITLE: &str = "π";

/// Upstream `CONFIG_DIR_NAME` (`pkg.piConfig?.configDir || ".pi"`).
pub const CONFIG_DIR_NAME: &str = ".pi";

/// Upstream `VERSION` (`pkg.version`); pinned to the workspace port version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// e.g., PI_CODING_AGENT_DIR (upstream `ENV_AGENT_DIR`).
pub const ENV_AGENT_DIR: &str = "PI_CODING_AGENT_DIR";

/// e.g., PI_CODING_AGENT_SESSION_DIR (upstream `ENV_SESSION_DIR`).
pub const ENV_SESSION_DIR: &str = "PI_CODING_AGENT_SESSION_DIR";

/// Upstream `expandTildePath` (a `normalizePath` wrapper).
pub fn expand_tilde_path(path: &str) -> String {
    normalize_path(path, &PathInputOptions::default())
}

/// Get the agent config directory (e.g., ~/.pi/agent/) — `PI_CODING_AGENT_DIR`
/// override wins (`getAgentDir`).
pub fn get_agent_dir() -> String {
    if let Ok(env_dir) = std::env::var(ENV_AGENT_DIR) {
        if !env_dir.is_empty() {
            return expand_tilde_path(&env_dir);
        }
    }
    join(&home_dir(), &format!("{CONFIG_DIR_NAME}/agent"))
}

/// Get path to sessions directory (`getSessionsDir`).
pub fn get_sessions_dir() -> String {
    join(&get_agent_dir(), "sessions")
}

// =============================================================================
// Runtime / install detection (deviation D18)
// =============================================================================

static IS_BUN_BINARY: AtomicBool = AtomicBool::new(false);
static IS_BUNDLED_NODE: AtomicBool = AtomicBool::new(false);

/// Upstream detects Bun-compiled-binary / esbuild-bundled-Node runtimes from
/// `import.meta.url` and bundler globals. A Rust binary has neither, so the
/// port exposes explicit process flags that a release harness may set at
/// startup; they default to `false` (plain binary).
pub fn set_runtime_context(is_bun_binary: bool, is_bundled_node: bool) {
    IS_BUN_BINARY.store(is_bun_binary, Ordering::Relaxed);
    IS_BUNDLED_NODE.store(is_bundled_node, Ordering::Relaxed);
}

/// Upstream `isBunBinary`.
pub fn is_bun_binary() -> bool {
    IS_BUN_BINARY.load(Ordering::Relaxed)
}

/// Upstream `isBundledNode`.
pub fn is_bundled_node() -> bool {
    IS_BUNDLED_NODE.load(Ordering::Relaxed)
}

/// Upstream `InstallMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallMethod {
    BunBinary,
    Npm,
    Pnpm,
    Yarn,
    Bun,
    Unknown,
}

/// Directory of the running executable (upstream `__dirname` /
/// `process.execPath` analogue).
fn exe_dir() -> Option<String> {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|dir| dir.to_string_lossy().into_owned()))
}

fn path_dir(path: &str) -> Option<String> {
    Path::new(path)
        .parent()
        .map(|dir| dir.to_string_lossy().into_owned())
}

fn path_basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn forward_slashes(path: &str) -> String {
    path.replace('\\', "/")
}

/// Upstream `detectInstallMethod`. The Bun binary check uses the explicit
/// runtime flag; path probes run against the package dir and executable path.
pub fn detect_install_method() -> InstallMethod {
    if is_bun_binary() {
        return InstallMethod::BunBinary;
    }

    let resolved_path = format!(
        "{}\0{}",
        get_package_dir(),
        std::env::current_exe()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_default()
    )
    .to_lowercase();
    let resolved_path = forward_slashes(&resolved_path);

    if resolved_path.contains("/pnpm/") || resolved_path.contains("/.pnpm/") {
        return InstallMethod::Pnpm;
    }
    if resolved_path.contains("/yarn/") || resolved_path.contains("/.yarn/") {
        return InstallMethod::Yarn;
    }
    if resolved_path.contains("/install/global/node_modules/") {
        return InstallMethod::Bun;
    }
    if resolved_path.contains("/npm/") || resolved_path.contains("/node_modules/") {
        return InstallMethod::Npm;
    }

    InstallMethod::Unknown
}

// =============================================================================
// Self-update commands
// =============================================================================

/// One command step of an upstream `SelfUpdateCommand`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfUpdateCommandStep {
    pub command: String,
    pub args: Vec<String>,
    pub display: String,
}

/// Upstream `SelfUpdateCommand`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfUpdateCommand {
    pub command: String,
    pub args: Vec<String>,
    pub display: String,
    pub steps: Option<Vec<SelfUpdateCommandStep>>,
}

/// Upstream `SelfUpdatePackageTarget`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelfUpdatePackageTarget {
    Simple(String),
    Filtered {
        package_name: String,
        install_spec: Option<String>,
    },
}

struct NormalizedTarget {
    package_name: String,
    install_spec: String,
}

/// Upstream `normalizeSelfUpdatePackageTarget`.
fn normalize_self_update_package_target(target: &SelfUpdatePackageTarget) -> NormalizedTarget {
    match target {
        SelfUpdatePackageTarget::Simple(name) => NormalizedTarget {
            package_name: name.clone(),
            install_spec: name.clone(),
        },
        SelfUpdatePackageTarget::Filtered {
            package_name,
            install_spec,
        } => NormalizedTarget {
            package_name: package_name.clone(),
            install_spec: install_spec.clone().unwrap_or_else(|| package_name.clone()),
        },
    }
}

/// Upstream `makeSelfUpdateCommand`.
fn make_self_update_command(
    install_step: SelfUpdateCommandStep,
    uninstall_step: Option<SelfUpdateCommandStep>,
) -> SelfUpdateCommand {
    let Some(uninstall_step) = uninstall_step else {
        return SelfUpdateCommand {
            command: install_step.command,
            args: install_step.args,
            display: install_step.display,
            steps: None,
        };
    };
    SelfUpdateCommand {
        command: install_step.command.clone(),
        args: install_step.args.clone(),
        display: format!("{} && {}", uninstall_step.display, install_step.display),
        steps: Some(vec![uninstall_step, install_step]),
    }
}

/// Upstream `makeSelfUpdateCommandStep`.
fn make_self_update_command_step(command: &str, args: &[String]) -> SelfUpdateCommandStep {
    SelfUpdateCommandStep {
        command: command.to_string(),
        args: args.to_vec(),
        display: std::iter::once(command.to_string())
            .chain(args.iter().cloned())
            .map(|arg| {
                if arg.chars().any(char::is_whitespace) {
                    format!("\"{arg}\"")
                } else {
                    arg
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// Upstream `getInferredNpmInstall`: derive `{ root, prefix }` from the
/// package dir's `node_modules[/@scope]/lib` path shape.
fn get_inferred_npm_install() -> Option<(String, String)> {
    let package_dir = get_package_dir();
    let parent = path_dir(&package_dir)?;
    let parent_name = path_basename(&parent);
    let mut root = None;
    if parent_name.starts_with('@') && path_basename(&path_dir(&parent)?) == "node_modules" {
        root = path_dir(&parent);
    } else if parent_name == "node_modules" {
        root = Some(parent);
    }
    let root = root?;
    let root_parent = path_dir(&root)?;
    if path_basename(&root_parent) == "lib" {
        return path_dir(&root_parent).map(|prefix| (root, prefix));
    }
    // Windows global npm prefixes use `<prefix>\node_modules`, which is
    // indistinguishable from local project installs by path shape alone. Do
    // not infer unsupported Windows custom prefixes without `npm root -g`
    // evidence.
    None
}

/// Upstream `readCommandOutput` (sync child process, trimmed stdout).
fn read_command_output(
    command: &str,
    args: &[String],
    require_success: bool,
) -> Result<Option<String>, String> {
    let output = std::process::Command::new(command)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();
    let result = match output {
        Ok(result) => result,
        Err(error) => {
            if require_success {
                let joined = format!("{command} {}", args.join(" "));
                return Err(format!("Failed to run {joined}: {error}"));
            }
            return Ok(None);
        }
    };
    if result.status.success() {
        let stdout = String::from_utf8_lossy(&result.stdout).trim().to_string();
        return Ok(if stdout.is_empty() {
            None
        } else {
            Some(stdout)
        });
    }
    if require_success {
        let mut argv: Vec<String> = vec![command.to_string()];
        argv.extend(args.iter().cloned());
        let joined = argv.join(" ");
        let stderr = String::from_utf8_lossy(&result.stderr).trim().to_string();
        let reason = if stderr.is_empty() {
            match result.status.code() {
                Some(code) => format!("exit code {code}"),
                None => "exit code unknown".to_string(),
            }
        } else {
            stderr
        };
        return Err(format!("Failed to run {joined}: {reason}"));
    }
    Ok(None)
}

/// Upstream `getSelfUpdateCommandForMethod`.
fn get_self_update_command_for_method(
    method: InstallMethod,
    installed_package_name: &str,
    update_package_target: &SelfUpdatePackageTarget,
    npm_command: Option<&[String]>,
) -> Result<Option<SelfUpdateCommand>, String> {
    let target = normalize_self_update_package_target(update_package_target);
    let same_package = target.package_name == installed_package_name;
    match method {
        InstallMethod::BunBinary => Ok(None),
        InstallMethod::Pnpm => {
            let pnpm_root =
                read_command_output("pnpm", &["root".to_string(), "-g".to_string()], false)?;
            let matched = if pnpm_root.is_some() {
                None
            } else {
                match_pnpm_global_dir(&get_package_dir())
            };
            let bin_dir_args: Vec<String> = match &matched {
                Some(matched) => vec![format!(
                    "--config.global-bin-dir={}",
                    std::env::var("PNPM_HOME")
                        .unwrap_or_else(|_| path_dir(&path_dir(matched).unwrap_or_default())
                            .unwrap_or_default())
                )],
                None => Vec::new(),
            };
            let mut install_args = vec![
                "install".to_string(),
                "-g".to_string(),
                "--ignore-scripts".to_string(),
                "--config.minimumReleaseAge=0".to_string(),
            ];
            install_args.extend(bin_dir_args.iter().cloned());
            install_args.push(target.install_spec.clone());
            let install_step = make_self_update_command_step("pnpm", &install_args);
            let uninstall_step = if same_package {
                None
            } else {
                let mut uninstall_args = vec!["remove".to_string(), "-g".to_string()];
                uninstall_args.extend(bin_dir_args.iter().cloned());
                uninstall_args.push(installed_package_name.to_string());
                Some(make_self_update_command_step("pnpm", &uninstall_args))
            };
            Ok(Some(make_self_update_command(install_step, uninstall_step)))
        }
        InstallMethod::Yarn => {
            let install_step = make_self_update_command_step(
                "yarn",
                &[
                    "global".to_string(),
                    "add".to_string(),
                    "--ignore-scripts".to_string(),
                    target.install_spec.clone(),
                ],
            );
            let uninstall_step = if same_package {
                None
            } else {
                Some(make_self_update_command_step(
                    "yarn",
                    &[
                        "global".to_string(),
                        "remove".to_string(),
                        installed_package_name.to_string(),
                    ],
                ))
            };
            Ok(Some(make_self_update_command(install_step, uninstall_step)))
        }
        InstallMethod::Bun => {
            let install_step = make_self_update_command_step(
                "bun",
                &[
                    "install".to_string(),
                    "-g".to_string(),
                    "--ignore-scripts".to_string(),
                    "--minimum-release-age=0".to_string(),
                    target.install_spec.clone(),
                ],
            );
            let uninstall_step = if same_package {
                None
            } else {
                Some(make_self_update_command_step(
                    "bun",
                    &[
                        "uninstall".to_string(),
                        "-g".to_string(),
                        installed_package_name.to_string(),
                    ],
                ))
            };
            Ok(Some(make_self_update_command(install_step, uninstall_step)))
        }
        InstallMethod::Npm => {
            let (command, npm_args): (String, Vec<String>) = match npm_command {
                Some(argv) if !argv.is_empty() => (argv[0].clone(), argv[1..].to_vec()),
                _ => ("npm".to_string(), Vec::new()),
            };
            let configured = npm_command.map(|argv| !argv.is_empty()).unwrap_or(false);
            let inferred = if configured {
                None
            } else {
                get_inferred_npm_install()
            };
            let mut prefix_args = npm_args.clone();
            if let Some((_, prefix)) = &inferred {
                prefix_args.push("--prefix".to_string());
                prefix_args.push(prefix.clone());
            }
            // pi.dev advertises releases immediately, so a configured npm age
            // gate would block the update. npm has no per-package age gate, so
            // this also lets new transitive dependency releases through.
            // Managed installs avoid this.
            let mut install_args = prefix_args.clone();
            install_args.extend([
                "install".to_string(),
                "-g".to_string(),
                "--ignore-scripts".to_string(),
                "--min-release-age=0".to_string(),
                target.install_spec.clone(),
            ]);
            let install_step = make_self_update_command_step(&command, &install_args);
            let uninstall_step = if same_package {
                None
            } else {
                let mut uninstall_args = prefix_args.clone();
                uninstall_args.extend([
                    "uninstall".to_string(),
                    "-g".to_string(),
                    installed_package_name.to_string(),
                ]);
                Some(make_self_update_command_step(&command, &uninstall_args))
            };
            Ok(Some(make_self_update_command(install_step, uninstall_step)))
        }
        InstallMethod::Unknown => Ok(None),
    }
}

/// Upstream's `/^(.*[\\/]global[\\/][^\\/]+)[\\/]\.pnpm[\\/]/` capture.
fn match_pnpm_global_dir(package_dir: &str) -> Option<String> {
    let normalized = forward_slashes(package_dir);
    let mut search_from = 0;
    while let Some(pos) = normalized[search_from..].find("/.pnpm/") {
        let absolute = search_from + pos;
        let prefix = &normalized[..absolute];
        if let Some(global_pos) = prefix.rfind("/global/") {
            let after_global = &prefix[global_pos + "/global/".len()..];
            if !after_global.is_empty() && !after_global.contains('/') {
                return Some(prefix.to_string());
            }
        }
        search_from = absolute + 1;
    }
    None
}

/// Upstream `getGlobalPackageRoots`.
fn get_global_package_roots(
    method: InstallMethod,
    npm_command: Option<&[String]>,
) -> Result<Vec<String>, String> {
    match method {
        InstallMethod::Npm => {
            let configured = npm_command.map(|argv| !argv.is_empty()).unwrap_or(false);
            let (command, npm_args): (String, Vec<String>) = match npm_command {
                Some(argv) if !argv.is_empty() => (argv[0].clone(), argv[1..].to_vec()),
                _ => ("npm".to_string(), Vec::new()),
            };
            if configured && command == "bun" {
                let bun_bin = read_command_output(
                    &command,
                    &[
                        npm_args.clone(),
                        vec!["pm".into(), "bin".into(), "-g".into()],
                    ]
                    .concat(),
                    true,
                )?;
                let mut roots = vec![join(&home_dir(), ".bun/install/global/node_modules")];
                if let Some(bun_bin) = bun_bin {
                    roots.push(join(
                        &path_dir(&bun_bin).unwrap_or_default(),
                        "install/global/node_modules",
                    ));
                }
                return Ok(roots);
            }
            let root = read_command_output(
                &command,
                &[npm_args, vec!["root".into(), "-g".into()]].concat(),
                configured,
            )?;
            let inferred = if configured {
                None
            } else {
                get_inferred_npm_install()
            };
            Ok([root, inferred.map(|(root, _)| root)]
                .into_iter()
                .flatten()
                .collect())
        }
        InstallMethod::Pnpm => {
            let root = read_command_output("pnpm", &["root".to_string(), "-g".to_string()], false)?;
            if let Some(root) = root {
                return Ok(vec![root.clone(), path_dir(&root).unwrap_or_default()]);
            }
            Ok(match_pnpm_global_dir(&get_package_dir())
                .into_iter()
                .collect())
        }
        InstallMethod::Yarn => {
            let dir =
                read_command_output("yarn", &["global".to_string(), "dir".to_string()], false)?;
            Ok(match dir {
                Some(dir) => vec![dir.clone(), join(&dir, "node_modules")],
                None => Vec::new(),
            })
        }
        InstallMethod::Bun => {
            let bun_bin = read_command_output(
                "bun",
                &["pm".to_string(), "bin".to_string(), "-g".to_string()],
                false,
            )?;
            let mut roots = vec![join(&home_dir(), ".bun/install/global/node_modules")];
            if let Some(bun_bin) = bun_bin {
                roots.push(join(
                    &path_dir(&bun_bin).unwrap_or_default(),
                    "install/global/node_modules",
                ));
            }
            Ok(roots)
        }
        InstallMethod::BunBinary | InstallMethod::Unknown => Ok(Vec::new()),
    }
}

/// Upstream `normalizeExistingPathForComparison`.
fn normalize_existing_path_for_comparison(path: &str, resolve_symlinks: bool) -> Option<String> {
    let resolved = resolve_path_cwd(path);
    if !Path::new(&resolved).exists() {
        return None;
    }
    let mut normalized = resolved;
    if resolve_symlinks {
        normalized = std::fs::canonicalize(&normalized)
            .ok()?
            .to_string_lossy()
            .into_owned();
    }
    if cfg!(windows) {
        normalized = normalized.to_lowercase();
    }
    Some(normalized)
}

/// Upstream `getPathComparisonCandidates`.
fn get_path_comparison_candidates(path: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    for candidate in [
        normalize_existing_path_for_comparison(path, false),
        normalize_existing_path_for_comparison(path, true),
    ]
    .into_iter()
    .flatten()
    {
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

/// Upstream `getEntrypointPackageDir`.
fn get_entrypoint_package_dir() -> Option<String> {
    let entrypoint = std::env::args().nth(1)?;
    let mut dir = path_dir(&entrypoint)?;
    loop {
        let parent = path_dir(&dir)?;
        if Path::new(&join(&dir, "package.json")).exists() {
            return Some(dir);
        }
        if parent == dir {
            return None;
        }
        dir = parent;
    }
}

/// Upstream `isSelfUpdatePathWritable`.
fn is_self_update_path_writable() -> bool {
    let package_dir = get_package_dir();
    let writable = |path: &str| {
        Path::new(path)
            .metadata()
            .map(|meta| meta.permissions().readonly())
            .map(|readonly| !readonly)
            .unwrap_or(false)
    };
    writable(&package_dir)
        && path_dir(&package_dir)
            .map(|dir| writable(&dir))
            .unwrap_or(false)
}

/// Upstream `isManagedByGlobalPackageManager`.
fn is_managed_by_global_package_manager(
    method: InstallMethod,
    npm_command: Option<&[String]>,
) -> Result<bool, String> {
    let mut package_dirs = vec![get_package_dir()];
    if let Some(entrypoint_dir) = get_entrypoint_package_dir() {
        package_dirs.push(entrypoint_dir);
    }
    let package_dir_candidates: Vec<String> = package_dirs
        .iter()
        .flat_map(|dir| get_path_comparison_candidates(dir))
        .collect();
    for root in get_global_package_roots(method, npm_command)? {
        for normalized_root in get_path_comparison_candidates(&root) {
            let root_prefix = if normalized_root.ends_with('/') {
                normalized_root.clone()
            } else {
                format!("{normalized_root}/")
            };
            if package_dir_candidates
                .iter()
                .any(|package_dir| package_dir.starts_with(&root_prefix))
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Upstream `getSelfUpdateCommand`.
pub fn get_self_update_command(
    package_name: &str,
    npm_command: Option<&[String]>,
    update_package_target: Option<&SelfUpdatePackageTarget>,
) -> Result<Option<SelfUpdateCommand>, String> {
    let method = detect_install_method();
    let default_target = SelfUpdatePackageTarget::Simple(package_name.to_string());
    let target = update_package_target.unwrap_or(&default_target);
    let command = get_self_update_command_for_method(method, package_name, target, npm_command)?;
    match command {
        Some(command)
            if is_managed_by_global_package_manager(method, npm_command)?
                && is_self_update_path_writable() =>
        {
            Ok(Some(command))
        }
        _ => Ok(None),
    }
}

/// Upstream `getSelfUpdateUnavailableInstruction`.
pub fn get_self_update_unavailable_instruction(
    package_name: &str,
    npm_command: Option<&[String]>,
    update_package_target: Option<&SelfUpdatePackageTarget>,
) -> Result<String, String> {
    let method = detect_install_method();
    let default_target = SelfUpdatePackageTarget::Simple(package_name.to_string());
    let target = update_package_target.unwrap_or(&default_target);
    let normalized = normalize_self_update_package_target(target);
    if method == InstallMethod::BunBinary {
        return Ok(
            "Download from: https://github.com/earendil-works/pi/releases/latest".to_string(),
        );
    }
    let command = get_self_update_command_for_method(method, package_name, target, npm_command)?;
    if let Some(command) = command {
        if is_managed_by_global_package_manager(method, npm_command)?
            && !is_self_update_path_writable()
        {
            return Ok(format!(
				"This installation is managed by a global {} install, but the install path is not writable. Update it yourself with: {}",
				install_method_name(method),
				command.display
			));
        }
        return Ok(format!(
			"This installation is not managed by a global {} install. Update it with the package manager, wrapper, or source checkout that provides it.",
			install_method_name(method)
		));
    }
    Ok(format!(
		"Update {} using the package manager, wrapper, or source checkout that provides this installation.",
		normalized.install_spec
	))
}

/// Upstream `getUpdateInstruction`.
pub fn get_update_instruction(package_name: &str) -> Result<String, String> {
    let method = detect_install_method();
    let default_target = SelfUpdatePackageTarget::Simple(package_name.to_string());
    let command = get_self_update_command_for_method(method, package_name, &default_target, None)?;
    if let Some(command) = command {
        return Ok(format!("Run: {}", command.display));
    }
    get_self_update_unavailable_instruction(package_name, None, None)
}

/// The TS string-literal form of an `InstallMethod` (used in messages).
fn install_method_name(method: InstallMethod) -> &'static str {
    match method {
        InstallMethod::BunBinary => "bun-binary",
        InstallMethod::Npm => "npm",
        InstallMethod::Pnpm => "pnpm",
        InstallMethod::Yarn => "yarn",
        InstallMethod::Bun => "bun",
        InstallMethod::Unknown => "unknown",
    }
}

// =============================================================================
// Package Asset Paths (shipped with executable)
// =============================================================================

/// Get the base directory for resolving package assets (themes, package.json,
/// README.md, CHANGELOG.md) — upstream `findNodePackageDir`.
pub fn find_node_package_dir(start_dir: &str) -> String {
    let mut dir = start_dir.to_string();
    loop {
        if Path::new(&join(&dir, "package.json")).exists() {
            if let Some(parent) = path_dir(&dir) {
                // build:binary places Bun's metadata inside dist/. Node still
                // needs the package root so its dist-relative asset paths do
                // not become dist/dist/.
                if path_basename(&dir) == "dist"
                    && Path::new(&join(&parent, "package.json")).exists()
                {
                    return parent;
                }
            }
            return dir;
        }
        match path_dir(&dir) {
            Some(parent) if parent != dir => dir = parent,
            _ => return start_dir.to_string(),
        }
    }
}

/// Upstream `getPackageDir`: `PI_PACKAGE_DIR` override, then the executable
/// directory's package root (upstream walks from `__dirname`; a Rust binary
/// has no dist layout, so the executable directory is the anchor).
pub fn get_package_dir() -> String {
    if let Ok(env_dir) = std::env::var("PI_PACKAGE_DIR") {
        if !env_dir.is_empty() {
            return normalize_path(&env_dir, &PathInputOptions::default());
        }
    }

    if is_bun_binary() {
        // Bun binary: process.execPath points to the compiled executable
        return exe_dir().unwrap_or_default();
    }
    let anchor = exe_dir().unwrap_or_else(|| ".".to_string());
    find_node_package_dir(&anchor)
}

/// Get path to built-in themes directory (shipped with package).
pub fn get_themes_dir() -> String {
    if is_bun_binary() {
        return join(&get_package_dir(), "theme");
    }
    // Theme is in modes/interactive/theme/ relative to src/ or dist/
    let package_dir = get_package_dir();
    let src_or_dist = if Path::new(&join(&package_dir, "src")).exists() {
        "src"
    } else {
        "dist"
    };
    join(
        &package_dir,
        &format!("{src_or_dist}/modes/interactive/theme"),
    )
}

/// Get path to HTML export template directory (shipped with package).
pub fn get_export_template_dir() -> String {
    if is_bun_binary() {
        return join(&get_package_dir(), "export-html");
    }
    let package_dir = get_package_dir();
    let src_or_dist = if Path::new(&join(&package_dir, "src")).exists() {
        "src"
    } else {
        "dist"
    };
    join(&package_dir, &format!("{src_or_dist}/core/export-html"))
}

/// Get path to package.json.
pub fn get_package_json_path() -> String {
    join(&get_package_dir(), "package.json")
}

/// Get path to README.md.
pub fn get_readme_path() -> String {
    resolve_path_cwd(&join(&get_package_dir(), "README.md"))
}

/// Get path to docs directory.
pub fn get_docs_path() -> String {
    resolve_path_cwd(&join(&get_package_dir(), "docs"))
}

/// Get path to examples directory.
pub fn get_examples_path() -> String {
    resolve_path_cwd(&join(&get_package_dir(), "examples"))
}

/// Get path to CHANGELOG.md.
pub fn get_changelog_path() -> String {
    resolve_path_cwd(&join(&get_package_dir(), "CHANGELOG.md"))
}

/// Get path to built-in interactive assets directory.
pub fn get_interactive_assets_dir() -> String {
    if is_bun_binary() {
        return join(&get_package_dir(), "assets");
    }
    let package_dir = get_package_dir();
    let src_or_dist = if Path::new(&join(&package_dir, "src")).exists() {
        "src"
    } else {
        "dist"
    };
    join(
        &package_dir,
        &format!("{src_or_dist}/modes/interactive/assets"),
    )
}

/// Get path to a bundled interactive asset.
pub fn get_bundled_interactive_asset_path(name: &str) -> String {
    join(&get_interactive_assets_dir(), name)
}

static EMBEDDED_QUICK_JS_WASM_PATH: Mutex<Option<String>> = Mutex::new(None);

/// Called by a compiled-binary entry with the path of the QuickJS wasm file
/// embedded in the executable (upstream `setEmbeddedQuickJSWasmPath`).
pub fn set_embedded_quick_js_wasm_path(path: &str) {
    *EMBEDDED_QUICK_JS_WASM_PATH.lock().unwrap() = Some(path.to_string());
}

/// Get path to `quickjs-wasi/quickjs.wasm`, the VM that runs codemode
/// scripts. Upstream resolves through Node's `createRequire`; the port
/// defaults to the package-relative path (deviation D18).
pub fn get_quick_js_wasm_path() -> String {
    if let Some(path) = EMBEDDED_QUICK_JS_WASM_PATH.lock().unwrap().clone() {
        return path;
    }
    join(&get_package_dir(), "quickjs-wasi/quickjs.wasm")
}

/// Upstream `resolveCodemodeWorkerSpecifier`. The bundled-Node URL object
/// becomes a plain path string (deviation D18).
pub fn resolve_codemode_worker_specifier(runtime: &str, module_url: &str) -> Option<String> {
    // Bun embeds explicit source entrypoints, but on Windows Bun 1.3 cannot
    // map an absolute `B:\~BUN` URL back to one. A relative string with the
    // original source extension works on every Bun platform.
    if runtime == "bun-binary" {
        return Some("./src/extensions/codemode/worker.ts".to_string());
    }
    if runtime == "bundled-node" {
        // `new URL("./codemode-worker.js", moduleUrl)` resolves relative to
        // the module's directory: the last path segment is replaced.
        let (scheme_end, rest) = match module_url.find("://") {
            Some(idx) => (idx + 3, &module_url[idx + 3..]),
            None => (0, module_url),
        };
        let base = match rest.rfind('/') {
            Some(idx) => &module_url[..scheme_end + idx],
            None => module_url,
        };
        return Some(format!("{base}/codemode-worker.js"));
    }
    None
}

/// Upstream `getCodemodeWorkerSpecifier`: the worker entry for the current
/// runtime, or `None` to use the worker that ships next to pi-codemode.
pub fn get_codemode_worker_specifier() -> Option<String> {
    let runtime = if is_bun_binary() {
        "bun-binary"
    } else if is_bundled_node() {
        "bundled-node"
    } else {
        "unbundled"
    };
    let specifier = resolve_codemode_worker_specifier(runtime, &get_package_dir())?;
    if runtime != "bundled-node" {
        return Some(specifier);
    }
    // Spawn workers from an in-memory copy. An update replaces or deletes the
    // file while this process keeps running (#10439).
    Some(format!(
        "data:text/javascript;base64,{}",
        encode_base64(&std::fs::read(specifier).ok()?)
    ))
}

/// Minimal standard base64 encoder (no external dependency).
fn encode_base64(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Upstream `InstallChange`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallChange {
    Updated { version: String },
    Removed,
}

/// The package.json this process started from, if one existed (resolved
/// lazily; upstream resolves it at module load).
fn startup_package_json_path() -> Option<String> {
    let path = get_package_json_path();
    Path::new(&path).exists().then_some(path)
}

/// Detect that the package this process runs from changed on disk, for
/// example after `pi update` in another terminal (upstream
/// `detectInstallChange`).
pub fn detect_install_change() -> Option<InstallChange> {
    // The Bun binary embeds its code, so replacing the executable does not
    // affect this process.
    if is_bun_binary() {
        return None;
    }
    let package_json_path = startup_package_json_path()?;
    detect_install_change_at(&package_json_path)
}

/// Upstream `detectInstallChange(packageJsonPath)` with an explicit path.
pub fn detect_install_change_at(package_json_path: &str) -> Option<InstallChange> {
    if is_bun_binary() {
        return None;
    }
    let content = match std::fs::read_to_string(package_json_path) {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Some(InstallChange::Removed)
        }
        Err(_) => return None,
    };
    let stripped = content.strip_prefix('\u{feff}').unwrap_or(&content);
    let installed: serde_json::Value = serde_json::from_str(stripped).ok()?;
    let version = installed
        .get("version")
        .and_then(Value::as_str)
        .map(str::to_string);
    match version {
        Some(version) if version != VERSION => Some(InstallChange::Updated { version }),
        _ => None,
    }
}

// =============================================================================
// App Config (from package.json piConfig)
// =============================================================================

/// Upstream `PACKAGE_NAME` (`pkg.name || "@earendil-works/pi-coding-agent"`).
pub const PACKAGE_NAME: &str = "@earendil-works/pi-coding-agent";

const DEFAULT_SHARE_VIEWER_URL: &str = "https://pi.dev/session/";

/// Get the share viewer URL for a gist ID.
pub fn get_share_viewer_url(gist_id: &str) -> String {
    let base_url = std::env::var("PI_SHARE_VIEWER_URL")
        .unwrap_or_else(|_| DEFAULT_SHARE_VIEWER_URL.to_string());
    format!("{base_url}#{gist_id}")
}

// =============================================================================
// User Config Paths (~/.pi/agent/*)
// =============================================================================

/// Get path to user's custom themes directory.
pub fn get_custom_themes_dir() -> String {
    join(&get_agent_dir(), "themes")
}

/// Get path to models.json.
pub fn get_models_path() -> String {
    join(&get_agent_dir(), "models.json")
}

/// Get path to auth.json.
pub fn get_auth_path() -> String {
    join(&get_agent_dir(), "auth.json")
}

/// Get path to settings.json.
pub fn get_settings_path() -> String {
    join(&get_agent_dir(), "settings.json")
}

/// Get path to tools directory.
pub fn get_tools_dir() -> String {
    join(&get_agent_dir(), "tools")
}

/// Get path to managed binaries directory (fd, rg).
pub fn get_bin_dir() -> String {
    join(&get_agent_dir(), "bin")
}

/// Get path to prompt templates directory.
pub fn get_prompts_dir() -> String {
    join(&get_agent_dir(), "prompts")
}

/// Get path to debug log file.
pub fn get_debug_log_path() -> String {
    join(&get_agent_dir(), &format!("{APP_NAME}-debug.log"))
}

#[cfg(test)]
mod tests {
    use super::*;

    static DIR_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    #[test]
    fn agent_dir_env_override_wins() {
        let _env = crate::utils::test_env_lock();
        std::env::set_var(ENV_AGENT_DIR, "/tmp/pi-agent-dir-override");
        assert_eq!(get_agent_dir(), "/tmp/pi-agent-dir-override");
        std::env::remove_var(ENV_AGENT_DIR);
        assert_eq!(get_sessions_dir(), join(&get_agent_dir(), "sessions"));
    }

    #[test]
    fn share_viewer_url_appends_gist_fragment() {
        assert_eq!(
            get_share_viewer_url("abc123"),
            "https://pi.dev/session/#abc123"
        );
    }

    #[test]
    fn pnpm_global_dir_match_mirrors_upstream_regex() {
        assert_eq!(
            match_pnpm_global_dir("/home/u/.local/share/pnpm/global/5/.pnpm/store"),
            Some("/home/u/.local/share/pnpm/global/5".to_string())
        );
        assert_eq!(match_pnpm_global_dir("/home/u/project/.pnpm/store"), None);
        assert_eq!(match_pnpm_global_dir("/global//.pnpm/x"), None);
    }

    #[test]
    fn self_update_display_quotes_whitespace_arguments() {
        let step = make_self_update_command_step(
            "npm",
            &[
                "install".to_string(),
                "-g".to_string(),
                "some pkg".to_string(),
            ],
        );
        assert_eq!(step.display, "npm install -g \"some pkg\"");
    }

    #[test]
    fn self_update_command_combines_uninstall_and_install_steps() {
        let install =
            make_self_update_command_step("bun", &["install".into(), "-g".into(), "new".into()]);
        let uninstall =
            make_self_update_command_step("bun", &["uninstall".into(), "-g".into(), "old".into()]);
        let command = make_self_update_command(install, Some(uninstall));
        let steps = command.steps.unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(
            command.display,
            "bun uninstall -g old && bun install -g new"
        );
    }

    #[test]
    fn codemode_worker_specifier_by_runtime() {
        assert_eq!(
            resolve_codemode_worker_specifier("bun-binary", "file:///x"),
            Some("./src/extensions/codemode/worker.ts".to_string())
        );
        assert_eq!(
            resolve_codemode_worker_specifier("bundled-node", "file:///x/dist/config.js"),
            Some("file:///x/dist/codemode-worker.js".to_string())
        );
        assert_eq!(
            resolve_codemode_worker_specifier("unbundled", "file:///x"),
            None
        );
    }

    #[test]
    fn base64_encoder_matches_standard_vectors() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
        assert_eq!(encode_base64(b"foob"), "Zm9vYg==");
        assert_eq!(encode_base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(encode_base64(b"foobar"), "Zm9vYmFy");
    }

    // ----- ported from test/config.test.ts -----

    #[test]
    fn find_node_package_dir_skips_binary_metadata_copied_into_dist() {
        let temp = std::env::temp_dir().join(format!(
            "pi-package-dir-{}-{}",
            std::process::id(),
            DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let dist_dir = temp.join("dist");
        let bundle_dir = dist_dir.join("bundle");
        std::fs::create_dir_all(&bundle_dir).unwrap();
        std::fs::write(temp.join("package.json"), "{}").unwrap();
        std::fs::write(dist_dir.join("package.json"), "{}").unwrap();

        assert_eq!(
            find_node_package_dir(&bundle_dir.to_string_lossy()),
            temp.to_string_lossy()
        );
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn install_change_reports_removed_instead_of_reading_further_up() {
        // Regression test for #10439: a deleted pnpm install must not fall
        // back to a package.json further up.
        let temp = std::env::temp_dir().join(format!(
            "pi-install-change-{}-{}",
            std::process::id(),
            DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let install_dir = temp.join("global").join("hash");
        std::fs::create_dir_all(&install_dir).unwrap();
        std::fs::write(temp.join("package.json"), r#"{"version":"0.0.1"}"#).unwrap();
        std::fs::remove_dir_all(&install_dir).unwrap();
        assert_eq!(
            detect_install_change_at(&install_dir.join("package.json").to_string_lossy()),
            Some(InstallChange::Removed)
        );
        let _ = std::fs::remove_dir_all(&temp);
    }
}
