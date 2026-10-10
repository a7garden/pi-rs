//! `package-manager-cli.ts` — the `pi install`/`remove`/`uninstall`/`update`/
//! `list`/`config` one-shot subcommands.
//!
//! Upstream wires these from `main.ts`: after the auth command and
//! `cleanupManagedInstall()`, `handlePackageCommand` runs, then
//! `handleConfigCommand`; a handled package command exits with
//! `process.exitCode ?? 0`. The store/network logic lives in
//! [`crate::core::package_manager`]; this module is the CLI layer (argument
//! parsing, output strings, exit codes).
//!
//! Deviations:
//!
//! - **D133 — chalk styling is dropped** (plain-text output), matching the
//!   crate-wide convention (cf. D81, `cli/args.rs` `printHelp`).
//! - **D134 — project trust resolution is the explicit override only.**
//!   `ProjectTrustStore`, `hasTrustRequiringProjectResources`,
//!   `loadProjectTrustExtensions`, and `resolveProjectTrusted` are deferred
//!   (D103), so `createCommandSettingsManager` applies
//!   `--approve`/`--no-approve`, defaulting to trusted (the crate-wide D103
//!   stance for the deferred trust store); there is no saved project-trust
//!   lookup, interactive trust prompt, or extension-derived trust discovery
//!   (hence no trust warnings and no `getCommandAppMode`). Settings errors
//!   report their message only (no stack).
//! - **D135 — the self-update note renders as its trimmed raw text.** The
//!   `@earendil-works/pi-tui` `Markdown` renderer (and its note theme) is
//!   excluded with the TUI; upstream's own `catch` fallback path is used.
//! - **D136 — managed-install plumbing.** `proper-lockfile` on
//!   `<managedRoot>/update` becomes a create-new `<managedRoot>/update.lock`
//!   file held with std advisory file locks (cf. D70); a held lock maps to
//!   upstream `ELOCKED`. `mkdtempSync` staging dirs are named
//!   `update-{pid}-{millis}-{n}`. The "Location of pi executable" line prints
//!   `argv[0]` (the Rust executable path) where Node printed `argv[1]` (the
//!   JS entry script).
//! - **D137 — `pi config` reports the config-selector deferral (D130).** The
//!   command runs upstream's argument parsing, trust gate, and settings-error
//!   reporting, then reports that the resource configuration TUI is excluded
//!   and exits 1 (upstream opens the selector and exits 0 when it closes).

use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_ai::models::Models as _;

use crate::config::{
    detect_install_method, get_agent_dir, get_package_dir, get_self_update_command,
    get_self_update_unavailable_instruction, InstallMethod, SelfUpdateCommand,
    SelfUpdateCommandStep, SelfUpdatePackageTarget, APP_NAME, CONFIG_DIR_NAME, PACKAGE_NAME,
    VERSION,
};
use crate::core::extensions::types::InlineExtension;
use crate::core::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::core::package_manager::{
    ConfiguredPackage, DefaultPackageManager, PackageManagerOptions, ProgressEventType,
};
use crate::core::settings_manager::{SettingsManager, SettingsManagerCreateOptions, SettingsScope};
use crate::core::source_info::SourceScope;
use crate::utils::child_process::spawn_process_sync;
use crate::utils::management_http::{Fetcher, ReqwestFetcher};
use crate::utils::paths::{canonicalize_path, join, resolve_path};
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::version_check::{
    format_version_check_error, get_latest_pi_release, is_newer_package_version, FetchCheckOptions,
};
#[cfg(windows)]
use crate::utils::windows_self_update::{
    cleanup_windows_self_update_quarantine, quarantine_windows_native_dependencies,
};

/// Upstream `PackageCommand` (`"install" | "remove" | "update" | "list"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageCommand {
    Install,
    Remove,
    Update,
    List,
}

/// Upstream `UpdateTarget`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateTarget {
    All,
    Self_,
    Extensions { source: Option<String> },
    Models,
}

const DEFAULT_INSTALLER_API_BASE: &str = "https://pi.dev/api/installer/releases";
const MANAGED_INSTALL_MARKER: &str = "managed-install.json";

/// Upstream `MANAGED_RELEASE_VERSION_RE`:
/// `/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/`.
fn is_managed_release_version(version: &str) -> bool {
    let (core, rest) = match version.find(['-', '+']) {
        Some(index) => (&version[..index], &version[index..]),
        None => (version, ""),
    };
    let mut parts = core.split('.');
    for _ in 0..3 {
        match parts.next() {
            Some(part) if !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()) => {}
            _ => return false,
        }
    }
    if parts.next().is_some() {
        return false;
    }
    let mut rest = rest;
    if let Some(prerelease_and_more) = rest.strip_prefix('-') {
        let (segment, tail) = match prerelease_and_more.find('+') {
            Some(index) => (
                &prerelease_and_more[..index],
                Some(&prerelease_and_more[index..]),
            ),
            None => (prerelease_and_more, None),
        };
        if !is_semver_segment(segment) {
            return false;
        }
        rest = tail.unwrap_or("");
    }
    if let Some(build) = rest.strip_prefix('+') {
        if !is_semver_segment(build) {
            return false;
        }
        rest = "";
    }
    rest.is_empty()
}

fn is_semver_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-'))
}

/// Upstream `getActiveManagedInstallRoot`.
fn get_active_managed_install_root() -> Result<Option<String>, String> {
    let Some(configured_root) = std::env::var("PI_MANAGED_INSTALL_ROOT")
        .ok()
        .map(|root| root.trim().to_string())
    else {
        return Ok(None);
    };
    if configured_root.is_empty() {
        return Ok(None);
    }

    let managed_root = resolve_path(
        &configured_root,
        &std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy(),
    );
    let releases_dir = canonicalize_path(&join(&managed_root, "releases"));
    // The launcher environment is inherited by child processes. Do not classify a
    // source checkout or another Pi installation launched from managed Pi as managed.
    if cwd_relative_path(&canonicalize_path(&get_package_dir()), &releases_dir).is_none() {
        return Ok(None);
    }

    let marker_path = join(&managed_root, MANAGED_INSTALL_MARKER);
    let raw = std::fs::read_to_string(&marker_path)
        .map_err(|_| format!("Managed install marker is missing or invalid: {marker_path}"))?;
    if !is_valid_managed_marker(&raw) {
        return Err(format!(
            "Managed install marker is missing or invalid: {marker_path}"
        ));
    }

    Ok(Some(managed_root))
}

/// Upstream marker shape check: `{ kind: "pi-managed-install", schemaVersion: 1,
/// layout: "releases-v1" }` (JSON parse + field match).
fn is_valid_managed_marker(raw: &str) -> bool {
    let Ok(marker) = serde_json::from_str::<serde_json::Value>(raw) else {
        return false;
    };
    marker.get("kind").and_then(serde_json::Value::as_str) == Some("pi-managed-install")
        && marker
            .get("schemaVersion")
            .and_then(serde_json::Value::as_i64)
            == Some(1)
        && marker.get("layout").and_then(serde_json::Value::as_str) == Some("releases-v1")
}

/// Upstream `getCwdRelativePath` (utils/paths.ts): a relative path from `cwd`
/// to `file_path` when inside, `None` otherwise.
fn cwd_relative_path(file_path: &str, cwd: &str) -> Option<String> {
    let resolved_cwd = resolve_path(
        cwd,
        &std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy(),
    );
    let resolved_path = resolve_path(file_path, &resolved_cwd);
    let relative = lexically_relative(&resolved_path, &resolved_cwd)?;
    let is_inside = relative.is_empty()
        || (relative != ".."
            && !relative.starts_with("../")
            && !Path::new(&relative).is_absolute());
    if is_inside {
        Some(if relative.is_empty() {
            ".".to_string()
        } else {
            relative
        })
    } else {
        None
    }
}

/// node `path.relative` (lexical; separators as given).
fn lexically_relative(path: &str, base: &str) -> Option<String> {
    fn split_segments(value: &str) -> Vec<&str> {
        value
            .split(['/', '\\'])
            .filter(|part| !part.is_empty() && *part != ".")
            .collect()
    }
    let (path_parts, base_parts) = (split_segments(path), split_segments(base));
    let mut common = 0;
    while common < path_parts.len()
        && common < base_parts.len()
        && path_parts[common] == base_parts[common]
    {
        common += 1;
    }
    let mut result: Vec<String> = vec!["..".to_string(); base_parts.len() - common];
    result.extend(path_parts[common..].iter().map(|part| (*part).to_string()));
    Some(result.join(std::path::MAIN_SEPARATOR_STR))
}

/// node `encodeURIComponent`.
fn encode_uri_component(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        let c = byte as char;
        if c.is_ascii_alphanumeric()
            || matches!(c, '-' | '_' | '.' | '!' | '~' | '*' | '\'' | '(' | ')')
        {
            encoded.push(c);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

/// Upstream `fetchInstallerArtifact`.
async fn fetch_installer_artifact(url: &str, label: &str) -> Result<String, String> {
    let fetcher = ReqwestFetcher::default();
    let response = fetcher
        .fetch(
            url,
            &[("User-Agent".to_string(), get_pi_user_agent(VERSION))],
            None,
        )
        .await
        .map_err(|error| error.to_string())?;
    if !(200..300).contains(&response.status) {
        return Err(format!(
            "Could not download managed installer {label} from {url}: HTTP {}",
            response.status
        ));
    }
    String::from_utf8(response.body).map_err(|error| error.to_string())
}

/// Upstream `runManagedNpmCi` — `npm ci` with inherited stdio in `stageDir`.
async fn run_managed_npm_ci(stage_dir: &str) -> Result<(), String> {
    let args = [
        "ci",
        "--ignore-scripts",
        "--min-release-age=0",
        "--omit=dev",
        "--include=optional",
        "--no-fund",
        "--no-audit",
        "--loglevel=error",
        "--progress=false",
    ];
    let mut command = tokio::process::Command::new("npm");
    command.args(args).current_dir(stage_dir);
    let code = match command.spawn() {
        Ok(mut child) => match child.wait().await {
            Ok(status) => status.code(),
            Err(_) => None,
        },
        Err(error) => return Err(error.to_string()),
    };
    match code {
        Some(0) => Ok(()),
        Some(code) => Err(format!("npm {} exited with code {code}", args.join(" "))),
        None => Err(format!("npm {} exited with code unknown", args.join(" "))),
    }
}

/// Upstream `verifyManagedRelease` — run the release's `pi --version`.
fn verify_managed_release(release_dir: &str, expected_version: &str) -> Result<(), String> {
    let bin = if cfg!(windows) {
        format!("{APP_NAME}.cmd")
    } else {
        APP_NAME.to_string()
    };
    let bin_path = join(&join(&join(release_dir, "node_modules"), ".bin"), &bin);
    let result = spawn_process_sync(&bin_path, &["--version"]);
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            return Err(format!(
                "Could not verify managed Pi {expected_version}: {error}"
            ));
        }
    };
    if output.status != Some(0) {
        let reason = if output.stderr.trim().is_empty() {
            format!(
                "exit code {}",
                output
                    .status
                    .map_or_else(|| "unknown".to_string(), |code| code.to_string())
            )
        } else {
            output.stderr.trim().to_string()
        };
        return Err(format!(
            "Could not verify managed Pi {expected_version}: {reason}"
        ));
    }
    let installed_version = output.stdout.trim();
    if installed_version != expected_version {
        return Err(format!(
			"Managed Pi smoke test returned version {installed_version}; expected {expected_version}."
		));
    }
    Ok(())
}

/// Upstream `activateManagedRelease` — atomically pin `current-version`.
fn activate_managed_release(managed_root: &str, version: &str) -> Result<(), String> {
    let current_path = join(managed_root, "current-version");
    let temporary_path = join(
        managed_root,
        &format!(
            "current-version.tmp.{}-{}",
            std::process::id(),
            now_millis()
        ),
    );
    let result = (|| -> std::io::Result<()> {
        std::fs::write(&temporary_path, format!("{version}\n"))?;
        std::fs::rename(&temporary_path, &current_path)
    })();
    let cleanup = std::fs::remove_file(&temporary_path);
    result.map_err(|error| error.to_string())?;
    cleanup.map_err(|error| error.to_string())
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

/// Keep the active release and the one running this update, which other open
/// sessions likely still use and which allows rolling back by editing
/// current-version (upstream `pruneManagedReleases`).
fn prune_managed_releases(managed_root: &str, active_version: &str) {
    let releases_root = join(managed_root, "releases");
    let Ok(entries) = std::fs::read_dir(&releases_root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == active_version || name == VERSION || !is_managed_release_version(&name) {
            continue;
        }
        // Files may be in use (e.g. loaded native modules on Windows); retry
        // on the next update.
        let _ = std::fs::remove_dir_all(entry.path());
    }
}

/// Upstream `cleanupManagedStaging` — drop stale `update-*` staging dirs.
fn cleanup_managed_staging(managed_root: &str) {
    let staging_root = join(managed_root, "staging");
    let Ok(entries) = std::fs::read_dir(&staging_root) else {
        // The staging directory does not exist yet or is not writable.
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("update-") {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Acquire the managed-install update lock (upstream `proper-lockfile` on
/// `<managedRoot>/update`; D136). A held lock is upstream `ELOCKED`.
fn lock_managed_update_sync(managed_root: &str) -> Result<std::fs::File, String> {
    let lock_path = join(managed_root, "update.lock");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .or_else(|_| std::fs::OpenOptions::new().write(true).open(&lock_path))
        .map_err(|error| error.to_string())?;
    file.try_lock().map(|()| file).map_err(|error| match error {
        std::fs::TryLockError::WouldBlock => {
            "Another managed Pi update is already running.".to_string()
        }
        std::fs::TryLockError::Error(error) => error.to_string(),
    })
}

fn release_managed_update_sync(lock_path: &str, file: std::fs::File) {
    let _ = file.unlock();
    let _ = std::fs::remove_file(lock_path);
}

/// Upstream `cleanupManagedInstall` — best-effort staging cleanup at startup.
pub fn cleanup_managed_install() {
    let managed_root = match get_active_managed_install_root() {
        Ok(managed_root) => managed_root,
        Err(_) => return,
    };
    let Some(managed_root) = managed_root else {
        return;
    };

    let lock_path = join(&managed_root, "update.lock");
    match lock_managed_update_sync(&managed_root) {
        Ok(lock) => {
            cleanup_managed_staging(&managed_root);
            release_managed_update_sync(&lock_path, lock);
        }
        Err(_) => {
            // A live update owns the staging directory, or cleanup is unavailable.
        }
    }
}

/// Upstream `runManagedSelfUpdate` — stage, verify, and activate a pinned
/// managed release.
async fn run_managed_self_update(managed_root: &str, version: &str) -> Result<(), String> {
    if !is_managed_release_version(version) {
        return Err(format!("Invalid managed release version: {version}"));
    }

    let lock_path = join(managed_root, "update.lock");
    let lock = lock_managed_update_sync(managed_root)?;

    let result = run_managed_self_update_locked(managed_root, version).await;
    release_managed_update_sync(&lock_path, lock);
    result
}

async fn run_managed_self_update_locked(managed_root: &str, version: &str) -> Result<(), String> {
    cleanup_managed_staging(managed_root);
    let installer_api_base = {
        let configured = std::env::var("PI_INSTALLER_API_BASE")
            .ok()
            .map(|base| base.trim().to_string());
        let base = configured
            .filter(|base| !base.is_empty())
            .unwrap_or_else(|| DEFAULT_INSTALLER_API_BASE.to_string());
        base.trim_end_matches('/').to_string()
    };
    let release_url = format!("{installer_api_base}/{}", encode_uri_component(version));
    let staging_root = join(managed_root, "staging");
    let releases_root = join(managed_root, "releases");
    std::fs::create_dir_all(&releases_root).map_err(|error| error.to_string())?;
    let release_dir = join(&releases_root, version);
    if Path::new(&release_dir).exists() {
        verify_managed_release(&release_dir, version)?;
        activate_managed_release(managed_root, version)?;
        prune_managed_releases(managed_root, version);
        return Ok(());
    }

    std::fs::create_dir_all(&staging_root).map_err(|error| error.to_string())?;
    let stage_dir = create_staging_dir(&staging_root)?;
    let staged = stage_managed_release(managed_root, &stage_dir, &release_url, version).await;
    if staged.is_err() {
        // Upstream `finally`: the stage dir is removed even after a successful
        // rename (where this is a no-op).
        let _ = std::fs::remove_dir_all(&stage_dir);
        return staged;
    }
    staged
}

/// `mkdtempSync(join(stagingRoot, "update-"))` (D136 for the name shape).
fn create_staging_dir(staging_root: &str) -> Result<String, String> {
    let pid = std::process::id();
    for counter in 0..1000u32 {
        let candidate = join(
            staging_root,
            &format!("update-{pid}-{}-{counter}", now_millis()),
        );
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.to_string()),
        }
    }
    Err("Could not create managed update staging directory".to_string())
}

/// Fetch `package.json`/`package-lock.json`, `npm ci`, verify, then rename the
/// staged dir into place and activate it.
async fn stage_managed_release(
    managed_root: &str,
    stage_dir: &str,
    release_url: &str,
    version: &str,
) -> Result<(), String> {
    let (package_json, package_lock) = {
        let package_json_url = format!("{release_url}/package.json");
        let package_lock_url = format!("{release_url}/package-lock.json");
        tokio::join!(
            fetch_installer_artifact(&package_json_url, "package.json"),
            fetch_installer_artifact(&package_lock_url, "package-lock.json"),
        )
    };
    let (package_json, package_lock) = (package_json?, package_lock?);
    std::fs::write(join(stage_dir, "package.json"), package_json)
        .map_err(|error| error.to_string())?;
    std::fs::write(join(stage_dir, "package-lock.json"), package_lock)
        .map_err(|error| error.to_string())?;

    run_managed_npm_ci(stage_dir).await?;
    verify_managed_release(stage_dir, version)?;
    let release_dir = join(&join(managed_root, "releases"), version);
    std::fs::rename(stage_dir, &release_dir).map_err(|error| error.to_string())?;
    activate_managed_release(managed_root, version)?;
    prune_managed_releases(managed_root, version);
    Ok(())
}

/// `handlePackageCommand` command outcome: `Err(Reported)` mirrors an error
/// that upstream already printed before setting `process.exitCode = 1`.
#[derive(Debug)]
enum CommandError {
    /// Print `Error: <message>` at the catch site.
    Message(String),
    /// The specific error output was already printed.
    Reported,
}

/// `parsePackageCommand` options.
#[derive(Debug, Clone, Default)]
pub struct PackageCommandOptions {
    pub command: Option<PackageCommand>,
    pub source: Option<String>,
    pub update_target: Option<UpdateTarget>,
    pub show_extensions_skipped_note: bool,
    pub local: bool,
    pub force: bool,
    /// `None` = absent, `Some(true)` = `--approve`, `Some(false)` = `--no-approve`.
    pub project_trust_override: Option<bool>,
    pub help: bool,
    pub invalid_option: Option<String>,
    pub invalid_argument: Option<String>,
    pub missing_option_value: Option<String>,
    pub conflicting_options: Option<String>,
}

/// Upstream `getPackageCommandUsage`.
pub fn get_package_command_usage(command: PackageCommand) -> String {
    match command {
		PackageCommand::Install => format!("{APP_NAME} install <source> [-l] [--approve|--no-approve]"),
		PackageCommand::Remove => format!("{APP_NAME} remove <source> [-l] [--approve|--no-approve]"),
		PackageCommand::Update => format!(
			"{APP_NAME} update [source|self|pi] [--self|--extensions|--models|--all] [--extension <source>] [--approve|--no-approve] [--force]"
		),
		PackageCommand::List => format!("{APP_NAME} list [--approve|--no-approve]"),
	}
}

/// Upstream `CONFIG_COMMAND_USAGE`.
pub fn config_command_usage() -> String {
    format!("{APP_NAME} config [-l] [--approve|--no-approve]")
}

/// Upstream `printConfigCommandHelp` (D133: uncolored).
pub fn config_command_help_text() -> String {
    let usage = config_command_usage();
    format!(
		"Usage:\n  {usage}\n\nOpen the resource configuration TUI to enable or disable package resources.\nWithout -l, starts in global settings (~/{CONFIG_DIR_NAME}/agent/settings.json).\nPress Tab in the TUI to switch between global and project-local modes.\n\nOptions:\n  -l, --local       Edit project overrides ({CONFIG_DIR_NAME}/settings.json)\n  -a, --approve     Trust project-local files for this command with -l\n  -na, --no-approve Ignore project-local files for this command with -l\n"
	)
}

pub fn print_config_command_help() {
    println!("{}", config_command_help_text());
}

/// Upstream `printPackageCommandHelp` (D133: uncolored).
pub fn package_command_help_text(command: PackageCommand) -> String {
    let options_block = |local_line: &str| {
        format!(
			"\nOptions:\n  -l, --local       {local_line}\n  -a, --approve     Trust project-local files for this command\n  -na, --no-approve Ignore project-local files for this command\n"
		)
    };
    match command {
		PackageCommand::Install => format!(
			"Usage:\n  {usage}\n\nInstall a package and add it to settings.\n{options}\nExamples:\n  {app} install npm:@foo/bar\n  {app} install git:github.com/user/repo\n  {app} install git:git@github.com:user/repo\n  {app} install https://github.com/user/repo\n  {app} install ssh://git@github.com:user/repo\n  {app} install ./local/path\n",
			usage = get_package_command_usage(command),
			options = options_block(&format!("Install project-locally ({CONFIG_DIR_NAME}/settings.json)")),
			app = APP_NAME,
		),
		PackageCommand::Remove => format!(
			"Usage:\n  {usage}\n\nRemove a package and its source from settings.\nAlias: {app} uninstall <source> [-l]\n{options}\nExamples:\n  {app} remove npm:@foo/bar\n  {app} uninstall npm:@foo/bar\n",
			usage = get_package_command_usage(command),
			options = options_block(&format!("Remove from project settings ({CONFIG_DIR_NAME}/settings.json)")),
			app = APP_NAME,
		),
		PackageCommand::Update => format!(
			"Usage:\n  {usage}\n\nUpdate pi, installed packages, or model catalogs.\n\nOptions:\n  --self                  Update pi only (default when no target is given)\n  --extensions            Update installed packages only\n  --models                Refresh model catalogs only\n  --all                   Update pi and installed packages\n  --extension <source>    Update one package only\n  -a, --approve           Trust project-local files for this command\n  -na, --no-approve       Ignore project-local files for this command\n  --force                 Reinstall pi even if the current version is latest\n\nShort forms:\n  {app} update                Update pi only\n  {app} update --all          Update pi and all extensions\n  {app} update --models       Refresh model catalogs only\n  {app} update <source>       Update one package\n  {app} update pi             Update pi only (self works as alias to pi)\n",
			usage = get_package_command_usage(command),
			app = APP_NAME,
		),
		PackageCommand::List => format!(
			"Usage:\n  {usage}\n\nList installed packages from user and project settings.\n\nOptions:\n  -a, --approve      Trust project-local files for this command\n  -na, --no-approve  Ignore project-local files for this command\n",
			usage = get_package_command_usage(command),
		),
	}
}

pub fn print_package_command_help(command: PackageCommand) {
    println!("{}", package_command_help_text(command));
}

/// Upstream `parsePackageCommand` — `None` when `args[0]` is not a package
/// command.
pub fn parse_package_command(args: &[String]) -> Option<PackageCommandOptions> {
    let (raw_command, rest) = args.split_first()?;
    let command = match raw_command.as_str() {
        "uninstall" => PackageCommand::Remove,
        "install" => PackageCommand::Install,
        "remove" => PackageCommand::Remove,
        "update" => PackageCommand::Update,
        "list" => PackageCommand::List,
        _ => return None,
    };

    let mut options = PackageCommandOptions {
        command: Some(command),
        ..Default::default()
    };
    let mut self_flag = false;
    let mut extensions_flag = false;
    let mut models_flag = false;
    let mut all_flag = false;
    let mut extension_flag_source: Option<String> = None;

    let mut index = 0;
    while index < rest.len() {
        let arg = &rest[index];
        index += 1;
        if arg == "-h" || arg == "--help" {
            options.help = true;
            continue;
        }

        if arg == "-l" || arg == "--local" {
            if matches!(command, PackageCommand::Install | PackageCommand::Remove) {
                options.local = true;
            } else {
                options.invalid_option.get_or_insert_with(|| arg.clone());
            }
            continue;
        }

        if arg == "--self" {
            if command == PackageCommand::Update {
                self_flag = true;
            } else {
                options.invalid_option.get_or_insert_with(|| arg.clone());
            }
            continue;
        }

        if arg == "--extensions" {
            if command == PackageCommand::Update {
                extensions_flag = true;
            } else {
                options.invalid_option.get_or_insert_with(|| arg.clone());
            }
            continue;
        }

        if arg == "--models" {
            if command == PackageCommand::Update {
                models_flag = true;
            } else {
                options.invalid_option.get_or_insert_with(|| arg.clone());
            }
            continue;
        }

        if arg == "--all" {
            if command == PackageCommand::Update {
                all_flag = true;
            } else {
                options.invalid_option.get_or_insert_with(|| arg.clone());
            }
            continue;
        }

        if arg == "--approve" || arg == "-a" {
            options.project_trust_override = Some(true);
            continue;
        }

        if arg == "--no-approve" || arg == "-na" {
            options.project_trust_override = Some(false);
            continue;
        }

        if arg == "--force" {
            if command == PackageCommand::Update {
                options.force = true;
            } else {
                options.invalid_option.get_or_insert_with(|| arg.clone());
            }
            continue;
        }

        if arg == "--extension" {
            if command != PackageCommand::Update {
                options.invalid_option.get_or_insert_with(|| arg.clone());
                continue;
            }

            let value = rest.get(index);
            let missing_value = match value {
                None => true,
                Some(value) => value.starts_with('-'),
            };
            if missing_value {
                options
                    .missing_option_value
                    .get_or_insert_with(|| arg.clone());
            } else if extension_flag_source.is_some() {
                options
                    .conflicting_options
                    .get_or_insert("--extension can only be provided once".to_string());
                index += 1;
            } else {
                extension_flag_source = Some(value.expect("checked above").clone());
                index += 1;
            }
            continue;
        }

        if arg.starts_with('-') {
            options.invalid_option.get_or_insert_with(|| arg.clone());
            continue;
        }

        if options.source.is_none() {
            options.source = Some(arg.clone());
        } else {
            options.invalid_argument.get_or_insert_with(|| arg.clone());
        }
    }

    if command == PackageCommand::Update {
        if all_flag
            && (self_flag || extensions_flag || models_flag || extension_flag_source.is_some())
        {
            options.conflicting_options.get_or_insert_with(|| {
                "--all cannot be combined with --self, --extensions, --models, or --extension"
                    .to_string()
            });
        }
        if all_flag && options.source.is_some() {
            options.conflicting_options.get_or_insert_with(|| {
                "--all cannot be combined with a positional source".to_string()
            });
        }

        if models_flag {
            if self_flag || extensions_flag || all_flag || extension_flag_source.is_some() {
                options.conflicting_options.get_or_insert_with(|| {
					"--models cannot be combined with --self, --extensions, --all, or --extension".to_string()
				});
            }
            if options.source.is_some() {
                options.conflicting_options.get_or_insert_with(|| {
                    "--models cannot be combined with a positional source".to_string()
                });
            }
            options.update_target = Some(UpdateTarget::Models);
        } else if let Some(extension_source) = extension_flag_source {
            if self_flag || extensions_flag || all_flag {
                options.conflicting_options.get_or_insert_with(|| {
                    "--extension cannot be combined with --self, --extensions, or --all".to_string()
                });
            }
            if options.source.is_some() {
                options.conflicting_options.get_or_insert_with(|| {
                    "--extension cannot be combined with a positional source".to_string()
                });
            }
            options.update_target = Some(UpdateTarget::Extensions {
                source: Some(extension_source),
            });
        } else if let Some(source) = options.source.clone() {
            let source_is_self = source == "self" || source == "pi";
            if source_is_self {
                options.update_target = Some(if extensions_flag {
                    UpdateTarget::All
                } else {
                    UpdateTarget::Self_
                });
            } else {
                if extensions_flag || self_flag || all_flag {
                    options.conflicting_options.get_or_insert_with(|| {
						"positional update targets cannot be combined with --self, --extensions, or --all".to_string()
					});
                }
                options.update_target = Some(UpdateTarget::Extensions {
                    source: Some(source),
                });
            }
        } else if all_flag {
            options.update_target = Some(UpdateTarget::All);
        } else if self_flag && extensions_flag {
            options.update_target = Some(UpdateTarget::All);
        } else if self_flag {
            options.update_target = Some(UpdateTarget::Self_);
        } else if extensions_flag {
            options.update_target = Some(UpdateTarget::Extensions { source: None });
        } else {
            options.update_target = Some(UpdateTarget::Self_);
            options.show_extensions_skipped_note = true;
        }
    }

    Some(options)
}

/// Upstream `updateTargetIncludesSelf`.
fn update_target_includes_self(target: &UpdateTarget) -> bool {
    matches!(target, UpdateTarget::All | UpdateTarget::Self_)
}

/// Upstream `updateTargetIncludesExtensions`.
fn update_target_includes_extensions(target: &UpdateTarget) -> bool {
    matches!(target, UpdateTarget::All | UpdateTarget::Extensions { .. })
}

/// Upstream `refreshModelCatalogs` — bounded (15s) forced catalog refresh.
async fn refresh_model_catalogs(agent_dir: &str) -> Result<(), String> {
    let refresh = async {
        let model_runtime = ModelRuntime::create(CreateModelRuntimeOptions {
            auth_path: Some(join(agent_dir, "auth.json")),
            models_path: Some(Some(join(agent_dir, "models.json"))),
            allow_model_network: false,
            ..Default::default()
        })
        .await
        .map_err(|value| thrown_value_message(&value))?;
        let result = model_runtime
            .refresh(Some(pi_ai::models::ModelsRefreshOptions {
                allow_network: Some(true),
                force: Some(true),
                ..Default::default()
            }))
            .await;
        if result.aborted {
            return Err("Model catalog refresh timed out.".to_string());
        }
        if !result.errors.is_empty() {
            let details = result
                .errors
                .iter()
                .map(|(provider, error)| format!("{provider}: {}", thrown_value_message(error)))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(format!("Could not refresh model catalogs: {details}"));
        }
        Ok(())
    };
    tokio::time::timeout(std::time::Duration::from_millis(15_000), refresh)
        .await
        .unwrap_or(Err("Model catalog refresh timed out.".to_string()))?;
    println!("Model catalogs refreshed");
    Ok(())
}

fn thrown_value_message(value: &pi_ai::utils::thrown::ThrownValue) -> String {
    match value {
        pi_ai::utils::thrown::ThrownValue::Error(error) => error.to_string(),
        pi_ai::utils::thrown::ThrownValue::Text(text) => text.clone(),
    }
}

/// Upstream `printSelfUpdateUnavailable` (D133: uncolored; D136: entrypoint is
/// `argv[0]`).
fn print_self_update_unavailable(
    npm_command: Option<&[String]>,
    update_package_target: &SelfUpdatePackageTarget,
) {
    eprintln!("error: {APP_NAME} cannot self-update this installation.");
    match get_self_update_unavailable_instruction(
        PACKAGE_NAME,
        npm_command,
        Some(update_package_target),
    ) {
        Ok(instruction) => eprintln!("{instruction}"),
        Err(error) => eprintln!("{error}"),
    }

    let entrypoint = std::env::args().next();
    if let Some(entrypoint) = entrypoint {
        eprintln!();
        eprintln!("Location of {APP_NAME} executable: {entrypoint}");
    }
}

/// Upstream `printSelfUpdateFallback` (D133: uncolored).
fn print_self_update_fallback(command: &SelfUpdateCommand) {
    eprintln!(
        "If this keeps failing, run this command yourself: {}",
        command.display
    );
}

/// Upstream `printPnpmSelfUpdateMetadataHint` (D133: uncolored).
fn print_pnpm_self_update_metadata_hint() {
    eprintln!(
        "If pnpm reports missing package versions, its cached registry metadata may be stale."
    );
    eprintln!("Run `pnpm store prune` and retry `{APP_NAME} update --self`.");
}

/// Upstream `printSelfUpdateNote` (D135: raw trimmed text).
pub fn print_self_update_note(note: &str) {
    let trimmed = note.trim();
    if trimmed.is_empty() {
        return;
    }

    println!();
    println!("Update note");
    println!("{trimmed}");
    println!();
}

/// Upstream `SelfUpdatePlan`.
#[derive(Debug, Clone)]
struct SelfUpdatePlan {
    package_name: String,
    install_spec: String,
    version: String,
    should_run: bool,
    note: Option<String>,
}

/// Upstream `getSelfUpdatePlan`.
async fn get_self_update_plan(force: bool) -> Result<SelfUpdatePlan, String> {
    let fetcher = Arc::new(ReqwestFetcher::default());
    let latest_release = match get_latest_pi_release(
        fetcher,
        VERSION,
        FetchCheckOptions {
            retry: true,
            timeout_ms: None,
        },
    )
    .await
    {
        Ok(release) => release,
        Err(error) => {
            return Err(format!(
                "Could not determine latest {APP_NAME} version: {}",
                format_version_check_error(&error)
            ));
        }
    };
    let Some(latest_release) = latest_release else {
        return Err(format!("Could not determine latest {APP_NAME} version."));
    };

    let package_name = latest_release
        .package_name
        .clone()
        .unwrap_or_else(|| PACKAGE_NAME.to_string());
    let install_spec = format!("{package_name}@{}", latest_release.version);
    if force
        || package_name != PACKAGE_NAME
        || is_newer_package_version(&latest_release.version, VERSION)
    {
        return Ok(SelfUpdatePlan {
            package_name,
            install_spec,
            version: latest_release.version,
            note: latest_release.note,
            should_run: true,
        });
    }

    println!("{APP_NAME} is already up to date (v{VERSION})");
    Ok(SelfUpdatePlan {
        package_name,
        install_spec,
        version: latest_release.version,
        note: None,
        should_run: false,
    })
}

/// Upstream `runSelfUpdate` — run each step with inherited stdio.
async fn run_self_update(command: &SelfUpdateCommand) -> Result<(), String> {
    println!("Updating {APP_NAME} with {}...", command.display);
    let default_step = SelfUpdateCommandStep {
        command: command.command.clone(),
        args: command.args.clone(),
        display: command.display.clone(),
    };
    for step in command
        .steps
        .as_deref()
        .unwrap_or(std::slice::from_ref(&default_step))
    {
        let mut child = tokio::process::Command::new(&step.command)
            .args(&step.args)
            .spawn()
            .map_err(|error| error.to_string())?;
        let status = child.wait().await.map_err(|error| error.to_string())?;
        match status.code() {
            Some(0) => {}
            Some(code) => return Err(format!("{} exited with code {code}", step.display)),
            None => {
                return Err(format!(
                    "{} terminated by signal {}",
                    step.display,
                    status_signal_name(&status)
                ))
            }
        }
    }
    Ok(())
}

/// Node reports the numeric signal; `ExitStatus` carries no portable name, so
/// the raw signal number is rendered (Unix only; `unknown` elsewhere).
#[cfg(unix)]
fn status_signal_name(status: &std::process::ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt as _;
    status
        .signal()
        .map_or_else(|| "unknown".to_string(), |signal| signal.to_string())
}

#[cfg(not(unix))]
fn status_signal_name(_status: &std::process::ExitStatus) -> String {
    "unknown".to_string()
}

/// Upstream `prepareWindowsNpmSelfUpdate` — a no-op off Windows, matching the
/// upstream `process.platform` gate.
#[cfg(windows)]
fn prepare_windows_npm_self_update() {
    let package_dir = get_package_dir();
    cleanup_windows_self_update_quarantine(&package_dir);
    quarantine_windows_native_dependencies(&package_dir);
}

#[cfg(not(windows))]
fn prepare_windows_npm_self_update() {}

/// Upstream `PackageCommandRuntimeOptions`.
#[derive(Default)]
pub struct PackageCommandRuntimeOptions {
    /// Upstream `extensionFactories`; only built-in named extensions
    /// (`builtin: true`) are consumed (the `config` command resolution).
    pub extension_factories: Vec<InlineExtension>,
}

/// Upstream `CommandSettingsResult`.
struct CommandSettingsResult {
    settings_manager: Arc<Mutex<SettingsManager>>,
    project_trust_warnings: Vec<String>,
}

/// Upstream `createCommandSettingsManager` options.
struct CommandSettingsOptions<'a> {
    cwd: &'a str,
    agent_dir: &'a str,
    project_trust_override: Option<bool>,
    /// Upstream `useSavedProjectTrustOnly`; both paths collapse to the same
    /// override-only trust resolution (D134).
    #[allow(dead_code)]
    use_saved_project_trust_only: bool,
}

/// Upstream `createCommandSettingsManager` (D134: no trust store, no extension
/// trust discovery — the explicit `--approve`/`--no-approve` override decides,
/// defaulting to trusted per the crate-wide D103 stance for the deferred
/// `ProjectTrustStore`).
fn create_command_settings_manager(options: CommandSettingsOptions<'_>) -> CommandSettingsResult {
    let mut settings_manager = SettingsManager::create(
        options.cwd,
        options.agent_dir,
        SettingsManagerCreateOptions {
            project_trusted: Some(false),
        },
    );
    // D134: upstream resolves the saved project trust (and possibly prompts);
    // the deferred trust modules reduce this to the explicit override.
    let project_trusted = options.project_trust_override.unwrap_or(true);
    settings_manager.set_project_trusted(project_trusted);
    CommandSettingsResult {
        settings_manager: Arc::new(Mutex::new(settings_manager)),
        project_trust_warnings: Vec::new(),
    }
}

/// Upstream `reportProjectTrustWarnings` (D133: uncolored).
fn report_project_trust_warnings(warnings: &[String]) {
    for warning in warnings {
        eprintln!("Warning: {warning}");
    }
}

/// Upstream `reportSettingsErrors` (D133: uncolored; D134: no stack line).
fn report_settings_errors(settings_manager: &mut SettingsManager, context: &str) {
    for error in settings_manager.drain_errors() {
        let scope = match error.scope {
            SettingsScope::Global => "global",
            SettingsScope::Project => "project",
        };
        eprintln!("Warning ({context}, {scope} settings): {}", error.error);
    }
}

/// Upstream `handleConfigCommand` — `Ok(false)` when `args[0]` is not
/// `config`; `Ok(true)` when handled (upstream exits 0 after the selector);
/// `Err(code)` for the handled error exits (D137: the TUI is excluded, so the
/// success path reports the deferral with exit code 1).
pub async fn handle_config_command(
    args: &[String],
    runtime_options: &PackageCommandRuntimeOptions,
) -> Result<bool, i32> {
    let Some((command, rest)) = args.split_first() else {
        return Ok(false);
    };
    if command != "config" {
        return Ok(false);
    }

    if rest.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_config_command_help();
        return Ok(true);
    }

    let config_usage = config_command_usage();
    let mut local = false;
    let mut project_trust_override: Option<bool> = None;
    for arg in rest {
        if arg == "-l" || arg == "--local" {
            local = true;
        } else if arg == "-a" || arg == "--approve" {
            project_trust_override = Some(true);
        } else if arg == "-na" || arg == "--no-approve" {
            project_trust_override = Some(false);
        } else if arg.starts_with('-') {
            eprintln!("Unknown option {arg} for \"config\".");
            eprintln!("Use \"{APP_NAME} --help\" or \"{config_usage}\".");
            return Err(1);
        } else {
            eprintln!("Unexpected argument {arg}.");
            eprintln!("Usage: {config_usage}");
            return Err(1);
        }
    }

    let cwd = std::env::current_dir()
        .map(|dir| dir.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let agent_dir = get_agent_dir();
    let command_settings = create_command_settings_manager(CommandSettingsOptions {
        cwd: &cwd,
        agent_dir: &agent_dir,
        project_trust_override,
        use_saved_project_trust_only: false,
    });
    report_project_trust_warnings(&command_settings.project_trust_warnings);
    if local
        && !command_settings
            .settings_manager
            .lock()
            .is_project_trusted()
    {
        eprintln!("Project is not trusted. Use --approve to modify local resource config.");
        return Err(1);
    }
    report_settings_errors(
        &mut command_settings.settings_manager.lock(),
        "config command",
    );
    let builtin_extensions = builtin_extension_names(&runtime_options.extension_factories);
    let global_settings_manager = SettingsManager::create(
        &cwd,
        &agent_dir,
        SettingsManagerCreateOptions {
            project_trusted: Some(false),
        },
    );
    let global_settings_manager = Arc::new(Mutex::new(global_settings_manager));
    let global_resolved = DefaultPackageManager::new(PackageManagerOptions {
        cwd: cwd.clone(),
        agent_dir: agent_dir.clone(),
        settings_manager: Arc::clone(&global_settings_manager),
        builtin_extensions: builtin_extensions.clone(),
    })
    .resolve(None)
    .await;
    let project_resolved = if command_settings
        .settings_manager
        .lock()
        .is_project_trusted()
    {
        DefaultPackageManager::new(PackageManagerOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            settings_manager: Arc::clone(&command_settings.settings_manager),
            builtin_extensions,
        })
        .resolve(None)
        .await
    } else {
        global_resolved.clone()
    };

    // D137: `selectConfig` (cli/config-selector.ts, D130) is the interactive
    // resource configuration TUI; it is excluded with the TUI, so report the
    // deferral instead of opening it (upstream would exit 0 when it closes).
    let _ = (global_resolved, project_resolved);
    eprintln!("Error: the resource configuration TUI is not available in this build (the TUI is excluded).");
    Err(1)
}

/// Upstream `isBuiltinExtension(input).map(name)` over inline extensions.
fn builtin_extension_names(extension_factories: &[InlineExtension]) -> Vec<String> {
    extension_factories
        .iter()
        .filter_map(|input| match input {
            InlineExtension::Named {
                name,
                builtin: Some(true),
                ..
            } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// Upstream `handlePackageCommand` — `Ok(false)` when `args[0]` is not a
/// package command; `Ok(true)` when handled successfully (upstream exits
/// `process.exitCode ?? 0`); `Err(code)` for a handled error exit.
/// `runtime_options` is consumed only by the deferred trust/extension path
/// (D134).
pub async fn handle_package_command(
    args: &[String],
    _runtime_options: &PackageCommandRuntimeOptions,
) -> Result<bool, i32> {
    let Some(options) = parse_package_command(args) else {
        return Ok(false);
    };
    let Some(command) = options.command else {
        return Ok(false);
    };

    if options.help {
        print_package_command_help(command);
        return Ok(true);
    }

    if let Some(invalid_option) = &options.invalid_option {
        eprintln!(
            "Unknown option {invalid_option} for \"{}\".",
            command_name(command)
        );
        eprintln!(
            "Use \"{APP_NAME} --help\" or \"{}\".",
            get_package_command_usage(command)
        );
        return Err(1);
    }

    if let Some(missing_option_value) = &options.missing_option_value {
        eprintln!("Missing value for {missing_option_value}.");
        eprintln!("Usage: {}", get_package_command_usage(command));
        return Err(1);
    }

    if let Some(invalid_argument) = &options.invalid_argument {
        eprintln!("Unexpected argument {invalid_argument}.");
        eprintln!("Usage: {}", get_package_command_usage(command));
        return Err(1);
    }

    if let Some(conflicting_options) = &options.conflicting_options {
        eprintln!("{conflicting_options}");
        eprintln!("Usage: {}", get_package_command_usage(command));
        return Err(1);
    }

    let source = options.source.clone();
    if matches!(command, PackageCommand::Install | PackageCommand::Remove) && source.is_none() {
        eprintln!("Missing {} source.", command_name(command));
        eprintln!("Usage: {}", get_package_command_usage(command));
        return Err(1);
    }

    if command == PackageCommand::Update
        && options.update_target.as_ref() == Some(&UpdateTarget::Models)
    {
        if let Err(message) = refresh_model_catalogs(&get_agent_dir()).await {
            eprintln!("Error: {message}");
            return Err(1);
        }
        return Ok(true);
    }

    let cwd = std::env::current_dir()
        .map(|dir| dir.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let agent_dir = get_agent_dir();
    let writes_project_package_config =
        matches!(command, PackageCommand::Install | PackageCommand::Remove) && options.local;
    let command_settings = create_command_settings_manager(CommandSettingsOptions {
        cwd: &cwd,
        agent_dir: &agent_dir,
        project_trust_override: options.project_trust_override,
        use_saved_project_trust_only: command == PackageCommand::Update,
    });
    report_project_trust_warnings(&command_settings.project_trust_warnings);
    if !command_settings
        .settings_manager
        .lock()
        .is_project_trusted()
        && writes_project_package_config
    {
        eprintln!("Project is not trusted. Use --approve to modify local package config.");
        return Err(1);
    }
    report_settings_errors(
        &mut command_settings.settings_manager.lock(),
        "package command",
    );
    let self_update_npm_command = command_settings.settings_manager.lock().get_npm_command();

    let package_manager = DefaultPackageManager::new(PackageManagerOptions {
        cwd,
        agent_dir,
        settings_manager: Arc::clone(&command_settings.settings_manager),
        builtin_extensions: Vec::new(),
    });

    package_manager.set_progress_callback(Some(Arc::new(|event| {
        if event.event_type == ProgressEventType::Start {
            if let Some(message) = &event.message {
                let mut stdout = std::io::stdout().lock();
                let _ = writeln!(stdout, "{message}");
                let _ = stdout.flush();
            }
        }
    })));

    let outcome: Result<(), CommandError> = match command {
        PackageCommand::Install => {
            let source = source.expect("validated above");
            package_manager
                .install_and_persist(&source, options.local)
                .await
                .map(|()| println!("Installed {source}"))
                .map_err(CommandError::Message)
        }
        PackageCommand::Remove => {
            let source = source.expect("validated above");
            match package_manager
                .remove_and_persist(&source, options.local)
                .await
            {
                Ok(true) => {
                    println!("Removed {source}");
                    Ok(())
                }
                Ok(false) => {
                    eprintln!("No matching package found for {source}");
                    Err(CommandError::Reported)
                }
                Err(message) => Err(CommandError::Message(message)),
            }
        }
        PackageCommand::List => {
            let configured_packages = package_manager.list_configured_packages();
            if configured_packages.is_empty() {
                println!("No packages installed.");
                return Ok(true);
            }
            print!("{}", render_package_listing(&configured_packages));
            Ok(())
        }
        PackageCommand::Update => {
            run_update_command(
                &package_manager,
                &options,
                self_update_npm_command.as_deref(),
            )
            .await
        }
    };

    match outcome {
        Ok(()) => Ok(true),
        Err(CommandError::Message(message)) => {
            eprintln!("Error: {message}");
            Err(1)
        }
        Err(CommandError::Reported) => Err(1),
    }
}

fn command_name(command: PackageCommand) -> &'static str {
    match command {
        PackageCommand::Install => "install",
        PackageCommand::Remove => "remove",
        PackageCommand::Update => "update",
        PackageCommand::List => "list",
    }
}

/// Upstream `list` output: `User packages:` / `Project packages:` sections.
pub fn render_package_listing(configured_packages: &[ConfiguredPackage]) -> String {
    let user_packages = configured_packages
        .iter()
        .filter(|pkg| pkg.scope == SourceScope::User);
    let project_packages = configured_packages
        .iter()
        .filter(|pkg| pkg.scope == SourceScope::Project);

    let mut output = String::new();
    let mut user_count = 0usize;
    for pkg in user_packages {
        if user_count == 0 {
            output.push_str("User packages:\n");
        }
        output.push_str(&format_package_entry(pkg));
        user_count += 1;
    }

    let mut project_count = 0usize;
    for pkg in project_packages {
        if project_count == 0 {
            if user_count > 0 {
                output.push('\n');
            }
            output.push_str("Project packages:\n");
        }
        output.push_str(&format_package_entry(pkg));
        project_count += 1;
    }
    output
}

fn format_package_entry(pkg: &ConfiguredPackage) -> String {
    let mut entry = if pkg.filtered {
        format!("  {} (filtered)\n", pkg.source)
    } else {
        format!("  {}\n", pkg.source)
    };
    if let Some(installed_path) = &pkg.installed_path {
        entry.push_str(&format!("    {installed_path}\n"));
    }
    entry
}

/// The `update` command body (extensions then self), with upstream's
/// command-specific error outputs surfaced as [`CommandError::Reported`].
async fn run_update_command(
    package_manager: &DefaultPackageManager,
    options: &PackageCommandOptions,
    self_update_npm_command: Option<&[String]>,
) -> Result<(), CommandError> {
    let target = options.update_target.clone().unwrap_or(UpdateTarget::Self_);
    if options.show_extensions_skipped_note {
        println!(
            "Extensions are skipped. Run {APP_NAME} update --extensions to update extensions."
        );
    }
    if update_target_includes_extensions(&target) {
        let update_source = match &target {
            UpdateTarget::Extensions { source } => source.clone(),
            _ => None,
        };
        package_manager
            .update(update_source.as_deref())
            .await
            .map_err(CommandError::Message)?;
        match &update_source {
            Some(update_source) => println!("Updated {update_source}"),
            None => println!("Updated packages"),
        }
    }
    if update_target_includes_self(&target) {
        let managed_install_root =
            get_active_managed_install_root().map_err(CommandError::Message)?;
        if managed_install_root.is_some() && options.force {
            eprintln!(
				"Managed {APP_NAME} installations do not support --force; rerun the installer to repair this installation."
			);
            return Err(CommandError::Reported);
        }
        let self_update_plan = get_self_update_plan(options.force)
            .await
            .map_err(CommandError::Message)?;
        if !self_update_plan.should_run {
            return Ok(());
        }
        if let Some(managed_install_root) = &managed_install_root {
            if let Some(note) = &self_update_plan.note {
                print_self_update_note(note);
            }
            println!("Updating managed {APP_NAME} installation...");
            if let Err(message) =
                run_managed_self_update(managed_install_root, &self_update_plan.version).await
            {
                eprintln!("Error: {message}");
                return Err(CommandError::Reported);
            }
            println!(
                "Updated {APP_NAME} from {VERSION} to {}",
                self_update_plan.version
            );
            return Ok(());
        }

        let install_method = detect_install_method();
        if cfg!(windows) && !matches!(install_method, InstallMethod::Npm | InstallMethod::Pnpm) {
            eprintln!(
                "{APP_NAME} self-update on Windows is only supported for npm and pnpm installs."
            );
            eprintln!(
                "Detected install method: {}. Update {APP_NAME} manually.",
                install_method_name(install_method)
            );
            return Err(CommandError::Reported);
        }
        let self_update_target = SelfUpdatePackageTarget::Filtered {
            package_name: self_update_plan.package_name.clone(),
            install_spec: Some(self_update_plan.install_spec.clone()),
        };
        let self_update_command = get_self_update_command(
            PACKAGE_NAME,
            self_update_npm_command,
            Some(&self_update_target),
        )
        .map_err(CommandError::Message)?;
        let Some(self_update_command) = self_update_command else {
            print_self_update_unavailable(self_update_npm_command, &self_update_target);
            return Err(CommandError::Reported);
        };
        if let Some(note) = &self_update_plan.note {
            print_self_update_note(note);
        }
        if install_method == InstallMethod::Npm {
            prepare_windows_npm_self_update();
        }
        if let Err(message) = run_self_update(&self_update_command).await {
            eprintln!("Error: {message}");
            if install_method == InstallMethod::Pnpm {
                print_pnpm_self_update_metadata_hint();
            }
            print_self_update_fallback(&self_update_command);
            return Err(CommandError::Reported);
        }
        println!(
            "Updated {APP_NAME} from {VERSION} to {}",
            self_update_plan.version
        );
        // The pi.dev installer migrates global npm installs to a managed install
        // that pins all dependencies. It does not migrate pnpm, yarn, or bun installs.
        if install_method == InstallMethod::Npm {
            let installer_command = if cfg!(windows) {
                "powershell -c \"irm https://pi.dev/install.ps1 | iex\""
            } else {
                "curl -fsSL https://pi.dev/install.sh | sh"
            };
            println!();
            println!("This npm installation of {APP_NAME} does not pin its dependencies.");
            println!("Run the installer to migrate to a managed installation that does:");
            println!();
            println!("  {installer_command}");
        }
    }
    Ok(())
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

#[cfg(test)]
#[path = "package_manager_cli_tests.rs"]
mod tests;
