//! One-time migrations that run on startup (`migrations.ts`).
//!
//! The session-header v1→v3 format migrations live in
//! [`crate::core::session_manager`] (ported with slice 7); this module is
//! the top-level startup orchestrator.

use crate::config::{get_agent_dir, CONFIG_DIR_NAME};
use crate::core::keybindings_migration::migrate_keybindings_config;
use crate::utils::paths::join;
use crate::utils::text::strip_bom;

const MIGRATION_GUIDE_URL: &str =
	"https://github.com/earendil-works/pi/blob/main/packages/coding-agent/CHANGELOG.md#extensions-migration";
const EXTENSIONS_DOC_URL: &str =
	"https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/extensions.md";

/// Result of [`run_migrations`] (upstream return object).
pub struct RunMigrationsResult {
	pub migrated_auth_providers: Vec<String>,
	pub deprecation_warnings: Vec<String>,
}

/// Migrate legacy oauth.json and settings.json apiKeys to auth.json
/// (`migrateAuthToAuthJson`). Returns the migrated provider names.
pub fn migrate_auth_to_auth_json(agent_dir: &str) -> Vec<String> {
	let auth_path = join(&agent_dir, "auth.json");
	let oauth_path = join(&agent_dir, "oauth.json");
	let settings_path = join(&agent_dir, "settings.json");

	// Skip if auth.json already exists
	if std::path::Path::new(&auth_path).exists() {
		return Vec::new();
	}

	let mut migrated = serde_json::Map::new();
	let mut providers: Vec<String> = Vec::new();

	// Migrate oauth.json
	if std::path::Path::new(&oauth_path).exists() {
		let parse = std::fs::read_to_string(&oauth_path)
			.ok()
			.map(|content| serde_json::from_str::<serde_json::Value>(strip_bom(&content)).ok())
			.flatten();
		if let Some(oauth) = parse {
			if let Some(object) = oauth.as_object() {
				for (provider, cred) in object {
					let mut entry = serde_json::Map::new();
					entry.insert("type".to_string(), serde_json::Value::String("oauth".to_string()));
					if let Some(cred_object) = cred.as_object() {
						for (key, value) in cred_object {
							entry.insert(key.clone(), value.clone());
						}
					}
					migrated.insert(provider.clone(), serde_json::Value::Object(entry));
					providers.push(provider.clone());
				}
				let _ = std::fs::rename(&oauth_path, format!("{oauth_path}.migrated"));
			}
		}
		// Parse failures skip the migration (upstream `catch {}`)
	}

	// Migrate settings.json apiKeys
	if std::path::Path::new(&settings_path).exists() {
		let parsed = std::fs::read_to_string(&settings_path)
			.ok()
			.map(|content| serde_json::from_str::<serde_json::Value>(strip_bom(&content)).ok())
			.flatten();
		if let Some(mut settings) = parsed {
			let api_keys = settings.get("apiKeys").and_then(|v| v.as_object()).cloned();
			if let Some(api_keys) = api_keys {
				for (provider, key) in &api_keys {
					if !migrated.contains_key(provider) && key.is_string() {
						let mut entry = serde_json::Map::new();
						entry.insert("type".to_string(), serde_json::Value::String("api_key".to_string()));
						entry.insert("key".to_string(), key.clone());
						migrated.insert(provider.clone(), serde_json::Value::Object(entry));
						providers.push(provider.clone());
					}
				}
				if let Some(object) = settings.as_object_mut() {
					object.remove("apiKeys");
					if let Ok(serialized) = serde_json::to_string_pretty(&settings) {
						let _ = std::fs::write(&settings_path, serialized);
					}
				}
			}
		}
	}

	if !migrated.is_empty() {
		if let Some(parent) = std::path::Path::new(&auth_path).parent() {
			let _ = std::fs::create_dir_all(parent);
		}
		if let Ok(serialized) = serde_json::to_string_pretty(&serde_json::Value::Object(migrated)) {
			let _ = std::fs::write(&auth_path, serialized);
			set_owner_only_permissions(&auth_path);
		}
	}

	providers
}

/// Best-effort `mode: 0o600` on Unix (no-op elsewhere).
fn set_owner_only_permissions(path: &str) {
	#[cfg(unix)]
	{
		use std::os::unix::fs::PermissionsExt;
		let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
	}
	#[cfg(not(unix))]
	{
		let _ = path;
	}
}

/// Migrate sessions from `~/.pi/agent/*.jsonl` to proper session
/// directories (`migrateSessionsFromAgentRoot`).
///
/// Bug in v0.30.0: Sessions were saved to `~/.pi/agent/` instead of
/// `~/.pi/agent/sessions/<encoded-cwd>/`.
pub fn migrate_sessions_from_agent_root(agent_dir: &str) {

	// Find all .jsonl files directly in agentDir (not in subdirectories)
	let files: Vec<String> = match std::fs::read_dir(&agent_dir) {
		Ok(entries) => entries
			.flatten()
			.filter(|entry| entry.file_type().map(|t| t.is_file()).unwrap_or(false))
			.map(|entry| entry.file_name().to_string_lossy().to_string())
			.filter(|name| name.ends_with(".jsonl"))
			.map(|name| join(&agent_dir, &name))
			.collect(),
		Err(_) => return,
	};

	if files.is_empty() {
		return;
	}

	for file in files {
		// Read first line to get session header
		let Ok(content) = std::fs::read_to_string(&file) else { continue };
		let Some(first_line) = content.lines().next().filter(|l| !l.trim().is_empty()) else { continue };
		let Ok(header) = serde_json::from_str::<serde_json::Value>(first_line) else { continue };
		if header.get("type").and_then(|t| t.as_str()) != Some("session") {
			continue;
		}
		let Some(cwd) = header.get("cwd").and_then(|c| c.as_str()) else { continue };

		// Compute the correct session directory (same encoding as
		// session-manager.ts)
		let trimmed = cwd.strip_prefix(['/', '\\']).unwrap_or(cwd).replace(['/', '\\', ':'], "-");
		let safe_path = format!("--{trimmed}--");
		let correct_dir = join(&agent_dir, &join("sessions", &safe_path));

		// Create directory if needed
		if !std::path::Path::new(&correct_dir).exists() {
			let _ = std::fs::create_dir_all(&correct_dir);
		}

		let file_name = std::path::Path::new(&file)
			.file_name()
			.map(|n| n.to_string_lossy().to_string())
			.unwrap_or_else(|| file.clone());
		let new_path = join(&correct_dir, &file_name);

		if std::path::Path::new(&new_path).exists() {
			continue; // Skip if target exists
		}

		let _ = std::fs::rename(&file, &new_path);
	}
}

/// Migrate commands/ to prompts/ if needed (`migrateCommandsToPrompts`).
/// Works for both regular directories and symlinks.
fn migrate_commands_to_prompts(base_dir: &str, label: &str) -> bool {
	let commands_dir = join(base_dir, "commands");
	let prompts_dir = join(base_dir, "prompts");

	if std::path::Path::new(&commands_dir).exists() && !std::path::Path::new(&prompts_dir).exists() {
		match std::fs::rename(&commands_dir, &prompts_dir) {
			Ok(()) => {
				println!("Migrated {label} commands/ → prompts/");
				return true;
			}
			Err(err) => {
				println!("Warning: Could not migrate {label} commands/ to prompts/: {err}");
			}
		}
	}
	false
}

/// Move fd/rg binaries from tools/ to bin/ if they exist
/// (`migrateToolsToBin`).
fn migrate_tools_to_bin(agent_dir: &str, bin_dir: &str) {
	let tools_dir = join(agent_dir, "tools");

	if !std::path::Path::new(&tools_dir).exists() {
		return;
	}

	let binaries = ["fd", "rg", "fd.exe", "rg.exe"];
	let mut moved_any = false;

	for bin in binaries {
		let old_path = join(&tools_dir, bin);
		let new_path = join(&bin_dir, bin);

		if std::path::Path::new(&old_path).exists() {
			if !std::path::Path::new(&bin_dir).exists() {
				let _ = std::fs::create_dir_all(&bin_dir);
			}
			if !std::path::Path::new(&new_path).exists() {
				if std::fs::rename(&old_path, &new_path).is_ok() {
					moved_any = true;
				}
			} else {
				// Target exists, just delete the old one
				let _ = std::fs::remove_file(&old_path);
			}
		}
	}

	if moved_any {
		println!("Migrated managed binaries tools/ → bin/");
	}
}

/// Check for deprecated hooks/ and tools/ directories
/// (`checkDeprecatedExtensionDirs`). Note: tools/ may contain fd/rg
/// binaries extracted by pi, so only warn if it has other files.
fn check_deprecated_extension_dirs(base_dir: &str, label: &str) -> Vec<String> {
	let hooks_dir = join(base_dir, "hooks");
	let tools_dir = join(base_dir, "tools");
	let mut warnings: Vec<String> = Vec::new();

	if std::path::Path::new(&hooks_dir).exists() {
		warnings.push(format!("{label} hooks/ directory found. Hooks have been renamed to extensions."));
	}

	if std::path::Path::new(&tools_dir).exists() {
		if let Ok(entries) = std::fs::read_dir(&tools_dir) {
			let custom_tools: Vec<String> = entries
				.flatten()
				.map(|entry| entry.file_name().to_string_lossy().to_string())
				.filter(|name| {
					let lower = name.to_lowercase();
					lower != "fd" && lower != "rg" && lower != "fd.exe" && lower != "rg.exe" && !name.starts_with('.')
				})
				.collect();
			if !custom_tools.is_empty() {
				warnings.push(format!(
					"{label} tools/ directory contains custom tools. Custom tools have been merged into extensions."
				));
			}
		}
	}

	warnings
}

/// Run extension system migrations (commands→prompts) and collect warnings
/// about deprecated directories (`migrateExtensionSystem`).
fn migrate_extension_system(agent_dir: &str, cwd: &str) -> Vec<String> {
	let project_dir = join(cwd, CONFIG_DIR_NAME);

	// Migrate commands/ to prompts/
	migrate_commands_to_prompts(&agent_dir, "Global");
	migrate_commands_to_prompts(&project_dir, "Project");

	// Check for deprecated directories
	let mut warnings = check_deprecated_extension_dirs(&agent_dir, "Global");
	warnings.extend(check_deprecated_extension_dirs(&project_dir, "Project"));
	warnings
}

/// Print deprecation warnings and wait for keypress
/// (`showDeprecationWarnings`).
///
/// The interactive "press any key" prompt is excluded with the TUI; the
/// warnings print and return immediately in non-interactive modes.
pub fn show_deprecation_warnings(warnings: &[String]) {
	if warnings.is_empty() {
		return;
	}

	for warning in warnings {
		println!("Warning: {warning}");
	}
	println!("\nMove your extensions to the extensions/ directory.");
	println!("Migration guide: {MIGRATION_GUIDE_URL}");
	println!("Documentation: {EXTENSIONS_DOC_URL}");
	println!();
}

/// Run all migrations (`runMigrations`). Called once on startup.
pub fn run_migrations(cwd: &str) -> RunMigrationsResult {
	run_migrations_with_agent_dir(&get_agent_dir(), cwd)
}

/// [`run_migrations`] against an explicit agent directory. Production passes
/// [`get_agent_dir`]; tests use it to avoid process-global env races.
pub fn run_migrations_with_agent_dir(agent_dir: &str, cwd: &str) -> RunMigrationsResult {
	let migrated_auth_providers = migrate_auth_to_auth_json(agent_dir);
	migrate_sessions_from_agent_root(agent_dir);
	migrate_tools_to_bin(agent_dir, &crate::config::get_bin_dir());
	migrate_keybindings_config_file(agent_dir);
	let deprecation_warnings = migrate_extension_system(agent_dir, cwd);
	RunMigrationsResult { migrated_auth_providers, deprecation_warnings }
}

/// Rewrite keybindings.json legacy names in place
/// (`migrateKeybindingsConfigFile`).
fn migrate_keybindings_config_file(agent_dir: &str) {
	let config_path = join(agent_dir, "keybindings.json");
	if !std::path::Path::new(&config_path).exists() {
		return;
	}

	let parsed = std::fs::read_to_string(&config_path)
		.ok()
		.map(|content| serde_json::from_str::<serde_json::Value>(strip_bom(&content)).ok())
		.flatten();
	let Some(parsed) = parsed else { return };
	if !parsed.is_object() || parsed.is_array() {
		return;
	}
	let Some(object) = parsed.as_object() else { return };
	let owned: serde_json::Map<String, serde_json::Value> = object.clone();
	let (config, migrated) = migrate_keybindings_config(&owned);
	if !migrated {
		return;
	}
	if let Ok(serialized) = serde_json::to_string_pretty(&serde_json::Value::Object(config)) {
		let _ = std::fs::write(&config_path, format!("{serialized}\n"));
	}
}
