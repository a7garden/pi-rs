//! Port of `packages/coding-agent/src/core/settings-manager.ts` — layered
//! (global `~/.pi/agent/settings.json` + project `<cwd>/.pi/settings.json`)
//! settings with deep merge, legacy migrations, modified-field tracking and
//! lock-guarded persistence.
//!
//! Fidelity notes (see crate-docs deviations D18–D20):
//! - [`Settings`] is an ordered JSON object, mirroring upstream's
//!   unvalidated, unknown-field-preserving `Settings` interface; typed
//!   getters validate lazily exactly like upstream accessors.
//! - Upstream's async write queue serializes writes through a promise chain;
//!   [`SettingsManager`] methods take `&mut self` and are synchronous, so
//!   writes are already serialized and `flush` is a no-op. Errors that
//!   upstream records asynchronously are recorded inline.
//! - File locking uses a `<path>.lock` create-new retry loop instead of
//!   `proper-lockfile`; lock/read/write failures surface as `Err` from
//!   [`SettingsStorage::with_lock`], mirroring upstream throws.

use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

use indexmap::{IndexMap, IndexSet};
use pi_agent::types::AgentThinkingLevel;
use pi_ai::types::Transport;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::CONFIG_DIR_NAME;
use crate::utils::paths::{join, normalize_path, resolve_path_cwd, PathInputOptions};

/// Upstream `DEFAULT_HTTP_IDLE_TIMEOUT_MS` (re-exported from
/// `core/http-dispatcher.ts`, which is not ported yet).
pub const DEFAULT_HTTP_IDLE_TIMEOUT_MS: u64 = 300_000;

/// Upstream default max server-requested retry delay (`pi-ai`
/// `DEFAULT_MAX_AGENT_RETRY_DELAY_MS`).
pub const DEFAULT_MAX_AGENT_RETRY_DELAY_MS: u64 = pi_ai::utils::retry::DEFAULT_MAX_AGENT_RETRY_DELAY_MS;

// =============================================================================
// Settings model
// =============================================================================

/// Upstream `Settings`: an unvalidated settings object. The Rust port keeps
/// it as an insertion-ordered JSON map so unknown fields and arbitrary
/// shapes survive round-trips exactly like upstream.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Settings(pub serde_json::Map<String, Value>);

impl Settings {
	pub fn new() -> Self {
		Self(serde_json::Map::new())
	}

	/// Parsed-settings constructor (used by tests and callers holding a
	/// `Value`).
	#[allow(dead_code)]
	pub(crate) fn from_value(value: Value) -> Self {
		match value {
			Value::Object(map) => Self(map),
			_ => Self::new(),
		}
	}

	pub fn get(&self, key: &str) -> Option<&Value> {
		self.0.get(key)
	}

	fn set(&mut self, key: &str, value: Value) {
		self.0.insert(key.to_string(), value);
	}

	fn remove(&mut self, key: &str) {
		self.0.remove(key);
	}
}

fn is_merge_object(value: &Value) -> bool {
	value.is_object()
}

/// Upstream `deepMergeObjects`.
fn deep_merge_objects(
	base: &serde_json::Map<String, Value>,
	overrides: &serde_json::Map<String, Value>,
) -> serde_json::Map<String, Value> {
	let mut result = base.clone();
	for (key, override_value) in overrides {
		let merged = match base.get(key) {
			Some(base_value) if is_merge_object(base_value) && is_merge_object(override_value) => Value::Object(
				deep_merge_objects(base_value.as_object().unwrap(), override_value.as_object().unwrap()),
			),
			_ => override_value.clone(),
		};
		result.insert(key.clone(), merged);
	}
	result
}

/// Tools enabled at startup when `defaultTools` does not change them
/// (upstream `DEFAULT_TOOL_NAMES`).
pub const DEFAULT_TOOL_NAMES: &[&str] = &["read", "bash", "edit", "write"];

fn is_tool_modifier(entry: &str) -> bool {
	entry.starts_with('+') || entry.starts_with('-')
}

fn value_string_list(value: Option<&Value>) -> Option<Vec<String>> {
	let value = value?;
	value.as_array().map(|entries| {
		entries
			.iter()
			.filter_map(|entry| entry.as_str().map(str::to_string))
			.collect()
	})
}

/// Upstream `mergeDefaultTools`: a list with plain tool names replaces the
/// inherited one; a list of only `+name`/`-name` entries is appended.
/// Settings files are not validated; a malformed value replaces instead of
/// throwing.
fn merge_default_tools(base: Option<&Value>, overrides: Option<&Value>) -> Option<Value> {
	let overrides = overrides?;
	let Some(base) = base else {
		return Some(overrides.clone());
	};
	let modifies = base.is_array()
		&& overrides.is_array()
		&& overrides
			.as_array()
			.unwrap()
			.iter()
			.all(|entry| entry.as_str().map(is_tool_modifier).unwrap_or(false));
	if !modifies {
		return Some(overrides.clone());
	}
	let mut merged = base.as_array().unwrap().clone();
	merged.extend(overrides.as_array().unwrap().iter().cloned());
	Some(Value::Array(merged))
}

/// Upstream `resolveDefaultTools`: plain names replace `DEFAULT_TOOL_NAMES`,
/// then `+name` adds and `-name` removes, in list order.
pub fn resolve_default_tools(entries: &[String]) -> Vec<String> {
	let plain: Vec<String> = entries.iter().filter(|entry| !is_tool_modifier(entry)).cloned().collect();
	let mut tools = if !plain.is_empty() || entries.is_empty() {
		plain
	} else {
		DEFAULT_TOOL_NAMES.iter().map(|name| (*name).to_string()).collect()
	};
	for entry in entries {
		if !is_tool_modifier(entry) {
			continue;
		}
		let name = &entry[1..];
		let index = tools.iter().position(|tool| tool == name);
		if entry.starts_with('+') && index.is_none() && !name.is_empty() {
			tools.push(name.to_string());
		} else if let Some(index) = index.filter(|_| entry.starts_with('-')) {
			tools.remove(index);
		}
	}
	tools
}

/// Upstream `deepMergeSettings`.
fn deep_merge_settings(base: &Settings, overrides: &Settings) -> Settings {
	let mut merged = Settings(deep_merge_objects(&base.0, &overrides.0));
	if let Some(default_tools) = merge_default_tools(base.get("defaultTools"), overrides.get("defaultTools")) {
		merged.set("defaultTools", default_tools);
	}
	merged
}

fn value_as_finite_f64(value: &Value) -> Option<f64> {
	let number = value.as_f64()?;
	number.is_finite().then_some(number)
}

/// Upstream `parseHttpIdleTimeoutMs` (`core/http-dispatcher.ts`): accepts a
/// finite non-negative number (floored) or a string (`"disabled"` → 0, empty
/// → `None`, otherwise numeric).
pub fn parse_http_idle_timeout_ms(value: &Value) -> Option<u64> {
	if let Some(text) = value.as_str() {
		let trimmed = text.trim();
		if trimmed.eq_ignore_ascii_case("disabled") {
			return Some(0);
		}
		if trimmed.is_empty() {
			return None;
		}
		return parse_http_idle_timeout_ms(&Value::from(trimmed.parse::<f64>().ok()?));
	}
	let number = value_as_finite_f64(value)?;
	(number >= 0.0).then_some(number.floor() as u64)
}

/// Upstream `parseTimeoutSetting`: `parseHttpIdleTimeoutMs` plus a throw on
/// any other non-`undefined` value.
fn parse_timeout_setting(value: Option<&Value>, setting_name: &str) -> Result<Option<u64>, String> {
	if let Some(value) = value {
		if let Some(timeout_ms) = parse_http_idle_timeout_ms(value) {
			return Ok(Some(timeout_ms));
		}
		return Err(format!("Invalid {setting_name} setting: {value}"));
	}
	Ok(None)
}

// =============================================================================
// Settings value enums (lazily validated string unions upstream)
// =============================================================================

macro_rules! string_enum {
	(
		$(#[$meta:meta])*
		pub enum $name:ident {
			$($(#[$variant_meta:meta])* $variant:ident = $literal:literal),+ $(,)?
		}
	) => {
		$(#[$meta])*
		#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
		pub enum $name {
			$($(#[$variant_meta])* #[serde(rename = $literal)] $variant,)+
		}

		// The full upstream accessor surface is ported; some helpers exist for
		// API parity even when current getters inline the parse.
		#[allow(dead_code)]
		impl $name {
			pub fn as_str(self) -> &'static str {
				match self {
					$($name::$variant => $literal,)+
				}
			}

			pub fn parse(value: &str) -> Option<Self> {
				match value {
					$($literal => Some($name::$variant),)+
					_ => None,
				}
			}

			fn from_value(value: &Value) -> Option<Self> {
				value.as_str().and_then(Self::parse)
			}

			fn to_value(self) -> Value {
				Value::from(self.as_str())
			}
		}
	};
}

string_enum! {
	/// Upstream `steeringMode` / `followUpMode` values.
	pub enum SteeringMode {
		All = "all",
		OneAtATime = "one-at-a-time",
	}
}

string_enum! {
	/// Cache-warming profile. "idle" also warms between agent runs
	/// (upstream `CacheWarmingMode`).
	pub enum CacheWarmingMode {
		Off = "off",
		Streaming = "streaming",
		Idle = "idle",
	}
}

string_enum! {
	/// Upstream `TuiMode`.
	pub enum TuiMode {
		Fullscreen = "fullscreen",
		Regular = "regular",
	}
}

string_enum! {
	/// Upstream `FullscreenExitOutput`.
	pub enum FullscreenExitOutput {
		Transcript = "transcript",
		ResumeHint = "resume-hint",
	}
}

string_enum! {
	/// Upstream `ScrollViewScrollbar`.
	pub enum ScrollViewScrollbar {
		Auto = "auto",
		Always = "always",
		Hidden = "hidden",
	}
}

string_enum! {
	/// Upstream `DoubleEscapeAction`.
	pub enum DoubleEscapeAction {
		Fork = "fork",
		Tree = "tree",
		None = "none",
	}
}

string_enum! {
	/// Upstream `TreeFilterMode`.
	pub enum TreeFilterMode {
		Default = "default",
		NoTools = "no-tools",
		UserOnly = "user-only",
		LabeledOnly = "labeled-only",
		All = "all",
	}
}

string_enum! {
	/// Upstream `MermaidRenderingMode`.
	pub enum MermaidRenderingMode {
		Off = "off",
		Final = "final",
		Streaming = "streaming",
	}
}

string_enum! {
	/// Upstream `CodemodeMode`.
	pub enum CodemodeMode {
		On = "on",
		Only = "only",
	}
}

string_enum! {
	/// Upstream `DefaultProjectTrust`.
	pub enum DefaultProjectTrust {
		Ask = "ask",
		Always = "always",
		Never = "never",
	}
}

string_enum! {
	/// Upstream terminal `images` selection (`"kitty" | "iterm2" | "auto"`;
	/// `false` is modeled as [`SettingsManager::get_terminal_images_none`]).
	pub enum TerminalImages {
		Kitty = "kitty",
		Iterm2 = "iterm2",
		Auto = "auto",
	}
}

/// Upstream `QuietStartup` (`boolean | "header"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuietStartup {
	No,
	Yes,
	Header,
}

/// Upstream `FullscreenWheelScrollLines` (`number | "auto"`, clamped 1–100).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WheelScrollLines {
	Auto,
	Lines(u32),
}

/// Terminal capability overrides derived from `terminal` settings. The typed
/// pi-tui `TerminalCapabilities` shape is TUI-scoped, so the port returns the
/// same key set as a plain struct (deviation D20).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TerminalCapabilityOverrides {
	/// `Some(None)` models upstream's explicit `images: null` (disable).
	pub images: Option<Option<TerminalImages>>,
	pub true_color: Option<bool>,
	pub hyperlinks: Option<bool>,
}

/// Package source for npm/git packages (upstream `PackageSource`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PackageSource {
	/// String form: load all resources from the package.
	Simple(String),
	/// Object form: filter which resources to load.
	Filtered {
		source: String,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		autoload: Option<bool>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		extensions: Option<Vec<String>>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		skills: Option<Vec<String>>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		prompts: Option<Vec<String>>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		themes: Option<Vec<String>>,
	},
}

/// Upstream `CompactionModelOverride`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionModelOverride {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub reserve_tokens: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub keep_recent_tokens: Option<u64>,
}

const DEFAULT_COMPACTION_RESERVE_TOKENS: u64 = 16384;
const DEFAULT_COMPACTION_KEEP_RECENT_TOKENS: u64 = 20000;

// =============================================================================
// Storage
// =============================================================================

/// Upstream `SettingsScope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SettingsScope {
	Global,
	Project,
}

/// Upstream `SettingsManagerCreateOptions`.
#[derive(Debug, Clone, Default)]
pub struct SettingsManagerCreateOptions {
	pub project_trusted: Option<bool>,
}

/// Upstream `SettingsError` (`error` flattened to its message).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsError {
	pub scope: SettingsScope,
	pub path: Option<String>,
	pub error: String,
}

/// Upstream `SettingsStorage`. `with_lock` hands the current file content (if
/// any) to `f` and writes the returned string (if any) back; a returned
/// `Err` mirrors an upstream throw inside `withLock`.
pub trait SettingsStorage {
	fn with_lock(
		&self,
		scope: SettingsScope,
		f: &mut dyn FnMut(Option<&str>) -> Option<String>,
	) -> Result<(), String>;
}

fn lock_path_for(path: &Path) -> PathBuf {
	let mut lock = path.as_os_str().to_os_string();
	lock.push(".lock");
	PathBuf::from(lock)
}

/// Upstream `FileSettingsStorage`: lock-guarded reads/writes of the two
/// settings files.
pub struct FileSettingsStorage {
	global_settings_path: PathBuf,
	project_settings_path: PathBuf,
}

impl FileSettingsStorage {
	/// Upstream `new FileSettingsStorage(cwd, agentDir)`.
	pub fn new(cwd: &str, agent_dir: &str) -> Self {
		let resolved_cwd = resolve_path_cwd(cwd);
		let resolved_agent_dir = resolve_path_cwd(agent_dir);
		Self {
			global_settings_path: PathBuf::from(join(&resolved_agent_dir, "settings.json")),
			project_settings_path: PathBuf::from(join(&resolved_cwd, &format!("{CONFIG_DIR_NAME}/settings.json"))),
		}
	}

	fn path_for(&self, scope: SettingsScope) -> &PathBuf {
		match scope {
			SettingsScope::Global => &self.global_settings_path,
			SettingsScope::Project => &self.project_settings_path,
		}
	}

	/// Upstream `acquireLockSyncWithRetry` (10 attempts, 20 ms apart).
	fn acquire_lock_sync_with_retry(path: &Path) -> Result<(), String> {
		const MAX_ATTEMPTS: usize = 10;
		const DELAY: Duration = Duration::from_millis(20);
		let lock_path = lock_path_for(path);
		let mut last_error = "Failed to acquire settings lock".to_string();
		for attempt in 1..=MAX_ATTEMPTS {
			match std::fs::OpenOptions::new().write(true).create_new(true).open(&lock_path) {
				Ok(_) => return Ok(()),
				Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists && attempt < MAX_ATTEMPTS => {
					last_error = format!("ELOCKED: {lock_path:?} is locked");
					let deadline = Instant::now() + DELAY;
					while Instant::now() < deadline {
						// Sleep synchronously to avoid changing callers to async.
					}
					sleep(Duration::from_millis(1));
				}
				Err(error) => return Err(error.to_string()),
			}
		}
		Err(last_error)
	}
}

impl SettingsStorage for FileSettingsStorage {
	fn with_lock(
		&self,
		scope: SettingsScope,
		f: &mut dyn FnMut(Option<&str>) -> Option<String>,
	) -> Result<(), String> {
		struct LockGuard<'a>(&'a Path);
		impl Drop for LockGuard<'_> {
			fn drop(&mut self) {
				let _ = std::fs::remove_file(lock_path_for(self.0));
			}
		}

		let path = self.path_for(scope);
		let file_exists = path.exists();
		let _guard = if file_exists {
			FileSettingsStorage::acquire_lock_sync_with_retry(path)?;
			Some(LockGuard(path))
		} else {
			None
		};
		let current = std::fs::read_to_string(path).ok();
		let next = f(current.as_deref());
		if let Some(next) = next {
			if let Some(dir) = path.parent() {
				std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
			}
			if _guard.is_none() {
				FileSettingsStorage::acquire_lock_sync_with_retry(path)?;
			}
			std::fs::write(path, next).map_err(|error| error.to_string())?;
			if _guard.is_none() {
				let _ = std::fs::remove_file(lock_path_for(path));
			}
		}
		Ok(())
	}
}

/// Upstream `InMemorySettingsStorage`.
#[derive(Default)]
pub struct InMemorySettingsStorage {
	global: parking_lot::Mutex<Option<String>>,
	project: parking_lot::Mutex<Option<String>>,
}

impl InMemorySettingsStorage {
	fn scope_cell(&self, scope: SettingsScope) -> &parking_lot::Mutex<Option<String>> {
		match scope {
			SettingsScope::Global => &self.global,
			SettingsScope::Project => &self.project,
		}
	}
}

impl SettingsStorage for InMemorySettingsStorage {
	fn with_lock(
		&self,
		scope: SettingsScope,
		f: &mut dyn FnMut(Option<&str>) -> Option<String>,
	) -> Result<(), String> {
		let mut cell = self.scope_cell(scope).lock();
		let next = f(cell.as_deref());
		if let Some(next) = next {
			*cell = Some(next);
		}
		Ok(())
	}
}

// =============================================================================
// SettingsManager
// =============================================================================

fn strip_bom(content: &str) -> &str {
	content.strip_prefix('\u{feff}').unwrap_or(content)
}

fn uuid_v4() -> String {
	let mut bytes = [0u8; 16];
	getrandom::getrandom(&mut bytes).expect("getrandom failed");
	bytes[6] = (bytes[6] & 0x0f) | 0x40;
	bytes[8] = (bytes[8] & 0x3f) | 0x80;
	let hex = |slice: &[u8]| slice.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
	format!(
		"{}-{}-{}-{}-{}",
		hex(&bytes[0..4]),
		hex(&bytes[4..6]),
		hex(&bytes[6..8]),
		hex(&bytes[8..10]),
		hex(&bytes[10..16])
	)
}

/// Upstream `SettingsManager`.
pub struct SettingsManager {
	storage: Box<dyn SettingsStorage>,
	global_settings: Settings,
	project_settings: Settings,
	settings: Settings,
	project_trusted: bool,
	/// Track global fields modified during session (insertion-ordered, like
	/// upstream's `Set`).
	modified_fields: IndexSet<String>,
	/// Track global nested field modifications.
	modified_nested_fields: IndexMap<String, IndexSet<String>>,
	modified_project_fields: IndexSet<String>,
	modified_project_nested_fields: IndexMap<String, IndexSet<String>>,
	/// Track if global settings file had parse errors.
	global_settings_load_error: Option<String>,
	project_settings_load_error: Option<String>,
	errors: Vec<SettingsError>,
	settings_paths: SettingsPaths,
}

#[derive(Default, Debug, Clone)]
struct SettingsPaths {
	global: Option<String>,
	project: Option<String>,
}

impl SettingsManager {
	fn new(
		storage: Box<dyn SettingsStorage>,
		initial_global: Settings,
		initial_project: Settings,
		global_load_error: Option<String>,
		project_load_error: Option<String>,
		initial_errors: Vec<SettingsError>,
		project_trusted: bool,
		settings_paths: SettingsPaths,
	) -> Self {
		let settings = deep_merge_settings(&initial_global, &initial_project);
		Self {
			storage,
			global_settings: initial_global,
			project_settings: initial_project,
			settings,
			project_trusted,
			modified_fields: IndexSet::new(),
			modified_nested_fields: IndexMap::new(),
			modified_project_fields: IndexSet::new(),
			modified_project_nested_fields: IndexMap::new(),
			global_settings_load_error: global_load_error,
			project_settings_load_error: project_load_error,
			errors: initial_errors,
			settings_paths,
		}
	}

	/// Create a SettingsManager that loads from files (upstream `create`).
	pub fn create(cwd: &str, agent_dir: &str, options: SettingsManagerCreateOptions) -> Self {
		let resolved_cwd = resolve_path_cwd(cwd);
		let resolved_agent_dir = resolve_path_cwd(agent_dir);
		let storage = Box::new(FileSettingsStorage::new(&resolved_cwd, &resolved_agent_dir));
		Self::from_storage_with_paths(
			storage,
			options,
			SettingsPaths {
				global: Some(join(&resolved_agent_dir, "settings.json")),
				project: Some(join(&resolved_cwd, &format!("{CONFIG_DIR_NAME}/settings.json"))),
			},
		)
	}

	/// Create a SettingsManager from an arbitrary storage backend
	/// (upstream `fromStorage`).
	pub fn from_storage(storage: Box<dyn SettingsStorage>, options: SettingsManagerCreateOptions) -> Self {
		Self::from_storage_with_paths(storage, options, SettingsPaths::default())
	}

	fn from_storage_with_paths(
		storage: Box<dyn SettingsStorage>,
		options: SettingsManagerCreateOptions,
		settings_paths: SettingsPaths,
	) -> Self {
		let project_trusted = options.project_trusted.unwrap_or(true);
		let global_load = Self::try_load_from_storage(storage.as_ref(), SettingsScope::Global, true);
		let project_load = Self::try_load_from_storage(storage.as_ref(), SettingsScope::Project, project_trusted);
		let mut initial_errors: Vec<SettingsError> = Vec::new();
		if let Some(error) = &global_load.error {
			initial_errors.push(SettingsError {
				scope: SettingsScope::Global,
				path: settings_paths.global.clone(),
				error: error.clone(),
			});
		}
		if let Some(error) = &project_load.error {
			initial_errors.push(SettingsError {
				scope: SettingsScope::Project,
				path: settings_paths.project.clone(),
				error: error.clone(),
			});
		}

		Self::new(
			storage,
			global_load.settings,
			project_load.settings,
			global_load.error,
			project_load.error,
			initial_errors,
			project_trusted,
			settings_paths,
		)
	}

	/// Create an in-memory SettingsManager (no file I/O).
	pub fn in_memory(settings: &Settings, options: SettingsManagerCreateOptions) -> Self {
		let storage = Box::new(InMemorySettingsStorage::default());
		let initial_settings = Self::migrate_settings(serde_json::to_value(settings).unwrap_or(Value::Object(Default::default())));
		let storage_ref: Box<dyn SettingsStorage> = storage;
		let serialized = serde_json::to_string_pretty(&initial_settings).unwrap();
		storage_ref
			.with_lock(SettingsScope::Global, &mut |_| Some(serialized.clone()))
			.expect("in-memory storage never fails");
		Self::from_storage(storage_ref, options)
	}

	fn load_from_storage(
		storage: &dyn SettingsStorage,
		scope: SettingsScope,
		project_trusted: bool,
	) -> Result<Settings, String> {
		if scope == SettingsScope::Project && !project_trusted {
			return Ok(Settings::new());
		}

		let mut content: Option<String> = None;
		storage.with_lock(scope, &mut |current| {
			content = current.map(str::to_string);
			None
		})?;

		let Some(content) = content else {
			return Ok(Settings::new());
		};
		if content.is_empty() {
			return Ok(Settings::new());
		}
		let parsed: Value = serde_json::from_str(strip_bom(&content)).map_err(|error| error.to_string())?;
		Ok(Self::migrate_settings(parsed))
	}

	fn try_load_from_storage(storage: &dyn SettingsStorage, scope: SettingsScope, project_trusted: bool) -> LoadedSettings {
		match Self::load_from_storage(storage, scope, project_trusted) {
			Ok(settings) => LoadedSettings {
				settings,
				error: None,
			},
			Err(error) => LoadedSettings {
				settings: Settings::new(),
				error: Some(error),
			},
		}
	}

	/// Migrate old settings format to new format (upstream
	/// `migrateSettings`).
	fn migrate_settings(value: Value) -> Settings {
		let mut settings = value.as_object().cloned().unwrap_or_default();
		// Migrate queueMode -> steeringMode
		if settings.contains_key("queueMode") && !settings.contains_key("steeringMode") {
			let value = settings.remove("queueMode").unwrap();
			settings.insert("steeringMode".into(), value);
		}

		// Migrate legacy websockets boolean -> transport enum
		if !settings.contains_key("transport") {
			if let Some(websockets) = settings.get("websockets").and_then(Value::as_bool) {
				settings.insert(
					"transport".into(),
					Value::from(if websockets { "websocket" } else { "sse" }),
				);
			}
		}
		settings.remove("websockets");

		// Migrate old skills object format to new array format
		if let Some(skills) = settings.get("skills").cloned() {
			if skills.is_object() {
				let skills_settings = skills.as_object().unwrap();
				let enable = skills_settings.get("enableSkillCommands").cloned();
				let custom = skills_settings.get("customDirectories").cloned();
				if let Some(enable) = enable {
					if !settings.contains_key("enableSkillCommands") {
						settings.insert("enableSkillCommands".into(), enable);
					}
				}
				let custom_dirs = custom.and_then(|value| value.as_array().cloned());
				match custom_dirs {
					Some(dirs) if !dirs.is_empty() => {
						settings.insert("skills".into(), Value::Array(dirs));
					}
					_ => {
						settings.remove("skills");
					}
				}
			}
		}

		// Migrate retry.maxDelayMs -> retry.provider.maxRetryDelayMs
		if let Some(retry) = settings.get_mut("retry").map(Value::as_object_mut).flatten() {
			let max_delay_ms = retry.get("maxDelayMs").and_then(Value::as_f64);
			let provider = retry.get("provider");
			let provider_overrides = if provider.map(Value::is_object).unwrap_or(false) {
				provider.unwrap().as_object().cloned().unwrap()
			} else {
				serde_json::Map::new()
			};
			let provider_has_override = provider
				.and_then(|value| value.get("maxRetryDelayMs"))
				.map(|value| !value.is_null())
				.unwrap_or(false);
			if let Some(max_delay_ms) = max_delay_ms.filter(|_| !provider_has_override) {
				let mut new_provider = provider_overrides;
				new_provider.insert("maxRetryDelayMs".into(), serde_json::json!(max_delay_ms as u64));
				retry.insert("provider".into(), Value::Object(new_provider));
			}
			retry.remove("maxDelayMs");
		}

		Settings(settings)
	}

	/// A copy of the effective settings: global and project settings merged,
	/// with overrides.
	pub fn get_settings(&self) -> Settings {
		self.settings.clone()
	}

	pub fn get_global_settings(&self) -> Settings {
		self.global_settings.clone()
	}

	pub fn get_project_settings(&self) -> Settings {
		self.project_settings.clone()
	}

	pub fn is_project_trusted(&self) -> bool {
		self.project_trusted
	}

	pub fn set_project_trusted(&mut self, trusted: bool) {
		if self.project_trusted == trusted {
			return;
		}

		self.project_trusted = trusted;
		self.modified_project_fields.clear();
		self.modified_project_nested_fields.clear();

		if !trusted {
			self.project_settings = Settings::new();
			self.project_settings_load_error = None;
			self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);
			return;
		}

		let project_load = Self::try_load_from_storage(self.storage.as_ref(), SettingsScope::Project, trusted);
		self.project_settings = project_load.settings;
		self.project_settings_load_error = project_load.error.clone();
		if let Some(error) = project_load.error {
			self.record_error(SettingsScope::Project, error);
		}
		self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);
	}

	/// Re-read both settings layers from storage (upstream `reload`).
	pub fn reload(&mut self) {
		let global_load = Self::try_load_from_storage(self.storage.as_ref(), SettingsScope::Global, true);
		if global_load.error.is_none() {
			self.global_settings = global_load.settings;
			self.global_settings_load_error = None;
		} else {
			self.global_settings_load_error = global_load.error.clone();
			self.record_error(SettingsScope::Global, global_load.error.unwrap());
		}

		self.modified_fields.clear();
		self.modified_nested_fields.clear();
		self.modified_project_fields.clear();
		self.modified_project_nested_fields.clear();

		let project_load = Self::try_load_from_storage(self.storage.as_ref(), SettingsScope::Project, self.project_trusted);
		if project_load.error.is_none() {
			self.project_settings = project_load.settings;
			self.project_settings_load_error = None;
		} else {
			self.project_settings_load_error = project_load.error.clone();
			self.record_error(SettingsScope::Project, project_load.error.unwrap());
		}

		self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);
	}

	/// Apply additional overrides on top of current settings.
	pub fn apply_overrides(&mut self, overrides: &Settings) {
		self.settings = deep_merge_settings(&self.settings, overrides);
	}

	fn mark_modified(&mut self, field: &str, nested_key: Option<&str>) {
		self.modified_fields.insert(field.to_string());
		if let Some(nested_key) = nested_key {
			self.modified_nested_fields
				.entry(field.to_string())
				.or_default()
				.insert(nested_key.to_string());
		}
	}

	fn mark_project_modified(&mut self, field: &str, nested_key: Option<&str>) {
		self.modified_project_fields.insert(field.to_string());
		if let Some(nested_key) = nested_key {
			self.modified_project_nested_fields
				.entry(field.to_string())
				.or_default()
				.insert(nested_key.to_string());
		}
	}

	fn assert_project_trusted_for_write(&self) -> Result<(), String> {
		if !self.project_trusted {
			return Err("Project is not trusted; refusing to write project settings".to_string());
		}
		Ok(())
	}

	fn record_error(&mut self, scope: SettingsScope, error: String) {
		let path = match scope {
			SettingsScope::Global => self.settings_paths.global.clone(),
			SettingsScope::Project => self.settings_paths.project.clone(),
		};
		self.errors.push(SettingsError { scope, path, error });
	}

	fn clear_modified_scope(&mut self, scope: SettingsScope) {
		match scope {
			SettingsScope::Global => {
				self.modified_fields.clear();
				self.modified_nested_fields.clear();
			}
			SettingsScope::Project => {
				self.modified_project_fields.clear();
				self.modified_project_nested_fields.clear();
			}
		}
	}

	fn enqueue_write(&mut self, scope: SettingsScope, task: impl FnOnce(&mut Self)) {
		if scope == SettingsScope::Project {
			if let Err(error) = self.assert_project_trusted_for_write() {
				self.record_error(scope, error);
				return;
			}
		}
		task(self);
		self.clear_modified_scope(scope);
	}

	fn persist_scoped_settings(
		&mut self,
		scope: SettingsScope,
		snapshot_settings: &Settings,
		modified_fields: &IndexSet<String>,
		modified_nested_fields: &IndexMap<String, IndexSet<String>>,
	) -> Result<(), String> {
		let snapshot_fields = modified_fields.clone();
		let snapshot_nested = modified_nested_fields.clone();
		let mut parse_error: Option<String> = None;
		let write_result = self.storage.with_lock(scope, &mut |current| {
			let current_file_settings = match current {
				Some(current) if !current.is_empty() => {
					let parsed: Value = match serde_json::from_str(strip_bom(current)) {
						Ok(parsed) => parsed,
						Err(error) => {
							parse_error = Some(error.to_string());
							return None;
						}
					};
					Self::migrate_settings(parsed).0
				}
				_ => serde_json::Map::new(),
			};
			let mut merged_settings = current_file_settings.clone();
			for field in &snapshot_fields {
				let Some(value) = snapshot_settings.get(field) else {
					continue;
				};
				if snapshot_nested.contains_key(field) && value.is_object() {
					let nested_modified = &snapshot_nested[field];
					let base_nested = current_file_settings
						.get(field.as_str())
						.and_then(Value::as_object)
						.cloned()
						.unwrap_or_default();
					let in_memory_nested = value.as_object().unwrap();
					let mut merged_nested = base_nested;
					for nested_key in nested_modified {
						if let Some(nested_value) = in_memory_nested.get(nested_key.as_str()) {
							merged_nested.insert(nested_key.clone(), nested_value.clone());
						}
					}
					merged_settings.insert(field.clone(), Value::Object(merged_nested));
				} else {
					merged_settings.insert(field.clone(), value.clone());
				}
			}
			Some(serde_json::to_string_pretty(&Settings(merged_settings)).unwrap())
		});
		if let Some(error) = parse_error {
			return Err(error);
		}
		write_result
	}

	fn save(&mut self) {
		self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);

		if self.global_settings_load_error.is_some() {
			return;
		}

		let snapshot_global_settings = self.global_settings.clone();
		let modified_fields = self.modified_fields.clone();
		let modified_nested_fields = self.modified_nested_fields.clone();

		self.enqueue_write(SettingsScope::Global, |this| {
			if let Err(error) =
				this.persist_scoped_settings(SettingsScope::Global, &snapshot_global_settings, &modified_fields, &modified_nested_fields)
			{
				this.record_error(SettingsScope::Global, error);
			}
		});
	}

	fn save_project_settings(&mut self, settings: Settings) {
		if let Err(error) = self.assert_project_trusted_for_write() {
			self.record_error(SettingsScope::Project, error);
			return;
		}
		self.project_settings = settings;
		self.settings = deep_merge_settings(&self.global_settings, &self.project_settings);

		if self.project_settings_load_error.is_some() {
			return;
		}

		let snapshot_project_settings = self.project_settings.clone();
		let modified_fields = self.modified_project_fields.clone();
		let modified_nested_fields = self.modified_project_nested_fields.clone();

		self.enqueue_write(SettingsScope::Project, |this| {
			if let Err(error) = this.persist_scoped_settings(
				SettingsScope::Project,
				&snapshot_project_settings,
				&modified_fields,
				&modified_nested_fields,
			) {
				this.record_error(SettingsScope::Project, error);
			}
		});
	}

	fn update_project_settings(&mut self, field: &str, update: impl FnOnce(&mut Settings)) {
		if self.assert_project_trusted_for_write().is_err() {
			self.record_error(
				SettingsScope::Project,
				"Project is not trusted; refusing to write project settings".to_string(),
			);
			return;
		}
		let mut project_settings = self.project_settings.clone();
		update(&mut project_settings);
		self.mark_project_modified(field, None);
		self.save_project_settings(project_settings);
	}

	/// Writes are synchronous in the port; retained for API parity.
	pub fn flush(&mut self) {}

	/// Drain recorded storage/parse errors (upstream `drainErrors`).
	pub fn drain_errors(&mut self) -> Vec<SettingsError> {
		std::mem::take(&mut self.errors)
	}

	fn set_global_field(&mut self, field: &str, value: Value, nested_key: Option<&str>) {
		self.global_settings.set(field, value);
		self.mark_modified(field, nested_key);
		self.save();
	}

	/// Upstream setters that assign `undefined` drop the key (JSON.stringify
	/// omits undefined fields).
	fn set_global_optional_field(&mut self, field: &str, value: Option<Value>) {
		match value {
			Some(value) => self.global_settings.set(field, value),
			None => self.global_settings.remove(field),
		}
		self.mark_modified(field, None);
		self.save();
	}

	// ----- simple scalar accessors -----

	pub fn get_last_changelog_version(&self) -> Option<String> {
		self.settings.get("lastChangelogVersion").and_then(Value::as_str).map(str::to_string)
	}

	pub fn set_last_changelog_version(&mut self, version: &str) {
		self.set_global_field("lastChangelogVersion", Value::from(version), None);
	}

	pub fn get_session_dir(&self) -> Option<String> {
		let session_dir = self.settings.get("sessionDir").and_then(Value::as_str)?;
		Some(normalize_path(session_dir, &PathInputOptions::default()))
	}

	pub fn get_default_provider(&self) -> Option<String> {
		self.settings.get("defaultProvider").and_then(Value::as_str).map(str::to_string)
	}

	pub fn get_default_model(&self) -> Option<String> {
		self.settings.get("defaultModel").and_then(Value::as_str).map(str::to_string)
	}

	pub fn set_default_provider(&mut self, provider: &str) {
		self.set_global_field("defaultProvider", Value::from(provider), None);
	}

	pub fn set_default_model(&mut self, model_id: &str) {
		self.set_global_field("defaultModel", Value::from(model_id), None);
	}

	pub fn set_default_model_and_provider(&mut self, provider: &str, model_id: &str) {
		self.global_settings.set("defaultProvider", Value::from(provider));
		self.global_settings.set("defaultModel", Value::from(model_id));
		self.mark_modified("defaultProvider", None);
		self.mark_modified("defaultModel", None);
		self.save();
	}

	fn get_mode_setting(&self, field: &str) -> Option<SteeringMode> {
		SteeringMode::from_value(self.settings.get(field)?)
	}

	pub fn get_steering_mode(&self) -> SteeringMode {
		self.get_mode_setting("steeringMode").unwrap_or(SteeringMode::OneAtATime)
	}

	pub fn set_steering_mode(&mut self, mode: SteeringMode) {
		self.set_global_field("steeringMode", mode.to_value(), None);
	}

	pub fn get_follow_up_mode(&self) -> SteeringMode {
		self.get_mode_setting("followUpMode").unwrap_or(SteeringMode::OneAtATime)
	}

	pub fn set_follow_up_mode(&mut self, mode: SteeringMode) {
		self.set_global_field("followUpMode", mode.to_value(), None);
	}

	pub fn get_theme_setting(&self) -> Option<String> {
		self.settings.get("theme").and_then(Value::as_str).map(str::to_string)
	}

	/// Theme names with a `/` are file paths, not registered themes.
	pub fn get_theme(&self) -> Option<String> {
		let theme = self.get_theme_setting()?;
		if theme.contains('/') {
			None
		} else {
			Some(theme)
		}
	}

	pub fn set_theme(&mut self, theme: &str) {
		self.set_global_field("theme", Value::from(theme), None);
	}

	pub fn get_default_thinking_level(&self) -> Option<AgentThinkingLevel> {
		let value = self.settings.get("defaultThinkingLevel")?.as_str()?;
		parse_thinking_level(value)
	}

	pub fn set_default_thinking_level(&mut self, level: AgentThinkingLevel) {
		self.set_global_field("defaultThinkingLevel", Value::from(thinking_level_str(level)), None);
	}

	pub fn get_model_thinking_level(&self, provider: &str, model_id: &str) -> Option<AgentThinkingLevel> {
		let key = format!("{provider}/{model_id}");
		let value = self.settings.get("modelThinkingLevels")?.get(key)?.as_str()?;
		parse_thinking_level(value)
	}

	pub fn get_all_model_thinking_levels(&self) -> Vec<(String, AgentThinkingLevel)> {
		let mut result = Vec::new();
		if let Some(levels) = self.settings.get("modelThinkingLevels").and_then(Value::as_object) {
			for (key, value) in levels {
				if let Some(level) = value.as_str().and_then(parse_thinking_level) {
					result.push((key.clone(), level));
				}
			}
		}
		result
	}

	pub fn set_model_thinking_level(&mut self, provider: &str, model_id: &str, level: AgentThinkingLevel) {
		let key = format!("{provider}/{model_id}");
		let levels = self
			.global_settings
			.get("modelThinkingLevels")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		let mut levels = levels;
		levels.insert(key, Value::from(thinking_level_str(level)));
		self.set_global_field("modelThinkingLevels", Value::Object(levels), None);
	}

	pub fn remove_model_thinking_level(&mut self, provider: &str, model_id: &str) {
		let key = format!("{provider}/{model_id}");
		let Some(levels) = self.global_settings.get("modelThinkingLevels").and_then(Value::as_object).cloned() else {
			return;
		};
		let mut levels = levels;
		levels.shift_remove(&key);
		if levels.is_empty() {
			self.global_settings.remove("modelThinkingLevels");
		} else {
			self.global_settings.set("modelThinkingLevels", Value::Object(levels));
		}
		self.mark_modified("modelThinkingLevels", None);
		self.save();
	}

	pub fn get_transport(&self) -> Transport {
		self.settings
			.get("transport")
			.and_then(|value| serde_json::from_value(value.clone()).ok())
			.unwrap_or(Transport::Auto)
	}

	pub fn set_transport(&mut self, transport: Transport) {
		let value = serde_json::to_value(transport).unwrap();
		self.set_global_field("transport", value, None);
	}

	pub fn get_compaction_enabled(&self) -> bool {
		self.settings
			.get("compaction")
			.and_then(|value| value.get("enabled"))
			.and_then(Value::as_bool)
			.unwrap_or(true)
	}

	pub fn set_compaction_enabled(&mut self, enabled: bool) {
		let mut compaction = self
			.global_settings
			.get("compaction")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		compaction.insert("enabled".into(), Value::from(enabled));
		self.set_global_field("compaction", Value::Object(compaction), Some("enabled"));
	}

	/// Upstream `getCompactionTokenSetting`: model override, then ordinary
	/// setting, then built-in default, with upstream's validation errors.
	fn get_compaction_token_setting(&self, field: &str, model: Option<(&str, &str)>) -> Result<u64, String> {
		let compaction = self.settings.get("compaction");
		let ordinary = compaction.and_then(|value| value.get(field)).cloned();
		if let Some(ordinary_value) = &ordinary {
			let valid = ordinary_value.as_u64().is_some();
			if !valid {
				return Err(format!(
					"Invalid compaction.{field} setting: {ordinary_value}. Expected a non-negative safe integer."
				));
			}
		}

		let model_key = model.map(|(provider, id)| format!("{provider}/{id}"));
		let entry = match &model_key {
			Some(key) => compaction.and_then(|value| value.get("modelOverrides")).and_then(|value| value.get(key.as_str())).cloned(),
			None => None,
		};
		if let Some(entry_value) = &entry {
			if !is_merge_object(entry_value) {
				let key = model_key.as_deref().unwrap_or("");
				return Err(format!(
					"Invalid compaction.modelOverrides[\"{key}\"] setting: {entry_value}. Expected an object."
				));
			}
		}
		let override_value = entry.as_ref().and_then(|value| value.get(field)).cloned();
		if let Some(override_value) = &override_value {
			if override_value.as_u64().is_none() {
				let key = model_key.as_deref().unwrap_or("");
				return Err(format!(
					"Invalid compaction.modelOverrides[\"{key}\"].{field} setting: {override_value}. Expected a non-negative safe integer."
				));
			}
		}
		let default = match field {
			"reserveTokens" => DEFAULT_COMPACTION_RESERVE_TOKENS,
			"keepRecentTokens" => DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
			_ => unreachable!("known compaction token field"),
		};
		Ok(override_value
			.as_ref()
			.and_then(Value::as_u64)
			.or(ordinary.as_ref().and_then(Value::as_u64))
			.unwrap_or(default))
	}

	pub fn get_compaction_reserve_tokens(&self, model: Option<(&str, &str)>) -> Result<u64, String> {
		self.get_compaction_token_setting("reserveTokens", model)
	}

	pub fn get_compaction_keep_recent_tokens(&self, model: Option<(&str, &str)>) -> Result<u64, String> {
		self.get_compaction_token_setting("keepRecentTokens", model)
	}

	/// Resolve each token setting through model override, ordinary setting,
	/// then built-in default.
	pub fn get_compaction_settings(
		&self,
		model: Option<(&str, &str)>,
	) -> Result<(bool, u64, u64), String> {
		Ok((
			self.get_compaction_enabled(),
			self.get_compaction_reserve_tokens(model)?,
			self.get_compaction_keep_recent_tokens(model)?,
		))
	}

	pub fn get_branch_summary_reserve_tokens(&self) -> u64 {
		self.settings
			.get("branchSummary")
			.and_then(|value| value.get("reserveTokens"))
			.and_then(Value::as_u64)
			.unwrap_or(16384)
	}

	pub fn get_branch_summary_skip_prompt(&self) -> bool {
		self.settings
			.get("branchSummary")
			.and_then(|value| value.get("skipPrompt"))
			.and_then(Value::as_bool)
			.unwrap_or(false)
	}

	pub fn get_retry_enabled(&self) -> bool {
		self.settings
			.get("retry")
			.and_then(|value| value.get("enabled"))
			.and_then(Value::as_bool)
			.unwrap_or(true)
	}

	pub fn set_retry_enabled(&mut self, enabled: bool) {
		let mut retry = self
			.global_settings
			.get("retry")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		retry.insert("enabled".into(), Value::from(enabled));
		self.set_global_field("retry", Value::Object(retry), Some("enabled"));
	}

	pub fn get_retry_settings(&self) -> (bool, u64, u64, u64) {
		let retry = self.settings.get("retry");
		(
			self.get_retry_enabled(),
			retry.and_then(|value| value.get("maxRetries")).and_then(Value::as_u64).unwrap_or(3),
			retry.and_then(|value| value.get("baseDelayMs")).and_then(Value::as_u64).unwrap_or(2000),
			retry
				.and_then(|value| value.get("maxAgentDelayMs"))
				.and_then(Value::as_u64)
				.unwrap_or(DEFAULT_MAX_AGENT_RETRY_DELAY_MS),
		)
	}

	pub fn get_provider_retry_settings(&self) -> (Option<u64>, Option<u64>, u64) {
		let provider = self.settings.get("retry").and_then(|value| value.get("provider"));
		(
			provider.and_then(|value| value.get("timeoutMs")).and_then(Value::as_u64),
			provider.and_then(|value| value.get("maxRetries")).and_then(Value::as_u64),
			provider
				.and_then(|value| value.get("maxRetryDelayMs"))
				.and_then(Value::as_u64)
				.unwrap_or(60000),
		)
	}

	pub fn get_http_idle_timeout_ms(&self) -> Result<u64, String> {
		Ok(parse_timeout_setting(self.settings.get("httpIdleTimeoutMs"), "httpIdleTimeoutMs")?
			.unwrap_or(DEFAULT_HTTP_IDLE_TIMEOUT_MS))
	}

	pub fn set_http_idle_timeout_ms(&mut self, timeout_ms: u64) {
		self.set_global_field("httpIdleTimeoutMs", Value::from(timeout_ms), None);
	}

	/// Read from global settings only because warming costs money.
	pub fn get_cache_warming_mode(&self) -> CacheWarmingMode {
		self.global_settings
			.get("cacheWarming")
			.and_then(CacheWarmingMode::from_value)
			.unwrap_or(CacheWarmingMode::Streaming)
	}

	pub fn set_cache_warming_mode(&mut self, mode: CacheWarmingMode) {
		self.set_global_field("cacheWarming", mode.to_value(), None);
	}

	pub fn get_web_socket_connect_timeout_ms(&self) -> Result<Option<u64>, String> {
		parse_timeout_setting(self.settings.get("websocketConnectTimeoutMs"), "websocketConnectTimeoutMs")
	}

	pub fn get_hide_thinking_block(&self) -> bool {
		self.settings.get("hideThinkingBlock").and_then(Value::as_bool).unwrap_or(false)
	}

	pub fn set_hide_thinking_block(&mut self, hide: bool) {
		self.set_global_field("hideThinkingBlock", Value::from(hide), None);
	}

	pub fn get_show_cache_miss_notices(&self) -> bool {
		self.settings.get("showCacheMissNotices").and_then(Value::as_bool).unwrap_or(false)
	}

	pub fn set_show_cache_miss_notices(&mut self, show: bool) {
		self.set_global_field("showCacheMissNotices", Value::from(show), None);
	}

	/// Upstream `getExternalEditorCommand`: configured editor, then
	/// `VISUAL`/`EDITOR`, then platform default.
	pub fn get_external_editor_command(&self) -> String {
		if let Some(configured) = self.settings.get("externalEditor").and_then(Value::as_str) {
			if !configured.trim().is_empty() {
				return configured.to_string();
			}
		}
		for key in ["VISUAL", "EDITOR"] {
			if let Ok(editor) = std::env::var(key) {
				if !editor.is_empty() {
					return editor;
				}
			}
		}
		if cfg!(windows) {
			"notepad".to_string()
		} else {
			"nano".to_string()
		}
	}

	pub fn get_shell_path(&self) -> Option<String> {
		let shell_path = self.settings.get("shellPath").and_then(Value::as_str)?;
		Some(normalize_path(shell_path, &PathInputOptions::default()))
	}

	pub fn set_shell_path(&mut self, path: Option<&str>) {
		let value = path.map(Value::from);
		self.set_global_optional_field("shellPath", value);
	}

	pub fn get_quiet_startup(&self) -> QuietStartup {
		match self.settings.get("quietStartup") {
			Some(Value::Bool(true)) => QuietStartup::Yes,
			Some(value) => QuietStartupHeader::from_value(value).unwrap_or(QuietStartup::No),
			_ => QuietStartup::No,
		}
	}

	pub fn set_quiet_startup(&mut self, quiet: QuietStartup) {
		let value = match quiet {
			QuietStartup::No => Value::from(false),
			QuietStartup::Yes => Value::from(true),
			QuietStartup::Header => Value::from("header"),
		};
		self.set_global_field("quietStartup", value, None);
	}

	pub fn get_default_project_trust(&self) -> DefaultProjectTrust {
		self.global_settings
			.get("defaultProjectTrust")
			.and_then(DefaultProjectTrust::from_value)
			.unwrap_or(DefaultProjectTrust::Ask)
	}

	pub fn set_default_project_trust(&mut self, default_project_trust: DefaultProjectTrust) {
		self.set_global_field("defaultProjectTrust", default_project_trust.to_value(), None);
	}

	pub fn get_shell_command_prefix(&self) -> Option<String> {
		self.settings.get("shellCommandPrefix").and_then(Value::as_str).map(str::to_string)
	}

	pub fn set_shell_command_prefix(&mut self, prefix: Option<&str>) {
		self.set_global_optional_field("shellCommandPrefix", prefix.map(Value::from));
	}

	pub fn get_npm_command(&self) -> Option<Vec<String>> {
		value_string_list(self.settings.get("npmCommand"))
	}

	pub fn set_npm_command(&mut self, command: Option<&[String]>) {
		self.set_global_optional_field("npmCommand", command.map(|entries| serde_json::to_value(entries).unwrap()));
	}

	pub fn get_collapse_changelog(&self) -> bool {
		self.settings.get("collapseChangelog").and_then(Value::as_bool).unwrap_or(false)
	}

	pub fn set_collapse_changelog(&mut self, collapse: bool) {
		self.set_global_field("collapseChangelog", Value::from(collapse), None);
	}

	pub fn get_enable_install_telemetry(&self) -> bool {
		self.settings.get("enableInstallTelemetry").and_then(Value::as_bool).unwrap_or(true)
	}

	pub fn set_enable_install_telemetry(&mut self, enabled: bool) {
		self.set_global_field("enableInstallTelemetry", Value::from(enabled), None);
	}

	pub fn get_enable_analytics(&self) -> bool {
		self.settings.get("enableAnalytics").and_then(Value::as_bool).unwrap_or(false)
	}

	pub fn get_tracking_id(&self) -> Option<String> {
		self.settings.get("trackingId").and_then(Value::as_str).map(str::to_string)
	}

	/// Set the analytics opt-in preference; generates a tracking identifier
	/// on first opt-in.
	pub fn set_enable_analytics(&mut self, enabled: bool) {
		self.global_settings.set("enableAnalytics", Value::from(enabled));
		self.mark_modified("enableAnalytics", None);
		if enabled && self.global_settings.get("trackingId").map(Value::is_null).unwrap_or(true) {
			self.global_settings.set("trackingId", Value::from(uuid_v4()));
			self.mark_modified("trackingId", None);
		}
		self.save();
	}

	/// Stable ID of this installation, e.g. sent to OpenAI as its agent host
	/// ID. Created on first use. Project settings are ignored so a committed
	/// project settings file cannot give every clone the same ID.
	pub fn get_or_create_device_id(&mut self) -> String {
		let existing = self
			.global_settings
			.get("deviceId")
			.and_then(Value::as_str)
			.map(str::to_string);
		if let Some(existing) = existing {
			return existing;
		}
		let device_id = uuid_v4();
		self.set_global_field("deviceId", Value::from(device_id.as_str()), None);
		device_id
	}

	pub fn get_packages(&self) -> Vec<PackageSource> {
		let Some(value) = self.settings.get("packages") else {
			return Vec::new();
		};
		serde_json::from_value(value.clone()).unwrap_or_default()
	}

	pub fn set_packages(&mut self, packages: &[PackageSource]) {
		let value = serde_json::to_value(packages).unwrap();
		self.set_global_field("packages", value, None);
	}

	pub fn set_project_packages(&mut self, packages: &[PackageSource]) {
		let value = serde_json::to_value(packages).unwrap();
		self.update_project_settings("packages", |settings| settings.set("packages", value));
	}

	fn string_list_field(&self, field: &str) -> Vec<String> {
		value_string_list(self.settings.get(field)).unwrap_or_default()
	}

	pub fn get_extension_paths(&self) -> Vec<String> {
		self.string_list_field("extensions")
	}

	pub fn set_extension_paths(&mut self, paths: &[String]) {
		self.set_global_field("extensions", serde_json::to_value(paths).unwrap(), None);
	}

	pub fn set_project_extension_paths(&mut self, paths: &[String]) {
		let value = serde_json::to_value(paths).unwrap();
		self.update_project_settings("extensions", |settings| settings.set("extensions", value));
	}

	pub fn get_skill_paths(&self) -> Vec<String> {
		self.string_list_field("skills")
	}

	pub fn set_skill_paths(&mut self, paths: &[String]) {
		self.set_global_field("skills", serde_json::to_value(paths).unwrap(), None);
	}

	pub fn set_project_skill_paths(&mut self, paths: &[String]) {
		let value = serde_json::to_value(paths).unwrap();
		self.update_project_settings("skills", |settings| settings.set("skills", value));
	}

	pub fn get_prompt_template_paths(&self) -> Vec<String> {
		self.string_list_field("prompts")
	}

	pub fn set_prompt_template_paths(&mut self, paths: &[String]) {
		self.set_global_field("prompts", serde_json::to_value(paths).unwrap(), None);
	}

	pub fn set_project_prompt_template_paths(&mut self, paths: &[String]) {
		let value = serde_json::to_value(paths).unwrap();
		self.update_project_settings("prompts", |settings| settings.set("prompts", value));
	}

	pub fn get_theme_paths(&self) -> Vec<String> {
		self.string_list_field("themes")
	}

	pub fn set_theme_paths(&mut self, paths: &[String]) {
		self.set_global_field("themes", serde_json::to_value(paths).unwrap(), None);
	}

	pub fn set_project_theme_paths(&mut self, paths: &[String]) {
		let value = serde_json::to_value(paths).unwrap();
		self.update_project_settings("themes", |settings| settings.set("themes", value));
	}

	pub fn get_enable_skill_commands(&self) -> bool {
		self.settings.get("enableSkillCommands").and_then(Value::as_bool).unwrap_or(true)
	}

	pub fn set_enable_skill_commands(&mut self, enabled: bool) {
		self.set_global_field("enableSkillCommands", Value::from(enabled), None);
	}

	/// Upstream `getTerminalCapabilityOverrides`.
	pub fn get_terminal_capability_overrides(&self) -> TerminalCapabilityOverrides {
		let terminal = self.settings.get("terminal");
		let images = terminal.and_then(|value| value.get("images"));
		let parsed_images = images.and_then(|value| value.as_str()).and_then(TerminalImages::parse);
		let parsed_images = match parsed_images {
			Some(mode @ (TerminalImages::Kitty | TerminalImages::Iterm2)) => Some(Some(mode)),
			Some(TerminalImages::Auto) | None => {
				if images == Some(&Value::Bool(false)) {
					Some(None)
				} else {
					None
				}
			}
		};
		let true_color = terminal
			.and_then(|value| value.get("trueColor"))
			.and_then(Value::as_bool);
		let hyperlinks = terminal
			.and_then(|value| value.get("hyperlinks"))
			.and_then(Value::as_bool);
		TerminalCapabilityOverrides {
			images: parsed_images,
			true_color,
			hyperlinks,
		}
	}

	pub fn get_show_images(&self) -> bool {
		self.settings
			.get("terminal")
			.and_then(|value| value.get("showImages"))
			.and_then(Value::as_bool)
			.unwrap_or(true)
	}

	pub fn set_show_images(&mut self, show: bool) {
		let mut terminal = self
			.global_settings
			.get("terminal")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		terminal.insert("showImages".into(), Value::from(show));
		self.set_global_field("terminal", Value::Object(terminal), Some("showImages"));
	}

	pub fn get_image_width_cells(&self) -> u32 {
		match self.settings.get("terminal").and_then(|value| value.get("imageWidthCells")) {
			Some(value) => match value_as_finite_f64(value) {
				Some(width) => (width.floor().max(1.0)) as u32,
				None => 60,
			},
			None => 60,
		}
	}

	pub fn set_image_width_cells(&mut self, width: u32) {
		let mut terminal = self
			.global_settings
			.get("terminal")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		terminal.insert("imageWidthCells".into(), Value::from(width.max(1)));
		self.set_global_field("terminal", Value::Object(terminal), Some("imageWidthCells"));
	}

	/// Settings takes precedence, then env var, then default false.
	pub fn get_clear_on_shrink(&self) -> bool {
		if let Some(value) = self.settings.get("terminal").and_then(|value| value.get("clearOnShrink")) {
			return value.as_bool().unwrap_or(false);
		}
		std::env::var("PI_CLEAR_ON_SHRINK").is_ok_and(|value| value == "1")
	}

	pub fn set_clear_on_shrink(&mut self, enabled: bool) {
		let mut terminal = self
			.global_settings
			.get("terminal")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		terminal.insert("clearOnShrink".into(), Value::from(enabled));
		self.set_global_field("terminal", Value::Object(terminal), Some("clearOnShrink"));
	}

	pub fn get_show_terminal_progress(&self) -> bool {
		self.settings
			.get("terminal")
			.and_then(|value| value.get("showTerminalProgress"))
			.and_then(Value::as_bool)
			.unwrap_or(false)
	}

	pub fn set_show_terminal_progress(&mut self, enabled: bool) {
		let mut terminal = self
			.global_settings
			.get("terminal")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		terminal.insert("showTerminalProgress".into(), Value::from(enabled));
		self.set_global_field("terminal", Value::Object(terminal), Some("showTerminalProgress"));
	}

	pub fn get_tui_mode(&self) -> TuiMode {
		self.settings
			.get("tuiMode")
			.and_then(|value| value.as_str().and_then(TuiMode::parse))
			.filter(|mode| *mode == TuiMode::Regular)
			.unwrap_or(TuiMode::Fullscreen)
	}

	pub fn set_tui_mode(&mut self, mode: TuiMode) {
		self.set_global_field("tuiMode", mode.to_value(), None);
	}

	pub fn get_fullscreen_exit_output(&self) -> FullscreenExitOutput {
		self.settings
			.get("fullscreenExitOutput")
			.and_then(FullscreenExitOutput::from_value)
			.filter(|output| *output == FullscreenExitOutput::ResumeHint)
			.unwrap_or(FullscreenExitOutput::Transcript)
	}

	pub fn set_fullscreen_exit_output(&mut self, output: FullscreenExitOutput) {
		self.set_global_field("fullscreenExitOutput", output.to_value(), None);
	}

	pub fn get_fullscreen_scrollbar(&self) -> ScrollViewScrollbar {
		self.settings
			.get("fullscreenScrollbar")
			.and_then(ScrollViewScrollbar::from_value)
			.unwrap_or(ScrollViewScrollbar::Auto)
	}

	pub fn set_fullscreen_scrollbar(&mut self, mode: ScrollViewScrollbar) {
		self.set_global_field("fullscreenScrollbar", mode.to_value(), None);
	}

	pub fn get_fullscreen_copy_on_select(&self) -> bool {
		self.settings
			.get("fullscreenCopyOnSelect")
			.and_then(Value::as_bool)
			.unwrap_or(true)
	}

	pub fn set_fullscreen_copy_on_select(&mut self, enabled: bool) {
		self.set_global_field("fullscreenCopyOnSelect", Value::from(enabled), None);
	}

	pub fn get_fullscreen_wheel_scroll_lines(&self) -> WheelScrollLines {
		match self.settings.get("fullscreenWheelScrollLines").and_then(value_as_finite_f64) {
			Some(lines) => WheelScrollLines::Lines(lines.floor().clamp(1.0, 100.0) as u32),
			None => WheelScrollLines::Auto,
		}
	}

	pub fn set_fullscreen_wheel_scroll_lines(&mut self, lines: WheelScrollLines) {
		let value = match lines {
			WheelScrollLines::Auto => Value::from("auto"),
			WheelScrollLines::Lines(count) => Value::from(count.clamp(1, 100)),
		};
		self.set_global_field("fullscreenWheelScrollLines", value, None);
	}

	pub fn get_image_auto_resize(&self) -> bool {
		self.settings
			.get("images")
			.and_then(|value| value.get("autoResize"))
			.and_then(Value::as_bool)
			.unwrap_or(true)
	}

	pub fn set_image_auto_resize(&mut self, enabled: bool) {
		let mut images = self
			.global_settings
			.get("images")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		images.insert("autoResize".into(), Value::from(enabled));
		self.set_global_field("images", Value::Object(images), Some("autoResize"));
	}

	pub fn get_block_images(&self) -> bool {
		self.settings
			.get("images")
			.and_then(|value| value.get("blockImages"))
			.and_then(Value::as_bool)
			.unwrap_or(false)
	}

	pub fn set_block_images(&mut self, blocked: bool) {
		let mut images = self
			.global_settings
			.get("images")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		images.insert("blockImages".into(), Value::from(blocked));
		self.set_global_field("images", Value::Object(images), Some("blockImages"));
	}

	pub fn get_enabled_models(&self) -> Option<Vec<String>> {
		value_string_list(self.settings.get("enabledModels"))
	}

	/// The resolved `defaultTools` selection, or `None` when no settings
	/// layer sets it.
	pub fn get_default_tools(&self) -> Option<Vec<String>> {
		let tools = self.settings.get("defaultTools")?;
		let entries = value_string_list(Some(tools)).unwrap_or_default();
		Some(resolve_default_tools(&entries))
	}

	pub fn set_enabled_models(&mut self, patterns: Option<&[String]>) {
		self.set_global_optional_field("enabledModels", patterns.map(|entries| serde_json::to_value(entries).unwrap()));
	}

	pub fn get_double_escape_action(&self) -> DoubleEscapeAction {
		self.settings
			.get("doubleEscapeAction")
			.and_then(DoubleEscapeAction::from_value)
			.unwrap_or(DoubleEscapeAction::Tree)
	}

	pub fn set_double_escape_action(&mut self, action: DoubleEscapeAction) {
		self.set_global_field("doubleEscapeAction", action.to_value(), None);
	}

	pub fn get_tree_filter_mode(&self) -> TreeFilterMode {
		self.settings
			.get("treeFilterMode")
			.and_then(TreeFilterMode::from_value)
			.unwrap_or(TreeFilterMode::Default)
	}

	pub fn set_tree_filter_mode(&mut self, mode: TreeFilterMode) {
		self.set_global_field("treeFilterMode", mode.to_value(), None);
	}

	pub fn get_show_hardware_cursor(&self) -> bool {
		self.settings
			.get("showHardwareCursor")
			.and_then(Value::as_bool)
			.unwrap_or_else(|| std::env::var("PI_HARDWARE_CURSOR").is_ok_and(|value| value == "1"))
	}

	pub fn set_show_hardware_cursor(&mut self, enabled: bool) {
		self.set_global_field("showHardwareCursor", Value::from(enabled), None);
	}

	pub fn get_editor_padding_x(&self) -> u32 {
		self.settings
			.get("editorPaddingX")
			.and_then(Value::as_u64)
			.map(|value| value as u32)
			.unwrap_or(0)
	}

	pub fn set_editor_padding_x(&mut self, padding: u32) {
		let clamped = padding.min(3);
		self.set_global_field("editorPaddingX", Value::from(clamped), None);
	}

	pub fn get_output_pad(&self) -> u8 {
		if self.settings.get("outputPad").and_then(Value::as_u64) == Some(0) {
			0
		} else {
			1
		}
	}

	pub fn set_output_pad(&mut self, padding: u8) {
		self.set_global_field("outputPad", Value::from(padding), None);
	}

	pub fn get_autocomplete_max_visible(&self) -> u32 {
		self.settings
			.get("autocompleteMaxVisible")
			.and_then(Value::as_u64)
			.map(|value| value as u32)
			.unwrap_or(5)
	}

	pub fn set_autocomplete_max_visible(&mut self, max_visible: u32) {
		let clamped = max_visible.clamp(3, 20);
		self.set_global_field("autocompleteMaxVisible", Value::from(clamped), None);
	}

	pub fn get_code_block_indent(&self) -> String {
		self.settings
			.get("markdown")
			.and_then(|value| value.get("codeBlockIndent"))
			.and_then(Value::as_str)
			.unwrap_or("  ")
			.to_string()
	}

	pub fn get_mermaid_rendering_mode(&self) -> MermaidRenderingMode {
		self.settings
			.get("markdown")
			.and_then(|value| value.get("mermaid"))
			.and_then(MermaidRenderingMode::from_value)
			.unwrap_or(MermaidRenderingMode::Streaming)
	}

	pub fn set_mermaid_rendering_mode(&mut self, mode: MermaidRenderingMode) {
		let mut markdown = self
			.global_settings
			.get("markdown")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		markdown.insert("mermaid".into(), mode.to_value());
		self.set_global_field("markdown", Value::Object(markdown), Some("mermaid"));
	}

	/// Upstream `WarningSettings` (currently only `anthropicExtraUsage`).
	pub fn get_warnings(&self) -> Option<bool> {
		self.settings
			.get("warnings")
			.and_then(|value| value.get("anthropicExtraUsage"))
			.and_then(Value::as_bool)
	}

	pub fn set_warnings(&mut self, anthropic_extra_usage: bool) {
		let warnings = serde_json::json!({ "anthropicExtraUsage": anthropic_extra_usage });
		self.set_global_field("warnings", warnings, None);
	}

	/// Upstream `codemode.mode`.
	pub fn get_codemode_mode(&self) -> CodemodeMode {
		self.settings
			.get("codemode")
			.and_then(|value| value.get("mode"))
			.and_then(CodemodeMode::from_value)
			.unwrap_or(CodemodeMode::On)
	}

	pub fn set_codemode_mode(&mut self, mode: CodemodeMode) {
		let mut codemode = self
			.global_settings
			.get("codemode")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		codemode.insert("mode".into(), mode.to_value());
		self.set_global_field("codemode", Value::Object(codemode), Some("mode"));
	}

	/// Upstream `codemode.inlineBudget` (default 3000).
	pub fn get_codemode_inline_budget(&self) -> u64 {
		self.settings
			.get("codemode")
			.and_then(|value| value.get("inlineBudget"))
			.and_then(Value::as_u64)
			.unwrap_or(3000)
	}
}

struct LoadedSettings {
	settings: Settings,
	error: Option<String>,
}

fn parse_thinking_level(value: &str) -> Option<AgentThinkingLevel> {
	Some(match value {
		"off" => AgentThinkingLevel::Off,
		"minimal" => AgentThinkingLevel::Minimal,
		"low" => AgentThinkingLevel::Low,
		"medium" => AgentThinkingLevel::Medium,
		"high" => AgentThinkingLevel::High,
		"xhigh" => AgentThinkingLevel::Xhigh,
		"max" => AgentThinkingLevel::Max,
		_ => return None,
	})
}

fn thinking_level_str(level: AgentThinkingLevel) -> &'static str {
	match level {
		AgentThinkingLevel::Off => "off",
		AgentThinkingLevel::Minimal => "minimal",
		AgentThinkingLevel::Low => "low",
		AgentThinkingLevel::Medium => "medium",
		AgentThinkingLevel::High => "high",
		AgentThinkingLevel::Xhigh => "xhigh",
		AgentThinkingLevel::Max => "max",
	}
}

/// Upstream `"header"` quiet-startup discriminator.
struct QuietStartupHeader;

impl QuietStartupHeader {
	fn from_value(value: &Value) -> Option<QuietStartup> {
		value.as_str().filter(|text| *text == "header").map(|_| QuietStartup::Header)
	}
}

// =============================================================================
// Tests (ports of upstream test/settings-manager.test.ts,
// test/settings-manager-bug.test.ts and the portable cases of
// test/settings-manager-compaction.test.ts)
// =============================================================================

#[cfg(test)]
mod settings_manager_tests {
	use super::*;
	use std::path::PathBuf;
	use std::sync::atomic::{AtomicU32, Ordering};

	static DIR_COUNTER: AtomicU32 = AtomicU32::new(0);
	static ENV_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

	fn fresh_dir(tag: &str) -> PathBuf {
		let dir = std::env::temp_dir().join(format!(
			"pi-settings-{}-{}-{}",
			tag,
			std::process::id(),
			DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
		));
		let _ = std::fs::remove_dir_all(&dir);
		std::fs::create_dir_all(dir.join("agent")).unwrap();
		std::fs::create_dir_all(dir.join("project").join(".pi")).unwrap();
		dir
	}

	fn cleanup(dir: &PathBuf) {
		let _ = std::fs::remove_dir_all(dir);
	}

	fn agent_dir_of(dir: &PathBuf) -> String {
		dir.join("agent").to_string_lossy().into_owned()
	}

	fn project_dir_of(dir: &PathBuf) -> String {
		dir.join("project").to_string_lossy().into_owned()
	}

	fn settings_path(dir: &PathBuf) -> PathBuf {
		dir.join("agent").join("settings.json")
	}

	fn project_settings_path(dir: &PathBuf) -> PathBuf {
		dir.join("project").join(".pi").join("settings.json")
	}

	fn write_file(path: &PathBuf, content: &str) {
		if let Some(parent) = path.parent() {
			std::fs::create_dir_all(parent).unwrap();
		}
		std::fs::write(path, content).unwrap();
	}

	fn read_json(path: &PathBuf) -> Value {
		serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
	}

	fn obj(pairs: &[(&str, Value)]) -> Value {
		Value::Object(pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect())
	}

	fn arr(items: &[&str]) -> Value {
		Value::Array(items.iter().map(|item| Value::from(*item)).collect())
	}

	fn in_memory_settings(settings: Value) -> SettingsManager {
		SettingsManager::in_memory(&Settings::from_value(settings), SettingsManagerCreateOptions::default())
	}

	fn model() -> (&'static str, &'static str) {
		("provider", "family/model")
	}

	// ----- preserves externally added settings -----

	#[test]
	fn preserves_enabled_models_when_changing_thinking_level() {
		let dir = fresh_dir("enabled-models");
		let path = settings_path(&dir);
		write_file(&path, &serde_json::to_string(&obj(&[("theme", "dark".into()), ("defaultModel", "claude-sonnet".into())])).unwrap());

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		// Simulate user editing settings.json externally to add enabledModels
		let mut current = read_json(&path);
		current["enabledModels"] = arr(&["claude-opus-4-5", "gpt-5.2-codex"]);
		write_file(&path, &serde_json::to_string_pretty(&current).unwrap());

		manager.set_default_thinking_level(AgentThinkingLevel::High);
		manager.flush();

		let saved = read_json(&path);
		assert_eq!(saved["enabledModels"], arr(&["claude-opus-4-5", "gpt-5.2-codex"]));
		assert_eq!(saved["defaultThinkingLevel"], "high");
		assert_eq!(saved["theme"], "dark");
		assert_eq!(saved["defaultModel"], "claude-sonnet");
		cleanup(&dir);
	}

	#[test]
	fn preserves_custom_settings_when_changing_theme() {
		let dir = fresh_dir("custom-settings");
		let path = settings_path(&dir);
		write_file(&path, &serde_json::to_string(&obj(&[("defaultModel", "claude-sonnet".into())])).unwrap());

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		let mut current = read_json(&path);
		current["shellPath"] = "/bin/zsh".into();
		current["extensions"] = arr(&["/path/to/extension.ts"]);
		write_file(&path, &serde_json::to_string_pretty(&current).unwrap());

		manager.set_theme("light");
		manager.flush();

		let saved = read_json(&path);
		assert_eq!(saved["shellPath"], "/bin/zsh");
		assert_eq!(saved["extensions"], arr(&["/path/to/extension.ts"]));
		assert_eq!(saved["theme"], "light");
		cleanup(&dir);
	}

	#[test]
	fn in_memory_changes_override_file_changes_for_same_key() {
		let dir = fresh_dir("in-memory-wins");
		let path = settings_path(&dir);
		write_file(&path, &serde_json::to_string(&obj(&[("theme", "dark".into())])).unwrap());

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		// Externally set thinking level to "low", then change via UI to "high"
		let mut current = read_json(&path);
		current["defaultThinkingLevel"] = "low".into();
		write_file(&path, &serde_json::to_string_pretty(&current).unwrap());

		manager.set_default_thinking_level(AgentThinkingLevel::High);
		manager.flush();

		assert_eq!(read_json(&path)["defaultThinkingLevel"], "high");
		cleanup(&dir);
	}

	// ----- deviceId -----

	#[test]
	fn creates_one_global_device_id_and_reuses_it() {
		let dir = fresh_dir("device-id");
		let path = settings_path(&dir);
		write_file(&path, &serde_json::to_string(&obj(&[("theme", "dark".into())])).unwrap());
		write_file(&project_settings_path(&dir), &serde_json::to_string(&obj(&[("deviceId", "project-device".into())])).unwrap());
		let mut first = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		let device_id = first.get_or_create_device_id();
		first.flush();

		assert_eq!(device_id.len(), 36);
		assert!(device_id.chars().all(|c| c == '-' || c.is_ascii_hexdigit()));
		assert_eq!(first.get_or_create_device_id(), device_id);
		let mut second = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(second.get_or_create_device_id(), device_id);
		assert_eq!(
			read_json(&path),
			obj(&[("theme", "dark".into()), ("deviceId", device_id.into())])
		);
		cleanup(&dir);
	}

	// ----- packages migration -----

	#[test]
	fn keeps_local_only_extensions_in_extensions_array() {
		let dir = fresh_dir("local-extensions");
		write_file(
			&settings_path(&dir),
			&serde_json::to_string(&obj(&[("extensions", arr(&["/local/ext.ts", "./relative/ext.ts"]))])).unwrap(),
		);

		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		assert_eq!(manager.get_packages(), Vec::<PackageSource>::new());
		assert_eq!(manager.get_extension_paths(), ["/local/ext.ts", "./relative/ext.ts"]);
		cleanup(&dir);
	}

	#[test]
	fn handles_packages_with_filtering_objects() {
		let dir = fresh_dir("package-filtering");
		write_file(
			&settings_path(&dir),
			&serde_json::to_string(&obj(&[(
				"packages",
				Value::Array(vec![
					Value::from("npm:simple-pkg"),
					serde_json::json!({
						"source": "npm:shitty-extensions",
						"extensions": ["extensions/oracle.ts"],
						"skills": [],
					}),
				]),
			)]))
			.unwrap(),
		);

		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		let packages = manager.get_packages();
		assert_eq!(packages.len(), 2);
		assert_eq!(packages[0], PackageSource::Simple("npm:simple-pkg".to_string()));
		assert_eq!(
			packages[1],
			PackageSource::Filtered {
				source: "npm:shitty-extensions".to_string(),
				autoload: None,
				extensions: Some(vec!["extensions/oracle.ts".to_string()]),
				skills: Some(vec![]),
				prompts: None,
				themes: None,
			}
		);
		cleanup(&dir);
	}

	// ----- reload -----

	#[test]
	fn reloads_global_settings_from_disk() {
		let dir = fresh_dir("reload");
		let path = settings_path(&dir);
		write_file(
			&path,
			&serde_json::to_string(&obj(&[("theme", "dark".into()), ("extensions", arr(&["/before.ts"]))])).unwrap(),
		);

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		write_file(
			&path,
			&serde_json::to_string(&obj(&[
				("theme", "light".into()),
				("extensions", arr(&["/after.ts"])),
				("defaultModel", "claude-sonnet".into()),
			]))
			.unwrap(),
		);

		manager.reload();

		assert_eq!(manager.get_theme().as_deref(), Some("light"));
		assert_eq!(manager.get_extension_paths(), ["/after.ts"]);
		assert_eq!(manager.get_default_model().as_deref(), Some("claude-sonnet"));
		cleanup(&dir);
	}

	#[test]
	fn keeps_previous_settings_and_reports_path_when_file_invalid() {
		let dir = fresh_dir("invalid-file");
		let path = settings_path(&dir);
		write_file(&path, &serde_json::to_string(&obj(&[("theme", "dark".into())])).unwrap());

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		write_file(&path, "{ invalid json");
		manager.reload();

		assert_eq!(manager.get_theme().as_deref(), Some("dark"));
		let errors = manager.drain_errors();
		assert_eq!(errors.len(), 1);
		assert_eq!(errors[0].scope, SettingsScope::Global);
		assert_eq!(errors[0].path.as_deref(), Some(path.to_string_lossy().as_ref()));
		cleanup(&dir);
	}

	// ----- theme setting -----

	#[test]
	fn stores_slash_separated_theme_settings_separately() {
		let dir = fresh_dir("theme-slash");
		let path = settings_path(&dir);
		write_file(&path, &serde_json::to_string(&obj(&[("theme", "light/dark".into())])).unwrap());

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		assert_eq!(manager.get_theme(), None);
		assert_eq!(manager.get_theme_setting().as_deref(), Some("light/dark"));

		manager.set_theme("solarized-light/tokyo-night");
		manager.flush();

		assert_eq!(read_json(&path)["theme"], "solarized-light/tokyo-night");
		cleanup(&dir);
	}

	// ----- error tracking -----

	#[test]
	fn collects_and_clears_load_errors_via_drain_errors() {
		let dir = fresh_dir("drain-errors");
		let global_path = settings_path(&dir);
		let project_path = project_settings_path(&dir);
		write_file(&global_path, "{ invalid global json");
		write_file(&project_path, "{ invalid project json");

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		let errors = manager.drain_errors();

		assert_eq!(errors.len(), 2);
		assert_eq!(errors[0].scope, SettingsScope::Global);
		assert_eq!(errors[0].path.as_deref(), Some(global_path.to_string_lossy().as_ref()));
		assert_eq!(errors[1].scope, SettingsScope::Project);
		assert_eq!(errors[1].path.as_deref(), Some(project_path.to_string_lossy().as_ref()));
		assert!(manager.drain_errors().is_empty());
		cleanup(&dir);
	}

	// ----- project trust -----

	#[test]
	fn skips_project_settings_when_project_is_not_trusted() {
		let dir = fresh_dir("untrusted");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("theme", "global".into())])).unwrap());
		write_file(&project_settings_path(&dir), &serde_json::to_string(&obj(&[("theme", "project".into())])).unwrap());

		let manager = SettingsManager::create(
			&project_dir_of(&dir),
			&agent_dir_of(&dir),
			SettingsManagerCreateOptions {
				project_trusted: Some(false),
			},
		);

		assert!(!manager.is_project_trusted());
		assert_eq!(manager.get_theme().as_deref(), Some("global"));
		assert_eq!(manager.get_project_settings(), Settings::new());
		cleanup(&dir);
	}

	#[test]
	fn reloads_project_settings_after_trust_changes_to_true() {
		let dir = fresh_dir("trust-reload");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("theme", "global".into())])).unwrap());
		write_file(&project_settings_path(&dir), &serde_json::to_string(&obj(&[("theme", "project".into())])).unwrap());
		let mut manager = SettingsManager::create(
			&project_dir_of(&dir),
			&agent_dir_of(&dir),
			SettingsManagerCreateOptions {
				project_trusted: Some(false),
			},
		);

		manager.set_project_trusted(true);

		assert!(manager.is_project_trusted());
		assert_eq!(manager.get_theme().as_deref(), Some("project"));
		cleanup(&dir);
	}

	#[test]
	fn fails_project_writes_when_project_is_not_trusted() {
		let dir = fresh_dir("untrusted-write");
		let project_path = project_settings_path(&dir);
		write_file(&project_path, &serde_json::to_string(&obj(&[("packages", arr(&["npm:existing"]))])).unwrap());
		let mut manager = SettingsManager::create(
			&project_dir_of(&dir),
			&agent_dir_of(&dir),
			SettingsManagerCreateOptions {
				project_trusted: Some(false),
			},
		);

		// Upstream throws synchronously; the port records the same error.
		manager.set_project_packages(&[PackageSource::Simple("npm:new".to_string())]);
		manager.flush();

		assert_eq!(manager.get_project_settings(), Settings::new());
		assert_eq!(read_json(&project_path), obj(&[("packages", arr(&["npm:existing"]))]));
		let errors = manager.drain_errors();
		assert!(errors.iter().any(|error| error.error == "Project is not trusted; refusing to write project settings"));
		cleanup(&dir);
	}

	#[test]
	fn reads_default_project_trust_from_global_settings_only() {
		let dir = fresh_dir("trust-global");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("defaultProjectTrust", "always".into())])).unwrap());
		write_file(&project_settings_path(&dir), &serde_json::to_string(&obj(&[("defaultProjectTrust", "never".into())])).unwrap());

		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		assert_eq!(manager.get_default_project_trust(), DefaultProjectTrust::Always);
		cleanup(&dir);
	}

	#[test]
	fn defaults_invalid_project_trust_settings_to_ask() {
		let dir = fresh_dir("trust-invalid");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("defaultProjectTrust", "sometimes".into())])).unwrap());

		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		assert_eq!(manager.get_default_project_trust(), DefaultProjectTrust::Ask);
		cleanup(&dir);
	}

	// ----- project settings directory creation -----

	#[test]
	fn does_not_create_pi_folder_when_only_reading() {
		let dir = fresh_dir("read-no-dir");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("theme", "dark".into())])).unwrap());
		std::fs::remove_dir_all(dir.join("project").join(".pi")).unwrap();

		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		assert!(!dir.join("project").join(".pi").exists());
		assert_eq!(manager.get_theme().as_deref(), Some("dark"));
		cleanup(&dir);
	}

	#[test]
	fn creates_pi_folder_when_writing_project_settings() {
		let dir = fresh_dir("write-dir");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("theme", "dark".into())])).unwrap());
		std::fs::remove_dir_all(dir.join("project").join(".pi")).unwrap();

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert!(!dir.join("project").join(".pi").exists());

		manager.set_project_packages(&[PackageSource::Filtered {
			source: "npm:test-pkg".to_string(),
			autoload: None,
			extensions: None,
			skills: None,
			prompts: None,
			themes: None,
		}]);
		manager.flush();

		assert!(dir.join("project").join(".pi").exists());
		assert!(project_settings_path(&dir).exists());
		cleanup(&dir);
	}

	// ----- terminal capability overrides -----

	#[test]
	fn maps_explicit_terminal_values_and_omits_auto_values() {
		let overrides = |terminal: Value| {
			in_memory_settings(obj(&[("terminal", terminal)]))
				.get_terminal_capability_overrides()
		};

		assert_eq!(
			overrides(serde_json::json!({ "images": false, "trueColor": false, "hyperlinks": false })),
			TerminalCapabilityOverrides {
				images: Some(None),
				true_color: Some(false),
				hyperlinks: Some(false),
			}
		);
		assert_eq!(
			overrides(serde_json::json!({ "images": "kitty", "trueColor": true, "hyperlinks": true })),
			TerminalCapabilityOverrides {
				images: Some(Some(TerminalImages::Kitty)),
				true_color: Some(true),
				hyperlinks: Some(true),
			}
		);
		assert_eq!(
			overrides(serde_json::json!({ "images": "auto", "trueColor": "auto", "hyperlinks": "auto" })),
			TerminalCapabilityOverrides::default()
		);
	}

	// ----- retry settings -----

	#[test]
	fn defaults_and_overrides_agent_retry_delay_cap() {
		let manager = in_memory_settings(Value::Object(Default::default()));
		assert_eq!(manager.get_retry_settings(), (true, 3, 2000, 60000));

		let manager = in_memory_settings(serde_json::json!({
			"retry": { "enabled": true, "maxRetries": 10, "baseDelayMs": 500, "maxAgentDelayMs": 5000 }
		}));
		assert_eq!(manager.get_retry_settings(), (true, 10, 500, 5000));
	}

	// ----- httpIdleTimeoutMs -----

	#[test]
	fn http_idle_timeout_defaults_to_five_minutes() {
		let dir = fresh_dir("http-default");
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_http_idle_timeout_ms().unwrap(), DEFAULT_HTTP_IDLE_TIMEOUT_MS);
		cleanup(&dir);
	}

	#[test]
	fn http_idle_timeout_uses_merged_settings() {
		let dir = fresh_dir("http-merged");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("httpIdleTimeoutMs", 300_000.into())])).unwrap());
		write_file(&project_settings_path(&dir), &serde_json::to_string(&obj(&[("httpIdleTimeoutMs", 0.into())])).unwrap());

		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		assert_eq!(manager.get_http_idle_timeout_ms().unwrap(), 0);
		cleanup(&dir);
	}

	#[test]
	fn http_idle_timeout_rejects_invalid_values() {
		let dir = fresh_dir("http-invalid");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("httpIdleTimeoutMs", (-1).into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		let error = manager.get_http_idle_timeout_ms().unwrap_err();
		assert!(error.starts_with("Invalid httpIdleTimeoutMs setting"), "{error}");
		cleanup(&dir);
	}

	// ----- cacheWarming -----

	#[test]
	fn cache_warming_defaults_to_streaming_and_ignores_project() {
		let dir = fresh_dir("cache-warming");
		let create = || SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		assert_eq!(create().get_cache_warming_mode(), CacheWarmingMode::Streaming);

		write_file(&project_settings_path(&dir), &serde_json::to_string(&obj(&[("cacheWarming", "idle".into())])).unwrap());
		assert_eq!(create().get_cache_warming_mode(), CacheWarmingMode::Streaming);

		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("cacheWarming", "idle".into())])).unwrap());
		assert_eq!(create().get_cache_warming_mode(), CacheWarmingMode::Idle);

		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("cacheWarming", "bogus".into())])).unwrap());
		assert_eq!(create().get_cache_warming_mode(), CacheWarmingMode::Streaming);
		cleanup(&dir);
	}

	#[test]
	fn cache_warming_persists_the_mode_globally() {
		let dir = fresh_dir("cache-warming-persist");
		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		manager.set_cache_warming_mode(CacheWarmingMode::Off);
		manager.flush();

		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_cache_warming_mode(), CacheWarmingMode::Off);
		assert_eq!(read_json(&settings_path(&dir)), obj(&[("cacheWarming", "off".into())]));
		cleanup(&dir);
	}

	// ----- externalEditor -----

	#[test]
	fn resolves_editor_commands_by_precedence() {
		let _env = ENV_LOCK.lock();
		std::env::set_var("VISUAL", "vim");
		std::env::set_var("EDITOR", "nano");
		assert_eq!(
			in_memory_settings(obj(&[("externalEditor", "code --wait".into())])).get_external_editor_command(),
			"code --wait"
		);
		assert_eq!(in_memory_settings(Value::Object(Default::default())).get_external_editor_command(), "vim");

		std::env::remove_var("VISUAL");
		std::env::set_var("EDITOR", "emacs");
		assert_eq!(in_memory_settings(Value::Object(Default::default())).get_external_editor_command(), "emacs");

		std::env::remove_var("VISUAL");
		std::env::remove_var("EDITOR");
	}

	// ----- TUI mode -----

	#[test]
	fn tui_mode_defaults_to_fullscreen_and_persists_regular() {
		let dir = fresh_dir("tui-mode");
		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());

		assert_eq!(manager.get_tui_mode(), TuiMode::Fullscreen);

		manager.set_tui_mode(TuiMode::Regular);
		manager.flush();

		assert_eq!(manager.get_tui_mode(), TuiMode::Regular);
		assert_eq!(read_json(&settings_path(&dir))["tuiMode"], "regular");
		cleanup(&dir);
	}

	#[test]
	fn tui_mode_falls_back_to_fullscreen_for_unsupported_values() {
		let dir = fresh_dir("tui-invalid");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("tuiMode", "other".into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_tui_mode(), TuiMode::Fullscreen);
		cleanup(&dir);
	}

	#[test]
	fn tui_mode_does_not_recognize_old_ui_mode() {
		let dir = fresh_dir("tui-uimode");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("uiMode", "regular".into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_tui_mode(), TuiMode::Fullscreen);
		cleanup(&dir);
	}

	// ----- fullscreen settings -----

	#[test]
	fn validates_and_persists_fullscreen_settings() {
		let dir = fresh_dir("fullscreen");
		let path = settings_path(&dir);
		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_fullscreen_exit_output(), FullscreenExitOutput::Transcript);
		assert_eq!(manager.get_fullscreen_scrollbar(), ScrollViewScrollbar::Auto);
		assert!(manager.get_fullscreen_copy_on_select());

		manager.set_fullscreen_exit_output(FullscreenExitOutput::ResumeHint);
		manager.set_fullscreen_scrollbar(ScrollViewScrollbar::Hidden);
		manager.set_fullscreen_copy_on_select(false);
		manager.flush();
		let saved = read_json(&path);
		assert_eq!(saved["fullscreenExitOutput"], "resume-hint");
		assert_eq!(saved["fullscreenScrollbar"], "hidden");
		assert_eq!(saved["fullscreenCopyOnSelect"], false);

		write_file(
			&path,
			&serde_json::to_string(&obj(&[("fullscreenExitOutput", "nothing".into()), ("fullscreenScrollbar", "sometimes".into())])).unwrap(),
		);
		let reloaded = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(reloaded.get_fullscreen_exit_output(), FullscreenExitOutput::Transcript);
		assert_eq!(reloaded.get_fullscreen_scrollbar(), ScrollViewScrollbar::Auto);
		assert!(reloaded.get_fullscreen_copy_on_select());
		cleanup(&dir);
	}

	#[test]
	fn persists_fullscreen_wheel_scroll_lines() {
		let dir = fresh_dir("wheel");
		let path = settings_path(&dir);
		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_fullscreen_wheel_scroll_lines(), WheelScrollLines::Auto);

		manager.set_fullscreen_wheel_scroll_lines(WheelScrollLines::Lines(3));
		manager.flush();
		assert_eq!(read_json(&path)["fullscreenWheelScrollLines"], 3);

		for (value, expected) in [
			(Value::from(7.9), WheelScrollLines::Lines(7)),
			(Value::from(0), WheelScrollLines::Lines(1)),
			(Value::from(1000), WheelScrollLines::Lines(100)),
			(Value::from("fast"), WheelScrollLines::Auto),
			(Value::Null, WheelScrollLines::Auto),
		] {
			write_file(&path, &serde_json::to_string(&obj(&[("fullscreenWheelScrollLines", value)])).unwrap());
			let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
			assert_eq!(manager.get_fullscreen_wheel_scroll_lines(), expected);
		}
		cleanup(&dir);
	}

	// ----- outputPad -----

	#[test]
	fn output_pad_defaults_to_one_and_persists_binary_values() {
		let dir = fresh_dir("output-pad");
		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_output_pad(), 1);

		manager.set_output_pad(0);
		manager.flush();

		assert_eq!(manager.get_output_pad(), 0);
		assert_eq!(read_json(&settings_path(&dir))["outputPad"], 0);
		cleanup(&dir);
	}

	#[test]
	fn output_pad_treats_unsupported_values_as_default() {
		let dir = fresh_dir("output-pad-invalid");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("outputPad", 2.into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_output_pad(), 1);
		cleanup(&dir);
	}

	// ----- markdown.mermaid -----

	#[test]
	fn mermaid_defaults_to_streaming_and_persists_modes() {
		let dir = fresh_dir("mermaid");
		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_mermaid_rendering_mode(), MermaidRenderingMode::Streaming);

		manager.set_mermaid_rendering_mode(MermaidRenderingMode::Final);
		manager.flush();

		assert_eq!(manager.get_mermaid_rendering_mode(), MermaidRenderingMode::Final);
		assert_eq!(read_json(&settings_path(&dir))["markdown"]["mermaid"], "final");
		cleanup(&dir);
	}

	#[test]
	fn mermaid_falls_back_to_streaming_for_unsupported_values() {
		let dir = fresh_dir("mermaid-invalid");
		write_file(
			&settings_path(&dir),
			&serde_json::to_string(&obj(&[("markdown", obj(&[("mermaid", "sometimes".into())]))])).unwrap(),
		);
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_mermaid_rendering_mode(), MermaidRenderingMode::Streaming);
		cleanup(&dir);
	}

	// ----- shellCommandPrefix -----

	#[test]
	fn loads_shell_command_prefix_from_settings() {
		let dir = fresh_dir("prefix");
		write_file(
			&settings_path(&dir),
			&serde_json::to_string(&obj(&[("shellCommandPrefix", "shopt -s expand_aliases".into())])).unwrap(),
		);
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_shell_command_prefix().as_deref(), Some("shopt -s expand_aliases"));
		cleanup(&dir);
	}

	#[test]
	fn shell_command_prefix_returns_none_when_unset() {
		let dir = fresh_dir("prefix-none");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("theme", "dark".into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_shell_command_prefix(), None);
		cleanup(&dir);
	}

	#[test]
	fn preserves_shell_command_prefix_when_saving_unrelated_settings() {
		let dir = fresh_dir("prefix-preserve");
		let path = settings_path(&dir);
		write_file(
			&path,
			&serde_json::to_string(&obj(&[("shellCommandPrefix", "shopt -s expand_aliases".into())])).unwrap(),
		);
		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		manager.set_theme("light");
		manager.flush();

		let saved = read_json(&path);
		assert_eq!(saved["shellCommandPrefix"], "shopt -s expand_aliases");
		assert_eq!(saved["theme"], "light");
		cleanup(&dir);
	}

	// ----- defaultTools -----

	#[test]
	fn default_tools_loads_global_and_project_replaces() {
		let dir = fresh_dir("tools-global");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("defaultTools", arr(&["read", "bash"]))])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_default_tools(), Some(vec!["read".to_string(), "bash".to_string()]));

		write_file(&project_settings_path(&dir), &serde_json::to_string(&obj(&[("defaultTools", arr(&["grep"]))])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_default_tools(), Some(vec!["grep".to_string()]));
		cleanup(&dir);
	}

	#[test]
	fn default_tools_preserves_an_empty_list() {
		let manager = in_memory_settings(obj(&[("defaultTools", Value::Array(vec![]))]));
		assert_eq!(manager.get_default_tools(), Some(vec![]));
		assert_eq!(in_memory_settings(Value::Object(Default::default())).get_default_tools(), None);
	}

	#[test]
	fn default_tools_applies_name_modifiers_to_defaults() {
		let manager = in_memory_settings(obj(&[("defaultTools", arr(&["+codemode", "-write"]))]));
		assert_eq!(
			manager.get_default_tools(),
			Some(vec!["read".to_string(), "bash".to_string(), "edit".to_string(), "codemode".to_string()])
		);

		let manager = in_memory_settings(obj(&[("defaultTools", arr(&["read", "+grep", "+read"]))]));
		assert_eq!(manager.get_default_tools(), Some(vec!["read".to_string(), "grep".to_string()]));
	}

	#[test]
	fn default_tools_layers_project_modifiers_on_global() {
		let dir = fresh_dir("tools-layers");
		write_file(
			&settings_path(&dir),
			&serde_json::to_string(&obj(&[("defaultTools", arr(&["read", "bash", "+codemode"]))])).unwrap(),
		);
		write_file(
			&project_settings_path(&dir),
			&serde_json::to_string(&obj(&[("defaultTools", arr(&["-codemode", "+tool_search"]))])).unwrap(),
		);

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(
			manager.get_default_tools(),
			Some(vec!["read".to_string(), "bash".to_string(), "tool_search".to_string()])
		);

		manager.apply_overrides(&Settings::from_value(obj(&[("defaultTools", arr(&["+codemode"]))])));
		assert_eq!(
			manager.get_default_tools(),
			Some(vec![
				"read".to_string(),
				"bash".to_string(),
				"tool_search".to_string(),
				"codemode".to_string()
			])
		);
		cleanup(&dir);
	}

	#[test]
	fn default_tools_applies_project_modifiers_to_builtins_without_global() {
		let dir = fresh_dir("tools-project-only");
		write_file(&project_settings_path(&dir), &serde_json::to_string(&obj(&[("defaultTools", arr(&["+codemode"]))])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(
			manager.get_default_tools(),
			Some(vec![
				"read".to_string(),
				"bash".to_string(),
				"edit".to_string(),
				"write".to_string(),
				"codemode".to_string()
			])
		);
		cleanup(&dir);
	}

	// ----- getSessionDir -----

	#[test]
	fn session_dir_cases() {
		let dir = fresh_dir("session-dir");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("theme", "dark".into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_session_dir(), None);

		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("sessionDir", "/tmp/sessions".into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_session_dir().as_deref(), Some("/tmp/sessions"));

		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("sessionDir", "/global/sessions".into())])).unwrap());
		write_file(&project_settings_path(&dir), &serde_json::to_string(&obj(&[("sessionDir", "./sessions".into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_session_dir().as_deref(), Some("./sessions"));

		std::fs::remove_file(&project_settings_path(&dir)).unwrap();
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("sessionDir", "~/sessions".into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(
			manager.get_session_dir().as_deref(),
			Some(crate::utils::paths::join(&crate::utils::paths::home_dir(), "sessions").as_str())
		);
		cleanup(&dir);
	}

	// ----- getShellPath -----

	#[test]
	fn shell_path_cases() {
		let dir = fresh_dir("shell-path");
		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("theme", "dark".into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_shell_path(), None);

		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("shellPath", "/bin/zsh".into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_shell_path().as_deref(), Some("/bin/zsh"));

		write_file(
			&settings_path(&dir),
			&serde_json::to_string(&obj(&[("shellPath", "~/.local/bin/agent-shell-sandbox".into())])).unwrap(),
		);
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(
			manager.get_shell_path().as_deref(),
			Some(crate::utils::paths::join(&crate::utils::paths::home_dir(), ".local/bin/agent-shell-sandbox").as_str())
		);

		write_file(&settings_path(&dir), &serde_json::to_string(&obj(&[("shellPath", "~".into())])).unwrap());
		let manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_shell_path().as_deref(), Some(crate::utils::paths::home_dir().as_str()));
		cleanup(&dir);
	}

	// ----- settings-manager-bug.test.ts -----

	#[test]
	fn bug_preserves_file_changes_to_packages_when_changing_unrelated_setting() {
		let dir = fresh_dir("bug-packages");
		let path = settings_path(&dir);
		write_file(
			&path,
			&serde_json::to_string(&obj(&[("theme", "dark".into()), ("packages", arr(&["npm:pi-mcp-adapter"]))])).unwrap(),
		);

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		assert_eq!(
			manager.get_packages(),
			vec![PackageSource::Simple("npm:pi-mcp-adapter".to_string())]
		);

		// User externally removes the package
		let mut current = read_json(&path);
		current["packages"] = Value::Array(vec![]);
		write_file(&path, &serde_json::to_string_pretty(&current).unwrap());
		assert_eq!(read_json(&path)["packages"], Value::Array(vec![]));

		// Unrelated change triggers save; packages must not be reverted
		manager.set_theme("light");
		manager.flush();

		let saved = read_json(&path);
		assert_eq!(saved["packages"], Value::Array(vec![]));
		assert_eq!(saved["theme"], "light");
		cleanup(&dir);
	}

	#[test]
	fn bug_preserves_file_changes_to_extensions() {
		let dir = fresh_dir("bug-extensions");
		let path = settings_path(&dir);
		write_file(&path, &serde_json::to_string(&obj(&[("theme", "dark".into()), ("extensions", arr(&["/old/extension.ts"]))])).unwrap());

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		let mut current = read_json(&path);
		current["extensions"] = arr(&["/new/extension.ts"]);
		write_file(&path, &serde_json::to_string_pretty(&current).unwrap());

		manager.set_default_thinking_level(AgentThinkingLevel::High);
		manager.flush();

		assert_eq!(read_json(&path)["extensions"], arr(&["/new/extension.ts"]));
		cleanup(&dir);
	}

	#[test]
	fn bug_preserves_external_project_changes_when_updating_unrelated_field() {
		let dir = fresh_dir("bug-project");
		let project_path = project_settings_path(&dir);
		write_file(
			&project_path,
			&serde_json::to_string(&obj(&[("extensions", arr(&["./old-extension.ts"])), ("prompts", arr(&["./old-prompt.md"]))])).unwrap(),
		);

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		let mut current = read_json(&project_path);
		current["prompts"] = arr(&["./new-prompt.md"]);
		write_file(&project_path, &serde_json::to_string_pretty(&current).unwrap());

		manager.set_project_extension_paths(&["./updated-extension.ts".to_string()]);
		manager.flush();

		let saved = read_json(&project_path);
		assert_eq!(saved["prompts"], arr(&["./new-prompt.md"]));
		assert_eq!(saved["extensions"], arr(&["./updated-extension.ts"]));
		cleanup(&dir);
	}

	#[test]
	fn bug_in_memory_project_changes_override_external_changes() {
		let dir = fresh_dir("bug-project-memory");
		let project_path = project_settings_path(&dir);
		write_file(&project_path, &serde_json::to_string(&obj(&[("extensions", arr(&["./initial-extension.ts"]))])).unwrap());

		let mut manager = SettingsManager::create(&project_dir_of(&dir), &agent_dir_of(&dir), SettingsManagerCreateOptions::default());
		let mut current = read_json(&project_path);
		current["extensions"] = arr(&["./external-extension.ts"]);
		write_file(&project_path, &serde_json::to_string_pretty(&current).unwrap());

		manager.set_project_extension_paths(&["./in-memory-extension.ts".to_string()]);
		manager.flush();

		assert_eq!(read_json(&project_path)["extensions"], arr(&["./in-memory-extension.ts"]));
		cleanup(&dir);
	}

	// ----- settings-manager-compaction.test.ts (portable cases) -----

	fn compaction_defaults() -> (bool, u64, u64) {
		(true, 16384, 20000)
	}

	#[test]
	fn compaction_uses_defaults_without_settings() {
		let manager = in_memory_settings(Value::Object(Default::default()));
		assert_eq!(manager.get_compaction_settings(None).unwrap(), compaction_defaults());
		assert_eq!(manager.get_compaction_settings(Some(model())).unwrap(), compaction_defaults());
	}

	#[test]
	fn compaction_resolves_fields_independently() {
		let manager = in_memory_settings(serde_json::json!({
			"compaction": {
				"reserveTokens": 8192,
				"keepRecentTokens": 10000,
				"modelOverrides": { "provider/family/model": { "reserveTokens": 400000 } },
			}
		}));
		assert_eq!(manager.get_compaction_settings(Some(model())).unwrap(), (true, 400000, 10000));
		assert_eq!(manager.get_compaction_reserve_tokens(Some(model())).unwrap(), 400000);
		assert_eq!(manager.get_compaction_keep_recent_tokens(Some(model())).unwrap(), 10000);
		assert_eq!(manager.get_compaction_settings(None).unwrap(), (true, 8192, 10000));

		let mut manager = manager;
		manager.apply_overrides(&Settings::from_value(serde_json::json!({
			"compaction": { "modelOverrides": { "provider/family/model": { "keepRecentTokens": 30000 } } }
		})));
		assert_eq!(manager.get_compaction_keep_recent_tokens(Some(model())).unwrap(), 30000);
		assert_eq!(manager.get_compaction_reserve_tokens(Some(model())).unwrap(), 400000);
	}

	#[test]
	fn compaction_falls_back_to_defaults_for_missing_fields() {
		let manager = in_memory_settings(serde_json::json!({
			"compaction": { "modelOverrides": { "provider/family/model": { "keepRecentTokens": 1024 } } }
		}));
		assert_eq!(manager.get_compaction_settings(Some(model())).unwrap(), (true, 16384, 1024));
	}

	#[test]
	fn compaction_matches_exact_model_ids_including_slashes() {
		let manager = in_memory_settings(serde_json::json!({
			"compaction": {
				"modelOverrides": {
					"provider/family/model": { "reserveTokens": 400000 },
					"provider/*": { "reserveTokens": 1 },
					"family/model": { "reserveTokens": 2 },
				}
			}
		}));
		assert_eq!(manager.get_compaction_reserve_tokens(Some(model())).unwrap(), 400000);
		for other in [("other", "family/model"), ("provider", "other"), ("provider", "family/Model")] {
			assert_eq!(manager.get_compaction_settings(Some(other)).unwrap(), compaction_defaults());
		}
	}

	#[test]
	fn compaction_merges_project_overrides_per_field() {
		let storage: Box<dyn SettingsStorage> = Box::new(InMemorySettingsStorage::default());
		storage
			.with_lock(SettingsScope::Global, &mut |_| {
				Some(serde_json::json!({
					"compaction": {
						"reserveTokens": 8192,
						"modelOverrides": {
							"provider/family/model": { "reserveTokens": 400000, "keepRecentTokens": 30000 },
							"provider/other": { "keepRecentTokens": 4096 },
						},
					}
				})
				.to_string())
			})
			.unwrap();
		storage
			.with_lock(SettingsScope::Project, &mut |_| {
				Some(
					serde_json::json!({
						"compaction": { "reserveTokens": 1024, "modelOverrides": { "provider/family/model": { "keepRecentTokens": 2000 } } }
					})
					.to_string(),
				)
			})
			.unwrap();
		let mut manager = SettingsManager::from_storage(storage, SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_compaction_settings(Some(model())).unwrap(), (true, 400000, 2000));
		assert_eq!(manager.get_compaction_settings(Some(("provider", "other"))).unwrap(), (true, 1024, 4096));
		manager.reload();
		assert_eq!(manager.get_compaction_keep_recent_tokens(Some(model())).unwrap(), 2000);
		manager.set_project_trusted(false);
		assert_eq!(manager.get_compaction_keep_recent_tokens(Some(model())).unwrap(), 30000);
	}

	#[test]
	fn compaction_keeps_enabled_global_and_preserves_overrides_on_toggle() {
		let storage: Box<dyn SettingsStorage> = Box::new(InMemorySettingsStorage::default());
		storage
			.with_lock(SettingsScope::Global, &mut |_| {
				Some(
					serde_json::json!({
						"compaction": { "modelOverrides": { "provider/family/model": { "enabled": false, "reserveTokens": 400000 } } }
					})
					.to_string(),
				)
			})
			.unwrap();
		let mut manager = SettingsManager::from_storage(storage, SettingsManagerCreateOptions::default());
		assert_eq!(manager.get_compaction_settings(Some(model())).unwrap().0, true);
		manager.set_compaction_enabled(false);
		manager.flush();
		manager.reload();
		assert_eq!(manager.get_compaction_settings(Some(model())).unwrap(), (false, 400000, 20000));
	}

	#[test]
	fn compaction_reports_invalid_model_override_token_values() {
		for field in ["reserveTokens", "keepRecentTokens"] {
			for value in [Value::Null, Value::from(-1), Value::from(1.5), Value::from("400000"), Value::from(true)] {
				let storage: Box<dyn SettingsStorage> = Box::new(InMemorySettingsStorage::default());
				let field = field.to_string();
				storage
					.with_lock(SettingsScope::Global, &mut |_| {
						Some(
							serde_json::json!({
								"compaction": { "modelOverrides": { "provider/family/model": { &field: value } } }
							})
							.to_string(),
						)
					})
					.unwrap();
				let manager = SettingsManager::from_storage(storage, SettingsManagerCreateOptions::default());
				let error = manager.get_compaction_settings(Some(model())).unwrap_err();
				let value_display = match &value {
					Value::Null => "null".to_string(),
					Value::Bool(true) => "true".to_string(),
					other => other.to_string(),
				};
				assert_eq!(
					error,
					format!("Invalid compaction.modelOverrides[\"provider/family/model\"].{field} setting: {value_display}. Expected a non-negative safe integer.")
				);
			}
		}
	}

	#[test]
	fn compaction_reports_invalid_ordinary_token_values() {
		for field in ["reserveTokens", "keepRecentTokens"] {
			let field = field.to_string();
			for value in [Value::Null, Value::from(-1), Value::from(1.5), Value::from("400000"), Value::from(true)] {
				let storage: Box<dyn SettingsStorage> = Box::new(InMemorySettingsStorage::default());
				let key = field.clone();
				storage
					.with_lock(SettingsScope::Global, &mut |_| {
						Some(
							serde_json::json!({
								"compaction": { &key: value, "modelOverrides": { "provider/family/model": { &key: 4096 } } }
							})
							.to_string(),
						)
					})
					.unwrap();
				let manager = SettingsManager::from_storage(storage, SettingsManagerCreateOptions::default());
				let value_display = match &value {
					Value::Null => "null".to_string(),
					Value::Bool(true) => "true".to_string(),
					other => other.to_string(),
				};
				let error = format!("Invalid compaction.{field} setting: {value_display}. Expected a non-negative safe integer.");
				assert_eq!(manager.get_compaction_settings(None).unwrap_err(), error);
				assert_eq!(manager.get_compaction_settings(Some(model())).unwrap_err(), error);
			}
		}
	}

	#[test]
	fn compaction_reports_malformed_model_entries() {
		for entry in [Value::Null, Value::from(false), Value::from(42), Value::from("invalid"), Value::Array(vec![])] {
			let storage: Box<dyn SettingsStorage> = Box::new(InMemorySettingsStorage::default());
			storage
				.with_lock(SettingsScope::Global, &mut |_| {
					Some(
						serde_json::json!({
							"compaction": { "modelOverrides": { "provider/family/model": entry } }
						})
						.to_string(),
					)
				})
				.unwrap();
			let manager = SettingsManager::from_storage(storage, SettingsManagerCreateOptions::default());
			let entry_display = match &entry {
				Value::Null => "null".to_string(),
				Value::Array(_) => "".to_string(),
				other => other.to_string(),
			};
			let expected_suffix = match &entry {
				Value::Array(_) => " setting: []. Expected an object.".to_string(),
				_ => format!(" setting: {entry_display}. Expected an object."),
			};
			let error = manager.get_compaction_settings(Some(model())).unwrap_err();
			assert!(
				error.starts_with("Invalid compaction.modelOverrides[\"provider/family/model\"]") && error.ends_with(&expected_suffix),
				"{error}"
			);
		}
	}

	#[test]
	fn compaction_accepts_zero_in_settings_and_overrides() {
		let mut manager = in_memory_settings(serde_json::json!({
			"compaction": { "reserveTokens": 0, "keepRecentTokens": 0 }
		}));
		assert_eq!(manager.get_compaction_settings(Some(model())).unwrap(), (true, 0, 0));
		manager.apply_overrides(&Settings::from_value(serde_json::json!({
			"compaction": {
				"reserveTokens": 1000,
				"keepRecentTokens": 1000,
				"modelOverrides": { "provider/family/model": { "reserveTokens": 0, "keepRecentTokens": 0 } }
			}
		})));
		assert_eq!(manager.get_compaction_settings(Some(model())).unwrap(), (true, 0, 0));
	}

	// ----- shared helpers (merge/migration unit coverage) -----

	#[test]
	fn migrate_settings_ports_legacy_fields() {
		let migrated = SettingsManager::migrate_settings(serde_json::json!({
			"queueMode": "all",
			"websockets": true,
			"skills": { "enableSkillCommands": false, "customDirectories": ["/skills"] },
			"retry": { "maxDelayMs": 5000 }
		}))
		.0;
		assert_eq!(migrated["steeringMode"], "all");
		assert!(migrated.get("queueMode").is_none());
		assert_eq!(migrated["transport"], "websocket");
		assert!(migrated.get("websockets").is_none());
		assert_eq!(migrated["skills"], arr(&["/skills"]));
		assert_eq!(migrated["enableSkillCommands"], false);
		assert_eq!(migrated["retry"]["provider"]["maxRetryDelayMs"], 5000);
		assert!(migrated["retry"].get("maxDelayMs").is_none());
	}

	#[test]
	fn resolve_default_tools_matches_upstream() {
		assert_eq!(
			resolve_default_tools(&["+codemode".to_string(), "-write".to_string()]),
			vec!["read".to_string(), "bash".to_string(), "edit".to_string(), "codemode".to_string()]
		);
		assert_eq!(resolve_default_tools(&[]), Vec::<String>::new());
	}
}
