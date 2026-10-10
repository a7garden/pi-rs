//! Port of `packages/coding-agent/src/core/trust-manager.ts` — the on-disk
//! project-trust store (`trust.json` in the agent dir), the trust option
//! table, and the trust-requiring-project-resources probe.
//!
//! # Deviations (per-file ledger)
//!
//! - **D111 — `proper-lockfile` is replaced by a create-exclusive lock
//!   marker.** `acquireTrustLockSync` becomes a `<trust.json>.lock`
//!   create-new retry loop (10 attempts, sleeping 20 ms between attempts
//!   instead of upstream's busy-wait); release unlinks the marker.
//!   Contention past the final attempt reports proper-lockfile's `ELOCKED`
//!   message (`Lock file is already being held`). Like upstream's unset
//!   `stale` option, a stale marker is never stolen.
//! - **D112 — store methods return `Result` instead of throwing.** JS
//!   exceptions become `Err(String)`. JSON parse failures keep the stable
//!   `Failed to read trust store {path}: ` prefix but carry the
//!   `serde_json` message body (JS engine messages are not reproducible).
//! - **D113 — written key order matches JS `Array.prototype.sort`.** Keys
//!   are sorted by UTF-16 code units (JS default string comparison), not
//!   UTF-8 byte order, and each key is serialized with `serde_json` string
//!   encoding (`JSON.stringify(key)` parity).
//! - **D114 — `process.env.HOME || homedir()` maps to the crate's
//!   [`crate::utils::paths::home_dir`] seam** (`HOME` when non-empty, else
//!   the platform home). The resource probe has a `_with_home` variant so
//!   tests exercise the same logic without mutating process-wide env.

use std::fs::OpenOptions;
use std::path::Path;
use std::time::Duration;

use serde_json::Value;

use crate::config::CONFIG_DIR_NAME;
use crate::utils::paths::{canonicalize_path, home_dir, join, resolve_path_cwd};
use crate::utils::text::strip_bom;

/// Upstream `ProjectTrustDecision` (`boolean | null`); `None` is `null`.
pub type ProjectTrustDecision = Option<bool>;

/// Upstream `ProjectTrustStoreEntry`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectTrustStoreEntry {
    pub path: String,
    pub decision: bool,
}

/// Upstream `ProjectTrustUpdate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectTrustUpdate {
    pub path: String,
    pub decision: ProjectTrustDecision,
}

/// Upstream `ProjectTrustOption`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectTrustOption {
    pub label: String,
    pub trusted: bool,
    pub updates: Vec<ProjectTrustUpdate>,
    pub saved_path: Option<String>,
}

/// Options of [`get_project_trust_options`] (upstream
/// `{ includeSessionOnly?: boolean }`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GetProjectTrustOptionsOptions {
    pub include_session_only: bool,
}

/// Upstream `TrustFile` (`Record<string, boolean | null | undefined>`).
type TrustFile = serde_json::Map<String, Value>;

/// Upstream `TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES` — entries under
/// `<cwd>/.pi` that make a project trust-gated.
const TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES: [&str; 8] = [
    "settings.json",
    "mcp.json",
    "extensions",
    "skills",
    "prompts",
    "themes",
    "SYSTEM.md",
    "APPEND_SYSTEM.md",
];

/// `dirname` with upstream loop semantics: `None` at the filesystem root
/// (where JS `dirname` returns its input unchanged).
fn parent_dir(path: &str) -> Option<String> {
    let parent = Path::new(path).parent()?;
    if parent.as_os_str().is_empty() {
        return None;
    }
    Some(parent.to_string_lossy().into_owned())
}

fn normalize_cwd(cwd: &str) -> String {
    canonicalize_path(&resolve_path_cwd(cwd))
}

fn find_nearest_trust_entry(data: &TrustFile, cwd: &str) -> Option<ProjectTrustStoreEntry> {
    let mut current_dir = normalize_cwd(cwd);
    loop {
        if let Some(Value::Bool(decision)) = data.get(&current_dir) {
            return Some(ProjectTrustStoreEntry {
                path: current_dir,
                decision: *decision,
            });
        }
        current_dir = parent_dir(&current_dir)?;
    }
}

/// Upstream `getProjectTrustParentPath`.
pub fn get_project_trust_parent_path(cwd: &str) -> Option<String> {
    let trust_path = normalize_cwd(cwd);
    parent_dir(&trust_path)
}

/// Upstream `getProjectTrustOptions` — the trust prompt option table.
pub fn get_project_trust_options(
    cwd: &str,
    options: GetProjectTrustOptionsOptions,
) -> Vec<ProjectTrustOption> {
    let trust_path = normalize_cwd(cwd);
    let mut trust_options = vec![ProjectTrustOption {
        label: "Trust".to_string(),
        trusted: true,
        updates: vec![ProjectTrustUpdate {
            path: trust_path.clone(),
            decision: Some(true),
        }],
        saved_path: Some(trust_path.clone()),
    }];
    if let Some(parent_path) = get_project_trust_parent_path(cwd) {
        trust_options.push(ProjectTrustOption {
            label: format!("Trust parent folder ({parent_path})"),
            trusted: true,
            updates: vec![
                ProjectTrustUpdate {
                    path: parent_path.clone(),
                    decision: Some(true),
                },
                ProjectTrustUpdate {
                    path: trust_path.clone(),
                    decision: None,
                },
            ],
            saved_path: Some(parent_path),
        });
    }
    if options.include_session_only {
        trust_options.push(ProjectTrustOption {
            label: "Trust (this session only)".to_string(),
            trusted: true,
            updates: Vec::new(),
            saved_path: None,
        });
    }
    trust_options.push(ProjectTrustOption {
        label: "Do not trust".to_string(),
        trusted: false,
        updates: vec![ProjectTrustUpdate {
            path: trust_path.clone(),
            decision: Some(false),
        }],
        saved_path: Some(trust_path),
    });
    if options.include_session_only {
        trust_options.push(ProjectTrustOption {
            label: "Do not trust (this session only)".to_string(),
            trusted: false,
            updates: Vec::new(),
            saved_path: None,
        });
    }
    trust_options
}

/// Upstream `readTrustFile` (D112: `Err` instead of throw).
fn read_trust_file(path: &str) -> Result<TrustFile, String> {
    if !Path::new(path).exists() {
        return Ok(serde_json::Map::new());
    }

    let content = std::fs::read_to_string(path)
        .map_err(|error| format!("Failed to read trust store {path}: {error}"))?;
    let parsed: Value = serde_json::from_str(strip_bom(&content))
        .map_err(|error| format!("Failed to read trust store {path}: {error}"))?;

    let Some(object) = parsed.as_object() else {
        return Err(format!("Invalid trust store {path}: expected an object"));
    };

    let mut data = serde_json::Map::new();
    for (key, value) in object {
        if !value.is_boolean() && !value.is_null() {
            // `JSON.stringify(key)` parity.
            let key_json = serde_json::to_string(key).expect("string key serializes");
            return Err(format!(
                "Invalid trust store {path}: value for {key_json} must be true, false, or null"
            ));
        }
        data.insert(key.clone(), value.clone());
    }
    Ok(data)
}

/// JS `Array.prototype.sort` string comparison: UTF-16 code-unit order
/// (D113).
fn compare_utf16(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// Upstream `writeTrustFile`: keys sorted, non-trust values skipped, two
/// space indent, trailing newline.
fn write_trust_file(path: &str, data: &TrustFile) -> Result<(), String> {
    let mut keys: Vec<&String> = data.keys().collect();
    keys.sort_by(|a, b| compare_utf16(a, b));
    let mut sorted = serde_json::Map::new();
    for key in keys {
        let value = &data[key];
        if value.is_boolean() || value.is_null() {
            sorted.insert(key.clone(), value.clone());
        }
    }
    let dir = parent_dir(path)
        .filter(|dir| !dir.is_empty())
        .unwrap_or_else(|| ".".to_string());
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let body = serde_json::to_string_pretty(&sorted).map_err(|error| error.to_string())?;
    std::fs::write(path, format!("{body}\n")).map_err(|error| error.to_string())
}

const TRUST_LOCK_MAX_ATTEMPTS: u32 = 10;
const TRUST_LOCK_DELAY: Duration = Duration::from_millis(20);
/// proper-lockfile's `ELOCKED` message (D111).
const TRUST_LOCKED_MESSAGE: &str = "Lock file is already being held";

/// Held trust-store lock; releasing (dropping) removes the marker file.
#[derive(Debug)]
struct TrustFileLock {
    lock_path: String,
}

impl Drop for TrustFileLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

/// Upstream `acquireTrustLockSync` (D111).
fn acquire_trust_lock_sync(path: &str) -> Result<TrustFileLock, String> {
    acquire_trust_lock_sync_with(path, TRUST_LOCK_MAX_ATTEMPTS, TRUST_LOCK_DELAY)
}

/// [`acquire_trust_lock_sync`] with explicit retry parameters (test seam).
fn acquire_trust_lock_sync_with(
    path: &str,
    max_attempts: u32,
    delay: Duration,
) -> Result<TrustFileLock, String> {
    let trust_dir = parent_dir(path)
        .filter(|dir| !dir.is_empty())
        .unwrap_or_else(|| ".".to_string());
    std::fs::create_dir_all(&trust_dir).map_err(|error| error.to_string())?;
    let lock_path = format!("{path}.lock");
    let mut last_error: Option<String> = None;

    for attempt in 1..=max_attempts {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(_) => return Ok(TrustFileLock { lock_path }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                last_error = Some(TRUST_LOCKED_MESSAGE.to_string());
                if attempt == max_attempts {
                    break;
                }
                std::thread::sleep(delay);
            }
            Err(error) => return Err(error.to_string()),
        }
    }

    Err(last_error.unwrap_or_else(|| "Failed to acquire trust store lock".to_string()))
}

/// Upstream `withTrustFileLock`.
fn with_trust_file_lock<T>(path: &str, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let _lock = acquire_trust_lock_sync(path)?;
    f()
}

/// Returns true when cwd has project-local resources that must be gated by
/// project trust: trust-requiring entries under `cwd/.pi`, or
/// `.agents/skills` in cwd or one of its ancestors. Returns false when no
/// such project resources exist. The user/global `~/.agents/skills`
/// directory is always treated as a trusted user resource and is ignored
/// here, even when cwd is `$HOME` (upstream
/// `hasTrustRequiringProjectResources`).
pub fn has_trust_requiring_project_resources(cwd: &str) -> bool {
    let home = std::env::var("HOME")
        .ok()
        .filter(|home| !home.is_empty())
        .unwrap_or_else(home_dir);
    has_trust_requiring_project_resources_with_home(cwd, &home)
}

/// [`has_trust_requiring_project_resources`] with an explicit home
/// directory (D114 test seam). Like upstream, the home directory is
/// canonicalized here (`canonicalizePath(resolvePath(...))`).
fn has_trust_requiring_project_resources_with_home(cwd: &str, home: &str) -> bool {
    let home = canonicalize_path(&resolve_path_cwd(home));
    let user_agents_skills_dir = Path::new(&home).join(".agents").join("skills");
    let mut current_dir = canonicalize_path(&resolve_path_cwd(cwd));

    let config_dir = Path::new(&current_dir).join(CONFIG_DIR_NAME);
    if TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES
        .iter()
        .any(|entry| config_dir.join(entry).exists())
    {
        return true;
    }

    loop {
        let agents_skills_dir = Path::new(&current_dir).join(".agents").join("skills");
        if agents_skills_dir != user_agents_skills_dir && agents_skills_dir.exists() {
            return true;
        }

        current_dir = match parent_dir(&current_dir) {
            Some(parent_dir) => parent_dir,
            None => return false,
        };
    }
}

/// Upstream `ProjectTrustStore` — file-backed decisions keyed by normalized
/// path, nearest ancestor wins (D112: `Result` instead of throw).
#[derive(Debug, Clone)]
pub struct ProjectTrustStore {
    trust_path: String,
}

impl ProjectTrustStore {
    pub fn new(agent_dir: &str) -> Self {
        ProjectTrustStore {
            trust_path: join(&resolve_path_cwd(agent_dir), "trust.json"),
        }
    }

    /// Upstream `get`.
    pub fn get(&self, cwd: &str) -> Result<ProjectTrustDecision, String> {
        Ok(self.get_entry(cwd)?.map(|entry| entry.decision))
    }

    /// Upstream `getEntry`.
    pub fn get_entry(&self, cwd: &str) -> Result<Option<ProjectTrustStoreEntry>, String> {
        with_trust_file_lock(&self.trust_path, || {
            let data = read_trust_file(&self.trust_path)?;
            Ok(find_nearest_trust_entry(&data, cwd))
        })
    }

    /// Upstream `set`.
    pub fn set(&self, cwd: &str, decision: ProjectTrustDecision) -> Result<(), String> {
        self.set_many(std::slice::from_ref(&ProjectTrustUpdate {
            path: cwd.to_string(),
            decision,
        }))
    }

    /// Upstream `setMany`.
    pub fn set_many(&self, decisions: &[ProjectTrustUpdate]) -> Result<(), String> {
        with_trust_file_lock(&self.trust_path, || {
            let mut data = read_trust_file(&self.trust_path)?;
            for update in decisions {
                let key = normalize_cwd(&update.path);
                match update.decision {
                    None => {
                        data.remove(&key);
                    }
                    Some(decision) => {
                        data.insert(key, Value::Bool(decision));
                    }
                }
            }
            write_trust_file(&self.trust_path, &data)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// Unique temp directory per test (upstream
    /// `join(tmpdir(), \`trust-test-...\`)`).
    fn temp_dir(tag: &str) -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("pi-trust-{tag}-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn update(path: &str, decision: ProjectTrustDecision) -> ProjectTrustUpdate {
        ProjectTrustUpdate {
            path: path.to_string(),
            decision,
        }
    }

    // -- Ported: "stores decisions and inherits from parent directories" ----

    #[test]
    fn stores_decisions_and_inherits_from_parent_directories() {
        let temp = temp_dir("store");
        let agent_dir = temp.join("agent");
        let parent_dir_path = temp.join("trusted-parent");
        let child_dir = parent_dir_path.join("project");
        std::fs::create_dir_all(&child_dir).expect("create child dir");

        let store = ProjectTrustStore::new(agent_dir.to_string_lossy().as_ref());
        assert_eq!(store.get(child_dir.to_string_lossy().as_ref()), Ok(None));
        store
            .set(parent_dir_path.to_string_lossy().as_ref(), Some(true))
            .expect("set parent");
        assert_eq!(
            store.get(child_dir.to_string_lossy().as_ref()),
            Ok(Some(true))
        );
        store
            .set(child_dir.to_string_lossy().as_ref(), Some(false))
            .expect("set child");
        assert_eq!(
            store.get(child_dir.to_string_lossy().as_ref()),
            Ok(Some(false))
        );
        store
            .set(child_dir.to_string_lossy().as_ref(), None)
            .expect("clear child");
        assert_eq!(
            store.get(child_dir.to_string_lossy().as_ref()),
            Ok(Some(true))
        );

        let _ = std::fs::remove_dir_all(&temp);
    }

    // -- Ported: "detects trust-requiring project resources" ----------------

    #[test]
    fn detects_trust_requiring_project_resources() {
        let temp = temp_dir("resources");
        let cwd = temp.join("project");
        std::fs::create_dir_all(&cwd).expect("create project cwd");
        std::fs::create_dir_all(temp.join(".pi").join("agent")).expect("create temp .pi/agent");
        std::fs::create_dir_all(temp.join(".agents").join("skills"))
            .expect("create ~/.agents/skills");

        // Upstream sets `process.env.HOME = tempDir`; the `_with_home`
        // variant covers the same logic without env mutation (D114).
        let home = temp.to_string_lossy().into_owned();
        let cwd_string = cwd.to_string_lossy().into_owned();
        assert!(!has_trust_requiring_project_resources_with_home(
            &home, &home
        ));
        assert!(!has_trust_requiring_project_resources_with_home(
            &cwd_string,
            &home
        ));

        std::fs::write(temp.join(".pi").join("settings.json"), "{}").expect("write settings");
        assert!(has_trust_requiring_project_resources_with_home(
            &home, &home
        ));
        let _ = std::fs::remove_file(temp.join(".pi").join("settings.json"));

        std::fs::create_dir_all(cwd.join(".pi")).expect("create .pi");
        std::fs::write(cwd.join(".pi").join("settings.json"), "{}").expect("write settings");
        assert!(has_trust_requiring_project_resources_with_home(
            &cwd_string,
            &home
        ));

        let _ = std::fs::remove_dir_all(cwd.join(".pi"));
        std::fs::create_dir_all(cwd.join(".agents").join("skills")).expect("create project skills");
        assert!(has_trust_requiring_project_resources_with_home(
            &cwd_string,
            &home
        ));

        let _ = std::fs::remove_dir_all(&temp);
    }

    // -- Module test: pin the trust option table ------------------------------

    #[test]
    fn trust_option_table_pins_labels_updates_and_saved_paths() {
        let cwd = "/tmp/pi-trust-options/project";

        let default = get_project_trust_options(cwd, GetProjectTrustOptionsOptions::default());
        assert_eq!(
            default,
            vec![
                ProjectTrustOption {
                    label: "Trust".to_string(),
                    trusted: true,
                    updates: vec![update(cwd, Some(true))],
                    saved_path: Some(cwd.to_string()),
                },
                ProjectTrustOption {
                    label: "Trust parent folder (/tmp/pi-trust-options)".to_string(),
                    trusted: true,
                    updates: vec![
                        update("/tmp/pi-trust-options", Some(true)),
                        update(cwd, None)
                    ],
                    saved_path: Some("/tmp/pi-trust-options".to_string()),
                },
                ProjectTrustOption {
                    label: "Do not trust".to_string(),
                    trusted: false,
                    updates: vec![update(cwd, Some(false))],
                    saved_path: Some(cwd.to_string()),
                },
            ]
        );

        let with_session_only = get_project_trust_options(
            cwd,
            GetProjectTrustOptionsOptions {
                include_session_only: true,
            },
        );
        assert_eq!(with_session_only.len(), 5);
        assert_eq!(with_session_only[2].label, "Trust (this session only)");
        assert!(with_session_only[2].trusted);
        assert!(with_session_only[2].updates.is_empty());
        assert_eq!(with_session_only[2].saved_path, None);
        assert_eq!(
            with_session_only[4].label,
            "Do not trust (this session only)"
        );
        assert!(!with_session_only[4].trusted);
        assert!(with_session_only[4].updates.is_empty());
        assert_eq!(with_session_only[4].saved_path, None);
    }

    #[test]
    fn root_cwd_has_no_parent_option() {
        assert_eq!(get_project_trust_parent_path("/"), None);
        let options = get_project_trust_options("/", GetProjectTrustOptionsOptions::default());
        assert_eq!(
            options
                .iter()
                .map(|option| option.label.as_str())
                .collect::<Vec<_>>(),
            vec!["Trust", "Do not trust"]
        );
    }

    // -- Module test: lock retry (D111) ---------------------------------------

    #[test]
    fn lock_retry_fails_after_max_attempts_and_releases_on_drop() {
        let temp = temp_dir("lock");
        let trust_path = temp.join("agent").join("trust.json");
        let trust_path = trust_path.to_string_lossy().into_owned();

        let lock = acquire_trust_lock_sync_with(&trust_path, 3, Duration::from_millis(1))
            .expect("first lock succeeds");
        assert!(Path::new(&format!("{trust_path}.lock")).exists());

        let error = acquire_trust_lock_sync_with(&trust_path, 3, Duration::from_millis(1))
            .expect_err("second lock blocks");
        assert_eq!(error, "Lock file is already being held");

        drop(lock);
        let reacquired = acquire_trust_lock_sync_with(&trust_path, 3, Duration::from_millis(1));
        assert!(reacquired.is_ok(), "lock releases on drop");

        drop(reacquired);
        assert!(!Path::new(&format!("{trust_path}.lock")).exists());
        let _ = std::fs::remove_dir_all(&temp);
    }

    // -- Module test: trust.json round-trip ------------------------------------

    #[test]
    fn trust_json_round_trips_sorted_with_trailing_newline() {
        let temp = temp_dir("round-trip");
        let agent_dir = temp.join("agent");
        let parent_dir_path = temp.join("zed-parent");
        let child_dir = parent_dir_path.join("project");
        let other_dir = temp.join("aaa-other");
        std::fs::create_dir_all(&child_dir).expect("create dirs");
        std::fs::create_dir_all(&other_dir).expect("create dirs");

        let store = ProjectTrustStore::new(agent_dir.to_string_lossy().as_ref());
        store
            .set_many(&[
                update(parent_dir_path.to_string_lossy().as_ref(), Some(true)),
                update(child_dir.to_string_lossy().as_ref(), Some(false)),
                update(other_dir.to_string_lossy().as_ref(), Some(true)),
            ])
            .expect("set many");

        // Keys are written sorted (D113); expect `aaa-other` before the
        // canonicalized `zed-parent` prefix.
        let trust_path = join(
            &resolve_path_cwd(agent_dir.to_string_lossy().as_ref()),
            "trust.json",
        );
        let raw = std::fs::read_to_string(&trust_path).expect("read trust.json");
        let other_key = normalize_cwd(other_dir.to_string_lossy().as_ref());
        let parent_key = normalize_cwd(parent_dir_path.to_string_lossy().as_ref());
        let child_key = normalize_cwd(child_dir.to_string_lossy().as_ref());
        assert!(raw.ends_with('\n'), "file ends with a newline");
        // `aaa-other` sorts before the canonicalized `zed-parent` prefix, and
        // the parent path sorts before its `.../project` child (prefix rule).
        let mut expected = String::from("{\n");
        expected.push_str(&format!("  \"{other_key}\": true,\n"));
        expected.push_str(&format!("  \"{parent_key}\": true,\n"));
        expected.push_str(&format!("  \"{child_key}\": false\n"));
        expected.push_str("}\n");
        assert_eq!(raw, expected);

        // Round-trip: the file parses back to the same decisions.
        assert_eq!(
            store.get(child_dir.to_string_lossy().as_ref()),
            Ok(Some(false))
        );
        assert_eq!(
            store.get(parent_dir_path.to_string_lossy().as_ref()),
            Ok(Some(true))
        );
        assert_eq!(
            store.get(other_dir.to_string_lossy().as_ref()),
            Ok(Some(true))
        );
        assert_eq!(
            store.get_entry(child_dir.to_string_lossy().as_ref()),
            Ok(Some(ProjectTrustStoreEntry {
                path: child_key,
                decision: false
            }))
        );

        let _ = std::fs::remove_dir_all(&temp);
    }

    // -- Module test: exact validation error strings (D112) --------------------

    #[test]
    fn read_rejects_malformed_values_and_non_objects_with_exact_messages() {
        let temp = temp_dir("invalid");
        let agent_dir = temp.join("agent");
        std::fs::create_dir_all(&agent_dir).expect("create agent dir");
        let trust_path = agent_dir.join("trust.json");
        let trust_path = trust_path.to_string_lossy().into_owned();

        std::fs::write(&trust_path, r#"{"x": "yes"}"#).expect("write trust.json");
        let error = ProjectTrustStore::new(agent_dir.to_string_lossy().as_ref())
            .get("/tmp")
            .expect_err("string value rejected");
        assert_eq!(
            error,
            format!(
                "Invalid trust store {trust_path}: value for \"x\" must be true, false, or null"
            )
        );

        std::fs::write(&trust_path, "[1]").expect("write trust.json");
        let error = ProjectTrustStore::new(agent_dir.to_string_lossy().as_ref())
            .get("/tmp")
            .expect_err("array rejected");
        assert_eq!(
            error,
            format!("Invalid trust store {trust_path}: expected an object")
        );

        std::fs::write(&trust_path, "{").expect("write trust.json");
        let error = ProjectTrustStore::new(agent_dir.to_string_lossy().as_ref())
            .get("/tmp")
            .expect_err("malformed JSON rejected");
        assert!(
            error.starts_with(&format!("Failed to read trust store {trust_path}: ")),
            "unexpected message: {error}"
        );

        let _ = std::fs::remove_dir_all(&temp);
    }
}
