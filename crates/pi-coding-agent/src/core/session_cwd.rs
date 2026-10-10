//! Port of `packages/coding-agent/src/core/session-cwd.ts` — detecting and
//! reporting a stored session working directory that no longer exists.
//!
//! Upstream `getMissingSessionCwdIssue`/`assertSessionCwdExists` take a
//! structural `{ getCwd, getSessionFile }` object; here that contract is the
//! [`SessionCwdSource`] trait (deviation D65) with a blanket impl for
//! [`SessionManager`], whose inherent accessors have the same names.
//!
//! Deferring with `AgentSessionRuntime` (see [`crate::core::agent_session`]):
//! the upstream suite's third case constructs `createAgentSessionRuntime`;
//! the runtime-creation wrapper lands with that module.

use std::fmt;

use crate::core::session_manager::SessionManager;

/// A session whose stored cwd is missing (`SessionCwdIssue`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionCwdIssue {
    /// Optional upstream (`sessionFile?: string`).
    pub session_file: Option<String>,
    pub session_cwd: String,
    pub fallback_cwd: String,
}

/// The `getCwd`/`getSessionFile` surface `session-cwd.ts` reads
/// (`SessionCwdSource`).
pub trait SessionCwdSource {
    fn get_cwd(&self) -> String;
    fn get_session_file(&self) -> Option<String>;
}

impl SessionCwdSource for SessionManager {
    fn get_cwd(&self) -> String {
        // Inherent method wins over the trait method.
        SessionManager::get_cwd(self).to_string()
    }

    fn get_session_file(&self) -> Option<String> {
        SessionManager::get_session_file(self).map(String::from)
    }
}

/// `getMissingSessionCwdIssue`: `Some` when the session has a file and a
/// stored cwd that names no existing path.
pub fn get_missing_session_cwd_issue(
    session_manager: &impl SessionCwdSource,
    fallback_cwd: &str,
) -> Option<SessionCwdIssue> {
    let session_file = session_manager.get_session_file()?;
    let session_cwd = session_manager.get_cwd();
    if session_cwd.is_empty() || std::path::Path::new(&session_cwd).exists() {
        return None;
    }

    Some(SessionCwdIssue {
        session_file: Some(session_file),
        session_cwd,
        fallback_cwd: fallback_cwd.to_string(),
    })
}

/// `formatMissingSessionCwdError`.
pub fn format_missing_session_cwd_error(issue: &SessionCwdIssue) -> String {
    let session_file = issue
        .session_file
        .as_ref()
        .map(|file| format!("\nSession file: {file}"))
        .unwrap_or_default();
    format!(
        "Stored session working directory does not exist: {}{}\nCurrent working directory: {}",
        issue.session_cwd, session_file, issue.fallback_cwd
    )
}

/// `formatMissingSessionCwdPrompt`.
pub fn format_missing_session_cwd_prompt(issue: &SessionCwdIssue) -> String {
    format!(
        "cwd from session file does not exist\n{}\n\ncontinue in current cwd\n{}",
        issue.session_cwd, issue.fallback_cwd
    )
}

/// `MissingSessionCwdError`: carries the issue, displays the formatted error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingSessionCwdError {
    pub issue: SessionCwdIssue,
}

impl MissingSessionCwdError {
    pub fn new(issue: SessionCwdIssue) -> MissingSessionCwdError {
        MissingSessionCwdError { issue }
    }
}

impl fmt::Display for MissingSessionCwdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format_missing_session_cwd_error(&self.issue))
    }
}

impl std::error::Error for MissingSessionCwdError {}

/// `assertSessionCwdExists`: throw [`MissingSessionCwdError`] when the stored
/// cwd is missing.
pub fn assert_session_cwd_exists(
    session_manager: &impl SessionCwdSource,
    fallback_cwd: &str,
) -> Result<(), MissingSessionCwdError> {
    match get_missing_session_cwd_issue(session_manager, fallback_cwd) {
        Some(issue) => Err(MissingSessionCwdError::new(issue)),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::session_manager::SessionManager;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn create_temp_dir(name: &str) -> PathBuf {
        // Counter disambiguates same-tick creations under parallel load.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "{name}-{}-{}-{}",
            nanos,
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_session_file(path: &PathBuf, cwd: &str) {
        let timestamp = crate::utils::time::to_iso_timestamp(crate::utils::time::now_ms());
        fs::write(
            path,
            format!(
                "{}\n",
                serde_json::json!({
                    "type": "session",
                    "version": 3,
                    "id": "session-id",
                    "timestamp": timestamp,
                    "cwd": cwd,
                })
            ),
        )
        .unwrap();
    }

    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    // Upstream `detects missing session cwd from persisted sessions`.
    #[test]
    fn detects_missing_session_cwd_from_persisted_sessions() {
        let fallback_dir = create_temp_dir("pi-session-cwd-fallback");
        let _cleanup = Cleanup(fallback_dir.clone());
        let fallback_cwd = fallback_dir.to_string_lossy().to_string();
        let missing_cwd = fallback_dir
            .join("does-not-exist")
            .to_string_lossy()
            .to_string();
        let session_dir = create_temp_dir("pi-session-cwd-session-dir");
        let _cleanup_session = Cleanup(session_dir.clone());
        let session_file = session_dir.join("session.jsonl");
        write_session_file(&session_file, &missing_cwd);

        let session_manager =
            SessionManager::open(&session_file.to_string_lossy(), None, None).unwrap();
        let issue = get_missing_session_cwd_issue(&session_manager, &fallback_cwd).unwrap();
        assert_eq!(
            issue,
            SessionCwdIssue {
                session_file: session_manager.get_session_file().map(String::from),
                session_cwd: missing_cwd,
                fallback_cwd,
            }
        );
    }

    // Upstream `supports overriding the effective cwd when opening a session`.
    #[test]
    fn supports_overriding_the_effective_cwd_when_opening_a_session() {
        let fallback_dir = create_temp_dir("pi-session-cwd-override");
        let _cleanup = Cleanup(fallback_dir.clone());
        let fallback_cwd = fallback_dir.to_string_lossy().to_string();
        let missing_cwd = fallback_dir
            .join("does-not-exist")
            .to_string_lossy()
            .to_string();
        let session_dir = create_temp_dir("pi-session-cwd-override-session-dir");
        let _cleanup_session = Cleanup(session_dir.clone());
        let session_file = session_dir.join("session.jsonl");
        write_session_file(&session_file, &missing_cwd);

        let session_manager =
            SessionManager::open(&session_file.to_string_lossy(), None, Some(&fallback_cwd))
                .unwrap();
        assert_eq!(session_manager.get_cwd(), fallback_cwd);
        assert!(get_missing_session_cwd_issue(&session_manager, &fallback_cwd).is_none());
    }

    // Upstream `throws a controlled error before runtime creation when the
    // stored cwd is missing` constructs `createAgentSessionRuntime`; it defers
    // with `core/agent-session-runtime.ts` (module docs).

    #[test]
    fn formats_error_prompt_and_throws_the_controlled_error() {
        let issue = SessionCwdIssue {
            session_file: Some("/tmp/session.jsonl".to_string()),
            session_cwd: "/gone".to_string(),
            fallback_cwd: "/here".to_string(),
        };
        assert_eq!(
			format_missing_session_cwd_error(&issue),
			"Stored session working directory does not exist: /gone\nSession file: /tmp/session.jsonl\nCurrent working directory: /here"
		);
        assert_eq!(
            format_missing_session_cwd_prompt(&issue),
            "cwd from session file does not exist\n/gone\n\ncontinue in current cwd\n/here"
        );

        struct Fixed;
        impl SessionCwdSource for Fixed {
            fn get_cwd(&self) -> String {
                "/gone".to_string()
            }
            fn get_session_file(&self) -> Option<String> {
                Some("/tmp/session.jsonl".to_string())
            }
        }
        let error = assert_session_cwd_exists(&Fixed, "/here").unwrap_err();
        assert_eq!(error.issue, issue);
        assert_eq!(error.to_string(), format_missing_session_cwd_error(&issue));

        struct NoFile;
        impl SessionCwdSource for NoFile {
            fn get_cwd(&self) -> String {
                "/gone".to_string()
            }
            fn get_session_file(&self) -> Option<String> {
                None
            }
        }
        // No session file: no issue, even with a missing cwd.
        assert!(get_missing_session_cwd_issue(&NoFile, "/here").is_none());
    }
}
