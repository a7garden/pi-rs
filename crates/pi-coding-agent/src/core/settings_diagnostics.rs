//! Port of `packages/coding-agent/src/core/settings-diagnostics.ts` —
//! collecting settings load errors as runtime diagnostics and
//! deduplicating them. No deviations: `SettingsError.error` already carries
//! the flattened upstream error message, and
//! [`AgentSessionRuntimeDiagnostic`]/[`DiagnosticType`] mirror the upstream
//! diagnostic shape.

use crate::core::agent_session_services::{AgentSessionRuntimeDiagnostic, DiagnosticType};
use crate::core::settings_manager::{SettingsManager, SettingsScope};

/// Upstream `collectSettingsDiagnostics` — drain recorded settings errors
/// into `warning` diagnostics.
pub fn collect_settings_diagnostics(
    settings_manager: &mut SettingsManager,
) -> Vec<AgentSessionRuntimeDiagnostic> {
    settings_manager
        .drain_errors()
        .into_iter()
        .map(|error| AgentSessionRuntimeDiagnostic {
            diagnostic_type: DiagnosticType::Warning,
            message: match &error.path {
                Some(path) => format!("Invalid settings file {path}: {}", error.error),
                None => format!(
                    "Invalid {} settings: {}",
                    scope_name(error.scope),
                    error.error
                ),
            },
        })
        .collect()
}

fn scope_name(scope: SettingsScope) -> &'static str {
    match scope {
        SettingsScope::Global => "global",
        SettingsScope::Project => "project",
    }
}

/// Upstream `deduplicateDiagnostics` — drop duplicate `type`/`message`
/// pairs while preserving first occurrences. Startup and runtime settings
/// managers can report the same file error.
pub fn deduplicate_diagnostics(
    diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
) -> Vec<AgentSessionRuntimeDiagnostic> {
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::with_capacity(diagnostics.len());
    for diagnostic in diagnostics {
        // Upstream key: `${type}\0${message}`.
        let key = format!(
            "{}\0{}",
            diagnostic.diagnostic_type.as_str(),
            diagnostic.message
        );
        if seen.insert(key) {
            unique.push(diagnostic);
        }
    }
    unique
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::settings_manager::{SettingsManagerCreateOptions, SettingsStorage};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_dir(tag: &str) -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "pi-settings-diagnostics-{tag}-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    // -- Ported: "includes the settings file path for file-backed storage" ---

    #[test]
    fn includes_the_settings_file_path_for_file_backed_storage() {
        let temp = temp_dir("file-backed");
        let agent_dir = temp.join("agent");
        let settings_path = agent_dir.join("settings.json");
        std::fs::create_dir_all(&agent_dir).expect("create agent dir");
        std::fs::write(&settings_path, "{").expect("write malformed settings");

        let mut manager = SettingsManager::create(
            temp.to_string_lossy().as_ref(),
            agent_dir.to_string_lossy().as_ref(),
            SettingsManagerCreateOptions::default(),
        );
        let diagnostics = collect_settings_diagnostics(&mut manager);

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].diagnostic_type, DiagnosticType::Warning);
        assert!(
            diagnostics[0].message.starts_with(&format!(
                "Invalid settings file {}: ",
                settings_path.to_string_lossy()
            )),
            "unexpected message: {}",
            diagnostics[0].message
        );

        let _ = std::fs::remove_dir_all(&temp);
    }

    // -- Ported: "falls back to the settings scope for storage without file
    //    paths" ---------------------------------------------------------------

    struct FailingGlobalStorage;

    impl SettingsStorage for FailingGlobalStorage {
        fn with_lock(
            &self,
            scope: SettingsScope,
            f: &mut dyn FnMut(Option<&str>) -> Option<String>,
        ) -> Result<(), String> {
            if scope == SettingsScope::Global {
                return Err("backend failed".to_string());
            }
            let _ = f(None);
            Ok(())
        }
    }

    #[test]
    fn falls_back_to_the_settings_scope_for_storage_without_file_paths() {
        let mut manager = SettingsManager::from_storage(
            Box::new(FailingGlobalStorage),
            SettingsManagerCreateOptions::default(),
        );
        let diagnostics = collect_settings_diagnostics(&mut manager);

        assert_eq!(
            diagnostics,
            vec![AgentSessionRuntimeDiagnostic {
                diagnostic_type: DiagnosticType::Warning,
                message: "Invalid global settings: backend failed".to_string(),
            }]
        );
    }

    // -- Ported: "deduplicates diagnostics by type and message" ---------------

    #[test]
    fn deduplicates_by_type_and_message_preserving_first_occurrence_and_order() {
        let warning = |diagnostic_type: DiagnosticType| AgentSessionRuntimeDiagnostic {
            diagnostic_type,
            message: "Invalid settings file /tmp/settings.json".to_string(),
        };

        let deduplicated = deduplicate_diagnostics(vec![
            warning(DiagnosticType::Warning),
            warning(DiagnosticType::Warning),
            warning(DiagnosticType::Error),
        ]);

        assert_eq!(
            deduplicated,
            vec![
                warning(DiagnosticType::Warning),
                warning(DiagnosticType::Error)
            ]
        );

        // Distinct types with the same message are distinct diagnostics; the
        // key never joins type and message ambiguously.
        let mixed = vec![
            AgentSessionRuntimeDiagnostic {
                diagnostic_type: DiagnosticType::Error,
                message: "a".to_string(),
            },
            AgentSessionRuntimeDiagnostic {
                diagnostic_type: DiagnosticType::Info,
                message: "a".to_string(),
            },
            AgentSessionRuntimeDiagnostic {
                diagnostic_type: DiagnosticType::Error,
                message: "a".to_string(),
            },
        ];
        assert_eq!(deduplicate_diagnostics(mixed).len(), 2);
    }
}
