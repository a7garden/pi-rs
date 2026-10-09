//! `cli/session-picker.ts` — deferred (TUI launcher).
//!
//! **D131:** the upstream module is the TUI session selector for `--resume`
//! (`selectSession`): it builds the startup TUI (`cli/startup-ui.ts`), binds
//! the `KeybindingsManager` (`core/keybindings.ts`, itself deferred — D132)
//! through pi-tui's `setKeybindings`, and hosts the interactive-mode
//! `SessionSelectorComponent`. There is no `pi-tui` crate and interactive
//! mode is excluded from this port, and the file carries no portable logic
//! beyond that wiring, so the whole module defers with the interactive mode.
//! `--resume` reports this deferral as an error at dispatch (D103). The
//! loader signature's data types are already ported in
//! [`crate::core::session_manager`]; only the TUI host is missing.
//!
//! Signature sketch (upstream shape; every name behind the TUI exclusion):
//!
//! ```text
//! /// async (onProgress?: SessionListProgress, signal?: AbortSignal)
//! ///   -> Vec<SessionInfo>   (core/session-manager.ts; types are ported)
//! type SessionsLoader = ...;
//!
//! /// Show TUI session selector; returns the selected session path,
//! /// or None if cancelled.
//! pub async fn select_session(
//! 	current_sessions_loader: SessionsLoader,
//! 	all_sessions_loader: SessionsLoader,
//! 	settings_manager: &SettingsManager,
//! ) -> Option<String>;
//! ```
