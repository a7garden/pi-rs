//! `cli/config-selector.ts` — deferred (TUI launcher).
//!
//! **D130:** the upstream module is the TUI launcher for `pi config`
//! (`selectConfig`): it wires `ProcessTerminal` + `TuiMainScreen`
//! (`@earendil-works/pi-tui`) around the interactive-mode
//! `ConfigSelectorComponent` and drives `initTheme`/`stopThemeWatcher`. There
//! is no `pi-tui` crate and interactive mode is excluded from this port, and
//! the file carries no portable data or logic beyond that component wiring
//! (`ConfigSelectorOptions` exists only as the launcher's argument bag, and
//! its `ScopedResolvedPaths` type lives in the excluded interactive
//! component), so the whole module defers with the interactive mode rather
//! than being half-ported. The `pi config` command reports this deferral at
//! dispatch (D104).
//!
//! Signature sketch (upstream shape; every name behind the TUI exclusion):
//!
//! ```text
//! pub struct ConfigSelectorOptions<'a> {
//! 	resolved_paths: ScopedResolvedPaths, // modes/interactive/components/config-selector.ts
//! 	settings_manager: &'a mut SettingsManager,
//! 	cwd: String,
//! 	agent_dir: String,
//! 	write_scope: WriteScope, // "global" | "project"
//! 	project_mode_available: bool,
//! }
//!
//! /// Show TUI config selector and return when closed.
//! pub async fn select_config(options: ConfigSelectorOptions<'_>);
//! ```
