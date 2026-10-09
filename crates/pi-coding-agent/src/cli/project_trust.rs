//! `cli/project-trust.ts` — build a [`ProjectTrustContext`] for trust
//! resolution.
//!
//! The `ui` hooks gate on `hasUI`/mode like upstream: in non-interactive
//! modes (the only modes this port dispatches — interactive/TUI is
//! excluded), `select`/`confirm`/`input` return `None`/`false` and `notify`
//! mirrors upstream's non-interactive `console.error` branch. The existing
//! [`NoopExtensionUi`] provides exactly those semantics, so it is reused.

use std::sync::Arc;

use crate::core::extensions::runner::NoopExtensionUi;
use crate::core::extensions::types::{ExtensionMode, ProjectTrustContext};
use crate::core::settings_manager::SettingsManager;

/// `createProjectTrustContext` — non-interactive shape. `mode` is mapped
/// like upstream (`interactive` → `tui`); `has_ui` is passed through and
/// keeps the upstream gating semantics.
pub fn create_project_trust_context(
	cwd: &str,
	mode: ExtensionMode,
	has_ui: bool,
	_settings_manager: &SettingsManager,
) -> ProjectTrustContext {
	ProjectTrustContext { cwd: cwd.to_string(), mode, has_ui, ui: Arc::new(NoopExtensionUi) }
}
