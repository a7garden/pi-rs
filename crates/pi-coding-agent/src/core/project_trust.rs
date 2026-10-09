//! Port of `packages/coding-agent/src/core/project-trust.ts` —
//! [`resolve_project_trusted`]: CLI override → resource probe → extension
//! `project_trust` event → saved decision → `defaultProjectTrust` →
//! interactive prompt.
//!
//! Upstream's exported `AppMode` union is not redeclared here; the crate's
//! `cli::main_dispatch::AppMode` already mirrors it.
//!
//! # Deviations (per-file ledger)
//!
//! - **D115 — extension event context pairing.** The Rust
//!   `emit_project_trust_event` seam threads a full `ExtensionContext` to
//!   handlers (upstream narrows to `ProjectTrustContext`; crate D28), so
//!   [`ResolveProjectTrustedOptions::extensions`] pairs the upstream
//!   `extensionsResult` with that context, while `project_trust_context`
//!   keeps the upstream role: `has_ui` gates the interactive prompt and
//!   `ui.select` renders it. Store failures surface as `Err` (trust-manager
//!   D112) where upstream throws.

use std::sync::Arc;

use crate::config::{APP_NAME, CONFIG_DIR_NAME};
use crate::core::extensions::runner::emit_project_trust_event;
use crate::core::extensions::types::{
	ExtensionContext, LoadExtensionsResult, ProjectTrustContext, ProjectTrustEvent, ProjectTrustEventDecision,
};
use crate::core::settings_manager::DefaultProjectTrust;
use crate::core::trust_manager::{
	get_project_trust_options, has_trust_requiring_project_resources, GetProjectTrustOptionsOptions, ProjectTrustOption,
	ProjectTrustStore,
};

/// Upstream `ResolveProjectTrustedOptions`.
pub struct ResolveProjectTrustedOptions {
	pub cwd: String,
	pub trust_store: Arc<ProjectTrustStore>,
	/// Upstream `trustOverride` (`--approve` / `--no-approve`).
	pub trust_override: Option<bool>,
	/// Upstream `defaultProjectTrust` (absent ⇒ ask).
	pub default_project_trust: Option<DefaultProjectTrust>,
	/// Upstream `extensionsResult`, paired with the runner context the emit
	/// seam requires (D115).
	pub extensions: Option<(LoadExtensionsResult, ExtensionContext)>,
	/// Upstream `projectTrustContext`.
	pub project_trust_context: ProjectTrustContext,
	/// Upstream `onExtensionError`.
	pub on_extension_error: Option<Box<dyn Fn(String) + Send>>,
}

fn format_project_trust_prompt(cwd: &str) -> String {
	format!("Trust project folder?\n{cwd}\n\nThis allows {APP_NAME} to load {CONFIG_DIR_NAME} settings and resources, install missing project packages, and execute project extensions.")
}

async fn select_project_trust_option(cwd: &str, ctx: &ProjectTrustContext) -> Option<ProjectTrustOption> {
	let options = get_project_trust_options(cwd, GetProjectTrustOptionsOptions { include_session_only: true });
	let labels: Vec<String> = options.iter().map(|option| option.label.clone()).collect();
	let selected = ctx.ui.select(format_project_trust_prompt(cwd), labels, None).await;
	options.into_iter().find(|option| selected.as_deref() == Some(option.label.as_str()))
}

fn save_project_trust_prompt_result(
	trust_store: &ProjectTrustStore,
	result: &ProjectTrustOption,
) -> Result<(), String> {
	if result.updates.is_empty() {
		return Ok(());
	}
	trust_store.set_many(&result.updates)
}

/// Upstream `resolveProjectTrusted`.
pub async fn resolve_project_trusted(options: ResolveProjectTrustedOptions) -> Result<bool, String> {
	let ResolveProjectTrustedOptions {
		cwd,
		trust_store,
		trust_override,
		default_project_trust,
		extensions,
		project_trust_context,
		on_extension_error,
	} = options;

	if let Some(trusted) = trust_override {
		return Ok(trusted);
	}
	if !has_trust_requiring_project_resources(&cwd) {
		return Ok(true);
	}

	if let Some((extensions_result, extension_context)) = extensions {
		let event = ProjectTrustEvent { cwd: cwd.clone() };
		let (result, errors) = emit_project_trust_event(&extensions_result, &event, extension_context).await;
		for error in errors {
			if let Some(on_extension_error) = &on_extension_error {
				on_extension_error(format!(
					"Extension \"{}\" project_trust error: {}",
					error.extension_path, error.error
				));
			}
		}
		if let Some(result) = result {
			let trusted = result.trusted == ProjectTrustEventDecision::Yes;
			if result.remember == Some(true) {
				trust_store.set(&cwd, Some(trusted))?;
			}
			return Ok(trusted);
		}
	}

	let decision = trust_store.get(&cwd)?;
	if let Some(decision) = decision {
		return Ok(decision);
	}

	match default_project_trust.unwrap_or(DefaultProjectTrust::Ask) {
		DefaultProjectTrust::Always => return Ok(true),
		DefaultProjectTrust::Never => return Ok(false),
		DefaultProjectTrust::Ask => {}
	}

	if !project_trust_context.has_ui {
		return Ok(false);
	}

	let selected = select_project_trust_option(&cwd, &project_trust_context).await;
	if let Some(selected) = selected {
		save_project_trust_prompt_result(&trust_store, &selected)?;
		return Ok(selected.trusted);
	}
	Ok(false)
}

#[cfg(test)]
mod tests {
	use std::path::PathBuf;
	use std::sync::atomic::{AtomicU32, Ordering};

	use pi_agent::BoxFuture;
	use serde_json::Value;

	use super::*;
	use crate::core::extensions::runner::NoopExtensionUi;
	use crate::core::extensions::types::{
		AutocompleteProviderFactory, CustomComponentFactory, CustomComponentOptions, EditorFactory, ExtensionMode,
		ExtensionUi, ExtensionUiDialogOptions, FooterFactory, HeaderFactory, NotifyKind, SetThemeArg, Theme,
		ThemeInfo, ThemeSwitchResult, TerminalInputHandler, WidgetContent, WorkingIndicatorOptions,
	};
	static COUNTER: AtomicU32 = AtomicU32::new(0);

	/// Unique temp directory per test.
	fn temp_dir(tag: &str) -> PathBuf {
		let id = COUNTER.fetch_add(1, Ordering::SeqCst);
		let dir = std::env::temp_dir().join(format!("pi-project-trust-{tag}-{}-{id}", std::process::id()));
		std::fs::create_dir_all(&dir).expect("create temp dir");
		dir
	}

	/// cwd with `.agents/skills`, so `hasTrustRequiringProjectResources`
	/// fires regardless of the real `$HOME`.
	fn resource_cwd(tag: &str) -> String {
		let cwd = temp_dir(tag);
		std::fs::create_dir_all(cwd.join(".agents").join("skills")).expect("create project skills");
		cwd.to_string_lossy().into_owned()
	}

	fn trust_store(agent_dir: &std::path::Path) -> Arc<ProjectTrustStore> {
		Arc::new(ProjectTrustStore::new(agent_dir.to_string_lossy().as_ref()))
	}

	fn options_for(
		cwd: &str,
		store: Arc<ProjectTrustStore>,
		ctx: ProjectTrustContext,
	) -> ResolveProjectTrustedOptions {
		ResolveProjectTrustedOptions {
			cwd: cwd.to_string(),
			trust_store: store,
			trust_override: None,
			default_project_trust: None,
			extensions: None,
			project_trust_context: ctx,
			on_extension_error: None,
		}
	}

	fn ctx_with_ui(has_ui: bool, ui: Arc<dyn ExtensionUi>) -> ProjectTrustContext {
		ProjectTrustContext { cwd: String::new(), mode: ExtensionMode::Print, has_ui, ui }
	}

	fn non_interactive_ctx() -> ProjectTrustContext {
		ctx_with_ui(false, Arc::new(NoopExtensionUi))
	}

	/// UI whose `select` resolves to a scripted label (or `None`); every
	/// other method is a no-op, mirroring `NoopExtensionUi`.
	struct ScriptedUi {
		choice: std::sync::Mutex<Option<String>>,
		prompt: std::sync::Mutex<Option<String>>,
	}

	impl ScriptedUi {
		fn choosing(label: &str) -> Arc<Self> {
			Arc::new(Self {
				choice: std::sync::Mutex::new(Some(label.to_string())),
				prompt: std::sync::Mutex::new(None),
			})
		}
	}

	impl ExtensionUi for ScriptedUi {
		fn select(
			&self,
			title: String,
			_options: Vec<String>,
			_opts: Option<ExtensionUiDialogOptions>,
		) -> BoxFuture<'static, Option<String>> {
			*self.prompt.lock().expect("lock") = Some(title);
			let choice = self.choice.lock().expect("lock").clone();
			Box::pin(async move { choice })
		}

		fn confirm(
			&self,
			_title: String,
			_message: String,
			_opts: Option<ExtensionUiDialogOptions>,
		) -> BoxFuture<'static, bool> {
			Box::pin(async { false })
		}

		fn input(
			&self,
			_title: String,
			_placeholder: Option<String>,
			_opts: Option<ExtensionUiDialogOptions>,
		) -> BoxFuture<'static, Option<String>> {
			Box::pin(async { None })
		}

		fn notify(&self, _message: String, _kind: Option<NotifyKind>) {}

		fn on_terminal_input(&self, _handler: TerminalInputHandler) -> crate::core::extensions::types::Unsubscribe {
			Box::new(|| {})
		}

		fn set_status(&self, _key: String, _text: Option<String>) {}
		fn set_working_message(&self, _message: Option<String>) {}
		fn set_working_visible(&self, _visible: bool) {}
		fn set_working_indicator(&self, _options: Option<WorkingIndicatorOptions>) {}
		fn set_hidden_thinking_label(&self, _label: Option<String>) {}
		fn set_widget(
			&self,
			_key: String,
			_content: Option<WidgetContent>,
			_options: Option<crate::core::extensions::types::ExtensionWidgetOptions>,
		) {
		}
		fn set_footer(&self, _factory: Option<FooterFactory>) {}
		fn set_header(&self, _factory: Option<HeaderFactory>) {}
		fn set_title(&self, _title: String) {}

		fn custom(
			&self,
			_factory: CustomComponentFactory,
			_options: Option<CustomComponentOptions>,
		) -> BoxFuture<'static, Value> {
			Box::pin(async { Value::Null })
		}

		fn paste_to_editor(&self, _text: String) {}
		fn set_editor_text(&self, _text: String) {}
		fn get_editor_text(&self) -> String {
			String::new()
		}

		fn editor(&self, _title: String, _prefill: Option<String>) -> BoxFuture<'static, Option<String>> {
			Box::pin(async { None })
		}

		fn add_autocomplete_provider(&self, _factory: AutocompleteProviderFactory) {}
		fn set_editor_component(&self, _factory: Option<EditorFactory>) {}
		fn get_editor_component(&self) -> Option<EditorFactory> {
			None
		}
		fn theme(&self) -> Theme {
			Theme
		}
		fn get_all_themes(&self) -> Vec<ThemeInfo> {
			Vec::new()
		}
		fn get_theme(&self, _name: &str) -> Option<Theme> {
			None
		}
		fn set_theme(&self, _theme: SetThemeArg) -> ThemeSwitchResult {
			ThemeSwitchResult { success: false, error: None }
		}
		fn get_tools_expanded(&self) -> bool {
			false
		}
		fn set_tools_expanded(&self, _expanded: bool) {}
	}

	#[tokio::test]
	async fn override_wins_without_probing_resources() {
		let cwd = resource_cwd("override");
		let temp = temp_dir("override-store");
		let store = trust_store(&temp);

		let mut options = options_for(&cwd, store.clone(), non_interactive_ctx());
		options.trust_override = Some(false);
		assert_eq!(resolve_project_trusted(options).await, Ok(false));

		let mut options = options_for(&cwd, store.clone(), non_interactive_ctx());
		options.trust_override = Some(true);
		assert_eq!(resolve_project_trusted(options).await, Ok(true));

		// No decision was persisted by the override path.
		assert_eq!(store.get(&cwd), Ok(None));

		let _ = std::fs::remove_dir_all(&temp);
		let _ = std::fs::remove_dir_all(&cwd);
	}

	#[tokio::test]
	async fn returns_true_without_trust_requiring_resources() {
		let cwd = temp_dir("plain").to_string_lossy().into_owned();
		let temp = temp_dir("plain-store");
		assert_eq!(
			resolve_project_trusted(options_for(&cwd, trust_store(&temp), non_interactive_ctx())).await,
			Ok(true)
		);
		let _ = std::fs::remove_dir_all(&cwd);
		let _ = std::fs::remove_dir_all(&temp);
	}

	#[tokio::test]
	async fn saved_decision_short_circuits_defaults() {
		let cwd = resource_cwd("saved");
		let temp = temp_dir("saved-store");
		let store = trust_store(&temp);

		store.set(&cwd, Some(false)).expect("save false");
		assert_eq!(
			resolve_project_trusted(options_for(&cwd, store.clone(), non_interactive_ctx())).await,
			Ok(false)
		);

		store.set(&cwd, Some(true)).expect("save true");
		assert_eq!(
			resolve_project_trusted(options_for(&cwd, store, non_interactive_ctx())).await,
			Ok(true)
		);

		let _ = std::fs::remove_dir_all(&temp);
		let _ = std::fs::remove_dir_all(&cwd);
	}

	#[tokio::test]
	async fn default_project_trust_gates_before_prompting() {
		let cwd = resource_cwd("default");
		let temp = temp_dir("default-store");

		let mut options = options_for(&cwd, trust_store(&temp), non_interactive_ctx());
		options.default_project_trust = Some(DefaultProjectTrust::Always);
		assert_eq!(resolve_project_trusted(options).await, Ok(true));

		let mut options = options_for(&cwd, trust_store(&temp), non_interactive_ctx());
		options.default_project_trust = Some(DefaultProjectTrust::Never);
		assert_eq!(resolve_project_trusted(options).await, Ok(false));

		// Default is `ask`; no UI ⇒ upstream `return false`.
		assert_eq!(
			resolve_project_trusted(options_for(&cwd, trust_store(&temp), non_interactive_ctx())).await,
			Ok(false)
		);

		let _ = std::fs::remove_dir_all(&temp);
		let _ = std::fs::remove_dir_all(&cwd);
	}

	#[tokio::test]
	async fn interactive_select_persists_and_returns_trusted() {
		let cwd = resource_cwd("select");
		let temp = temp_dir("select-store");

		// Session-only trust: returns true without persisting.
		let store = trust_store(&temp.join("agent1"));
		let ui = ScriptedUi::choosing("Trust (this session only)");
		let result = resolve_project_trusted(options_for(&cwd, store.clone(), ctx_with_ui(true, ui.clone()))).await;
		assert_eq!(result, Ok(true));
		assert_eq!(store.get(&cwd), Ok(None));

		// Session-only refusal: returns false without persisting.
		let store = trust_store(&temp.join("agent2"));
		let ui = ScriptedUi::choosing("Do not trust (this session only)");
		let result = resolve_project_trusted(options_for(&cwd, store.clone(), ctx_with_ui(true, ui.clone()))).await;
		assert_eq!(result, Ok(false));
		assert_eq!(store.get(&cwd), Ok(None));

		// The prompt carries the upstream wording with the raw cwd.
		let prompt = ui.prompt.lock().expect("lock").clone().expect("prompt shown");
		assert!(
			prompt.starts_with(&format!("Trust project folder?\n{cwd}\n\nThis allows pi to load .pi settings")),
			"unexpected prompt: {prompt}"
		);

		// Persisting options write through the store (fresh store per case:
		// a saved decision would short-circuit before the prompt).
		let store = trust_store(&temp.join("agent3"));
		let ui = ScriptedUi::choosing("Trust");
		let result = resolve_project_trusted(options_for(&cwd, store.clone(), ctx_with_ui(true, ui))).await;
		assert_eq!(result, Ok(true));
		assert_eq!(store.get(&cwd), Ok(Some(true)));

		let store = trust_store(&temp.join("agent4"));
		let ui = ScriptedUi::choosing("Do not trust");
		let result = resolve_project_trusted(options_for(&cwd, store.clone(), ctx_with_ui(true, ui))).await;
		assert_eq!(result, Ok(false));
		assert_eq!(store.get(&cwd), Ok(Some(false)));

		// Cancelled selection (`undefined` upstream) falls back to false.
		let store = trust_store(&temp.join("agent5"));
		let ui = Arc::new(ScriptedUi { choice: std::sync::Mutex::new(None), prompt: std::sync::Mutex::new(None) });
		let result = resolve_project_trusted(options_for(&cwd, store, ctx_with_ui(true, ui))).await;
		assert_eq!(result, Ok(false));

		let _ = std::fs::remove_dir_all(&temp);
		let _ = std::fs::remove_dir_all(&cwd);
	}
}
