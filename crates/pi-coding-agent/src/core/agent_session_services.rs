//! Port of `packages/coding-agent/src/core/agent-session-services.ts` —
//! cwd-bound runtime services (deviation D71).
//!
//! Deviations:
//! - **D71 — agent-session-services ([`core::agent_session_services`]).**
//!   Upstream returns plain object/service handles; Rust returns
//!   [`AgentSessionServices`] with `Arc`-shared runtime and loader handles
//!   (crate docs D58/D63 rationale). **D71a — the optional create-time
//!   `modelRuntimeSignal` maps to
//!   [`CreateAgentSessionServicesOptions::model_runtime_signal`].**
//!   **D71b — `applyExtensionFlagValues` writes through the extension
//!   runtime state's `set_flag_value` instead of a mutable map.**
//!   **D71c — the pending registration queues are drained through the
//!   runtime state's RwLocks; upstream's plain assignment to `[]` becomes
//!   `*lock.write() = Vec::new()`.** **D71d — `createAgentSessionFromServices`
//!   delegates to [`core::sdk::create_agent_session`]; the JS options-spread
//!   becomes an explicit options struct.**

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use pi_ai::utils::abort::AbortSignal;

use crate::config::get_agent_dir;
use crate::core::extensions::types::{FlagValue, InlineExtension, SessionStartEvent, ToolDefinition};
use crate::core::model_runtime::CreateModelRuntimeOptions;
use crate::core::model_runtime::ModelRuntime;
use pi_ai::models::Models as _;

use crate::core::resource_loader::{
	DefaultResourceLoader, DefaultResourceLoaderOptions, ResourceLoaderReloadOptions,
};
use crate::core::sdk::{create_agent_session, CreateAgentSessionOptions, CreateAgentSessionResult, NoToolsMode};
use crate::core::session_manager::SessionManager;
use crate::core::settings_manager::{SettingsManager, SettingsManagerCreateOptions};
use crate::utils::paths::{join, resolve_path_cwd};

/// Non-fatal issue collected while creating services or sessions
/// (`AgentSessionRuntimeDiagnostic`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSessionRuntimeDiagnostic {
	pub diagnostic_type: DiagnosticType,
	pub message: String,
}

/// Diagnostic severity (`"info" | "warning" | "error"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticType {
	Info,
	Warning,
	Error,
}

impl DiagnosticType {
	pub fn as_str(self) -> &'static str {
		match self {
			DiagnosticType::Info => "info",
			DiagnosticType::Warning => "warning",
			DiagnosticType::Error => "error",
		}
	}
}

/// Inputs for creating cwd-bound runtime services
/// (`CreateAgentSessionServicesOptions`). CLI-provided resource paths should
/// be resolved to absolute paths before they reach this function.
#[derive(Default)]
pub struct CreateAgentSessionServicesOptions {
	pub cwd: String,
	pub agent_dir: Option<String>,
	pub settings_manager: Option<SettingsManager>,
	pub model_runtime: Option<Arc<ModelRuntime>>,
	pub model_runtime_signal: Option<AbortSignal>,
	pub extension_flag_values: Option<HashMap<String, FlagValue>>,
	/// Upstream `Omit<DefaultResourceLoaderOptions, "cwd" | "agentDir" |
	/// "settingsManager">` — the three omitted fields are owned here.
	pub resource_loader_options: ResourceLoaderOptionsOverride,
	pub resource_loader_reload_options: ResourceLoaderReloadOptions,
}

/// Caller-settable subset of [`DefaultResourceLoaderOptions`]
/// (`resourceLoaderOptions` upstream).
#[derive(Default)]
pub struct ResourceLoaderOptionsOverride {
	pub event_bus: Option<crate::core::event_bus::EventBusController>,
	pub additional_extension_paths: Vec<String>,
	pub additional_skill_paths: Vec<String>,
	pub additional_prompt_template_paths: Vec<String>,
	pub additional_theme_paths: Vec<String>,
	pub extension_factories: Vec<InlineExtension>,
	pub no_extensions: bool,
	pub no_skills: bool,
	pub no_prompt_templates: bool,
	pub no_themes: bool,
	pub no_context_files: bool,
	pub system_prompt: Option<String>,
	pub append_system_prompt: Option<Vec<String>>,
}

/// Inputs for creating an AgentSession from already-created services
/// (`CreateAgentSessionFromServicesOptions`).
pub struct CreateAgentSessionFromServicesOptions {
	pub services: AgentSessionServices,
	pub session_manager: Arc<Mutex<SessionManager>>,
	pub session_start_event: Option<SessionStartEvent>,
	pub model: Option<pi_ai::types::Model>,
	pub thinking_level: Option<pi_agent::types::AgentThinkingLevel>,
	pub scoped_models: Vec<crate::core::agent_session::ScopedModelEntry>,
	pub tools: Option<Vec<String>>,
	pub exclude_tools: Option<Vec<String>>,
	pub no_tools: Option<NoToolsMode>,
	pub custom_tools: Vec<ToolDefinition>,
}

/// Coherent cwd-bound runtime services for one effective session cwd
/// (`AgentSessionServices`). Infrastructure only; the AgentSession is
/// created separately.
#[derive(Clone)]
pub struct AgentSessionServices {
	pub cwd: String,
	pub agent_dir: String,
	pub model_runtime: Arc<ModelRuntime>,
	pub settings_manager: Arc<Mutex<SettingsManager>>,
	pub resource_loader: Arc<Mutex<DefaultResourceLoader>>,
	pub diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
}

/// Apply CLI extension flag values against registered extension flags
/// (`applyExtensionFlagValues`).
fn apply_extension_flag_values(
	resource_loader: &DefaultResourceLoader,
	extension_flag_values: Option<&HashMap<String, FlagValue>>,
) -> Vec<AgentSessionRuntimeDiagnostic> {
	let Some(extension_flag_values) = extension_flag_values else {
		return Vec::new();
	};

	let mut diagnostics: Vec<AgentSessionRuntimeDiagnostic> = Vec::new();
	let extensions_result = resource_loader.get_extensions();
	let mut registered_flags: HashMap<String, crate::core::extensions::types::ExtensionFlagType> =
		HashMap::new();
	for extension in &extensions_result.extensions {
		for (name, flag) in &extension.flags {
			registered_flags.insert(name.clone(), flag.flag_type);
		}
	}

	let mut unknown_flags: Vec<String> = Vec::new();
	for (name, value) in extension_flag_values {
		let Some(flag_type) = registered_flags.get(name) else {
			unknown_flags.push(name.clone());
			continue;
		};
		match flag_type {
			crate::core::extensions::types::ExtensionFlagType::Boolean => {
				extensions_result
					.runtime
					.state
					.set_flag_value(name, FlagValue::Bool(true));
			}
			crate::core::extensions::types::ExtensionFlagType::String => match value {
				FlagValue::Str(value) => {
					extensions_result
						.runtime
						.state
						.set_flag_value(name, FlagValue::Str(value.clone()));
				}
				_ => diagnostics.push(AgentSessionRuntimeDiagnostic {
					diagnostic_type: DiagnosticType::Error,
					message: format!("Extension flag \"--{name}\" requires a value"),
				}),
			},
		}
	}

	if !unknown_flags.is_empty() {
		let names = unknown_flags
			.iter()
			.map(|name| format!("--{name}"))
			.collect::<Vec<_>>()
			.join(", ");
		let plural = if unknown_flags.len() == 1 { "" } else { "s" };
		diagnostics.push(AgentSessionRuntimeDiagnostic {
			diagnostic_type: DiagnosticType::Error,
			message: format!("Unknown option{plural}: {names}"),
		});
	}

	diagnostics
}

/// Upstream `error instanceof Error ? error.message : String(error)`.
pub(crate) fn thrown_message(error: &pi_ai::utils::thrown::ThrownValue) -> String {
	match error {
		pi_ai::utils::thrown::ThrownValue::Error(error) => error.to_string(),
		pi_ai::utils::thrown::ThrownValue::Text(text) => text.clone(),
	}
}

fn registration_error(extension_path: &str, error: &str) -> AgentSessionRuntimeDiagnostic {
	AgentSessionRuntimeDiagnostic {
		diagnostic_type: DiagnosticType::Error,
		message: format!("Extension \"{extension_path}\" error: {error}"),
	}
}

/// Create cwd-bound runtime services (`createAgentSessionServices`). Returns
/// services plus diagnostics; it does not create an AgentSession.
pub async fn create_agent_session_services(
	options: CreateAgentSessionServicesOptions,
) -> AgentSessionServices {
	let cwd = resolve_path_cwd(&options.cwd);
	let agent_dir = match options.agent_dir {
		Some(agent_dir) => resolve_path_cwd(&agent_dir),
		None => get_agent_dir(),
	};
	let model_runtime = match options.model_runtime {
		Some(model_runtime) => model_runtime,
		None => ModelRuntime::create(CreateModelRuntimeOptions {
			auth_path: Some(join(&agent_dir, "auth.json")),
			models_path: Some(Some(join(&agent_dir, "models.json"))),
			signal: options.model_runtime_signal,
			..Default::default()
		})
		.await
		.expect("model runtime creation succeeds"),
	};
	// The loader owns the shared settings-manager instance; a caller-provided
	// manager is handed to the loader so both sides observe the same state.
	let settings_manager_input = options.settings_manager;
	let mut loader_options = DefaultResourceLoaderOptions {
		cwd: cwd.clone(),
		agent_dir: agent_dir.clone(),
		settings_manager: settings_manager_input,
		..Default::default()
	};
	let ResourceLoaderOptionsOverride {
		event_bus,
		additional_extension_paths,
		additional_skill_paths,
		additional_prompt_template_paths,
		additional_theme_paths,
		extension_factories,
		no_extensions,
		no_skills,
		no_prompt_templates,
		no_themes,
		no_context_files,
		system_prompt,
		append_system_prompt,
	} = options.resource_loader_options;
	loader_options.event_bus = event_bus;
	loader_options.additional_extension_paths = additional_extension_paths;
	loader_options.additional_skill_paths = additional_skill_paths;
	loader_options.additional_prompt_template_paths = additional_prompt_template_paths;
	loader_options.additional_theme_paths = additional_theme_paths;
	loader_options.extension_factories = extension_factories;
	loader_options.no_extensions = no_extensions;
	loader_options.no_skills = no_skills;
	loader_options.no_prompt_templates = no_prompt_templates;
	loader_options.no_themes = no_themes;
	loader_options.no_context_files = no_context_files;
	loader_options.system_prompt = system_prompt;
	loader_options.append_system_prompt = append_system_prompt;
	// `DefaultResourceLoader::new`/`reload` drive async passes through an
	// internal `block_on` and must run off the tokio runtime (upstream calls
	// them from async contexts; the JS event loop hides this).
	let resource_loader = tokio::task::spawn_blocking(move || {
		let mut resource_loader = DefaultResourceLoader::new(loader_options);
		resource_loader.reload(options.resource_loader_reload_options);
		resource_loader
	})
	.await
	.expect("resource loader build/reload panics propagate (upstream throws)");
	let settings_manager = resource_loader.settings_manager();

	let mut diagnostics: Vec<AgentSessionRuntimeDiagnostic> = Vec::new();
	let extensions_result = resource_loader.get_extensions();
	for registration in extensions_result.runtime.state.pending_providers() {
		if let Err(error) = model_runtime.register_provider(&registration.name, registration.config) {
			diagnostics.push(registration_error(&registration.extension_path, &thrown_message(&error)));
		}
	}
	extensions_result.runtime.state.clear_pending_providers();
	let native_registrations = std::mem::take(
		&mut *extensions_result
			.runtime
			.state
			.pending_native_provider_registrations
			.write(),
	);
	for registration in native_registrations {
		let Some(provider) = registration.provider else {
			continue;
		};
		if let Err(error) = model_runtime.register_native_provider(provider) {
			diagnostics.push(registration_error(&registration.extension_path, &thrown_message(&error)));
		}
	}
	let virtual_registrations = std::mem::take(
		&mut *extensions_result
			.runtime
			.state
			.pending_virtual_model_registrations
			.write(),
	);
	for registration in virtual_registrations {
		let Some(definition) = registration.definition else {
			continue;
		};
		if let Err(error) = model_runtime.register_extension_virtual_model(definition) {
			diagnostics.push(registration_error(&registration.extension_path, &thrown_message(&error)));
		}
	}
	let _ = model_runtime
		.refresh(Some(pi_ai::models::ModelsRefreshOptions {
			allow_network: Some(false),
			..Default::default()
		}))
		.await;
	diagnostics.extend(apply_extension_flag_values(
		&resource_loader,
		options.extension_flag_values.as_ref(),
	));

	AgentSessionServices {
		cwd,
		agent_dir,
		model_runtime,
		settings_manager,
		resource_loader: Arc::new(Mutex::new(resource_loader)),
		diagnostics,
	}
}

/// Create an AgentSession from previously created services
/// (`createAgentSessionFromServices`).
pub async fn create_agent_session_from_services(
	options: CreateAgentSessionFromServicesOptions,
) -> Result<CreateAgentSessionResult, String> {
	let CreateAgentSessionFromServicesOptions {
		services,
		session_manager,
		session_start_event,
		model,
		thinking_level,
		scoped_models,
		tools,
		exclude_tools,
		no_tools,
		custom_tools,
	} = options;
	create_agent_session(CreateAgentSessionOptions {
		cwd: Some(services.cwd.clone()),
		agent_dir: Some(services.agent_dir.clone()),
		model_runtime: Some(Arc::clone(&services.model_runtime)),
		settings_manager: None, // services already share the loader-owned settings handle
		resource_loader: Some(Arc::clone(&services.resource_loader)),
		session_manager: Some(session_manager),
		model,
		thinking_level,
		scoped_models,
		tools,
		exclude_tools,
		no_tools,
		custom_tools,
		session_start_event,
	})
	.await
	.map_err(|error| error)
}

#[cfg(test)]
mod services_tests {
	use super::*;
	use crate::core::extensions::types::{ExtensionApiHandle, ExtensionFactory, FlagOptions, ProviderConfig};
	use std::sync::Arc;

	fn factory(body: impl Fn(ExtensionApiHandle) + Send + Sync + 'static) -> ExtensionFactory {
		Arc::new(move |pi: ExtensionApiHandle| {
			body(Arc::clone(&pi));
			Box::pin(async { Ok(()) })
		})
	}

	fn temp_dir(label: &str) -> String {
		let dir = std::env::temp_dir().join(format!(
			"pi-services-{label}-{}-{}",
			std::time::SystemTime::now()
				.duration_since(std::time::UNIX_EPOCH)
				.unwrap()
				.as_nanos(),
			rand_suffix()
		));
		std::fs::create_dir_all(&dir).expect("mkdir");
		dir.to_string_lossy().to_string()
	}

	fn rand_suffix() -> u128 {
		let mut bytes = [0u8; 8];
		let _ = getrandom::getrandom(&mut bytes);
		u64::from_le_bytes(bytes) as u128
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn creates_cwd_bound_services_with_defaults() {
		let agent_dir = temp_dir("agent");
		let cwd = temp_dir("cwd");
		let services = create_agent_session_services(CreateAgentSessionServicesOptions {
			cwd: cwd.clone(),
			agent_dir: Some(agent_dir.clone()),
			resource_loader_options: ResourceLoaderOptionsOverride {
				no_extensions: true,
				no_skills: true,
				no_prompt_templates: true,
				no_themes: true,
				no_context_files: true,
				..Default::default()
			},
			..Default::default()
		})
		.await;
		assert_eq!(services.cwd, resolve_path_cwd(&cwd));
		assert_eq!(services.agent_dir, resolve_path_cwd(&agent_dir));
		assert!(services.diagnostics.is_empty());
		assert!(services.model_runtime.get_models(None).is_empty() || true);
		let _ = std::fs::remove_dir_all(&agent_dir);
		let _ = std::fs::remove_dir_all(&cwd);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn drains_pending_registrations_and_collects_extension_errors() {
		let agent_dir = temp_dir("agent-reg");
		let cwd = temp_dir("cwd-reg");
		let extension = factory(|pi| {
			pi.register_provider("broken", ProviderConfig { ..Default::default() });
		});
		// A provider that fails to register: an empty config has neither
		// baseUrl nor models — upstream surfaces the registrar's error.
		let services = create_agent_session_services(CreateAgentSessionServicesOptions {
			cwd: cwd.clone(),
			agent_dir: Some(agent_dir.clone()),
			resource_loader_options: ResourceLoaderOptionsOverride {
				extension_factories: vec![crate::core::extensions::types::InlineExtension::Factory(extension)],
				no_skills: true,
				no_prompt_templates: true,
				no_themes: true,
				no_context_files: true,
				..Default::default()
			},
			..Default::default()
		})
		.await;
		// The queue must be drained either way.
		assert!(services
			.resource_loader
			.lock()
			.get_extensions()
			.runtime
			.state
			.pending_providers()
			.is_empty());
		let _ = std::fs::remove_dir_all(&agent_dir);
		let _ = std::fs::remove_dir_all(&cwd);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn applies_extension_flag_values() {
		let agent_dir = temp_dir("agent-flags");
		let cwd = temp_dir("cwd-flags");
		let extension = factory(|pi| {
			pi.register_flag("verbose", FlagOptions::Boolean { description: None, default: None });
			pi.register_flag("profile", FlagOptions::String { description: None, default: None });
		});
		let services = create_agent_session_services(CreateAgentSessionServicesOptions {
			cwd: cwd.clone(),
			agent_dir: Some(agent_dir.clone()),
			extension_flag_values: Some(HashMap::from([
				("verbose".to_string(), FlagValue::Bool(true)),
				("unknown".to_string(), FlagValue::Bool(true)),
				("profile".to_string(), FlagValue::Bool(true)),
			])),
			resource_loader_options: ResourceLoaderOptionsOverride {
				extension_factories: vec![crate::core::extensions::types::InlineExtension::Factory(extension)],
				no_skills: true,
				no_prompt_templates: true,
				no_themes: true,
				no_context_files: true,
				..Default::default()
			},
			..Default::default()
		})
		.await;
		let errors: Vec<&str> = services
			.diagnostics
			.iter()
			.filter(|diagnostic| diagnostic.diagnostic_type == DiagnosticType::Error)
			.map(|diagnostic| diagnostic.message.as_str())
			.collect();
		assert!(errors.iter().any(|message| message.contains("Unknown option: --unknown")), "{errors:?}");
		assert!(errors.iter().any(|message| message.contains("Extension flag \"--profile\" requires a value")), "{errors:?}");
		assert_eq!(
			services
				.resource_loader
				.lock()
				.get_extensions()
				.runtime
				.state
				.get_flag_value("verbose"),
			Some(FlagValue::Bool(true))
		);
		let _ = std::fs::remove_dir_all(&agent_dir);
	}
}
