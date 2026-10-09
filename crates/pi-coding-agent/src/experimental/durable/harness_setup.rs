//! Port of `experimental/durable/harness-setup.ts` — pi's harness assembly:
//! HTTP setup, live harness settings, the coding registry, per-directory
//! execution environments, and the initial-model resolution.
//!
//! Deviations:
//! - **D173 — live settings travel through
//!   [`pi_durable::harness::types::HarnessSettingsProvider`].** Upstream
//!   `createHarnessSettings` returns a `HarnessSettings` object whose fields
//!   are getters read at every use; the erased surface's equivalent live
//!   source is `HarnessOptions.settings_provider`. `stream.timeoutMs` keeps
//!   the upstream `idle === 0 ? 2147483647 : idle` mapping.
//! - [`configure_harness_http`] reuses
//!   [`crate::core::http_dispatcher::apply_http_proxy_settings`] +
//!   [`crate::core::http_dispatcher::configure_http_dispatcher`] (D119); the
//!   idle-timeout getter's parse failure surfaces as this function's error
//!   (upstream `getHttpIdleTimeoutMs()` throws).
//! - `findInitialAgentModel`'s CLI branch resolves through
//!   [`crate::core::model_resolver::resolve_cli_model`] directly so the
//!   failure returns as this function's error, matching the upstream
//!   `Could not resolve model` throw (the crate's `find_initial_model`
//!   prints and exits for that branch). The settings-defaults branch uses
//!   [`crate::core::model_resolver::find_initial_model`]; its
//!   `AgentThinkingLevel` converts to pi-ai's `ModelThinkingLevel` at the
//!   seam ([`agent_thinking_level_to_model`], mirroring `core/sdk.rs`).
//! - `NodeExecutionEnv` maps to [`pi_durable::env::node::NodeExecutionEnv`];
//!   the erased `HarnessOptions.env` is an async builder, so
//!   [`ExecutionEnvs::builder`] adapts the sync per-directory memo to it.
//! - The vacation cluster shares these pieces verbatim (upstream
//!   `vacation/harness-setup.ts` is the same file minus
//!   `ExecutionEnvs`/`createCodingRegistry`); `vacation::harness_setup`
//!   re-exports the shared surface.

use std::collections::HashMap;
use std::sync::Arc;

use pi_chord::context::Context;
use pi_durable::env::node::NodeExecutionEnv;
use pi_durable::env::{ExecutionEnv, NodeEnvOptions};
use pi_durable::harness::registry::{create_registry, RegistryImpl};
use pi_durable::harness::types::Registry;
use pi_durable::harness::types::{
	CompactionPolicyOverrides, ConversationStreamOptions, EnvBuilder, EnvTarget, HarnessSettings,
	HarnessSettingsProvider, ProgressPolicyOverrides, QueueMode, RetryPolicyOverrides,
};
use pi_durable::tasks::BoxFuture;
use pi_agent::types::AgentThinkingLevel;
use pi_ai::types::ModelThinkingLevel;
use serde_json::Value;

use crate::core::http_dispatcher::{
	apply_http_proxy_settings, configure_http_dispatcher, DEFAULT_HTTP_IDLE_TIMEOUT_MS,
};
use crate::core::model_resolver::{
	find_initial_model, resolve_cli_model, FindInitialModelOptions, ResolveCliModelOptions,
};
use crate::core::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::core::settings_manager::{SettingsManager, SettingsManagerCreateOptions};

use super::prompt::create_pi_prompt;

/// The settings handle every cluster shares (the `SharedSettingsManager`
/// shape of the services layer).
pub type SharedSettingsManager = Arc<parking_lot::Mutex<SettingsManager>>;

/// `AgentThinkingLevel` → pi-ai `ModelThinkingLevel` (the `core/sdk.rs`
/// seam; defined here because the sdk helper is crate-private).
pub(crate) fn agent_thinking_level_to_model(level: AgentThinkingLevel) -> ModelThinkingLevel {
	match level {
		AgentThinkingLevel::Off => ModelThinkingLevel::Off,
		AgentThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
		AgentThinkingLevel::Low => ModelThinkingLevel::Low,
		AgentThinkingLevel::Medium => ModelThinkingLevel::Medium,
		AgentThinkingLevel::High => ModelThinkingLevel::High,
		AgentThinkingLevel::Xhigh => ModelThinkingLevel::Xhigh,
		AgentThinkingLevel::Max => ModelThinkingLevel::Max,
	}
}

/// pi's HTTP setup: proxy, idle timeouts, and one dispatcher for fetch.
/// Without it, some provider streams break off (upstream
/// `configureHarnessHttp`).
pub fn configure_harness_http(settings_manager: &SettingsManager) -> Result<(), String> {
	let global_settings = settings_manager.get_global_settings();
	let http_proxy = global_settings.get("httpProxy").and_then(Value::as_str);
	apply_http_proxy_settings(http_proxy);
	configure_http_dispatcher(settings_manager.get_http_idle_timeout_ms()?)?;
	Ok(())
}

/// Harness settings read at every use from pi's settings as loaded at
/// startup (upstream `createHarnessSettings`).
pub fn create_harness_settings(settings_manager: SharedSettingsManager) -> HarnessSettingsProvider {
	Arc::new(move || {
		let settings = settings_manager.lock();
		let (provider_timeout_ms, provider_max_retries, provider_max_retry_delay_ms) =
			settings.get_provider_retry_settings();
		let idle = settings.get_http_idle_timeout_ms().unwrap_or(DEFAULT_HTTP_IDLE_TIMEOUT_MS);
		let (retry_enabled, retry_max_retries, retry_base_delay_ms, retry_max_agent_delay_ms) =
			settings.get_retry_settings();
		let compaction = settings.get_compaction_settings(None).unwrap_or_else(|_| {
			// Upstream `getCompactionSettings()` always resolves through
			// settings defaults; a malformed override falls back to them
			// (built-in `enabled/reserve/keep`).
			(true, 16_384, 20_000)
		});
		HarnessSettings {
			extensions: None,
			stream: Some(ConversationStreamOptions {
				timeout_ms: Some(provider_timeout_ms.unwrap_or(if idle == 0 { 2_147_483_647 } else { idle })),
				max_retry_delay_ms: Some(provider_max_retry_delay_ms),
				max_retries: provider_max_retries.map(|retries| retries as u32),
				..ConversationStreamOptions::default()
			}),
			retry: Some(RetryPolicyOverrides {
				enabled: Some(retry_enabled),
				max_retries: Some(retry_max_retries as u32),
				base_delay_ms: Some(retry_base_delay_ms),
				max_agent_delay_ms: Some(retry_max_agent_delay_ms),
			}),
			compaction: Some(CompactionPolicyOverrides {
				enabled: Some(compaction.0),
				reserve_tokens: Some(compaction.1),
				keep_recent_tokens: Some(compaction.2),
				background_tokens: None,
			}),
			progress: Some(ProgressPolicyOverrides::default()),
			tool_execution: None,
			steering_mode: Some(queue_mode(settings.get_steering_mode())),
			follow_up_mode: Some(queue_mode(settings.get_follow_up_mode())),
		}
	})
}

fn queue_mode(mode: crate::core::settings_manager::SteeringMode) -> QueueMode {
	match mode {
		crate::core::settings_manager::SteeringMode::All => QueueMode::All,
		crate::core::settings_manager::SteeringMode::OneAtATime => QueueMode::OneAtATime,
	}
}

/// A registry with pi's coding tools and system prompt (upstream
/// `createCodingRegistry`).
pub fn create_coding_registry(settings_manager: SharedSettingsManager, cwd: &str) -> RegistryImpl {
	let registry = create_registry();
	registry.install(pi_durable::tools::coding_tools());
	registry.install(create_pi_prompt(settings_manager, cwd.to_string()));
	registry
}

/// One execution environment per directory, shared by every conversation in
/// it (upstream `ExecutionEnvs`).
pub struct ExecutionEnvs {
	default_cwd: String,
	envs: parking_lot::Mutex<HashMap<String, Arc<dyn ExecutionEnv>>>,
}

impl ExecutionEnvs {
	pub fn new(default_cwd: impl Into<String>) -> Self {
		Self { default_cwd: default_cwd.into(), envs: parking_lot::Mutex::new(HashMap::new()) }
	}

	/// Upstream `env({ cwd = this.#defaultCwd })`; the erased
	/// `HarnessOptions.env` form.
	pub fn builder(self: &Arc<Self>) -> EnvBuilder {
		let envs = Arc::clone(self);
		Arc::new(move |target: EnvTarget, _context: Context| {
			let envs = Arc::clone(&envs);
			Box::pin(async move { Ok(Some(envs.env_for(target))) }) as BoxFuture<Result<Option<Arc<dyn ExecutionEnv>>, String>>
		})
	}

	fn env_for(self: &Arc<Self>, target: EnvTarget) -> Arc<dyn ExecutionEnv> {
		let cwd = target.cwd.unwrap_or_else(|| self.default_cwd.clone());
		if let Some(found) = self.envs.lock().get(&cwd) {
			return Arc::clone(found);
		}
		let env: Arc<dyn ExecutionEnv> = Arc::new(NodeExecutionEnv::new(NodeEnvOptions {
			cwd: cwd.clone(),
			..NodeEnvOptions::default()
		}));
		self.envs.lock().insert(cwd, Arc::clone(&env));
		env
	}

	/// Clean up every environment created for this cluster (upstream
	/// `cleanup`).
	pub async fn cleanup(&self, context: &Context) {
		let envs: Vec<Arc<dyn ExecutionEnv>> = self.envs.lock().drain().map(|(_, env)| env).collect();
		for env in envs {
			// `cleanup` exists on both `FileSystem` and `Shell`; upstream's
			// `env.cleanup` is the `FileSystem` one.
			pi_durable::env::FileSystem::cleanup(env.as_ref(), context).await;
		}
	}
}

/// Upstream `InitialModel`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InitialModel {
	pub model: Option<pi_durable::harness::types::ModelRef>,
	pub thinking_level: Option<ModelThinkingLevel>,
	pub fallback_message: Option<String>,
}

/// The `--provider`/`--model` argument shape (upstream the optional `cli`
/// parameter).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliModel {
	pub provider: Option<String>,
	pub model: String,
}

/// The model a new root conversation starts with: an explicit
/// `--provider`/`--model`, or pi's default resolution (upstream
/// `findInitialAgentModel`).
pub fn find_initial_agent_model(
	settings_manager: &SettingsManager,
	model_runtime: &ModelRuntime,
	cli: Option<CliModel>,
) -> Result<InitialModel, String> {
	if let Some(cli) = cli {
		let resolved = resolve_cli_model(ResolveCliModelOptions {
			cli_provider: cli.provider.as_deref(),
			cli_model: Some(&cli.model),
			cli_thinking: None,
			model_runtime,
		});
		if resolved.model.is_none() {
			return Err(format!(
				"Could not resolve model: {}",
				resolved.error.unwrap_or_else(|| cli.model.clone())
			));
		}
		let model = resolved.model.expect("checked Some above");
		return Ok(InitialModel {
			model: Some(pi_durable::harness::types::ModelRef {
				provider: model.base.provider.clone(),
				model_id: model.base.id.clone(),
			}),
			thinking_level: resolved.thinking_level.map(agent_thinking_level_to_model),
			fallback_message: None,
		});
	}
	let initial = find_initial_model(FindInitialModelOptions {
		cli_provider: None,
		cli_model: None,
		scoped_models: &[],
		is_continuing: false,
		default_provider: settings_manager.get_default_provider().as_deref(),
		default_model_id: settings_manager.get_default_model().as_deref(),
		default_thinking_level: settings_manager.get_default_thinking_level(),
		model_thinking_levels: None,
		model_runtime,
	});
	Ok(InitialModel {
		model: initial.model.as_ref().map(|model| pi_durable::harness::types::ModelRef {
			provider: model.base.provider.clone(),
			model_id: model.base.id.clone(),
		}),
		thinking_level: if initial.model.is_some() {
			Some(agent_thinking_level_to_model(initial.thinking_level))
		} else {
			None
		},
		fallback_message: initial.fallback_message,
	})
}

/// `ModelRuntime.create()` for the clusters (upstream the bare await; the
/// default options mirror upstream's `create()` defaults).
pub async fn create_model_runtime() -> Result<Arc<ModelRuntime>, String> {
	ModelRuntime::create(CreateModelRuntimeOptions::default())
		.await
		.map_err(|thrown| crate::core::agent_session_services::thrown_message(&thrown))
}

/// `SettingsManager.create(cwd)` for the clusters (the port's constructor
/// also takes the agent dir; the crate default applies).
pub fn create_settings_manager(cwd: &str) -> SharedSettingsManager {
	Arc::new(parking_lot::Mutex::new(SettingsManager::create(
		cwd,
		&crate::config::get_agent_dir(),
		SettingsManagerCreateOptions { project_trusted: None },
	)))
}
