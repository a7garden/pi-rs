//! Port of `experimental/services/slash-commands-provider.ts` — the process-local
//! slash command registry and the built-in command facets.
//!
//! Deviations (D144):
//! - Upstream freezes contributed objects; Rust contributions are shared
//!   closure structs, so freezing is implicit.
//! - `register`/`replace` return `Err` instead of throwing; `list()` returns an
//!   owned snapshot instead of a frozen array.
//! - The remote `Models`/`AgentController`/`PresentationPlugins` requirements
//!   of the built-in facet resolve through chord's remote client handles and
//!   JSON-encoded member calls (the upstream `env.use` proxy surface).
//! - The selection dialog (`PresentationUI.select`) is consumed through the
//!   local handle trait; the TUI implementation is deferred with interactive
//!   mode.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::context::Context;
use pi_chord::facets::host::LocalRequirement;
use pi_chord::types::Facet;
use pi_chord::json::JsonValue;
use pi_chord::services::errors::{error, AnyError};
use pi_chord::types::{ServiceClientHandle, Unsubscribe};

use super::agent_controller::AgentOperationResponse;
use super::models::{ModelRef, ModelSummary};
use super::presentation_ui::{PresentationSelectItem, PresentationUiHandle};
use super::slash_commands::{
	ArgumentCompletionsFn, SlashCommandCompletion, SlashCommandContribution, SlashCommandRunFn,
	SlashCommandRunResult,
};

/// Cloneable remote-service accessor lending its `RemoteRequirement` out of a
/// mutex cell for each synchronous handle acquisition (D148: requirements are
/// not `Clone`).
#[derive(Clone)]
pub struct RemoteServiceAccess {
	cell: Arc<Mutex<Option<pi_chord::facets::host::RemoteRequirement>>>,
}

impl RemoteServiceAccess {
	fn new(requirement: pi_chord::facets::host::RemoteRequirement) -> RemoteServiceAccess {
		RemoteServiceAccess { cell: Arc::new(Mutex::new(Some(requirement))) }
	}

	/// Acquire the consumer client handle (access check runs per call).
	fn handle(&self) -> Result<Arc<dyn ServiceClientHandle>, AnyError> {
		let requirement = self
			.cell
			.lock()
			.take()
			.ok_or_else(|| error("Remote service requirement is already in use"))?;
		let acquired = requirement.client();
		*self.cell.lock() = Some(requirement);
		acquired
	}

	/// Invoke one remote member by name.
	pub async fn invoke(
		&self,
		member: &str,
		args: Vec<JsonValue>,
		context: &Context,
	) -> Result<Option<JsonValue>, String> {
		let handle = self.handle().map_err(|handle_error| handle_error.to_string())?;
		handle.invoke(member, args, context).await.map_err(|invoke_error| invoke_error.to_string())
	}

	/// Read one replicated state member's current value.
	pub async fn state_value(&self, member: &str) -> Result<Option<JsonValue>, String> {
		let handle = self.handle().map_err(|handle_error| handle_error.to_string())?;
		let state = handle.state(member).map_err(|state_error| state_error.to_string())?;
		Ok(state.and_then(|state| state.value().cloned()))
	}
}

/// Cloneable local-handle accessor for `PresentationUI` (D148).
#[derive(Clone)]
struct PresentationUiAccess {
	cell: Arc<Mutex<Option<LocalRequirement<PresentationUiHandle>>>>,
}

impl PresentationUiAccess {
	fn new(requirement: LocalRequirement<PresentationUiHandle>) -> PresentationUiAccess {
		PresentationUiAccess { cell: Arc::new(Mutex::new(Some(requirement))) }
	}

	fn with<R>(&self, f: impl FnOnce(&PresentationUiHandle) -> R) -> Result<R, AnyError> {
		let requirement = self
			.cell
			.lock()
			.take()
			.ok_or_else(|| error("PresentationUI requirement is already in use"))?;
		let result = requirement.with(f);
		*self.cell.lock() = Some(requirement);
		result
	}

	fn show_status(&self, message: &str, context: Context) {
		if let Ok(handle) = self.with(|ui| ui.0.clone()) {
			handle.show_status(message, context);
		}
	}

	async fn select(
		&self,
		title: &str,
		items: &[PresentationSelectItem],
		selected_value: Option<&str>,
		context: Context,
	) -> Result<Option<String>, String> {
		let handle = self.with(|ui| ui.0.clone()).map_err(|ui_error| ui_error.to_string())?;
		handle.select(title, items, selected_value, context).await
	}
}

const THINKING_DESCRIPTIONS: &[(pi_ai::types::ModelThinkingLevel, &str)] = &[
	(pi_ai::types::ModelThinkingLevel::Off, "No reasoning"),
	(pi_ai::types::ModelThinkingLevel::Minimal, "Very brief reasoning"),
	(pi_ai::types::ModelThinkingLevel::Low, "Light reasoning"),
	(pi_ai::types::ModelThinkingLevel::Medium, "Moderate reasoning"),
	(pi_ai::types::ModelThinkingLevel::High, "Deep reasoning"),
	(pi_ai::types::ModelThinkingLevel::Xhigh, "Extra-high reasoning"),
	(pi_ai::types::ModelThinkingLevel::Max, "Maximum reasoning"),
];

struct RegisteredSlashCommand {
	command: SlashCommandContribution,
	closed: bool,
}

type CommandListener = Arc<dyn Fn(&SlashCommandSnapshot) + Send + Sync>;

/// Owned `list()` result (upstream frozen array).
#[derive(Clone, Default)]
pub struct SlashCommandSnapshot {
	pub commands: Vec<SlashCommandContribution>,
}

/// Upstream `SlashCommandRegistry` implementing the `SlashCommands` service.
#[derive(Clone, Default)]
pub struct SlashCommandRegistry {
	commands: Arc<Mutex<BTreeMap<String, Vec<Arc<Mutex<RegisteredSlashCommand>>>>>>,
	listeners: Arc<Mutex<Vec<(u64, CommandListener)>>>,
	next_listener_id: Arc<AtomicU64>,
}

impl SlashCommandRegistry {
	pub fn new() -> Arc<SlashCommandRegistry> {
		Arc::new(SlashCommandRegistry::default())
	}

	/// Upstream `register(command)`; `Err` mirrors the duplicate throw.
	pub fn register(&self, command: SlashCommandContribution) -> Result<Unsubscribe, String> {
		validate(&command)?;
		if self.commands.lock().contains_key(&command.name) {
			return Err(format!("Slash command /{} is already registered", command.name));
		}
		Ok(self.add(command))
	}

	/// Upstream `replace(command)` — stage a same-name replacement while the
	/// previous facet generation retires.
	pub fn replace(&self, command: SlashCommandContribution) -> Result<Unsubscribe, String> {
		validate(&command)?;
		Ok(self.add(command))
	}

	/// Upstream `list()`.
	pub fn list(&self) -> SlashCommandSnapshot {
		SlashCommandSnapshot {
			commands: self
				.commands
				.lock()
				.values()
				.filter_map(|entries| entries.first())
				.map(|entry| entry.lock().command.clone())
				.collect(),
		}
	}

	/// Upstream `subscribe(listener)`; the listener fires immediately with the
	/// current list, and the returned disposer removes exactly that listener.
	pub fn subscribe(&self, listener: CommandListener) -> Unsubscribe {
		listener(&self.list());
		let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
		self.listeners.lock().push((id, listener));
		let listeners = Arc::clone(&self.listeners);
		Box::new(move || {
			listeners.lock().retain(|(registered, _)| *registered != id);
		})
	}

	fn add(&self, command: SlashCommandContribution) -> Unsubscribe {
		let entry = Arc::new(Mutex::new(RegisteredSlashCommand { command, closed: false }));
		let name = entry.lock().command.name.clone();
		{
			let mut commands = self.commands.lock();
			commands.entry(name.clone()).or_default().push(Arc::clone(&entry));
		}
		let is_first = {
			let commands = self.commands.lock();
			commands.get(&name).map(|entries| entries.len() == 1).unwrap_or(false)
		};
		if is_first {
			self.publish();
		}
		let registry = self.clone();
		Box::new(move || {
			let mut guard = entry.lock();
			if guard.closed {
				return;
			}
			guard.closed = true;
			drop(guard);
			let mut removed_all = false;
			{
				let mut commands = registry.commands.lock();
				if let Some(entries) = commands.get_mut(&name) {
					let is_front = entries.first().map(|front| Arc::ptr_eq(front, &entry)).unwrap_or(false);
					if !is_front {
						return;
					}
					while entries.first().map(|front| front.lock().closed).unwrap_or(false) {
						entries.remove(0);
					}
					if entries.is_empty() {
						commands.remove(&name);
						removed_all = true;
					}
				}
			}
			let _ = removed_all;
			registry.publish();
		})
	}

	fn publish(&self) {
		let snapshot = self.list();
		for (_, listener) in self.listeners.lock().iter() {
			listener(&snapshot);
		}
	}
}

fn validate(command: &SlashCommandContribution) -> Result<(), String> {
	if !super::slash_commands::is_valid_command_name(&command.name) {
		return Err(format!("Invalid slash command name: {}", command.name));
	}
	Ok(())
}

/// `createSlashCommandsRuntimeFacet(registry?)`.
pub fn create_slash_commands_runtime_facet(registry: Arc<SlashCommandRegistry>) -> Facet {
	pi_chord::api::define_facet(pi_chord::types::Facet::new("@pi/slash-commands-runtime", move |env| {
		let registry = Arc::clone(&registry);
		env.provide_local(&super::slash_commands::SLASH_COMMANDS, registry)?;
		Ok(())
	}))
}

/// Options for [`create_built_in_slash_commands_facet`].
pub struct BuiltInSlashCommandsOptions {
	/// Upstream `reloadPresentationPlugins(data)`.
	pub reload_presentation_plugins:
		Arc<dyn Fn(JsonValue) -> futures_util::future::BoxFuture<'static, Result<(), String>> + Send + Sync>,
}

/// `createBuiltInSlashCommandsFacet(options)` — the model, thinking, compact,
/// and reload commands over the presentation host's services.
pub fn create_built_in_slash_commands_facet(options: BuiltInSlashCommandsOptions) -> Facet {
	let options = Arc::new(options);
	pi_chord::api::define_facet(pi_chord::types::Facet::new("@pi/slash-commands-builtin", move |env| {
		let commands: LocalRequirement<super::slash_commands_provider::SlashCommandRegistry> =
			env.use_local(&super::slash_commands::SLASH_COMMANDS)?;
		let models = env.use_remote(super::models::MODELS.id())?;
		let controller = env.use_remote(super::agent_controller::AGENT_CONTROLLER.id())?;
		let ui = env.use_local(&super::presentation_ui::PRESENTATION_UI)?;
		let presentation_plugins = env.use_remote(super::plugins::PRESENTATION_PLUGINS.id())?;
		let session_plugins: LocalRequirement<super::plugins::SessionPluginsService> =
			env.use_local(&super::plugins::SESSION_PLUGINS)?;
		let models_access = models;
		let models = RemoteServiceAccess::new(models_access);
		let controller = RemoteServiceAccess::new(controller);
		let presentation_plugins = RemoteServiceAccess::new(presentation_plugins);
		let ui = PresentationUiAccess::new(ui);
		let session_plugins = Arc::new(Mutex::new(Some(session_plugins)));
		let options = Arc::clone(&options);
		env.on_activate(Arc::new(move || {
			let model = model_command(models.clone(), ui.clone());
			let thinking = thinking_command(models.clone(), ui.clone());
			let compact = compact_command(controller.clone(), ui.clone());
			let reload = reload_command(
				ui.clone(),
				presentation_plugins.clone(),
				session_plugins.clone(),
				Arc::clone(&options),
			);
			// Upstream stages same-name replacements through `commands.replace`
			// inside `onActivate`; the disposers are owned by the facet.
			for command in [model, thinking, compact, reload] {
				if let Err(replace_error) = commands.with(|registry| registry.replace(command)) {
					return Box::pin(async move {
						Err(replace_error) as Result<(), AnyError>
					});
				}
			}
			Box::pin(async { Ok(()) })
		}))?;
		Ok(())
	}))
}

/// Upstream `modelCommand(models, ui)`.
fn model_command(models: RemoteServiceAccess, ui: PresentationUiAccess) -> SlashCommandContribution {
	SlashCommandContribution {
		name: "model".into(),
		description: Some("Select model".into()),
		argument_hint: Some("<provider/model>".into()),
		get_argument_completions: Some(completions_fn(models.clone())),
		run: run_fn(models, ui),
	}
}

fn completions_fn(models: RemoteServiceAccess) -> ArgumentCompletionsFn {
	Arc::new(move |prefix| {
		let models = models.clone();
		Box::pin(async move {
			let available = read_model_summaries(&models).await?;
			let normalized = prefix.to_lowercase();
			Ok(Some(
				available
					.into_iter()
					.filter(|model| {
						format!("{}/{} {}", model.provider, model.model_id, model.name)
							.to_lowercase()
							.contains(&normalized)
					})
					.map(|model| SlashCommandCompletion {
						value: format!("{}/{}", model.provider, model.model_id),
						label: model.model_id,
						description: Some(model.provider),
					})
					.collect(),
			))
		})
	})
}

fn run_fn(models: RemoteServiceAccess, ui: PresentationUiAccess) -> SlashCommandRunFn {
	Arc::new(move |args, context| {
		let models = models.clone();
		let ui = ui.clone();
		Box::pin(async move {
			let available = read_model_summaries(&models).await?;
			let mut selected = exact_model(&available, &args);
			if !args.is_empty() && selected.is_none() {
				return Err(format!("Unknown model: {args}"));
			}
			if selected.is_none() {
				let configured = read_configuration(&models).await?;
				let items: Vec<PresentationSelectItem> = available
					.iter()
					.map(|model| {
						let is_selected = configured
							.model
							.as_ref()
							.map(|current| {
								current.provider == model.provider && current.model_id == model.model_id
							})
							.unwrap_or(false);
						PresentationSelectItem {
							value: format!("{}/{}", model.provider, model.model_id),
							label: if is_selected {
								format!("{} (selected)", model.name)
							} else {
								model.name.clone()
							},
							description: Some(format!("{}/{}", model.provider, model.model_id)),
						}
					})
					.collect();
				let selected_value = configured
					.model
					.as_ref()
					.map(|current| format!("{}/{}", current.provider, current.model_id));
				let value = ui
					.select("Select model:", &items, selected_value.as_deref(), context.clone())
					.await?;
				let Some(value) = value else {
					return Ok(None);
				};
				selected = exact_model(&available, &value);
				if selected.is_none() {
					return Err(format!("Unknown model: {value}"));
				}
			}
			let selected = selected.expect("selected model");
			let reference = ModelRef {
				provider: selected.provider.clone(),
				model_id: selected.model_id.clone(),
			};
			invoke_method(
				&models,
				super::models_provider::members::SELECT,
				vec![serde_json::to_value(&reference).expect("ModelRef serializes")],
				&context,
			)
			.await?;
			ui.show_status(
				&format!("Selected {}/{}.", selected.provider, selected.model_id),
				context,
			);
			Ok(None)
		})
	})
}

/// Upstream `thinkingCommand(models, ui)`.
fn thinking_command(models: RemoteServiceAccess, ui: PresentationUiAccess) -> SlashCommandContribution {
	SlashCommandContribution {
		name: "thinking".into(),
		description: Some("Set thinking level".into()),
		argument_hint: Some("<level>".into()),
		get_argument_completions: None,
		run: Arc::new(move |args, context| {
			let models = models.clone();
			let ui = ui.clone();
			Box::pin(async move {
				let levels = invoke_method(
					&models,
					super::models_provider::members::GET_THINKING_LEVELS,
					vec![],
					&context,
				)
				.await?;
				let levels: Vec<String> = match levels {
					Some(value) => serde_json::from_value(value).unwrap_or_default(),
					None => Vec::new(),
				};
				let args_lower = args.to_lowercase();
				let mut selected = levels.iter().find(|level| **level == args_lower).cloned();
				if !args.is_empty() && selected.is_none() {
					return Err(format!(
						"Unknown thinking level \"{args}\". Available levels: {}.",
						levels.join(", ")
					));
				}
				if selected.is_none() {
					let current = read_configuration(&models).await?.thinking_level;
					let items: Vec<PresentationSelectItem> = levels
						.iter()
						.map(|level| {
							let description = THINKING_DESCRIPTIONS
								.iter()
								.find(|(candidate, _)| pi_level_matches(*candidate, level))
								.map(|(_, description)| (*description).to_string());
							PresentationSelectItem {
								value: level.clone(),
								label: if pi_level_matches(current, level) {
									format!("{level} (selected)")
								} else {
									level.clone()
								},
								description,
							}
						})
						.collect();
					let selected_value = pi_level_name(&current);
					let value = ui
						.select("Select thinking level:", &items, Some(&selected_value), context.clone())
						.await?;
					let Some(value) = value else {
						return Ok(None);
					};
					selected = levels.iter().find(|level| **level == value).cloned();
					if selected.is_none() {
						return Err(format!("Unknown thinking level: {value}"));
					}
				}
				let selected = selected.expect("selected thinking level");
				invoke_method(
					&models,
					super::models_provider::members::SELECT_THINKING,
					vec![JsonValue::String(selected.clone())],
					&context,
				)
				.await?;
				ui.show_status(&format!("Thinking level: {selected}."), context);
				Ok(None)
			})
		}),
	}
}

/// Upstream `compactCommand(controller, ui)`.
fn compact_command(controller: RemoteServiceAccess, ui: PresentationUiAccess) -> SlashCommandContribution {
	SlashCommandContribution {
		name: "compact".into(),
		description: Some("Manually compact the session context".into()),
		argument_hint: Some("<instructions>".into()),
		get_argument_completions: None,
		run: Arc::new(move |args, context| {
			let controller = controller.clone();
			let ui = ui.clone();
			Box::pin(async move {
				ui.show_status("Compacting…", context.clone());
				let request = super::agent_controller::AgentCompactionRequest {
					custom_instructions: if args.is_empty() { None } else { Some(args) },
				};
				let response = invoke_method(
					&controller,
					"compact",
					vec![serde_json::to_value(&request).expect("AgentCompactionRequest serializes")],
					&context,
				)
				.await?;
				let response: AgentOperationResponse = match response {
					Some(value) => serde_json::from_value(value).map_err(|error| error.to_string())?,
					None => return Ok(None),
				};
				Ok(Some(SlashCommandRunResult::Operation(response)))
			})
		}),
	}
}

fn reload_command(
	ui: PresentationUiAccess,
	presentation_plugins: RemoteServiceAccess,
	session_plugins: Arc<Mutex<Option<LocalRequirement<super::plugins::SessionPluginsService>>>>,
	options: Arc<BuiltInSlashCommandsOptions>,
) -> SlashCommandContribution {
	let requirement_cell = session_plugins;
	SlashCommandContribution {
		name: "reload".into(),
		description: Some("Reload server-selected plugins".into()),
		argument_hint: None,
		get_argument_completions: None,
		run: Arc::new(move |_args, context| {
			let ui = ui.clone();
			let presentation_plugins = presentation_plugins.clone();
			let requirement_cell = Arc::clone(&requirement_cell);
			let options = Arc::clone(&options);
			Box::pin(async move {
				ui.show_status("Reloading plugins…", context.clone());
				let data = invoke_method(
					&presentation_plugins,
					super::plugins::members::RELOAD,
					vec![],
					&context,
				)
				.await?
				.unwrap_or(JsonValue::Null);
				// `LocalRequirement` is not `Clone`; lend it out of the cell for
				// the synchronous handle acquisition, then put it back.
				let requirement = requirement_cell
					.lock()
					.take()
					.ok_or_else(|| "Session plugins requirement is already in use".to_string())?;
				let acquired = requirement.with(|plugins| plugins.clone_handle());
				*requirement_cell.lock() = Some(requirement);
				let plugins = acquired.map_err(|error| error.to_string())?;
				plugins.reload(context.clone()).await?;
				(options.reload_presentation_plugins)(data).await?;
				ui.show_status("Reloaded plugins.", context);
				Ok(None)
			})
		}),
	}
}

/// Upstream `exactModel(models, query)`.
fn exact_model(models: &[ModelSummary], query: &str) -> Option<ModelSummary> {
	if query.is_empty() {
		return None;
	}
	let normalized = query.to_lowercase();
	let matches: Vec<ModelSummary> = models
		.iter()
		.filter(|model| {
			format!("{}/{}", model.provider, model.model_id).to_lowercase() == normalized
				|| model.model_id.to_lowercase() == normalized
		})
		.cloned()
		.collect();
	if matches.len() == 1 {
		matches.into_iter().next()
	} else {
		None
	}
}

/// Read `Models.state.catalog.availableModels` through the remote state view.
async fn read_model_summaries(models: &RemoteServiceAccess) -> Result<Vec<ModelSummary>, String> {
	let value = read_models_state_value(models).await?;
	match value {
		Some(value) => {
			let state: super::models::ModelsState =
				serde_json::from_value(value).map_err(|error| error.to_string())?;
			Ok(state.catalog.available_models)
		}
		None => Ok(Vec::new()),
	}
}

async fn read_configuration(models: &RemoteServiceAccess) -> Result<super::models::ModelsConfiguration, String> {
	let value = read_models_state_value(models).await?;
	match value {
		Some(value) => {
			let state: super::models::ModelsState =
				serde_json::from_value(value).map_err(|error| error.to_string())?;
			Ok(state.configuration)
		}
		None => Ok(super::models::ModelsConfiguration {
			model: None,
			thinking_level: pi_ai::types::ModelThinkingLevel::Off,
		}),
	}
}

async fn read_models_state_value(models: &RemoteServiceAccess) -> Result<Option<JsonValue>, String> {
	models.state_value(super::models_provider::members::STATE).await
}

async fn invoke_method(
	handle: &RemoteServiceAccess,
	member: &str,
	args: Vec<JsonValue>,
	context: &Context,
) -> Result<Option<JsonValue>, String> {
	handle.invoke(member, args, context).await
}

fn pi_level_matches(level: pi_ai::types::ModelThinkingLevel, name: &str) -> bool {
	pi_level_of(name) == Some(level)
}

fn pi_level_name(level: &pi_ai::types::ModelThinkingLevel) -> String {
	serde_json::to_value(level)
		.ok()
		.and_then(|value| value.as_str().map(str::to_string))
		.unwrap_or_else(|| "off".to_string())
}

fn pi_level_of(name: &str) -> Option<pi_ai::types::ModelThinkingLevel> {
	serde_json::from_value(JsonValue::String(name.to_string())).ok()
}

#[cfg(test)]
mod tests {
	use super::*;

	fn command(name: &str) -> SlashCommandContribution {
		SlashCommandContribution {
			name: name.to_string(),
			description: None,
			argument_hint: None,
			get_argument_completions: None,
			run: Arc::new(|_args, _context| Box::pin(async { Ok(None) })),
		}
	}

	#[test]
	fn register_rejects_duplicates_then_replace_stages() {
		let registry = SlashCommandRegistry::new();
		registry.register(command("model")).unwrap();
		assert!(registry.register(command("model")).is_err());
		assert!(registry.replace(command("model")).is_ok());
		assert_eq!(registry.list().commands.len(), 1);
	}

	#[test]
	fn register_rejects_invalid_names() {
		let registry = SlashCommandRegistry::new();
		assert!(registry.register(command("Bad Name")).is_err());
	}

	#[test]
	fn first_disposer_publishes_and_unlists() {
		let registry = SlashCommandRegistry::new();
		let disposer = registry.register(command("model")).unwrap();
		assert_eq!(registry.list().commands.len(), 1);
		disposer();
		assert_eq!(registry.list().commands.len(), 0);
	}

	#[test]
	fn second_generation_replace_publishes_only_when_front_retires() {
		let registry = SlashCommandRegistry::new();
		let first = registry.register(command("model")).unwrap();
		let second = registry.replace(command("model")).unwrap();
		assert_eq!(registry.list().commands.len(), 1);
		// Retiring the non-front generation does not change the list.
		second();
		assert_eq!(registry.list().commands.len(), 1);
		// Retiring the front generation removes both retired entries.
		first();
		assert_eq!(registry.list().commands.len(), 0);
	}

	#[tokio::test]
	async fn subscribe_fires_immediately_and_disposer_removes_only_its_listener() {
		let registry = SlashCommandRegistry::new();
		registry.register(command("model")).unwrap();
		let seen = Arc::new(Mutex::new(0usize));
		let seen_clone = Arc::clone(&seen);
		let disposer = registry.subscribe(Arc::new(move |snapshot| {
			*seen_clone.lock() = snapshot.commands.len();
		}));
		assert_eq!(*seen.lock(), 1);
		disposer();
		let other_seen = Arc::new(Mutex::new(0usize));
		let other_clone = Arc::clone(&other_seen);
		let _other = registry.subscribe(Arc::new(move |snapshot| {
			*other_clone.lock() = snapshot.commands.len();
		}));
		registry.replace(command("thinking")).unwrap();
		assert_eq!(*other_seen.lock(), 2);
		assert_eq!(*seen.lock(), 1, "disposed listener must not fire");
	}

	#[test]
	fn exact_model_matches_full_or_short_query() {
		let models = vec![ModelSummary {
			provider: "p".into(),
			model_id: "m".into(),
			name: "Model".into(),
			reasoning: true,
		}];
		assert_eq!(exact_model(&models, "p/m").unwrap().model_id, "m");
		assert_eq!(exact_model(&models, "M").unwrap().name, "Model");
		assert!(exact_model(&models, "nope").is_none());
		assert!(exact_model(&models, "").is_none());
	}
}
