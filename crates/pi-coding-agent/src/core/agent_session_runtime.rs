//! Port of `packages/coding-agent/src/core/agent-session-runtime.ts` — the
//! session-lifetime owner around an [`AgentSession`] (deviation D72).
//!
//! Deviations:
//! - **D72 — agent-session-runtime ([`AgentSessionRuntime`],
//!   [`SessionImportFileNotFoundError`], [`create_agent_session_runtime`]).**
//!   Upstream tears the current session down and rebuilds via a factory
//!   closure; the factory is an `Arc`-ed async [`CreateAgentSessionRuntimeFactory`].
//!   **D72a — `SessionImportFileNotFoundError` carries the path and message
//!   (`File not found: <path>`); the ported error implements
//!   [`std::error::Error`].** **D72b — `existsSync`/`copyFileSync` with
//!   `COPYFILE_EXCL` become [`std::path::Path::exists`] and
//!   `std::fs::copy` with create-new semantics; the `name-N` suffix scan
//!   keeps upstream's collision-avoidance loop.** **D72c —
//!   `extractUserMessageText` filters text parts from user message content.**

use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures_util::future::BoxFuture;
use parking_lot::Mutex;

use pi_agent::types::AgentMessage;

use crate::core::agent_session::AgentSession;
use crate::core::agent_session_services::{
	AgentSessionRuntimeDiagnostic, AgentSessionServices,
};
use crate::core::extensions::runner::emit_session_shutdown_event;
use crate::core::extensions::types::{
	ForkPosition, ProjectTrustContext, ReplacedSessionContext, SessionShutdownReason,
	SessionStartEvent, SessionStartReason, SessionSwitchReason,
};
use crate::core::sdk::{CreateAgentSessionResult};
use crate::core::session_cwd::{assert_session_cwd_exists, SessionCwdSource};
use crate::core::session_manager::{NewSessionOptions as SmNewSessionOptions, SessionManager};
use crate::utils::paths::{join, resolve_path_cwd};

/// Result returned by runtime creation
/// (`CreateAgentSessionRuntimeResult`).
pub struct CreateAgentSessionRuntimeResult {
	pub session: Arc<AgentSession>,
	pub services: AgentSessionServices,
	pub diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
	pub model_fallback_message: Option<String>,
}

impl CreateAgentSessionRuntimeResult {
	/// Split a [`CreateAgentSessionResult`] plus its services into the
	/// runtime result shape (mirrors upstream's spread).
	pub fn from_session_result(
		result: CreateAgentSessionResult,
		services: AgentSessionServices,
	) -> Self {
		CreateAgentSessionRuntimeResult {
			model_fallback_message: result.model_fallback_message,
			session: result.session,
			diagnostics: services.diagnostics.clone(),
			services,
		}
	}
}

/// Creates a full runtime for a target cwd and session manager
/// (`CreateAgentSessionRuntimeFactory`). The factory closes over
/// process-global fixed inputs and is reused for later new/resume/fork/import
/// flows.
pub struct CreateAgentSessionRuntimeOptions {
	pub cwd: String,
	pub agent_dir: String,
	pub session_manager: Arc<Mutex<SessionManager>>,
	pub session_start_event: Option<SessionStartEvent>,
	pub project_trust_context: Option<ProjectTrustContext>,
}

pub type CreateAgentSessionRuntimeFactory = Arc<
	dyn Fn(CreateAgentSessionRuntimeOptions) -> BoxFuture<'static, Result<CreateAgentSessionRuntimeResult, String>>
		+ Send
		+ Sync,
>;

/// Thrown when `/import` references a JSONL file path that does not exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionImportFileNotFoundError {
	pub file_path: String,
}

impl SessionImportFileNotFoundError {
	pub fn new(file_path: impl Into<String>) -> Self {
		SessionImportFileNotFoundError { file_path: file_path.into() }
	}
}

impl std::fmt::Display for SessionImportFileNotFoundError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "File not found: {}", self.file_path)
	}
}

impl std::error::Error for SessionImportFileNotFoundError {}

/// User-visible text of a user message (`extractUserMessageText`).
fn extract_user_message_text(content: &AgentMessage) -> String {
	let AgentMessage::Llm(pi_ai::types::Message::User(user)) = content else {
		return String::new();
	};
	match &user.content {
		pi_ai::types::UserContentField::Text(text) => text.clone(),
		pi_ai::types::UserContentField::Blocks(blocks) => blocks
			.iter()
			.filter_map(|block| match block {
				pi_ai::types::UserContent::Text(text) => Some(text.text.clone()),
				_ => None,
			})
			.collect(),
	}
}

/// Why the current session is being replaced (`SessionReplacementReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplacementReason {
	Resume,
	New,
	Fork,
}

impl ReplacementReason {
	fn shutdown_reason(self) -> SessionShutdownReason {
		match self {
			ReplacementReason::Resume => SessionShutdownReason::Resume,
			ReplacementReason::New => SessionShutdownReason::New,
			ReplacementReason::Fork => SessionShutdownReason::Fork,
		}
	}

	fn start_reason(self) -> SessionStartReason {
		match self {
			ReplacementReason::Resume => SessionStartReason::Resume,
			ReplacementReason::New => SessionStartReason::New,
			ReplacementReason::Fork => SessionStartReason::Fork,
		}
	}
}

/// Outcome of a cancelled-or-completed switch (`{ cancelled: boolean }`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwitchOutcome {
	pub cancelled: bool,
}

/// Outcome of a fork (`{ cancelled, selectedText? }`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ForkOutcome {
	pub cancelled: bool,
	pub selected_text: Option<String>,
}

/// Options for [`AgentSessionRuntime::switch_session`].
pub struct SwitchSessionOptions {
	pub cwd_override: Option<String>,
	pub with_session: Option<Box<dyn FnOnce(ReplacedSessionContext) -> BoxFuture<'static, ()> + Send>>,
	pub project_trust_context_factory:
		Option<Box<dyn Fn(&str) -> ProjectTrustContext + Send>>,
}

/// Options for [`AgentSessionRuntime::new_session`].
pub struct NewSessionOptions {
	pub parent_session: Option<String>,
	pub setup: Option<Box<dyn FnOnce(&mut SessionManager) -> BoxFuture<'static, ()> + Send>>,
	pub with_session: Option<Box<dyn FnOnce(ReplacedSessionContext) -> BoxFuture<'static, ()> + Send>>,
}

/// Options for [`AgentSessionRuntime::fork`].
pub struct ForkOptions {
	pub position: Option<ForkPosition>,
	pub with_session: Option<Box<dyn FnOnce(ReplacedSessionContext) -> BoxFuture<'static, ()> + Send>>,
}

/// Owns the current AgentSession plus its cwd-bound services
/// (`AgentSessionRuntime`). Session replacement methods tear the current
/// runtime down first, then create and apply the next runtime.
pub struct AgentSessionRuntime {
	rebind_session: Option<Arc<dyn Fn(&AgentSession) -> BoxFuture<'static, ()> + Send + Sync>>,
	before_session_invalidate: Option<Box<dyn Fn() + Send>>,
	session: Arc<AgentSession>,
	services: AgentSessionServices,
	create_runtime: CreateAgentSessionRuntimeFactory,
	diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
	model_fallback_message: Option<String>,
}

impl AgentSessionRuntime {
	pub fn new(
		session: Arc<AgentSession>,
		services: AgentSessionServices,
		create_runtime: CreateAgentSessionRuntimeFactory,
		diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
		model_fallback_message: Option<String>,
	) -> Self {
		AgentSessionRuntime {
			rebind_session: None,
			before_session_invalidate: None,
			session,
			services,
			create_runtime,
			diagnostics,
			model_fallback_message,
		}
	}

	pub fn services(&self) -> &AgentSessionServices {
		&self.services
	}

	pub fn session(&self) -> &AgentSession {
		&self.session
	}

	pub fn cwd(&self) -> &str {
		&self.services.cwd
	}

	pub fn diagnostics(&self) -> &[AgentSessionRuntimeDiagnostic] {
		&self.diagnostics
	}

	pub fn model_fallback_message(&self) -> Option<&str> {
		self.model_fallback_message.as_deref()
	}

	/// `setRebindSession`.
	pub fn set_rebind_session(
		&mut self,
		rebind_session: Option<Arc<dyn Fn(&AgentSession) -> BoxFuture<'static, ()> + Send + Sync>>,
	) {
		self.rebind_session = rebind_session;
	}

	/// `setBeforeSessionInvalidate`: a synchronous callback that runs after
	/// `session_shutdown` handlers finish but before the current session is
	/// invalidated.
	pub fn set_before_session_invalidate(&mut self, before_session_invalidate: Option<Box<dyn Fn() + Send>>) {
		self.before_session_invalidate = before_session_invalidate;
	}

	async fn emit_before_switch(
		&self,
		reason: SessionSwitchReason,
		target_session_file: Option<&str>,
	) -> SwitchOutcome {
		// `session_before_*` events dispatch through the typed runner methods
		// (the generic `emit` path filters them, runner.rs
		// `is_session_before_event`).
		let runner = self.session.extension_runner();
		if !runner.has_handlers("session_before_switch") {
			return SwitchOutcome { cancelled: false };
		}
		let result = runner
			.emit_session_before_switch(&crate::core::extensions::types::SessionBeforeSwitchEvent {
				reason,
				target_session_file: target_session_file.map(|file| file.to_string()),
			})
			.await;
		SwitchOutcome {
			cancelled: result.and_then(|result| result.cancel) == Some(true),
		}
	}

	async fn emit_before_fork(&self, entry_id: &str, position: ForkPosition) -> SwitchOutcome {
		let runner = self.session.extension_runner();
		if !runner.has_handlers("session_before_fork") {
			return SwitchOutcome { cancelled: false };
		}
		let result = runner
			.emit_session_before_fork(&crate::core::extensions::types::SessionBeforeForkEvent {
				entry_id: entry_id.to_string(),
				position,
			})
			.await;
		SwitchOutcome {
			cancelled: result.and_then(|result| result.cancel) == Some(true),
		}
	}

	/// Settle any active response first so the aborted turn (including tool
	/// results) is persisted to the outgoing session before it is replaced
	/// (`teardownCurrent`).
	async fn teardown_current(&self, reason: ReplacementReason, target_session_file: Option<&str>) {
		let _ = self.session.abort().await;
		emit_session_shutdown_event(
			self.session.extension_runner().as_ref(),
			&crate::core::extensions::types::SessionShutdownEvent {
				reason: reason.shutdown_reason(),
				target_session_file: target_session_file.map(|file| file.to_string()),
			},
		)
		.await;
		if let Some(before_session_invalidate) = &self.before_session_invalidate {
			before_session_invalidate();
		}
		self.session.dispose();
	}

	fn apply(&mut self, result: CreateAgentSessionRuntimeResult) {
		let CreateAgentSessionRuntimeResult {
			session,
			services,
			diagnostics,
			model_fallback_message,
		} = result;
		self.session = session;
		self.services = services;
		self.diagnostics = diagnostics;
		self.model_fallback_message = model_fallback_message;
	}

	async fn finish_session_replacement(
		&self,
		with_session: Option<Box<dyn FnOnce(ReplacedSessionContext) -> BoxFuture<'static, ()> + Send>>,
	) -> Result<(), String> {
		if let Some(rebind_session) = &self.rebind_session {
			rebind_session(self.session_ref()).await;
		}
		if let Some(with_session) = with_session {
			let ctx = self
				.session
				.create_replaced_session_context()
				.map_err(|error| error.to_string())?;
			with_session(ctx).await;
		}
		Ok(())
	}

	fn session_ref(&self) -> &AgentSession {
		&self.session
	}

	/// Switch to an existing session file (`switchSession`).
	pub async fn switch_session(
		&mut self,
		session_path: &str,
		options: Option<SwitchSessionOptions>,
	) -> Result<SwitchOutcome, String> {
		let options = options.unwrap_or(SwitchSessionOptions {
			cwd_override: None,
			with_session: None,
			project_trust_context_factory: None,
		});
		let before_result = self
			.emit_before_switch(SessionSwitchReason::Resume, Some(session_path))
			.await;
		if before_result.cancelled {
			return Ok(before_result);
		}

		let previous_session_file = self.session.session_file().map(|file| file.to_string());
		let session_manager = Arc::new(Mutex::new(SessionManager::open(
			session_path,
			None,
			options.cwd_override.as_deref(),
		)
		.map_err(|error| error.to_string())?));
		{
			let guard = session_manager.lock();
			assert_session_cwd_exists(&*guard, self.cwd()).map_err(|error| error.to_string())?;
		}
		let new_session_file = session_manager.lock().get_session_file().map(|file| file.to_string());
		let next_cwd = session_manager.lock().get_cwd().to_string();
		self.teardown_current(ReplacementReason::Resume, new_session_file.as_deref()).await;
		let (project_trust_context, with_session) = match options.project_trust_context_factory {
			Some(factory) => (Some(factory(&next_cwd)), options.with_session),
			None => (None, options.with_session),
		};
		let result = (self.create_runtime.clone())(CreateAgentSessionRuntimeOptions {
			cwd: next_cwd,
			agent_dir: self.services.agent_dir.clone(),
			session_manager,
			session_start_event: Some(SessionStartEvent {
				reason: SessionStartReason::Resume,
				previous_session_file,
			}),
			project_trust_context,
		})
		.await?;
		self.apply(result);
		self.finish_session_replacement(with_session).await;
		Ok(SwitchOutcome { cancelled: false })
	}

	/// Start a fresh session (`newSession`).
	pub async fn new_session(&mut self, options: Option<NewSessionOptions>) -> Result<SwitchOutcome, String> {
		let NewSessionOptions {
			parent_session,
			setup,
			with_session,
		} = options.unwrap_or(NewSessionOptions {
			parent_session: None,
			setup: None,
			with_session: None,
		});
		let before_result = self.emit_before_switch(SessionSwitchReason::New, None).await;
		if before_result.cancelled {
			return Ok(before_result);
		}

		let previous_session_file = self.session.session_file().map(|file| file.to_string());
		let session_dir = self.session.session_manager().lock().get_session_dir();
		let session_manager = Arc::new(Mutex::new(if self.session.session_manager().lock().is_persisted() {
			SessionManager::create(self.cwd(), Some(&session_dir), None)
				.map_err(|error| error.to_string())?
		} else {
			SessionManager::in_memory(Some(self.cwd()), None, None)
				.map_err(|error| error.to_string())?
		}));
		if let Some(parent_session) = &parent_session {
			session_manager
				.lock()
				.new_session(Some(&SmNewSessionOptions {
					id: None,
					parent_session: Some(parent_session.clone()),
				}))
				.map_err(|error| error.to_string())?;
		}

		let new_session_file = session_manager.lock().get_session_file().map(|file| file.to_string());
		self.teardown_current(ReplacementReason::New, new_session_file.as_deref()).await;
		let result = (self.create_runtime.clone())(CreateAgentSessionRuntimeOptions {
			cwd: self.cwd().to_string(),
			agent_dir: self.services.agent_dir.clone(),
			session_manager,
			session_start_event: Some(SessionStartEvent {
				reason: SessionStartReason::New,
				previous_session_file,
			}),
			project_trust_context: None,
		})
		.await?;
		self.apply(result);
		if let Some(setup) = setup {
			let session_manager = self.session.session_manager();
			let mut guard = session_manager.lock();
			setup(&mut guard).await;
			drop(guard);
			self.session.refresh_context();
		}
		self.finish_session_replacement(with_session).await;
		Ok(SwitchOutcome { cancelled: false })
	}

	/// Fork from a transcript entry (`fork`).
	pub async fn fork(&mut self, entry_id: &str, options: Option<ForkOptions>) -> Result<ForkOutcome, String> {
		let ForkOptions {
			position,
			with_session,
		} = options.unwrap_or(ForkOptions {
			position: None,
			with_session: None,
		});
		let position = position.unwrap_or(ForkPosition::Before);
		let before_result = self.emit_before_fork(entry_id, position).await;
		if before_result.cancelled {
			return Ok(ForkOutcome {
				cancelled: true,
				selected_text: None,
			});
		}

		let selected_entry = {
			let guard = self.session.session_manager().lock();
			guard
				.get_entry(entry_id)
				.cloned()
				.ok_or_else(|| "Invalid entry ID for forking".to_string())?
		};

		let target_leaf_id: Option<String>;
		let mut selected_text: Option<String> = None;
		if position == ForkPosition::At {
			target_leaf_id = Some(selected_entry.id().to_string());
		} else {
			let crate::core::session_manager::SessionEntry::Message(message_entry) = &selected_entry else {
				return Err("Invalid entry ID for forking".to_string());
			};
			if !matches!(
				message_entry.message,
				AgentMessage::Llm(pi_ai::types::Message::User(_))
			) {
				return Err("Invalid entry ID for forking".to_string());
			}
			target_leaf_id = message_entry.base.parent_id.clone();
			selected_text = Some(extract_user_message_text(&message_entry.message));
		}

		let previous_session_file = self.session.session_file().map(|file| file.to_string());
		if self.session.session_manager().lock().is_persisted() {
			let current_session_file = self
				.session
				.session_file()
				.ok_or_else(|| "Persisted session is missing a session file".to_string())?
				.to_string();
			let session_dir = self.session.session_manager().lock().get_session_dir();
			let Some(target_leaf_id) = target_leaf_id else {
				let session_manager = Arc::new(Mutex::new(SessionManager::create(
					self.cwd(),
					Some(&session_dir),
					None,
				)
				.map_err(|error| error.to_string())?));
				session_manager
					.lock()
					.new_session(Some(&SmNewSessionOptions {
						id: None,
						parent_session: Some(current_session_file),
					}))
					.map_err(|error| error.to_string())?;
				let new_session_file = session_manager.lock().get_session_file().map(|file| file.to_string());
				self.teardown_current(ReplacementReason::Fork, new_session_file.as_deref()).await;
				let result = (self.create_runtime.clone())(CreateAgentSessionRuntimeOptions {
					cwd: self.cwd().to_string(),
					agent_dir: self.services.agent_dir.clone(),
					session_manager,
					session_start_event: Some(SessionStartEvent {
						reason: SessionStartReason::Fork,
						previous_session_file,
					}),
					project_trust_context: None,
				})
				.await?;
				self.apply(result);
				self.finish_session_replacement(with_session).await;
				return Ok(ForkOutcome {
					cancelled: false,
					selected_text,
				});
			};

			if !Path::new(&current_session_file).exists() {
				return Err(
					"This session has not been saved yet. Send a message before cloning or forking it."
						.to_string(),
				);
			}
			let session_manager = Arc::new(Mutex::new(SessionManager::open(
				&current_session_file,
				Some(&session_dir),
				None,
			)
			.map_err(|error| error.to_string())?));
			session_manager
				.lock()
				.create_branched_session(&target_leaf_id)
				.map_err(|error| error.to_string())?
				.ok_or_else(|| "Failed to create forked session".to_string())?;
			let new_session_file = session_manager.lock().get_session_file().map(|file| file.to_string());
			let next_cwd = session_manager.lock().get_cwd().to_string();
			self.teardown_current(ReplacementReason::Fork, new_session_file.as_deref()).await;
			let result = (self.create_runtime.clone())(CreateAgentSessionRuntimeOptions {
				cwd: next_cwd,
				agent_dir: self.services.agent_dir.clone(),
				session_manager,
				session_start_event: Some(SessionStartEvent {
					reason: SessionStartReason::Fork,
					previous_session_file,
				}),
				project_trust_context: None,
			})
			.await?;
			self.apply(result);
			self.finish_session_replacement(with_session).await;
			return Ok(ForkOutcome {
				cancelled: false,
				selected_text,
			});
		}

		// In-memory session: reuse the same manager.
		{
			let mut guard = self.session.session_manager().lock();
			if let Some(target_leaf_id) = &target_leaf_id {
				guard
					.create_branched_session(target_leaf_id)
					.map_err(|error| error.to_string())?;
			} else {
				guard
					.new_session(Some(&SmNewSessionOptions {
						id: None,
						parent_session: previous_session_file.clone(),
					}))
					.map_err(|error| error.to_string())?;
			}
		}
		let session_file = self.session.session_manager().lock().get_session_file().map(|file| file.to_string());
		self.teardown_current(ReplacementReason::Fork, session_file.as_deref()).await;
		let session_manager = Arc::clone(self.session.session_manager());
		let result = (self.create_runtime.clone())(CreateAgentSessionRuntimeOptions {
			cwd: self.cwd().to_string(),
			agent_dir: self.services.agent_dir.clone(),
			session_manager,
			session_start_event: Some(SessionStartEvent {
				reason: SessionStartReason::Fork,
				previous_session_file,
			}),
			project_trust_context: None,
		})
		.await?;
		self.apply(result);
		self.finish_session_replacement(with_session).await;
		Ok(ForkOutcome {
			cancelled: false,
			selected_text,
		})
	}

	/// Import a session JSONL file and switch runtime state to the imported
	/// session (`importFromJsonl`). Returns `cancelled: true` when cancelled
	/// by `session_before_switch`.
	pub async fn import_from_jsonl(
		&mut self,
		input_path: &str,
		cwd_override: Option<&str>,
	) -> Result<SwitchOutcome, String> {
		let resolved_path = resolve_path_cwd(input_path);
		if !Path::new(&resolved_path).exists() {
			return Err(SessionImportFileNotFoundError::new(resolved_path).to_string());
		}

		let session_dir = self.session.session_manager().lock().get_session_dir();
		if !Path::new(&session_dir).exists() {
			std::fs::create_dir_all(&session_dir).map_err(|error| error.to_string())?;
		}

		let mut destination_path: PathBuf = Path::new(&join(&session_dir, &basename(&resolved_path))).to_path_buf();
		let source_already_stored =
			destination_path.canonicalize().ok().map(|path| path.to_string_lossy().to_string())
				== Some(resolved_path.clone());
		if !source_already_stored {
			let name = destination_path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
			let ext = destination_path.extension().map(|s| format!(".{}", s.to_string_lossy())).unwrap_or_default();
			let mut suffix = 1;
			while destination_path.exists() {
				destination_path = Path::new(&session_dir).join(format!("{name}-{suffix}{ext}"));
				suffix += 1;
			}
		}
		let destination_string = destination_path.to_string_lossy().to_string();
		let before_result = self
			.emit_before_switch(SessionSwitchReason::Resume, Some(&destination_string))
			.await;
		if before_result.cancelled {
			return Ok(before_result);
		}

		let previous_session_file = self.session.session_file().map(|file| file.to_string());
		if !source_already_stored {
			copy_file_excl(&resolved_path, &destination_path).map_err(|error| error.to_string())?;
		}

		let session_manager = Arc::new(Mutex::new(SessionManager::open(
			&destination_string,
			Some(&session_dir),
			cwd_override,
		)
		.map_err(|error| error.to_string())?));
		{
			let guard = session_manager.lock();
			assert_session_cwd_exists(&*guard, self.cwd()).map_err(|error| error.to_string())?;
		}
		let new_session_file = session_manager.lock().get_session_file().map(|file| file.to_string());
		let next_cwd = session_manager.lock().get_cwd().to_string();
		self.teardown_current(ReplacementReason::Resume, new_session_file.as_deref()).await;
		let result = (self.create_runtime.clone())(CreateAgentSessionRuntimeOptions {
			cwd: next_cwd,
			agent_dir: self.services.agent_dir.clone(),
			session_manager,
			session_start_event: Some(SessionStartEvent {
				reason: SessionStartReason::Resume,
				previous_session_file,
			}),
			project_trust_context: None,
		})
		.await?;
		self.apply(result);
		self.finish_session_replacement(None).await;
		Ok(SwitchOutcome { cancelled: false })
	}

	/// Tear the runtime down for quit (`dispose`).
	pub async fn dispose(&self) {
		emit_session_shutdown_event(
			self.session.extension_runner().as_ref(),
			&crate::core::extensions::types::SessionShutdownEvent {
				reason: SessionShutdownReason::Quit,
				target_session_file: None,
			},
		)
		.await;
		if let Some(before_session_invalidate) = &self.before_session_invalidate {
			before_session_invalidate();
		}
		self.session.dispose();
	}
}

/// `basename` (upstream `node:path` basename).
fn basename(path: &str) -> String {
	Path::new(path)
		.file_name()
		.map(|name| name.to_string_lossy().to_string())
		.unwrap_or_else(|| path.to_string())
}

/// `copyFileSync(src, dest, COPYFILE_EXCL)`: fail if the destination exists.
fn copy_file_excl(source: &str, destination: &Path) -> std::io::Result<()> {
	let mut options = std::fs::OpenOptions::new();
	options.write(true).create_new(true);
	let mut writer = options.open(destination)?;
	let mut reader = std::fs::File::open(source)?;
	std::io::copy(&mut reader, &mut writer)?;
	Ok(())
}

/// Create the initial runtime from a runtime factory and initial session
/// target (`createAgentSessionRuntime`). The same factory is stored on the
/// returned runtime and reused for later flows.
pub async fn create_agent_session_runtime(
	create_runtime: CreateAgentSessionRuntimeFactory,
	options: CreateAgentSessionRuntimeOptions,
) -> Result<AgentSessionRuntime, String> {
	assert_session_cwd_exists(&*options.session_manager.lock(), &options.cwd)
		.map_err(|error| error.to_string())?;
	let result = create_runtime(options).await?;
	Ok(AgentSessionRuntime::new(
		result.session,
		result.services,
		create_runtime,
		result.diagnostics,
		result.model_fallback_message,
	))
}

#[cfg(test)]
mod runtime_tests {
	use super::*;

	// Port of the SessionImportFileNotFoundError contract (message + path).
	#[test]
	fn import_error_carries_path_and_message() {
		let error = SessionImportFileNotFoundError::new("/tmp/missing.jsonl");
		assert_eq!(error.file_path, "/tmp/missing.jsonl");
		assert_eq!(error.to_string(), "File not found: /tmp/missing.jsonl");
	}

	#[test]
	fn extracts_user_message_text() {
		let message = AgentMessage::Llm(pi_ai::types::Message::User(pi_ai::types::UserMessage {
			content: pi_ai::types::UserContentField::Blocks(vec![
				pi_ai::types::UserContent::Text(pi_ai::types::TextContent {
					text: "hello ".to_string(),
					text_signature: None,
				}),
				pi_ai::types::UserContent::Image(pi_ai::types::ImageContent { data: std::string::String::new(), mime_type: "image/png".to_string() }),
				pi_ai::types::UserContent::Text(pi_ai::types::TextContent {
					text: "world".to_string(),
					text_signature: None,
				}),
			]),
			timestamp: 0,
		}));
		assert_eq!(extract_user_message_text(&message), "hello world");
	}
}


#[cfg(test)]
mod runtime_events_tests {
	//! Port of `test/agent-session-runtime-events.test.ts` — the four
	//! `AgentSessionRuntime` lifecycle cases. Upstream drives
	//! `session.prompt()` against a scripted faux provider; the port keeps
	//! that 1:1 (faux streaming through `ModelRuntime`). JS seams: the faux
	//! provider registers through the pi-ai api registry
	//! (`registerFauxProvider` → `register_faux_provider`); the stale-ctx
	//! assertion (case 3) runs through `catch_unwind` like the runner tests.

	use super::*;
	use crate::core::agent_session::ExtensionBindings;
	use crate::core::agent_session_services::{
		create_agent_session_from_services, CreateAgentSessionFromServicesOptions,
		CreateAgentSessionServicesOptions, ResourceLoaderOptionsOverride,
	};
	use crate::core::extensions::types::{ExtensionApiHandle, ExtensionContext, ExtensionFactory, InlineExtension};
	use crate::core::agent_session_services::create_agent_session_services;
	use crate::core::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
	use pi_ai::auth::types::CredentialStore;
	use futures_util::future::BoxFuture;
	use parking_lot::Mutex as PlMutex;
	use pi_ai::providers::faux::{faux_assistant_message, FauxMessageOptions};
	use std::sync::Arc;

	type RecordedEvents = Arc<PlMutex<Vec<serde_json::Value>>>;

	struct TestRuntime {
		runtime: AgentSessionRuntime,
		temp_dir: String,
	}

	impl Drop for TestRuntime {
		fn drop(&mut self) {
			let _ = std::fs::remove_dir_all(&self.temp_dir);
		}
	}

	fn temp_dir() -> String {
		// Nanosecond timestamps alone collide when two hosts are created in the
		// same tick under parallel load; the process-wide counter makes names
		// unique.
		static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
		let dir = std::env::temp_dir().join(format!(
			"pi-runtime-events-{}-{}-{}",
			std::time::SystemTime::now()
				.duration_since(std::time::UNIX_EPOCH)
				.unwrap()
				.as_nanos(),
			std::process::id(),
			SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
		));
		std::fs::create_dir_all(&dir).expect("mkdir");
		dir.to_string_lossy().to_string()
	}



	fn handler(
		events: RecordedEvents,
		mutate: fn(&RecordedEvents, &serde_json::Value) -> Option<serde_json::Value>,
	) -> crate::core::extensions::types::HandlerFn {
		Arc::new(move |event, _ctx: ExtensionContext| {
			let events = Arc::clone(&events);
			Box::pin(async move { Ok(mutate(&events, &event)) })
		})
	}

	fn push_event(events: &RecordedEvents, event: &serde_json::Value) -> Option<serde_json::Value> {
		events.lock().push(event.clone());
		None
	}

	fn lifecycle_factory(
		events: &RecordedEvents,
		cancel_next_fork: Option<Arc<std::sync::atomic::AtomicBool>>,
	) -> ExtensionFactory {
		let events = Arc::clone(events);
		Arc::new(move |pi: ExtensionApiHandle| {
			pi.on(
				"session_before_switch",
				handler(
					Arc::clone(&events),
					|events, event| {
						events.lock().push(event.clone());
						Some(serde_json::json!({ "cancel": false }))
					},
				),
			);
			pi.on(
				"session_shutdown",
				handler(Arc::clone(&events), push_event),
			);
			pi.on("session_start", handler(Arc::clone(&events), push_event));
			if let Some(cancel_next_fork) = &cancel_next_fork {
				let events = Arc::clone(&events);
				let cancel_next_fork = Arc::clone(cancel_next_fork);
				pi.on(
					"session_before_fork",
					Arc::new(move |event, _ctx: ExtensionContext| {
						let events = Arc::clone(&events);
						let cancel_next_fork = Arc::clone(&cancel_next_fork);
						Box::pin(async move {
							events.lock().push(event.clone());
							let cancel =
								cancel_next_fork.swap(false, std::sync::atomic::Ordering::SeqCst);
							if cancel {
								Ok(Some(serde_json::json!({ "cancel": true })))
							} else {
								Ok(None)
							}
						})
					}),
				);
			}
			Box::pin(async { Ok::<(), String>(()) })
		})
	}

	fn startup_only_factory(events: &RecordedEvents) -> ExtensionFactory {
		let events = Arc::clone(events);
		Arc::new(move |pi: ExtensionApiHandle| {
			pi.on("session_start", handler(Arc::clone(&events), push_event));
			Box::pin(async { Ok::<(), String>(()) })
		})
	}

	/// `createRuntimeHost` equivalent: faux provider, in-memory auth, cwd-bound
	/// services behind the runtime factory.
	async fn create_runtime_host(
		extension_factory: ExtensionFactory,
		_cancel_next_fork: Option<Arc<std::sync::atomic::AtomicBool>>,
	) -> TestRuntime {
		let temp = temp_dir();
		let registration =
			pi_ai::compat::register_faux_provider(pi_ai::providers::faux::RegisterFauxProviderOptions {
				models: vec![pi_ai::providers::faux::FauxModelDefinition {
					id: "faux-main".to_string(),
					..Default::default()
				}],
				..Default::default()
			});
		let model = registration.core.get_model("faux-main").unwrap();
		registration.core.set_responses(
			["one", "two", "three"]
				.iter()
				.map(|text| {
					let mut message = faux_assistant_message(*text, FauxMessageOptions::default());
					message.api = model.api().to_string();
					message.provider = model.provider().to_string();
					message.model = model.id().to_string();
					message.into()
				})
				.collect(),
		);

		let credentials: Arc<dyn CredentialStore> =
			Arc::new(pi_ai::auth::credential_store::InMemoryCredentialStore::default());
		credentials
			.modify(
				model.provider(),
				Box::new(|_current: Option<&pi_ai::auth::types::Credential>| {
					Box::pin(async move {
						Ok(Some(pi_ai::auth::types::Credential::ApiKey(
							pi_ai::auth::types::ApiKeyCredential {
								key: Some("faux-key".to_string()),
								env: None,
							},
						)))
					})
				}),
				None,
			)
			.await
			.expect("credential preload");
		let model_runtime = ModelRuntime::create(CreateModelRuntimeOptions {
			credentials: Some(credentials),
			models_path: Some(Some(format!("{temp}/models.json"))),
			..Default::default()
		})
		.await
		.expect("model runtime");
		model_runtime
			.register_provider(
				model.provider(),
				crate::core::extensions::types::ProviderConfig {
					base_url: Some(model.base.base_url.clone()),
					api: Some(model.api().to_string()),
					models: Some(vec![crate::core::extensions::types::ProviderModelConfig::Chat(
						crate::core::extensions::types::ProviderChatModelConfig {
							base: crate::core::extensions::types::ProviderModelConfigBase {
								id: model.id().to_string(),
								name: model.base.name.clone(),
								input: vec![pi_ai::types::ModelInput::Text],
								..Default::default()
							},
							model_type: None,
							api: Some(model.api().to_string()),
							reasoning: model.reasoning,
							thinking_level_map: None,
							prompt_cache: None,
							context_window: model.context_window,
							max_tokens: model.max_tokens,
							sampling_params: None,
							compat: None,
						},
					)]),
					..Default::default()
				},
			)
			.expect("faux provider registers");

		let factory_model = model.clone();
		let factory_temp = temp.clone();
		let create_runtime: CreateAgentSessionRuntimeFactory = Arc::new(move |options| {
			let model = factory_model.clone();
			let temp = factory_temp.clone();
			let extension_factory = Arc::clone(&extension_factory);
			let model_runtime = Arc::clone(&model_runtime);
			Box::pin(async move {
				// Fresh inline-factory load per runtime (upstream
				// createAgentSessionServices builds a new loader per call; the
				// port's loader cache keys inline factories, so clear before
				// each cwd-bound services build). The gate serializes against
				// the loader module's cache-reading tests.
				{
					let _cache_gate =
						crate::core::extensions::loader::cache_test_gate();
					crate::core::extensions::loader::clear_extension_cache();
				}
				let services = create_agent_session_services(CreateAgentSessionServicesOptions {
					cwd: options.cwd.clone(),
					agent_dir: Some(temp),
					model_runtime: Some(model_runtime),
					resource_loader_options: ResourceLoaderOptionsOverride {
						extension_factories: vec![InlineExtension::Factory(Arc::clone(&extension_factory))],
						no_skills: true,
						no_prompt_templates: true,
						no_themes: true,
						no_context_files: true,
						..Default::default()
					},
					..Default::default()
				})
				.await;
				let result = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
					services: services.clone(),
					session_manager: Arc::clone(&options.session_manager),
					session_start_event: options.session_start_event,
					model: Some(model),
					thinking_level: None,
					scoped_models: Vec::new(),
					tools: None,
					exclude_tools: None,
					no_tools: None,
					custom_tools: Vec::new(),
				})
				.await
				.expect("session creation");
				Ok(CreateAgentSessionRuntimeResult::from_session_result(result, services))
			})
		});

		let session_manager =
			SessionManager::create(&temp, None, None).expect("session manager");
		let runtime = create_agent_session_runtime(
			create_runtime,
			CreateAgentSessionRuntimeOptions {
				cwd: temp.clone(),
				agent_dir: temp.clone(),
				session_manager: Arc::new(PlMutex::new(session_manager)),
				session_start_event: None,
				project_trust_context: None,
			},
		)
		.await
		.expect("runtime host");
		runtime
			.session()
			.bind_extensions(ExtensionBindings {
				ui_context: None,
				mode: None,
				command_context_actions: None,
				abort_handler: None,
				shutdown_handler: None,
				on_error: None,
			})
			.await
			.expect("extensions bind");
		// Keep the faux registration alive for the host lifetime.
		std::mem::forget(registration);
		TestRuntime { runtime, temp_dir: temp }
	}

	fn event_type(event: &serde_json::Value) -> &str {
		event.get("type").and_then(serde_json::Value::as_str).unwrap_or_default()
	}

	fn reason(event: &serde_json::Value) -> &str {
		event.get("reason").and_then(serde_json::Value::as_str).unwrap_or_default()
	}

	// Case 1: "emits session_before_switch and session_start for new and
	// resume flows".
	#[tokio::test(flavor = "multi_thread")]
	async fn emits_switch_and_start_events_for_new_and_resume() {
		let events: RecordedEvents = Arc::new(PlMutex::new(Vec::new()));
		let mut host = create_runtime_host(lifecycle_factory(&events, None), None).await;

		assert_eq!(events.lock().len(), 1);
		assert_eq!(event_type(&events.lock()[0]), "session_start");
		assert_eq!(reason(&events.lock()[0]), "startup");
		events.lock().clear();

		host.runtime.session().prompt("hello", None).await.expect("prompt");
		let original_session_file = host
			.runtime
			.session()
			.session_file()
			.expect("persisted session file");

		let new_session_result = host.runtime.new_session(None).await.expect("new session");
		assert!(!new_session_result.cancelled);
		host.runtime
			.session()
			.bind_extensions(ExtensionBindings {
				ui_context: None,
				mode: None,
				command_context_actions: None,
				abort_handler: None,
				shutdown_handler: None,
				on_error: None,
			})
			.await
			.expect("rebind");
		let second_session_file = host
			.runtime
			.session()
			.session_file()
			.expect("second session file");
		{
			let events = events.lock();
			assert_eq!(events.len(), 3, "{events:?}");
			assert_eq!(event_type(&events[0]), "session_before_switch");
			assert_eq!(reason(&events[0]), "new");
			assert!(
				events[0]
					.get("targetSessionFile")
					.map(serde_json::Value::is_null)
					.unwrap_or(true)
			);
			assert_eq!(event_type(&events[1]), "session_shutdown");
			assert_eq!(reason(&events[1]), "new");
			assert_eq!(
				events[1].get("targetSessionFile").and_then(serde_json::Value::as_str),
				Some(second_session_file.as_str())
			);
			assert_eq!(event_type(&events[2]), "session_start");
			assert_eq!(reason(&events[2]), "new");
			assert_eq!(
				events[2].get("previousSessionFile").and_then(serde_json::Value::as_str),
				Some(original_session_file.as_str())
			);
		}

		events.lock().clear();
		let switch_result = host
			.runtime
			.switch_session(&original_session_file, None)
			.await
			.expect("switch session");
		assert!(!switch_result.cancelled);
		host.runtime
			.session()
			.bind_extensions(ExtensionBindings {
				ui_context: None,
				mode: None,
				command_context_actions: None,
				abort_handler: None,
				shutdown_handler: None,
				on_error: None,
			})
			.await
			.expect("resume rebind");
		let events = events.lock();
		assert_eq!(events.len(), 3, "{events:?}");
		assert_eq!(event_type(&events[0]), "session_before_switch");
		assert_eq!(reason(&events[0]), "resume");
		assert_eq!(
			events[0].get("targetSessionFile").and_then(serde_json::Value::as_str),
			Some(original_session_file.as_str())
		);
		assert_eq!(event_type(&events[1]), "session_shutdown");
		assert_eq!(reason(&events[1]), "resume");
		assert_eq!(event_type(&events[2]), "session_start");
		assert_eq!(reason(&events[2]), "resume");
		assert_eq!(
			events[2].get("previousSessionFile").and_then(serde_json::Value::as_str),
			Some(second_session_file.as_str())
		);
	}

	// Case 2: "honors session_before_switch cancellation".
	#[tokio::test(flavor = "multi_thread")]
	async fn honors_before_switch_cancellation() {
		let events: RecordedEvents = Arc::new(PlMutex::new(Vec::new()));
		let cancel_factory: ExtensionFactory = {
			let events = Arc::clone(&events);
			Arc::new(move |pi: ExtensionApiHandle| {
				let switch_events = Arc::clone(&events);
				pi.on(
					"session_before_switch",
					Arc::new(move |event, _ctx: ExtensionContext| {
						let events = Arc::clone(&switch_events);
						Box::pin(async move {
							events.lock().push(event.clone());
							Ok(Some(serde_json::json!({ "cancel": true })))
						})
					}),
				);
				let start_events = Arc::clone(&events);
				pi.on(
					"session_start",
					Arc::new(move |event, _ctx: ExtensionContext| {
						let events = Arc::clone(&start_events);
						Box::pin(async move {
							events.lock().push(event.clone());
							Ok(None)
						})
					}),
				);
				Box::pin(async { Ok::<(), String>(()) })
			})
		};
		let mut host = create_runtime_host(cancel_factory, None).await;

		assert_eq!(events.lock().len(), 1);
		events.lock().clear();

		host.runtime.session().prompt("hello", None).await.expect("prompt");
		let original_session_file = host
			.runtime
			.session()
			.session_file()
			.expect("session file");

		let result = host.runtime.new_session(None).await.expect("new session");
		assert!(result.cancelled);
		assert_eq!(
			host.runtime.session().session_file(),
			Some(original_session_file.clone())
		);
		let events = events.lock();
		assert_eq!(events.len(), 1, "{events:?}");
		assert_eq!(event_type(&events[0]), "session_before_switch");
		assert_eq!(reason(&events[0]), "new");
		assert!(
				events[0]
					.get("targetSessionFile")
					.map(serde_json::Value::is_null)
					.unwrap_or(true)
			);
	}

	// Case 3: "runs beforeSessionInvalidate after session_shutdown and before
	// rebindSession".
	#[tokio::test(flavor = "multi_thread")]
	async fn orders_shutdown_invalidate_rebind() {
		let phases: Arc<PlMutex<Vec<&'static str>>> = Arc::new(PlMutex::new(Vec::new()));
		let shutdown_factory: ExtensionFactory = {
			let phases = Arc::clone(&phases);
			Arc::new(move |pi: ExtensionApiHandle| {
				let phases = Arc::clone(&phases);
				pi.on(
					"session_shutdown",
					Arc::new(move |_event, _ctx: ExtensionContext| {
						let phases = Arc::clone(&phases);
						Box::pin(async move {
							phases.lock().push("session_shutdown");
							Ok(None)
						})
					}),
				);
				Box::pin(async { Ok::<(), String>(()) })
			})
		};
		let mut host = create_runtime_host(shutdown_factory, None).await;

		let rebind_phases = Arc::clone(&phases);
		let invalidate_phases = Arc::clone(&phases);
		host.runtime.set_before_session_invalidate(Some(Box::new(move || {
			invalidate_phases.lock().push("beforeSessionInvalidate");
		})));
		host.runtime.set_rebind_session(Some(Arc::new(move |_session| {
			let rebind_phases = Arc::clone(&rebind_phases);
			Box::pin(async move {
				rebind_phases.lock().push("rebindSession");
			})
		})));

		host.runtime.new_session(None).await.expect("new session");

		assert_eq!(
			*phases.lock(),
			vec!["session_shutdown", "beforeSessionInvalidate", "rebindSession"]
		);
		host.runtime.set_before_session_invalidate(None);
		host.runtime.set_rebind_session(None);
	}

	// Case 4: "emits session_before_fork and session_start and honors
	// cancellation".
	#[tokio::test(flavor = "multi_thread")]
	async fn fork_events_and_cancellation() {
		let events: RecordedEvents = Arc::new(PlMutex::new(Vec::new()));
		let cancel_next_fork = Arc::new(std::sync::atomic::AtomicBool::new(false));
		let mut host =
			create_runtime_host(lifecycle_factory(&events, Some(Arc::clone(&cancel_next_fork))), Some(Arc::clone(&cancel_next_fork)))
				.await;

		assert_eq!(events.lock().len(), 1);
		events.lock().clear();

		host.runtime.session().prompt("hello", None).await.expect("prompt");
		let user_messages = host.runtime.session().get_user_messages_for_forking();
		assert!(!user_messages.is_empty(), "expected forkable user messages");
		let (entry_id, text) = &user_messages[0];
		assert_eq!(text, "hello");
		let previous_session_file = host
			.runtime
			.session()
			.session_file()
			.expect("session file");

		let success_result = host.runtime.fork(entry_id, None).await.expect("fork");
		assert!(!success_result.cancelled);
		assert_eq!(success_result.selected_text.as_deref(), Some("hello"));
		host.runtime
			.session()
			.bind_extensions(ExtensionBindings {
				ui_context: None,
				mode: None,
				command_context_actions: None,
				abort_handler: None,
				shutdown_handler: None,
				on_error: None,
			})
			.await
			.expect("fork rebind");
		{
			let events = events.lock();
			assert_eq!(events.len(), 3, "{events:?}");
			assert_eq!(event_type(&events[0]), "session_before_fork");
			assert_eq!(events[0].get("entryId").and_then(serde_json::Value::as_str), Some(entry_id.as_str()));
			assert_eq!(events[0].get("position").and_then(serde_json::Value::as_str), Some("before"));
			assert_eq!(event_type(&events[1]), "session_shutdown");
			assert_eq!(reason(&events[1]), "fork");
			assert_eq!(event_type(&events[2]), "session_start");
			assert_eq!(reason(&events[2]), "fork");
			assert_eq!(
				events[2].get("previousSessionFile").and_then(serde_json::Value::as_str),
				Some(previous_session_file.as_str())
			);
		}

		events.lock().clear();
		cancel_next_fork.store(true, std::sync::atomic::Ordering::SeqCst);
		let cancel_result = host
			.runtime
			.fork(entry_id, None)
			.await
			.expect("cancelled fork");
		assert!(cancel_result.cancelled);
		assert_eq!(cancel_result.selected_text, None);
		{
			let events = events.lock();
			assert_eq!(events.len(), 1, "{events:?}");
			assert_eq!(event_type(&events[0]), "session_before_fork");
		}

		events.lock().clear();
		cancel_next_fork.store(true, std::sync::atomic::Ordering::SeqCst);
		let cancel_at_result = host
			.runtime
			.fork("missing-entry", Some(ForkOptions {
				position: Some(crate::core::extensions::types::ForkPosition::At),
				with_session: None,
			}))
			.await
			.expect("cancelled at-fork");
		assert!(cancel_at_result.cancelled);
		{
			let events = events.lock();
			assert_eq!(events.len(), 1, "{events:?}");
			assert_eq!(events[0].get("entryId").and_then(serde_json::Value::as_str), Some("missing-entry"));
			assert_eq!(events[0].get("position").and_then(serde_json::Value::as_str), Some("at"));
		}
	}
}
