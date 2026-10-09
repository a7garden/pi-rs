//! Port of `packages/coding-agent/src/modes/rpc/rpc-mode.ts` — headless
//! operation with a JSON stdin/stdout protocol.
//!
//! Protocol:
//! - Commands: JSON objects with a `type` field, optional `id` for correlation
//! - Responses: `{ type: "response", command, success }` with `data`/`error`
//! - Events: session events streamed as they occur (`toJsonEvent` projection)
//! - Extension UI: `extension_ui_request` lines; the client answers with
//!   `extension_ui_response`
//!
//! Deviations:
//! - **D92** — the upstream loop terminates with `process.exit`; the Rust
//!   loop ([`run_rpc_mode_with_io`]) returns the exit code and the production
//!   wrapper ([`run_rpc_mode`]) exits. stdin/signals ride the [`RpcInput`]/
//!   [`ModeStdio`] seams (upstream tests `vi.mock` these modules).
//! - **D93** — command bodies deserialize through [`RpcCommandEnvelope`];
//!   non-`Serialize` aggregates (`SessionStats`) project manually; unknown
//!   thinking levels / queue modes are rejected where upstream passes them
//!   through (typed deserialization).
//! - **D93b** — the `bash` command honors `user_bash` `result` overrides;
//!   extension-provided remote `operations` have no runtime bridge on the
//!   erased channel (D43), so local execution runs instead.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::BoxFuture;
use parking_lot::Mutex;
use serde_json::{json, Map, Value as JsonValue};

use pi_agent::types::{AgentThinkingLevel, QueueMode};
use pi_ai::utils::uuid::uuidv7_now;

use crate::core::agent_session::{
	AgentSession, AgentSessionEvent, CycleDirection, ExecuteBashOptions, ExtensionBindings, ModelMutationOptions,
	PromptOptions, StreamingBehavior,
};
use crate::core::extensions::types::ExtensionError;
use crate::core::extensions::types::{ExtensionMode, ExtensionUi, ExtensionUiDialogOptions, InputSource, NotifyKind, Unsubscribe};
use crate::core::extensions::types::{
	CustomComponentFactory, CustomComponentOptions, EditorFactory, FooterFactory, HeaderFactory, SetThemeArg,
	ThemeSwitchResult, WidgetContent, WorkingIndicatorOptions,
};
use crate::utils::shell::kill_tracked_detached_children;

use super::super::json_event::{compaction_result_to_json, to_json_event};
use super::jsonl::{read_jsonl_lines, serialize_json_line};
use super::rpc_types::{
	extension_ui_request, prompt_disposition_wire, queued_disposition_wire, session_tree_node_to_json, BashCommand,
	CompactCommand, ExportHtmlCommand, ForkCommand, GetEntriesCommand, MessageCommand, NewSessionCommand,
	PromptCommand, QueueModeCommand, RpcCommandEnvelope, RpcCommandSource, RpcExtensionUIResponse, RpcResponse,
	RpcSessionState, RpcSlashCommand, SetEnabledCommand, SetModelCommand, SetSessionNameCommand,
	SetThinkingLevelCommand, SwitchSessionCommand,
};
use super::super::{command_context_actions, ModeCore, ModeStdio, RuntimeHost, StdioModeIo};

// ============================================================================
// Input seam
// ============================================================================

/// stdin seam of RPC mode (`attachJsonlLineReader(process.stdin, …)` +
/// `process.stdin.on("end")`; D92).
pub trait RpcInput: Send + Sync + 'static {
	/// Start feeding LF-framed lines; returns the detach closure.
	fn attach(
		&self,
		on_line: Arc<dyn Fn(String) + Send + Sync>,
		on_end: Arc<dyn Fn() + Send + Sync>,
	) -> Box<dyn FnOnce() + Send>;
}

/// Production input: blocking reader thread over stdin.
pub struct StdinInput;

impl RpcInput for StdinInput {
	fn attach(
		&self,
		on_line: Arc<dyn Fn(String) + Send + Sync>,
		on_end: Arc<dyn Fn() + Send + Sync>,
	) -> Box<dyn FnOnce() + Send> {
		let running = Arc::new(AtomicBool::new(true));
		let reader_running = Arc::clone(&running);
		std::thread::spawn(move || {
			read_jsonl_lines(std::io::stdin().lock(), move |line| {
				if reader_running.load(Ordering::SeqCst) {
					on_line(line);
				}
			});
			on_end();
		});
		let detach_running = Arc::clone(&running);
		Box::new(move || {
			detach_running.store(false, Ordering::SeqCst);
		})
	}
}

// ============================================================================
// Mode state
// ============================================================================

type PendingExtensionRequests =
	Arc<Mutex<HashMap<String, tokio::sync::oneshot::Sender<RpcExtensionUIResponse>>>>;

/// Loop state shared with signal/input/UI closures.
struct RpcModeState {
	core: Arc<ModeCore>,
	session: Mutex<Arc<AgentSession>>,
	unsubscribe: Mutex<Option<Box<dyn FnOnce() + Send>>>,
	unsubscribe_backpressure: Mutex<Option<Box<dyn FnOnce() + Send>>>,
	pending_extension_requests: PendingExtensionRequests,
	shutdown_requested: AtomicBool,
	shutting_down: AtomicBool,
	signal_cleanup: Mutex<Option<Box<dyn FnOnce() + Send>>>,
	detach_input: Mutex<Option<Box<dyn FnOnce() + Send>>>,
	exit_requested: tokio::sync::Notify,
	exit_code: AtomicI32,
}

impl RpcModeState {
	fn output(&self, value: &JsonValue) {
		self.core.io.write_raw_stdout(&serialize_json_line(value));
	}

	fn success(&self, id: Option<&str>, command: &str) -> RpcResponse {
		RpcResponse::success(id.map(str::to_string), command)
	}

	fn success_with(&self, id: Option<&str>, command: &str, data: JsonValue) -> RpcResponse {
		RpcResponse::success_with(id.map(str::to_string), command, data)
	}

	fn error(&self, id: Option<&str>, command: &str, message: impl Into<String>) -> RpcResponse {
		RpcResponse::error(id.map(str::to_string), command, message)
	}

	/// `shutdown`: tear everything down once and report the exit code.
	async fn shutdown(&self, exit_code: i32, sigterm: bool) {
		if self.shutting_down.swap(true, Ordering::SeqCst) {
			// Upstream re-enters `process.exit`; in-process, the first
			// shutdown already owns teardown.
			return;
		}
		if let Some(cleanup) = self.signal_cleanup.lock().take() {
			cleanup();
		}
		if let Some(unsubscribe) = self.unsubscribe.lock().take() {
			unsubscribe();
		}
		if let Some(unsubscribe_backpressure) = self.unsubscribe_backpressure.lock().take() {
			unsubscribe_backpressure();
		}
		self.core.host.dispose().await;
		if let Some(detach_input) = self.detach_input.lock().take() {
			detach_input();
		}
		if !sigterm {
			self.core.io.flush_raw_stdout().await;
		}
		self.exit_code.store(exit_code, Ordering::SeqCst);
		self.exit_requested.notify_waiters();
	}

	/// `checkShutdownRequested`.
	async fn check_shutdown_requested(self: &Arc<Self>) {
		if !self.shutdown_requested.load(Ordering::SeqCst) {
			return;
		}
		self.shutdown(0, false).await;
	}
}

/// Run in RPC mode. Listens for JSON commands on stdin, outputs events and
/// responses on stdout. Never returns (`process.exit` inside [`shutdown`]).
pub async fn run_rpc_mode(host: Arc<dyn RuntimeHost>) -> ! {
	let stdio = Arc::new(StdioModeIo::new());
	stdio.take_over_stdout();
	let io: Arc<dyn ModeStdio> = stdio;
	let code = run_rpc_mode_with_io(host, io, Arc::new(StdinInput)).await;
	std::process::exit(code)
}

/// [`run_rpc_mode`] over explicit stdio/input seams; returns the exit code
/// instead of exiting (D92; upstream tests mock these modules).
pub async fn run_rpc_mode_with_io(
	host: Arc<dyn RuntimeHost>,
	io: Arc<dyn ModeStdio>,
	input: Arc<dyn RpcInput>,
) -> i32 {
	let initial_session = host.session();
	let state = Arc::new(RpcModeState {
		core: Arc::new(ModeCore { host, io }),
		session: Mutex::new(initial_session),
		unsubscribe: Mutex::new(None),
		unsubscribe_backpressure: Mutex::new(None),
		pending_extension_requests: Arc::new(Mutex::new(HashMap::new())),
		shutdown_requested: AtomicBool::new(false),
		shutting_down: AtomicBool::new(false),
		signal_cleanup: Mutex::new(None),
		detach_input: Mutex::new(None),
		exit_requested: tokio::sync::Notify::new(),
		exit_code: AtomicI32::new(0),
	});

	// `runtimeHost.setRebindSession`.
	let rebind_state = Arc::clone(&state);
	let rebind_hook: super::super::RebindSessionHook = Arc::new(move |_session: &AgentSession| {
		let state = Arc::clone(&rebind_state);
		Box::pin(async move {
			let _ = rebind_session(&state).await;
		})
	});
	state.core.host.set_rebind_session(Some(rebind_hook));

	let _ = rebind_session(&state).await;

	// `registerSignalHandlers`: SIGTERM (and SIGHUP off-Windows) → shutdown.
	let signal_state = Arc::clone(&state);
	let signal_cleanup = state.core.io.register_signal_handlers(Arc::new(
		move |exit_code: i32| -> BoxFuture<'static, ()> {
			let state = Arc::clone(&signal_state);
			Box::pin(async move {
				kill_tracked_detached_children();
				state.shutdown(exit_code, exit_code == 129).await;
			})
		},
	));
	*state.signal_cleanup.lock() = Some(signal_cleanup);

	// Input lines flow through a channel so command handling can be spawned
	// per line (upstream: `void handleInputLine(line)`).
	let (line_tx, mut line_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
	let end_state = Arc::clone(&state);
	let detach = input.attach(
		Arc::new(move |line: String| {
			let _ = line_tx.send(line);
		}),
		Arc::new(move || {
			let state = Arc::clone(&end_state);
			tokio::spawn(async move { state.shutdown(0, false).await });
		}),
	);
	*state.detach_input.lock() = Some(detach);

	let exit_code = loop {
		tokio::select! {
			line = line_rx.recv() => match line {
				Some(line) => {
					let state = Arc::clone(&state);
					tokio::spawn(handle_input_line(state, line));
				}
				None => {
					state.shutdown(0, false).await;
					break state.exit_code.load(Ordering::SeqCst);
				}
			},
			_ = state.exit_requested.notified() => {
				break state.exit_code.load(Ordering::SeqCst);
			}
		}
	};
	exit_code
}

/// `handleInputLine`.
async fn handle_input_line(state: Arc<RpcModeState>, line: String) {
	let parsed: Result<JsonValue, String> = serde_json::from_str(&line).map_err(|error| error.to_string());
	let parsed = match parsed {
		Ok(parsed) => parsed,
		Err(message) => {
			let response = state.error(None, "parse", format!("Failed to parse command: {message}"));
			state.output(&serde_json::to_value(&response).unwrap_or(JsonValue::Null));
			state.core.io.wait_for_raw_stdout_backpressure().await;
			return;
		}
	};

	// Handle extension UI responses.
	if parsed.get("type").and_then(JsonValue::as_str) == Some("extension_ui_response") {
		if let Ok(response) = serde_json::from_value::<RpcExtensionUIResponse>(parsed) {
			if let Some(pending) = state.pending_extension_requests.lock().remove(&response.id) {
				let _ = pending.send(response);
			}
		}
		return;
	}

	let envelope: RpcCommandEnvelope = serde_json::from_value(parsed).unwrap_or(RpcCommandEnvelope {
		id: None,
		kind: "undefined".to_string(),
		body: JsonValue::Null,
	});
	let command_id = envelope.id.clone();
	let command_kind = envelope.kind.clone();

	match handle_command(&state, &envelope).await {
		Ok(Some(response)) => {
			state.output(&serde_json::to_value(&response).unwrap_or(JsonValue::Null));
			state.core.io.wait_for_raw_stdout_backpressure().await;
		}
		Ok(None) => {}
		Err(message) => {
			let response = state.error(command_id.as_deref(), &command_kind, message);
			state.output(&serde_json::to_value(&response).unwrap_or(JsonValue::Null));
			state.core.io.wait_for_raw_stdout_backpressure().await;
		}
	}

	state.check_shutdown_requested().await;
}

/// `body<T>(envelope)`: decode the command body (D93); missing fields fall
/// back to `Default` (upstream reads possibly-absent members).
fn command_body<T: serde::de::DeserializeOwned + Default>(envelope: &RpcCommandEnvelope) -> T {
	serde_json::from_value(envelope.body.clone()).unwrap_or_default()
}

/// `handleCommand`: returns the immediate response, if any.
async fn handle_command(
	state: &Arc<RpcModeState>,
	envelope: &RpcCommandEnvelope,
) -> Result<Option<RpcResponse>, String> {
	let id = envelope.id.as_deref();
	let session = Arc::clone(&state.session.lock());

	match envelope.kind.as_str() {
		// =================================================================
		// Prompting
		// =================================================================
		"prompt" => {
			// Start prompt handling immediately, but emit the authoritative
			// response only after prompt preflight succeeds. Queued and
			// immediately handled prompts also count as success.
			let body: PromptCommand = command_body(envelope);
			let preflight_succeeded = Arc::new(AtomicBool::new(false));
			let preflight_state = Arc::clone(state);
			let preflight_succeeded_flag = Arc::clone(&preflight_succeeded);
			let preflight_id = id.map(str::to_string);
			let preflight = Arc::new(move |disposition: crate::core::agent_session::PromptDisposition| {
				preflight_succeeded_flag.store(true, Ordering::SeqCst);
				let response =
					preflight_state.success_with(preflight_id.as_deref(), "prompt", json!({ "disposition": prompt_disposition_wire(disposition) }));
				preflight_state.output(&serde_json::to_value(&response).unwrap_or(JsonValue::Null));
			});
			let streaming_behavior = match body.streaming_behavior.as_deref() {
				Some("steer") => Some(StreamingBehavior::Steer),
				Some("followUp") => Some(StreamingBehavior::FollowUp),
				_ => None,
			};
			let result = session
				.prompt(
					body.message,
					Some(PromptOptions {
						images: body.images,
						streaming_behavior,
						source: Some(InputSource::Rpc),
						preflight_result: Some(preflight),
						..Default::default()
					}),
				)
				.await;
			if let Err(message) = result {
				if !preflight_succeeded.load(Ordering::SeqCst) {
					return Err(message);
				}
			}
			Ok(None)
		}

		"steer" => {
			let body: MessageCommand = command_body(envelope);
			let disposition = session.steer(body.message, body.images, Some(InputSource::Rpc)).await?;
			Ok(Some(state.success_with(id, "steer", json!({ "disposition": queued_disposition_wire(disposition) }))))
		}

		"follow_up" => {
			let body: MessageCommand = command_body(envelope);
			let disposition = session.follow_up(body.message, body.images, Some(InputSource::Rpc)).await?;
			Ok(Some(
				state.success_with(id, "follow_up", json!({ "disposition": queued_disposition_wire(disposition) })),
			))
		}

		"abort" => {
			session.abort().await;
			Ok(Some(state.success(id, "abort")))
		}

		"clear_queue" => {
			let (steering, follow_up) = session.clear_queue();
			Ok(Some(state.success_with(id, "clear_queue", json!({ "steering": steering, "followUp": follow_up }))))
		}

		"new_session" => {
			let body: NewSessionCommand = command_body(envelope);
			let options = body.parent_session.map(|parent_session| {
				crate::core::agent_session_runtime::NewSessionOptions {
					parent_session: Some(parent_session),
					setup: None,
					with_session: None,
				}
			});
			let result = state.core.host.new_session(options).await?;
			if !result.cancelled {
				rebind_session(state).await?;
			}
			Ok(Some(state.success_with(id, "new_session", json!({ "cancelled": result.cancelled }))))
		}

		// =================================================================
		// State
		// =================================================================
		"get_state" => {
			let session_state = RpcSessionState {
				model: session.model(),
				thinking_level: session.thinking_level(),
				is_streaming: session.is_streaming(),
				is_compacting: session.is_compacting(),
				steering_mode: session.steering_mode(),
				follow_up_mode: session.follow_up_mode(),
				session_file: session.session_file(),
				session_id: session.session_id(),
				session_name: session.session_name(),
				auto_compaction_enabled: session.auto_compaction_enabled(),
				message_count: session.messages().len(),
				pending_message_count: session.pending_message_count(),
			};
			Ok(Some(state.success_with(
				id,
				"get_state",
				serde_json::to_value(&session_state).unwrap_or(JsonValue::Null),
			)))
		}

		// =================================================================
		// Model
		// =================================================================
		"set_model" => {
			let body: SetModelCommand = command_body(envelope);
			let models = session.model_runtime().get_available_snapshot();
			let model = models
				.into_iter()
				.find(|model| model.provider() == body.provider && model.id() == body.model_id);
			let Some(model) = model else {
				return Ok(Some(state.error(id, "set_model", format!("Model not found: {}/{}", body.provider, body.model_id))));
			};
			session.set_model(model.clone(), ModelMutationOptions::default()).await?;
			Ok(Some(state.success_with(
				id,
				"set_model",
				serde_json::to_value(&model).unwrap_or(JsonValue::Null),
			)))
		}

		"cycle_model" => {
			let result = session.cycle_model(CycleDirection::Forward, ModelMutationOptions::default()).await?;
			Ok(Some(state.success_with(id, "cycle_model", cycle_result_to_json(result))))
		}

		"get_available_models" => {
			let models = session.model_runtime().get_available_snapshot();
			let models: Vec<JsonValue> =
				models.into_iter().map(|model| serde_json::to_value(&model).unwrap_or(JsonValue::Null)).collect();
			Ok(Some(state.success_with(id, "get_available_models", json!({ "models": models }))))
		}

		// =================================================================
		// Thinking
		// =================================================================
		"set_thinking_level" => {
			let body: SetThinkingLevelCommand = command_body(envelope);
			let level = parse_thinking_level(&body.level)?;
			session.set_thinking_level(level, ModelMutationOptions::default());
			Ok(Some(state.success(id, "set_thinking_level")))
		}

		"cycle_thinking_level" => {
			let level = session.cycle_thinking_level(ModelMutationOptions::default());
			Ok(Some(state.success_with(
				id,
				"cycle_thinking_level",
				match level {
					Some(level) => json!({ "level": level }),
					None => JsonValue::Null,
				},
			)))
		}

		"get_available_thinking_levels" => {
			let levels = session.get_available_thinking_levels();
			Ok(Some(state.success_with(id, "get_available_thinking_levels", json!({ "levels": levels }))))
		}

		// =================================================================
		// Queue Modes
		// =================================================================
		"set_steering_mode" => {
			let body: QueueModeCommand = command_body(envelope);
			session.set_steering_mode(parse_queue_mode(&body.mode)?);
			Ok(Some(state.success(id, "set_steering_mode")))
		}

		"set_follow_up_mode" => {
			let body: QueueModeCommand = command_body(envelope);
			session.set_follow_up_mode(parse_queue_mode(&body.mode)?);
			Ok(Some(state.success(id, "set_follow_up_mode")))
		}

		// =================================================================
		// Compaction
		// =================================================================
		"compact" => {
			let body: CompactCommand = command_body(envelope);
			let result = session.compact(body.custom_instructions).await?;
			Ok(Some(state.success_with(id, "compact", compaction_result_to_json(&result))))
		}

		"set_auto_compaction" => {
			let body: SetEnabledCommand = command_body(envelope);
			session.set_auto_compaction_enabled(body.enabled);
			Ok(Some(state.success(id, "set_auto_compaction")))
		}

		// =================================================================
		// Retry
		// =================================================================
		"set_auto_retry" => {
			let body: SetEnabledCommand = command_body(envelope);
			session.set_auto_retry_enabled(body.enabled);
			Ok(Some(state.success(id, "set_auto_retry")))
		}

		"abort_retry" => {
			session.abort_retry();
			Ok(Some(state.success(id, "abort_retry")))
		}

		// =================================================================
		// Bash
		// =================================================================
		"bash" => {
			let body: BashCommand = command_body(envelope);
			let bash_cwd = session.session_manager().lock().get_cwd().to_string();
			let event_result = session
				.extension_runner()
				.emit_user_bash(&crate::core::extensions::types::UserBashEvent {
					command: body.command.clone(),
					exclude_from_context: body.exclude_from_context.unwrap_or(false),
					cwd: bash_cwd,
				})
				.await?;

			if let Some(result_value) = event_result.get("result") {
				let bash_result: crate::core::bash_executor::BashResult =
					serde_json::from_value(result_value.clone()).map_err(|error| error.to_string())?;
				session.record_bash_result(&body.command, &bash_result, body.exclude_from_context);
				return Ok(Some(state.success_with(
					id,
					"bash",
					serde_json::to_value(&bash_result).unwrap_or(JsonValue::Null),
				)));
			}

			// D93b: extension-provided remote `operations` have no runtime
			// bridge on the erased channel; execute locally.
			let result = session
				.execute_bash(
					body.command.clone(),
					None,
					Some(ExecuteBashOptions {
						exclude_from_context: body.exclude_from_context,
						id: id.map(str::to_string),
						operations: None,
					}),
				)
				.await?;
			Ok(Some(state.success_with(
				id,
				"bash",
				serde_json::to_value(&result).unwrap_or(JsonValue::Null),
			)))
		}

		"abort_bash" => {
			session.abort_bash();
			Ok(Some(state.success(id, "abort_bash")))
		}

		// =================================================================
		// Session
		// =================================================================
		"get_session_stats" => {
			let stats = session.get_session_stats();
			Ok(Some(state.success_with(id, "get_session_stats", session_stats_to_json(&stats))))
		}

		"export_html" => {
			let body: ExportHtmlCommand = command_body(envelope);
			let path = session.export_to_html(body.output_path.as_deref(), None).await?;
			Ok(Some(state.success_with(id, "export_html", json!({ "path": path }))))
		}

		"switch_session" => {
			let body: SwitchSessionCommand = command_body(envelope);
			let result = state.core.host.switch_session(&body.session_path, None).await?;
			if !result.cancelled {
				rebind_session(state).await?;
			}
			Ok(Some(state.success_with(id, "switch_session", json!({ "cancelled": result.cancelled }))))
		}

		"fork" => {
			let body: ForkCommand = command_body(envelope);
			let result = state.core.host.fork(&body.entry_id, None).await?;
			if !result.cancelled {
				rebind_session(state).await?;
			}
			Ok(Some(state.success_with(
				id,
				"fork",
				json!({ "text": result.selected_text, "cancelled": result.cancelled }),
			)))
		}

		"clone" => {
			let leaf_id = session.session_manager().lock().get_leaf_id().map(str::to_string);
			let Some(leaf_id) = leaf_id else {
				return Ok(Some(state.error(id, "clone", "Cannot clone session: no current entry selected")));
			};
			let result = state
				.core
				.host
				.fork(
					&leaf_id,
					Some(crate::core::agent_session_runtime::ForkOptions {
						position: Some(crate::core::extensions::types::ForkPosition::At),
						with_session: None,
					}),
				)
				.await?;
			if !result.cancelled {
				rebind_session(state).await?;
			}
			Ok(Some(state.success_with(id, "clone", json!({ "cancelled": result.cancelled }))))
		}

		"get_fork_messages" => {
			let messages = session.get_user_messages_for_forking();
			let messages: Vec<JsonValue> = messages
				.into_iter()
				.map(|(entry_id, text)| json!({ "entryId": entry_id, "text": text }))
				.collect();
			Ok(Some(state.success_with(id, "get_fork_messages", json!({ "messages": messages }))))
		}

		"get_entries" => {
			let body: GetEntriesCommand = command_body(envelope);
			let session_manager = session.session_manager();
			let mut entries = session_manager.lock().get_entries();
			if let Some(since) = &body.since {
				let since_index = entries.iter().position(|entry| entry.id() == since);
				let Some(since_index) = since_index else {
					return Ok(Some(state.error(id, "get_entries", format!("Entry not found: {since}"))));
				};
				entries = entries.split_off(since_index + 1);
			}
			let entries_json: Vec<JsonValue> =
				entries.into_iter().map(|entry| serde_json::to_value(&entry).unwrap_or(JsonValue::Null)).collect();
			let leaf_id = session_manager.lock().get_leaf_id().map(str::to_string);
			Ok(Some(state.success_with(
				id,
				"get_entries",
				json!({ "entries": entries_json, "leafId": leaf_id }),
			)))
		}

		"get_tree" => {
			let session_manager = session.session_manager();
			let tree = session_manager.lock().get_tree();
			let tree: Vec<JsonValue> = tree.iter().map(session_tree_node_to_json).collect();
			let leaf_id = session_manager.lock().get_leaf_id().map(str::to_string);
			Ok(Some(state.success_with(id, "get_tree", json!({ "tree": tree, "leafId": leaf_id }))))
		}

		"get_last_assistant_text" => {
			let text = session.get_last_assistant_text();
			Ok(Some(state.success_with(id, "get_last_assistant_text", json!({ "text": text }))))
		}

		"set_session_name" => {
			let body: SetSessionNameCommand = command_body(envelope);
			let name = body.name.trim().to_string();
			if name.is_empty() {
				return Ok(Some(state.error(id, "set_session_name", "Session name cannot be empty")));
			}
			session.set_session_name(&name);
			Ok(Some(state.success(id, "set_session_name")))
		}

		// =================================================================
		// Messages
		// =================================================================
		"get_messages" => Ok(Some(state.success_with(
			id,
			"get_messages",
			json!({ "messages": session.messages() }),
		))),

		// =================================================================
		// Commands (available for invocation via prompt)
		// =================================================================
		"get_commands" => {
			let mut commands: Vec<RpcSlashCommand> = Vec::new();

			for command in session.extension_runner().get_registered_commands() {
				commands.push(RpcSlashCommand {
					name: command.invocation_name.clone(),
					description: command.description.clone(),
					source: RpcCommandSource::Extension,
					source_info: command.source_info.clone(),
				});
			}

			for template in session.prompt_templates() {
				commands.push(RpcSlashCommand {
					name: template.name,
					description: Some(template.description),
					source: RpcCommandSource::Prompt,
					source_info: template.source_info,
				});
			}

			for skill in session.resource_loader().lock().get_skills().items {
				commands.push(RpcSlashCommand {
					name: format!("skill:{}", skill.name),
					description: Some(skill.description),
					source: RpcCommandSource::Skill,
					source_info: skill.source_info,
				});
			}

			Ok(Some(state.success_with(
				id,
				"get_commands",
				serde_json::to_value(&commands).map(|value| json!({ "commands": value })).unwrap_or(JsonValue::Null),
			)))
		}

		kind => Ok(Some(state.error(id, kind, format!("Unknown command: {kind}")))),
	}
}

/// `rebindSession`: bind the RPC extension UI + actions, resubscribe events.
async fn rebind_session(state: &Arc<RpcModeState>) -> Result<(), String> {
	let session = state.core.host.session();
	*state.session.lock() = Arc::clone(&session);

	let core = ModeCore { host: Arc::clone(&state.core.host), io: Arc::clone(&state.core.io) };
	let shutdown_state = Arc::clone(state);
	let error_state = Arc::clone(state);
	session
		.bind_extensions(ExtensionBindings {
			ui_context: Some(Arc::new(RpcUi::new(Arc::downgrade(state))) as Arc<dyn ExtensionUi>),
			mode: Some(ExtensionMode::Rpc),
			command_context_actions: Some(command_context_actions(&core, &session)),
			shutdown_handler: Some(Arc::new(move || {
				shutdown_state.shutdown_requested.store(true, Ordering::SeqCst);
			})),
			on_error: Some(Arc::new(move |error: &ExtensionError| {
				let line = json!({
					"type": "extension_error",
					"extensionPath": error.extension_path,
					"event": error.event,
					"error": error.error,
				});
				error_state.output(&line);
			})),
			..Default::default()
		})
		.await?;

	if let Some(unsubscribe) = state.unsubscribe.lock().take() {
		unsubscribe();
	}
	if let Some(unsubscribe_backpressure) = state.unsubscribe_backpressure.lock().take() {
		unsubscribe_backpressure();
	}

	let output_state = Arc::clone(state);
	let settled_state = Arc::clone(state);
	*state.unsubscribe.lock() = Some(session.subscribe(move |event: AgentSessionEvent| {
		let line = match to_json_event(&event) {
			Ok(line) => line,
			Err(message) => panic!("{message}"),
		};
		output_state.output(&line);
		if matches!(event, AgentSessionEvent::AgentSettled) {
			let state = Arc::clone(&settled_state);
			tokio::spawn(async move { state.check_shutdown_requested().await });
		}
	}));
	let io = Arc::clone(&state.core.io);
	*state.unsubscribe_backpressure.lock() = Some(session.agent().subscribe(move |_, _| {
		let io = Arc::clone(&io);
		Box::pin(async move { io.wait_for_raw_stdout_backpressure().await })
	}));

	Ok(())
}

// ============================================================================
// Projections (D93)
// ============================================================================

fn cycle_result_to_json(result: Option<crate::core::agent_session::ModelCycleResult>) -> JsonValue {
	match result {
		None => JsonValue::Null,
		Some(result) => json!({
			"model": serde_json::to_value(&result.model).unwrap_or(JsonValue::Null),
			"thinkingLevel": result.thinking_level,
			"isScoped": result.is_scoped,
		}),
	}
}

/// `AgentThinkingLevel` wire names are Serialize-only, so commands parse by
/// name (D93).
pub(crate) fn parse_thinking_level(level: &str) -> Result<AgentThinkingLevel, String> {
	match level {
		"off" => Ok(AgentThinkingLevel::Off),
		"minimal" => Ok(AgentThinkingLevel::Minimal),
		"low" => Ok(AgentThinkingLevel::Low),
		"medium" => Ok(AgentThinkingLevel::Medium),
		"high" => Ok(AgentThinkingLevel::High),
		"xhigh" => Ok(AgentThinkingLevel::Xhigh),
		"max" => Ok(AgentThinkingLevel::Max),
		other => Err(format!("Unknown thinking level: {other}")),
	}
}

fn parse_queue_mode(mode: &str) -> Result<QueueMode, String> {
	match mode {
		"all" => Ok(QueueMode::All),
		"one-at-a-time" => Ok(QueueMode::OneAtATime),
		other => Err(format!("Unknown queue mode: {other}")),
	}
}

fn session_stats_to_json(stats: &crate::core::agent_session::SessionStats) -> JsonValue {
	let mut object = Map::new();
	if let Some(session_file) = &stats.session_file {
		object.insert("sessionFile".into(), json!(session_file));
	}
	object.insert("sessionId".into(), json!(stats.session_id));
	object.insert("userMessages".into(), json!(stats.user_messages));
	object.insert("assistantMessages".into(), json!(stats.assistant_messages));
	object.insert("toolCalls".into(), json!(stats.tool_calls));
	object.insert("toolResults".into(), json!(stats.tool_results));
	object.insert("totalMessages".into(), json!(stats.total_messages));
	object.insert(
		"tokens".into(),
		json!({
			"input": stats.tokens.input,
			"output": stats.tokens.output,
			"cacheRead": stats.tokens.cache_read,
			"cacheWrite": stats.tokens.cache_write,
			"total": stats.tokens.total,
		}),
	);
	object.insert("cost".into(), json!(stats.cost));
	if let Some(context_usage) = &stats.context_usage {
		object.insert(
			"contextUsage".into(),
			serde_json::to_value(context_usage).unwrap_or(JsonValue::Null),
		);
	}
	JsonValue::Object(object)
}

// ============================================================================
// Extension UI context
// ============================================================================

/// RPC-mode extension UI context (`createExtensionUIContext`).
struct RpcUi {
	state: std::sync::Weak<RpcModeState>,
}

impl RpcUi {
	fn new(state: std::sync::Weak<RpcModeState>) -> Self {
		RpcUi { state }
	}

	fn with_state(&self, f: impl FnOnce(&RpcModeState)) {
		if let Some(state) = self.state.upgrade() {
			f(&state);
		}
	}

	/// `createDialogPromise`. Static over an owned `Weak` so the future is
	/// `'static` without borrowing `&self`.
	fn create_dialog<T: Send + 'static>(
		state: std::sync::Weak<RpcModeState>,
		opts: Option<ExtensionUiDialogOptions>,
		default: T,
		request: JsonValue,
		parse: impl Fn(&RpcExtensionUIResponse) -> T + Send + 'static,
	) -> BoxFuture<'static, T> {
		Box::pin(async move {
		let opts = opts.unwrap_or_default();
		if let Some(signal) = &opts.signal {
			if signal.aborted() {
				return default;
			}
		}

		let id = uuidv7_now().unwrap_or_default();
		let (sender, receiver) = tokio::sync::oneshot::channel::<RpcExtensionUIResponse>();
		let pending = state
			.upgrade()
			.map(|state| Arc::clone(&state.pending_extension_requests));
		if let Some(pending) = &pending {
			pending.lock().insert(id.clone(), sender);
		}
		if let Some(state) = state.upgrade() {
			state.output(&extension_ui_request(&id, request));
		}

		let cleanup = |pending: &Option<PendingExtensionRequests>, id: &str| {
			if let Some(pending) = pending {
				pending.lock().remove(id);
			}
		};

		let signal_future = async {
			match &opts.signal {
				Some(signal) => signal.aborted_future().await,
				None => std::future::pending::<()>().await,
			}
		};
		let timeout_future = async {
			match opts.timeout {
				Some(timeout) => tokio::time::sleep(Duration::from_millis(timeout)).await,
				None => std::future::pending::<()>().await,
			}
		};

		let result = tokio::select! {
			response = receiver => match response {
				Ok(response) => parse(&response),
				Err(_) => default,
			},
			_ = signal_future => default,
			_ = timeout_future => default,
		};

		cleanup(&pending, &id);
		result
		})
	}
}

impl ExtensionUi for RpcUi {
	fn select(
		&self,
		title: String,
		options: Vec<String>,
		opts: Option<ExtensionUiDialogOptions>,
	) -> BoxFuture<'static, Option<String>> {
		let state = self.state.clone();
		Self::create_dialog(
			state,
			opts,
			None,
			json!({ "method": "select", "title": title, "options": options }),
			|response| {
				if response.is_cancelled() {
					None
				} else {
					response.value.clone()
				}
			},
		)
	}

	fn confirm(&self, title: String, message: String, opts: Option<ExtensionUiDialogOptions>) -> BoxFuture<'static, bool> {
		let state = self.state.clone();
		Self::create_dialog(
			state,
			opts,
			false,
			json!({ "method": "confirm", "title": title, "message": message }),
			|response| {
				if response.is_cancelled() {
					false
				} else {
					response.confirmed.unwrap_or(false)
				}
			},
		)
	}

	fn input(
		&self,
		title: String,
		placeholder: Option<String>,
		opts: Option<ExtensionUiDialogOptions>,
	) -> BoxFuture<'static, Option<String>> {
		let state = self.state.clone();
		Self::create_dialog(
			state,
			opts,
			None,
			json!({ "method": "input", "title": title, "placeholder": placeholder }),
			|response| {
				if response.is_cancelled() {
					None
				} else {
					response.value.clone()
				}
			},
		)
	}

	fn notify(&self, message: String, kind: Option<NotifyKind>) {
		// Fire and forget - no response needed.
		self.with_state(|state| {
			let id = uuidv7_now().unwrap_or_default();
			let notify_type = kind.map(|kind| match kind {
				NotifyKind::Info => "info",
				NotifyKind::Warning => "warning",
				NotifyKind::Error => "error",
			});
			state.output(&extension_ui_request(
				&id,
				json!({ "method": "notify", "message": message, "notifyType": notify_type }),
			));
		});
	}

	fn on_terminal_input(&self, _handler: crate::core::extensions::types::TerminalInputHandler) -> Unsubscribe {
		// Raw terminal input not supported in RPC mode.
		Box::new(|| {})
	}

	fn set_status(&self, key: String, text: Option<String>) {
		// Fire and forget - no response needed.
		self.with_state(|state| {
			let id = uuidv7_now().unwrap_or_default();
			state.output(&extension_ui_request(
				&id,
				json!({ "method": "setStatus", "statusKey": key, "statusText": text }),
			));
		});
	}

	fn set_working_message(&self, _message: Option<String>) {
		// Working message not supported in RPC mode - requires TUI loader access.
	}

	fn set_working_visible(&self, _visible: bool) {
		// Working visibility not supported in RPC mode - requires TUI loader access.
	}

	fn set_working_indicator(&self, _options: Option<WorkingIndicatorOptions>) {
		// Working indicator customization not supported in RPC mode.
	}

	fn set_hidden_thinking_label(&self, _label: Option<String>) {
		// Hidden thinking label not supported in RPC mode.
	}

	fn set_widget(
		&self,
		key: String,
		content: Option<WidgetContent>,
		options: Option<crate::core::extensions::types::ExtensionWidgetOptions>,
	) {
		// Only support string arrays in RPC mode - factory functions are ignored.
		let widget_lines = match &content {
			None => JsonValue::Null,
			Some(WidgetContent::Text(lines)) => json!(lines),
			Some(WidgetContent::Factory(_)) => return,
		};
		self.with_state(|state| {
			let id = uuidv7_now().unwrap_or_default();
			let placement = options.and_then(|options| options.placement).map(|placement| match placement {
				crate::core::extensions::types::WidgetPlacement::AboveEditor => "aboveEditor",
				crate::core::extensions::types::WidgetPlacement::BelowEditor => "belowEditor",
			});
			state.output(&extension_ui_request(
				&id,
				json!({
					"method": "setWidget",
					"widgetKey": key,
					"widgetLines": widget_lines,
					"widgetPlacement": placement,
				}),
			));
		});
	}

	fn set_footer(&self, _factory: Option<FooterFactory>) {
		// Custom footer not supported in RPC mode - requires TUI access.
	}

	fn set_header(&self, _factory: Option<HeaderFactory>) {
		// Custom header not supported in RPC mode - requires TUI access.
	}

	fn set_title(&self, title: String) {
		// Fire and forget - host can implement terminal title control.
		self.with_state(|state| {
			let id = uuidv7_now().unwrap_or_default();
			state.output(&extension_ui_request(&id, json!({ "method": "setTitle", "title": title })));
		});
	}

	fn custom(&self, _factory: CustomComponentFactory, _options: Option<CustomComponentOptions>) -> BoxFuture<'static, JsonValue> {
		// Custom UI not supported in RPC mode.
		Box::pin(async move { JsonValue::Null })
	}

	fn paste_to_editor(&self, text: String) {
		// Paste handling not supported in RPC mode - falls back to setEditorText.
		self.set_editor_text(text);
	}

	fn set_editor_text(&self, text: String) {
		// Fire and forget - host can implement editor control.
		self.with_state(|state| {
			let id = uuidv7_now().unwrap_or_default();
			state.output(&extension_ui_request(&id, json!({ "method": "set_editor_text", "text": text })));
		});
	}

	fn get_editor_text(&self) -> String {
		// Synchronous method can't wait for RPC response.
		String::new()
	}

	fn editor(&self, title: String, prefill: Option<String>) -> BoxFuture<'static, Option<String>> {
		let state = self.state.clone();
		Box::pin(async move {
			let id = uuidv7_now().unwrap_or_default();
			let pending = state.upgrade().map(|state| Arc::clone(&state.pending_extension_requests));
			let (sender, receiver) = tokio::sync::oneshot::channel::<RpcExtensionUIResponse>();
			if let Some(pending) = &pending {
				pending.lock().insert(id.clone(), sender);
			}
			if let Some(state) = state.upgrade() {
				state.output(&extension_ui_request(
					&id,
					json!({ "method": "editor", "title": title, "prefill": prefill }),
				));
			}

			let response = receiver.await.ok();
			if let Some(pending) = &pending {
				pending.lock().remove(&id);
			}
			response.and_then(|response| {
				if response.is_cancelled() {
					None
				} else {
					response.value
				}
			})
		})
	}

	fn add_autocomplete_provider(&self, _factory: crate::core::extensions::types::AutocompleteProviderFactory) {
		// Autocomplete provider composition is not supported in RPC mode.
	}

	fn set_editor_component(&self, _factory: Option<EditorFactory>) {
		// Custom editor components not supported in RPC mode.
	}

	fn get_editor_component(&self) -> Option<EditorFactory> {
		// Custom editor components not supported in RPC mode.
		None
	}

	fn theme(&self) -> crate::core::extensions::types::Theme {
		// The opaque theme stand-in (D29); the interactive theme module is
		// TUI and stays unported.
		crate::core::extensions::types::Theme
	}

	fn get_all_themes(&self) -> Vec<crate::core::extensions::types::ThemeInfo> {
		Vec::new()
	}

	fn get_theme(&self, _name: &str) -> Option<crate::core::extensions::types::Theme> {
		None
	}

	fn set_theme(&self, _theme: SetThemeArg) -> ThemeSwitchResult {
		// Theme switching not supported in RPC mode.
		ThemeSwitchResult {
			success: false,
			error: Some("Theme switching not supported in RPC mode".to_string()),
		}
	}

	fn get_tools_expanded(&self) -> bool {
		// Tool expansion not supported in RPC mode - no TUI.
		false
	}

	fn set_tools_expanded(&self, _expanded: bool) {
		// Tool expansion not supported in RPC mode - no TUI.
	}
}

#[cfg(test)]
mod tests {
	//! Ports of:
	//! - `test/rpc-prompt-response-semantics.test.ts` (7 cases)
	//! - `test/suite/regressions/5868-rpc-unknown-command-id.test.ts`
	//! - `test/suite/regressions/7150-rpc-prompt-during-compaction.test.ts`
	//!
	//! The upstream `vi.mock` of `core/output-guard` and `modes/rpc/jsonl`
	//! becomes the [`TestRpc`] seam, and the suite `createHarness` becomes a
	//! real [`AgentSession`] on test doubles (same components the semantics
	//! suite builds upstream).

	use std::sync::Arc;
	use std::time::{Duration, Instant};

	use futures_util::future::BoxFuture;
	use parking_lot::Mutex;
	use serde_json::{json, Value as JsonValue};

	use pi_agent::types::AgentMessage;
	use pi_ai::types::StopReason;

	use crate::core::agent_session::{
		AgentSession, AgentSessionConfig, AgentSessionEvent, PromptDisposition, PromptOptions, ScopedModelEntry,
	};
	use crate::core::extensions::types::{ExtensionFactory, InputSource};
	use crate::core::session_manager::SessionManager;
	use crate::core::settings_manager::{Settings, SettingsManager, SettingsManagerCreateOptions};

	use crate::modes::test_support::{
		assistant_message, delayed_stream_fn, loader_with_factories, temp_dir, test_model_runtime, text_model,
	};
	use crate::modes::{ModeStdio, RebindSessionHook, RuntimeHost};
	use super::{run_rpc_mode_with_io, RpcInput};

	// ========================================================================
	// Test seams (upstream: vi.mock + vi.fn)
	// ========================================================================

	/// Captured stdout lines, stderr text, and the stdin line handler.
	#[derive(Default)]
	struct TestRpc {
		lines: Mutex<Vec<String>>,
		stderr: Mutex<Vec<String>>,
		line_handler: Arc<Mutex<Option<Arc<dyn Fn(String) + Send + Sync>>>>,
		on_end: Arc<Mutex<Option<Arc<dyn Fn() + Send + Sync>>>>,
	}

	impl TestRpc {
		/// Parsed output lines (`parseOutputLines` upstream: stdout may carry
		/// multi-line payloads).
		fn output_lines(&self) -> Vec<JsonValue> {
			self.lines
				.lock()
				.iter()
				.flat_map(|line| line.split('\n'))
				.filter(|line| !line.trim().is_empty())
				.map(|line| serde_json::from_str(line).unwrap())
				.collect()
		}

		/// `rpcIo.lineHandler(JSON.stringify(...))`.
		fn handle_line(&self, line: &str) {
			let handler = self.line_handler.lock().clone();
			if let Some(handler) = handler {
				handler(line.to_string());
			}
		}

		/// `vi.waitFor`: poll until `predicate` holds over the output lines.
		async fn wait_for(&self, predicate: impl Fn(&[JsonValue]) -> bool) -> Vec<JsonValue> {
			let deadline = Instant::now() + Duration::from_secs(5);
			loop {
				let lines = self.output_lines();
				if predicate(&lines) {
					return lines;
				}
				assert!(
					Instant::now() < deadline,
					"timeout waiting for RPC output; got {:#}",
					serde_json::to_string(&self.output_lines()).unwrap()
				);
				tokio::time::sleep(Duration::from_millis(10)).await;
			}
		}

		async fn wait_line_handler(&self) {
			let deadline = Instant::now() + Duration::from_secs(5);
			while self.line_handler.lock().is_none() {
				assert!(Instant::now() < deadline, "timeout waiting for the line handler");
				tokio::time::sleep(Duration::from_millis(5)).await;
			}
		}
	}

	impl ModeStdio for TestRpc {
		fn write_raw_stdout(&self, text: &str) {
			self.lines.lock().push(text.to_string());
		}

		fn write_stderr(&self, text: &str) {
			self.stderr.lock().push(text.to_string());
		}

		fn wait_for_raw_stdout_backpressure(&self) -> BoxFuture<'static, ()> {
			Box::pin(async {})
		}

		fn flush_raw_stdout(&self) -> BoxFuture<'static, ()> {
			Box::pin(async {})
		}

		fn register_signal_handlers(
			&self,
			_on_signal: Arc<dyn Fn(i32) -> BoxFuture<'static, ()> + Send + Sync>,
		) -> Box<dyn FnOnce() + Send> {
			Box::new(|| {})
		}

		fn exit(&self, code: i32) -> ! {
			unreachable!("rpc loop shutdown does not call exit({code}); loop returns the code")
		}
	}

	impl RpcInput for TestRpc {
		fn attach(
			&self,
			on_line: Arc<dyn Fn(String) + Send + Sync>,
			on_end: Arc<dyn Fn() + Send + Sync>,
		) -> Box<dyn FnOnce() + Send> {
			*self.line_handler.lock() = Some(on_line);
			*self.on_end.lock() = Some(on_end);
			let line_handler = Arc::clone(&self.line_handler);
			let on_end = Arc::clone(&self.on_end);
			Box::new(move || {
				*line_handler.lock() = None;
				if let Some(on_end) = on_end.lock().take() {
					on_end();
				}
			})
		}
	}

	/// `createRuntimeHost`'s host double: session switches report
	/// `cancelled: true` like the `vi.fn` doubles upstream.
	struct FakeRuntimeHost {
		session: Arc<AgentSession>,
	}

	impl RuntimeHost for FakeRuntimeHost {
		fn session(&self) -> Arc<AgentSession> {
			Arc::clone(&self.session)
		}

		fn new_session(
			&self,
			_options: Option<crate::core::agent_session_runtime::NewSessionOptions>,
		) -> BoxFuture<'static, Result<crate::core::agent_session_runtime::SwitchOutcome, String>> {
			Box::pin(async move { Ok(crate::core::agent_session_runtime::SwitchOutcome { cancelled: true }) })
		}

		fn fork(
			&self,
			_entry_id: &str,
			_options: Option<crate::core::agent_session_runtime::ForkOptions>,
		) -> BoxFuture<'static, Result<crate::core::agent_session_runtime::ForkOutcome, String>> {
			Box::pin(async move {
				Ok(crate::core::agent_session_runtime::ForkOutcome {
					cancelled: true,
					selected_text: Some(String::new()),
				})
			})
		}

		fn switch_session(
			&self,
			_session_path: &str,
			_options: Option<crate::core::agent_session_runtime::SwitchSessionOptions>,
		) -> BoxFuture<'static, Result<crate::core::agent_session_runtime::SwitchOutcome, String>> {
			Box::pin(async move { Ok(crate::core::agent_session_runtime::SwitchOutcome { cancelled: true }) })
		}

		fn dispose(&self) -> BoxFuture<'static, ()> {
			Box::pin(async {})
		}

		fn set_rebind_session(&self, _rebind_session: Option<RebindSessionHook>) {}
	}

	// ========================================================================
	// Harness (upstream createRuntimeHost)
	// ========================================================================

	struct HarnessOptions {
		with_auth: bool,
		response_delay_ms: u64,
		model: Option<pi_ai::types::Model>,
		stream_response: Option<pi_ai::types::AssistantMessage>,
		factories: Vec<ExtensionFactory>,
		settings: Option<Settings>,
	}

	impl Default for HarnessOptions {
		fn default() -> Self {
			HarnessOptions {
				with_auth: true,
				response_delay_ms: 0,
				model: None,
				stream_response: None,
				factories: Vec::new(),
				settings: None,
			}
		}
	}

	async fn create_runtime_host(
		options: HarnessOptions,
	) -> (Arc<FakeRuntimeHost>, Arc<AgentSession>, Arc<Mutex<SessionManager>>) {
		let dir = temp_dir("rpc-prompt");
		let model = options.model.clone().unwrap_or_else(|| text_model(200_000));
		let response = options.stream_response.clone().unwrap_or_else(|| assistant_message("done"));
		let stream_fn = delayed_stream_fn(options.response_delay_ms, response);
		let factories = options.factories;
		let loader = loader_with_factories(&dir, factories).await;
		let model_runtime = test_model_runtime(model.clone(), options.with_auth).await;
		let agent = Arc::new(pi_agent::Agent::new(pi_agent::AgentOptions {
			stream_fn: Some(stream_fn),
			initial_state: pi_agent::AgentInitialState {
				system_prompt: Some("Test".to_string()),
				model: Some(model),
				thinking_level: None,
				tools: Vec::new(),
				messages: Vec::new(),
			},
			..Default::default()
		}));
		let settings = options.settings.unwrap_or_else(Settings::new);
		let session_manager = Arc::new(Mutex::new(SessionManager::in_memory(None, None, None).unwrap()));
		let session = AgentSession::new(AgentSessionConfig {
			agent,
			session_manager: Arc::clone(&session_manager),
			settings_manager: Arc::new(Mutex::new(SettingsManager::in_memory(
				&settings,
				SettingsManagerCreateOptions::default(),
			))),
			cwd: dir,
			scoped_models: Vec::<ScopedModelEntry>::new(),
			resource_loader: loader,
			custom_tools: Vec::new(),
			model_runtime,
			cache_warmer: None,
			initial_active_tool_names: None,
			uses_default_tools: false,
			allowed_tool_names: None,
			excluded_tool_names: None,
			base_tools_override: None,
			extension_runner_ref: None,
			session_start_event: None,
		});
		(Arc::new(FakeRuntimeHost { session: Arc::clone(&session) }), session, session_manager)
	}

	/// `startRpcMode`: spawn the mode loop over the captured seams and wait
	/// for the line handler.
	async fn start_rpc_mode(host: Arc<FakeRuntimeHost>, io: Arc<TestRpc>) {
		let stdio: Arc<dyn ModeStdio> = io.clone();
		let input: Arc<dyn RpcInput> = io.clone();
		tokio::spawn(run_rpc_mode_with_io(
			host as Arc<dyn RuntimeHost>,
			stdio,
			input,
		));
		io.wait_line_handler().await;
	}

	/// Responses for id + prompt command (`getPromptResponses`).
	fn prompt_responses<'a>(lines: &'a [JsonValue], id: &str) -> Vec<&'a JsonValue> {
		lines
			.iter()
			.filter(|record| {
				record.get("id").and_then(JsonValue::as_str) == Some(id)
					&& record.get("type").and_then(JsonValue::as_str) == Some("response")
					&& record.get("command").and_then(JsonValue::as_str) == Some("prompt")
			})
			.collect()
	}

	fn filter_type<'a>(lines: &'a [JsonValue], event_type: &str) -> Vec<&'a JsonValue> {
		lines
			.iter()
			.filter(|record| record.get("type").and_then(JsonValue::as_str) == Some(event_type))
			.collect()
	}

	fn contains(lines: &[JsonValue], expected: &JsonValue) -> bool {
		lines.iter().any(|line| line == expected)
	}

	fn sleep_ms(ms: u64) {
		std::thread::sleep(Duration::from_millis(ms));
	}

	// ========================================================================
	// rpc-prompt-response-semantics.test.ts
	// ========================================================================

	/// Unconfigured-provider model (preflight rejects).
	fn fake_provider_model() -> pi_ai::types::Model {
		let mut model = text_model(0);
		model.base.provider = "fake-provider".to_string();
		model.base.id = "fake-model".to_string();
		model.base.name = "Fake Model".to_string();
		model
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn emits_one_failure_response_when_prompt_preflight_rejects() {
		let (host, _session, _manager) = create_runtime_host(HarnessOptions {
			with_auth: false,
			model: Some(fake_provider_model()),
			..Default::default()
		})
		.await;
		let io = Arc::new(TestRpc::default());
		start_rpc_mode(host, Arc::clone(&io)).await;

		io.handle_line(&json!({ "id": "b1", "type": "prompt", "message": "Hello" }).to_string());

		let lines = io
			.wait_for(|lines| prompt_responses(lines, "b1").len() == 1)
			.await;
		let response = prompt_responses(&lines, "b1")[0];
		assert_eq!(response.get("id").and_then(JsonValue::as_str), Some("b1"));
		assert_eq!(response.get("success").and_then(JsonValue::as_bool), Some(false));
		let error = response.get("error").and_then(JsonValue::as_str).unwrap();
		assert!(
			error.contains("No API key found for fake-provider."),
			"error: {error}"
		);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn emits_one_started_response_when_prompt_preflight_succeeds() {
		let (host, _session, _manager) = create_runtime_host(HarnessOptions::default()).await;
		let io = Arc::new(TestRpc::default());
		start_rpc_mode(host, Arc::clone(&io)).await;

		io.handle_line(&json!({ "id": "b2", "type": "prompt", "message": "Hello" }).to_string());

		let lines = io
			.wait_for(|lines| prompt_responses(lines, "b2").len() == 1)
			.await;
		let response = prompt_responses(&lines, "b2")[0];
		assert_eq!(
			response,
			&json!({ "id": "b2", "type": "response", "command": "prompt", "success": true, "data": { "disposition": "started" } })
		);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn reports_extension_commands_and_intercepted_input_as_handled() {
		let handled_factory: ExtensionFactory = Arc::new(|pi| {
			pi.register_command(
				"handled",
				crate::core::extensions::types::RegisteredCommandOptions {
					description: None,
					get_argument_completions: None,
					handler: Arc::new(|_args, _context| Box::pin(async move {})),
				},
			);
			let _ = pi.on(
				"input",
				Arc::new(|event: JsonValue, _context| {
					Box::pin(async move {
						if event.get("text").and_then(JsonValue::as_str) == Some("handled input") {
							Ok(Some(json!({ "action": "handled" })))
						} else {
							Ok(None)
						}
					})
				}),
			);
			Box::pin(async move { Ok(()) })
		});
		let (host, _session, _manager) = create_runtime_host(HarnessOptions {
			with_auth: false,
			factories: vec![handled_factory],
			..Default::default()
		})
		.await;
		let io = Arc::new(TestRpc::default());
		start_rpc_mode(host, Arc::clone(&io)).await;

		io.handle_line(&json!({ "id": "command", "type": "prompt", "message": "/handled" }).to_string());
		io.wait_for(|lines| {
			prompt_responses(lines, "command")
				.first()
				.is_some_and(|response| {
					response.get("data").and_then(|data| data.get("disposition")).and_then(JsonValue::as_str)
						== Some("handled")
				})
		})
		.await;

		io.handle_line(&json!({ "id": "input", "type": "prompt", "message": "handled input" }).to_string());
		io.wait_for(|lines| {
			prompt_responses(lines, "input")
				.first()
				.is_some_and(|response| {
					response.get("data").and_then(|data| data.get("disposition")).and_then(JsonValue::as_str)
						== Some("handled")
				})
		})
		.await;

		// #9098: handled prompts never start a run.
		let lines = sleep_and_collect(&io, 100).await;
		assert_eq!(filter_type(&lines, "agent_start").len(), 0);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn emits_one_success_response_when_prompt_is_queued_during_streaming() {
		let (host, _session, _manager) =
			create_runtime_host(HarnessOptions { response_delay_ms: 100, ..Default::default() }).await;
		let io = Arc::new(TestRpc::default());
		start_rpc_mode(host, Arc::clone(&io)).await;

		io.handle_line(&json!({ "id": "b3-start", "type": "prompt", "message": "Start" }).to_string());
		io.wait_for(|lines| prompt_responses(lines, "b3-start").len() == 1).await;

		io.handle_line(
			&json!({ "id": "b3", "type": "prompt", "message": "Queue this", "streamingBehavior": "followUp" })
				.to_string(),
		);
		let lines = io
			.wait_for(|lines| prompt_responses(lines, "b3").len() == 1)
			.await;
		let response = prompt_responses(&lines, "b3")[0];
		assert_eq!(
			response.get("data").and_then(|data| data.get("disposition")).and_then(JsonValue::as_str),
			Some("queued")
		);

		sleep_ms(150);
	}

	/// #9803: a handler can consume A while independently queueing B; report
	/// A's outcome.
	async fn reports_handled_even_when_extension_queues_another_message(kind: &str) {
		let queue_factory: ExtensionFactory = Arc::new(|pi| {
			let _ = pi.on(
				"input",
				Arc::new(|event: JsonValue, _context| {
					Box::pin(async move {
						let text = event.get("text").and_then(JsonValue::as_str).unwrap_or_default();
						let source = event.get("source").and_then(JsonValue::as_str).unwrap_or_default();
						if text == "A" && source == "rpc" {
							return Ok(Some(json!({ "action": "handled" })));
						}
						Ok(None)
					})
				}),
			);
			Box::pin(async move { Ok(()) })
		});
		let (host, session, _manager) = create_runtime_host(HarnessOptions {
			response_delay_ms: 500,
			factories: vec![queue_factory],
			..Default::default()
		})
		.await;
		let io = Arc::new(TestRpc::default());
		start_rpc_mode(host, Arc::clone(&io)).await;

		io.handle_line(&json!({ "id": "start", "type": "prompt", "message": "Start" }).to_string());
		io.wait_for(|lines| prompt_responses(lines, "start").len() == 1).await;

		io.handle_line(&json!({ "id": "A", "type": kind, "message": "A" }).to_string());
		// Extension-origin queueing of B (`pi.sendUserMessage` upstream); the
		// port drives the same queue path with the extension input source
		// (agent_session_tests adaptation).
		let queue_result = if kind == "steer" {
			session.steer("B", None, Some(InputSource::Extension)).await
		} else {
			session.follow_up("B", None, Some(InputSource::Extension)).await
		};
		assert!(queue_result.is_ok(), "queue B: {queue_result:?}");
		let lines = io
			.wait_for(|lines| {
				contains(
					lines,
					&json!({
						"id": "A", "type": "response", "command": kind,
						"success": true, "data": { "disposition": "handled" },
					}),
				) && contains(
					lines,
					&json!({
						"type": "queue_update",
						"steering": if kind == "steer" { json!(["B"]) } else { json!([]) },
						"followUp": if kind == "follow_up" { json!(["B"]) } else { json!([]) },
					}),
				)
			})
			.await;
		// Exactly one response for A.
		assert_eq!(filter_type(&lines, "response").iter().filter(|r| r.get("id").and_then(JsonValue::as_str) == Some("A")).count(), 1);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn reports_steer_as_handled_even_when_an_extension_queues_another_message() {
		reports_handled_even_when_extension_queues_another_message("steer").await;
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn reports_follow_up_as_handled_even_when_an_extension_queues_another_message() {
		reports_handled_even_when_extension_queues_another_message("follow_up").await;
	}

	/// Input transformation queues the transformed text.
	async fn reports_queued_after_input_transformation(kind: &str) {
		let transform_factory: ExtensionFactory = Arc::new(|pi| {
			let _ = pi.on(
				"input",
				Arc::new(|event: JsonValue, _context| {
					Box::pin(async move {
						if event.get("text").and_then(JsonValue::as_str) == Some("A") {
							Ok(Some(json!({ "action": "transform", "text": "B" })))
						} else {
							Ok(None)
						}
					})
				}),
			);
			Box::pin(async move { Ok(()) })
		});
		let (host, _session, _manager) = create_runtime_host(HarnessOptions {
			with_auth: false,
			factories: vec![transform_factory],
			..Default::default()
		})
		.await;
		let io = Arc::new(TestRpc::default());
		start_rpc_mode(host, Arc::clone(&io)).await;

		io.handle_line(&json!({ "id": "A", "type": kind, "message": "A" }).to_string());
		io.wait_for(|lines| {
			contains(
				lines,
				&json!({
					"id": "A", "type": "response", "command": kind,
					"success": true, "data": { "disposition": "queued" },
				}),
			) && contains(
				lines,
				&json!({
					"type": "queue_update",
					"steering": if kind == "steer" { json!(["B"]) } else { json!([]) },
					"followUp": if kind == "follow_up" { json!(["B"]) } else { json!([]) },
				}),
			)
		})
		.await;
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn reports_steer_as_queued_after_input_transformation() {
		reports_queued_after_input_transformation("steer").await;
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn reports_follow_up_as_queued_after_input_transformation() {
		reports_queued_after_input_transformation("follow_up").await;
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn returns_and_clears_queued_steering_and_follow_up_messages() {
		let (host, _session, _manager) =
			create_runtime_host(HarnessOptions { response_delay_ms: 500, ..Default::default() }).await;
		let io = Arc::new(TestRpc::default());
		start_rpc_mode(host, Arc::clone(&io)).await;

		io.handle_line(&json!({ "id": "clear-start", "type": "prompt", "message": "Start" }).to_string());
		io.wait_for(|lines| prompt_responses(lines, "clear-start").len() == 1).await;

		io.handle_line(
			&json!({
				"id": "clear-steering", "type": "prompt",
				"message": "Change direction", "streamingBehavior": "steer",
			})
			.to_string(),
		);
		io.wait_for(|lines| {
			prompt_responses(lines, "clear-steering")
				.first()
				.is_some_and(|response| {
					response.get("data").and_then(|data| data.get("disposition")).and_then(JsonValue::as_str)
						== Some("queued")
				})
		})
		.await;

		io.handle_line(
			&json!({
				"id": "clear-follow-up", "type": "prompt",
				"message": "Summarize when finished", "streamingBehavior": "followUp",
			})
			.to_string(),
		);
		io.wait_for(|lines| prompt_responses(lines, "clear-follow-up").len() == 1).await;

		io.handle_line(&json!({ "id": "clear", "type": "clear_queue" }).to_string());
		io.wait_for(|lines| {
			contains(
				lines,
				&json!({
					"id": "clear", "type": "response", "command": "clear_queue",
					"success": true,
					"data": { "steering": ["Change direction"], "followUp": ["Summarize when finished"] },
				}),
			)
		})
		.await;

		sleep_ms(600);
		let lines = io.output_lines();
		assert_eq!(filter_type(&lines, "agent_start").len(), 1);
	}

	async fn sleep_and_collect(io: &Arc<TestRpc>, ms: u64) -> Vec<JsonValue> {
		sleep_ms(ms);
		io.output_lines()
	}

	// ========================================================================
	// suite/regressions/5868
	// ========================================================================

	#[tokio::test(flavor = "multi_thread")]
	async fn preserves_the_request_id_on_unknown_command_errors() {
		let (host, _session, _manager) = create_runtime_host(HarnessOptions::default()).await;
		let io = Arc::new(TestRpc::default());
		start_rpc_mode(host, Arc::clone(&io)).await;

		io.handle_line(&json!({ "id": "test", "type": "foobar" }).to_string());

		io.wait_for(|lines| {
			contains(
				lines,
				&json!({
					"id": "test", "type": "response", "command": "foobar",
					"success": false, "error": "Unknown command: foobar",
				}),
			)
		})
		.await;
	}

	// ========================================================================
	// suite/regressions/7150
	// ========================================================================

	#[tokio::test(flavor = "multi_thread")]
	async fn rejects_an_rpc_prompt_while_manual_compaction_is_in_progress() {
		let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
		let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
		let started_slot = Arc::new(Mutex::new(Some(started_tx)));
		let release_slot = Arc::new(Mutex::new(Some(release_rx)));
		let first_kept = Arc::new(Mutex::new(String::new()));
		let first_kept_shared = Arc::clone(&first_kept);
		let blocking_factory: ExtensionFactory = Arc::new(move |pi| {
			let started_slot = Arc::clone(&started_slot);
			let release_slot = Arc::clone(&release_slot);
			let first_kept = Arc::clone(&first_kept_shared);
			let _ = pi.on(
				"session_before_compact",
				Arc::new(move |_event: JsonValue, _context| {
					let started_slot = Arc::clone(&started_slot);
					let release_slot = Arc::clone(&release_slot);
					let first_kept = Arc::clone(&first_kept);
					Box::pin(async move {
						if let Some(started_tx) = started_slot.lock().take() {
							let _ = started_tx.send(());
						}
						// Block the compaction until the probe prompt ran.
						let release_rx = release_slot.lock().take();
						if let Some(release_rx) = release_rx {
							let _ = release_rx.await;
						}
						Ok(Some(json!({
							"compaction": {
								"summary": "manual compacted",
								"firstKeptEntryId": first_kept.lock().clone(),
								"tokensBefore": 10,
								"details": {},
							}
						})))
					})
				}),
			);
			Box::pin(async move { Ok(()) })
		});

		let settings = Settings::from_value(json!({
			"compaction": { "enabled": true, "keepRecentTokens": 1, "reserveTokens": 1 },
		}));
		let (host, session, session_manager) = create_runtime_host(HarnessOptions {
			settings: Some(settings),
			factories: vec![blocking_factory],
			stream_response: Some(assistant_message("probe response")),
			..Default::default()
		})
		.await;

		let timestamp = crate::modes::test_support::now_ms();
		let user_id = session_manager
			.lock()
			.append_message(pi_agent::types::AgentMessage::user(pi_ai::types::UserMessage {
				content: pi_ai::types::UserContentField::Text("old user message".to_string()),
				timestamp: timestamp - 1000,
			}))
			.unwrap();
		let assistant_entry_id = session_manager
			.lock()
			.append_message(pi_agent::types::AgentMessage::assistant({
				let mut message = assistant_message("old assistant response");
				message.timestamp = timestamp - 500;
				message
			}))
			.unwrap();
		*first_kept.lock() = assistant_entry_id.clone();

		// Record session events for the no-run assertions.
		let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
		let event_recorder = Arc::clone(&events);
		let _unsubscribe = session.subscribe(move |event: AgentSessionEvent| {
			let name = match event {
				AgentSessionEvent::AgentStart => "agent_start".to_string(),
				AgentSessionEvent::AgentSettled => "agent_settled".to_string(),
				_ => String::new(),
			};
			if !name.is_empty() {
				event_recorder.lock().push(name);
			}
		});

		let compact_session = Arc::clone(&session);
		let compact = tokio::spawn(async move { compact_session.compact(None).await });
		started_rx.await.unwrap();

		// Probe prompt while compaction holds the session.
		let preflight: Arc<Mutex<Option<PromptDisposition>>> = Arc::new(Mutex::new(None));
		let preflight_probe = Arc::clone(&preflight);
		let prompt_result = session
			.prompt(
				"PROBE-7150",
				Some(PromptOptions {
					source: Some(InputSource::Rpc),
					preflight_result: Some(Arc::new(move |disposition| {
						*preflight_probe.lock() = Some(disposition);
					})),
					..Default::default()
				}),
			)
			.await;

		release_tx.send(()).unwrap();
		let compaction = compact.await.unwrap();
		assert!(compaction.is_ok(), "compaction should complete: {compaction:?}");

		// `preflightResult` stays unset and the prompt carries the
		// compaction-in-progress error.
		assert!(prompt_result.is_err());
		assert!(
			prompt_result.as_ref().unwrap_err().contains("compaction is in progress"),
			"error: {prompt_result:?}"
		);
		assert!(preflight.lock().is_none());
		// Neither the live session nor the persisted entries contain the probe.
		let live_texts: Vec<String> = session
			.messages()
			.into_iter()
			.filter_map(|message| match message {
				AgentMessage::Llm(pi_ai::types::Message::User(user)) => match &user.content {
					pi_ai::types::UserContentField::Text(text) => Some(text.clone()),
					_ => None,
				},
				_ => None,
			})
			.collect();
		assert!(!live_texts.iter().any(|text| text.contains("PROBE-7150")));
		let persisted: Vec<String> = session_manager
			.lock()
			.get_entries()
			.into_iter()
			.filter_map(|entry| match serde_json::to_value(&entry) {
				Ok(value) => Some(value),
				Err(_) => None,
			})
			.filter(|value| value.get("message").and_then(|m| m.get("role")).and_then(JsonValue::as_str) == Some("user"))
			.filter_map(|value| {
				value
					.pointer("/message/content")
					.and_then(|content| content.as_array())
					.map(|parts| {
						parts
							.iter()
							.filter_map(|part| part.get("text").and_then(JsonValue::as_str))
							.collect::<Vec<_>>()
							.join("\n")
					})
			})
			.collect();
		assert!(!persisted.iter().any(|text| text.contains("PROBE-7150")));
		assert_eq!(events.lock().len(), 0);
		let _ = (user_id, StopReason::Stop, host);
	}
}
