//! Port of `packages/coding-agent/src/modes/print-mode.ts` — print mode
//! (single-shot): send prompts, output result, exit.
//!
//! Used for:
//! - `pi -p "prompt"` - text output
//! - `pi --mode json "prompt"` - JSON event stream
//!
//! Deviations:
//! - **D92** — signals and stdout go through the [`ModeStdio`] seam (upstream:
//!   `process` globals); the production entry [`run_print_mode`] installs
//!   [`StdioModeIo`]. The rebind hook reports rebind failures through the
//!   runtime's own switch path (upstream the callback's rejection propagates
//!   inside `runtimeHost`).
//! - **D93b — mode command actions** (see [`super`] docs): infallible actions
//!   log errors to stderr and report `cancelled: false`; the extension
//!   `setup` hook is not forwarded across the landed runtime seam.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures_util::future::BoxFuture;
use parking_lot::Mutex;

use pi_agent::types::AgentMessage;
use pi_ai::types::{AssistantContent, ImageContent, Message, StopReason};

use crate::core::agent_session::{AgentSession, AgentSessionEvent, ExtensionBindings, PromptOptions};
use crate::core::extensions::types::{ExtensionError, ExtensionMode};
use crate::utils::shell::kill_tracked_detached_children;

use super::json_event::to_json_event;
use super::{command_context_actions, ModeCore, ModeStdio, RebindSessionHook, RuntimeHost, StdioModeIo};

/// Output mode of print mode (`"text" | "json"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintOutputMode {
	Text,
	Json,
}

/// Options for print mode (`PrintModeOptions`).
#[derive(Clone)]
pub struct PrintModeOptions {
	/// Output mode: text for final response only, json for all events.
	pub mode: PrintOutputMode,
	/// Additional prompts sent after `initial_message` (`messages`).
	pub messages: Vec<String>,
	/// First message to send (may contain @file content).
	pub initial_message: Option<String>,
	/// Images attached to the initial message.
	pub initial_images: Option<Vec<ImageContent>>,
}

impl Default for PrintModeOptions {
	fn default() -> Self {
		PrintModeOptions {
			mode: PrintOutputMode::Text,
			messages: Vec::new(),
			initial_message: None,
			initial_images: None,
		}
	}
}

/// Mutable mode state shared with the signal/rebind closures.
struct PrintModeState {
	core: Arc<ModeCore>,
	mode: PrintOutputMode,
	session: Mutex<Arc<AgentSession>>,
	unsubscribe: Mutex<Option<Box<dyn FnOnce() + Send>>>,
	unsubscribe_backpressure: Mutex<Option<Box<dyn FnOnce() + Send>>>,
	disposed: AtomicBool,
}

impl PrintModeState {
	/// `disposeRuntime`.
	async fn dispose_runtime(&self) {
		if self.disposed.swap(true, Ordering::SeqCst) {
			return;
		}
		if let Some(unsubscribe) = self.unsubscribe.lock().take() {
			unsubscribe();
		}
		if let Some(unsubscribe_backpressure) = self.unsubscribe_backpressure.lock().take() {
			unsubscribe_backpressure();
		}
		self.core.host.dispose().await;
	}
}

/// `Request ${stopReason}` wire text for the error/aborted branch.
fn stop_reason_wire(reason: StopReason) -> &'static str {
	match reason {
		StopReason::Error => "error",
		StopReason::Aborted => "aborted",
		_ => "stop",
	}
}

/// Run in print (single-shot) mode. Sends prompts to the agent and outputs
/// the result. Returns the process exit code (the CLI exits with it).
pub async fn run_print_mode(host: Arc<dyn RuntimeHost>, options: PrintModeOptions) -> i32 {
	let io: Arc<dyn ModeStdio> = Arc::new(StdioModeIo::new());
	run_print_mode_with_io(host, io, options).await
}

/// [`run_print_mode`] over an explicit stdio seam (D91/D92; upstream tests
/// mock the output-guard module instead).
pub async fn run_print_mode_with_io(
	host: Arc<dyn RuntimeHost>,
	io: Arc<dyn ModeStdio>,
	options: PrintModeOptions,
) -> i32 {
	let PrintModeOptions { mode, messages, initial_message, initial_images } = options;
	let state = Arc::new(PrintModeState {
		core: Arc::new(ModeCore { host: Arc::clone(&host), io: Arc::clone(&io) }),
		mode,
		session: Mutex::new(host.session()),
		unsubscribe: Mutex::new(None),
		unsubscribe_backpressure: Mutex::new(None),
		disposed: AtomicBool::new(false),
	});

	// `registerSignalHandlers`: SIGTERM (and SIGHUP off-Windows) kill tracked
	// children, dispose the runtime, and exit.
	let signal_state = Arc::clone(&state);
	let cleanup_signals = io.register_signal_handlers(Arc::new(
		move |exit_code: i32| -> BoxFuture<'static, ()> {
			let state = Arc::clone(&signal_state);
			Box::pin(async move {
				kill_tracked_detached_children();
				state.dispose_runtime().await;
				state.core.io.exit(exit_code);
			})
		},
	));

	let rebind_state = Arc::clone(&state);
	let rebind_hook: RebindSessionHook = Arc::new(move |_session: &AgentSession| {
		let state = Arc::clone(&rebind_state);
		Box::pin(async move {
			let _ = rebind_session(&state).await;
		})
	});
	host.set_rebind_session(Some(rebind_hook));

	/// Exit-code carrier standing in for the upstream `exitCode` local (D92:
	/// the async block returns the code instead of mutating a captured
	/// variable across the await boundary).
	struct ExitCode(i32);
	let outcome: Result<ExitCode, String> = async {
		if mode == PrintOutputMode::Json {
			// `session.sessionManager.getHeader()`.
			let session = Arc::clone(&state.session.lock());
			let header = session.session_manager().lock().get_header();
			if let Some(header) = header {
				let line = serde_json::to_string(&header).unwrap_or_default();
				io.write_raw_stdout(&format!("{line}\n"));
			}
		}

		rebind_session(&state).await?;

		let session = Arc::clone(&state.session.lock());
		if let Some(initial_message) = &initial_message {
			session
				.prompt(
					initial_message.clone(),
					Some(PromptOptions {
						images: initial_images.clone(),
						..Default::default()
					}),
				)
				.await?;
		}

		for message in &messages {
			session.prompt(message.clone(), None).await?;
		}

		if mode == PrintOutputMode::Text {
			let state_snapshot = session.state();
			let last_message = state_snapshot.messages.last();
			if let Some(AgentMessage::Llm(Message::Assistant(assistant))) = last_message {
				if assistant.stop_reason == StopReason::Error || assistant.stop_reason == StopReason::Aborted {
					let reason = assistant
						.error_message
						.clone()
						.unwrap_or_else(|| format!("Request {}", stop_reason_wire(assistant.stop_reason)));
					io.write_stderr(&reason);
					return Ok(ExitCode(1));
				}
				for content in &assistant.content {
					if let AssistantContent::Text(text) = content {
						io.write_raw_stdout(&format!("{}\n", text.text));
					}
				}
			}
		}

		Ok(ExitCode(0))
	}
	.await;

	let exit_code = match outcome {
		Ok(ExitCode(code)) => code,
		Err(error) => {
			io.write_stderr(&error);
			1
		}
	};

	// `finally`: uninstall signals, dispose runtime, flush stdout.
	cleanup_signals();
	state.dispose_runtime().await;
	io.flush_raw_stdout().await;
	exit_code
}

/// `rebindSession`: bind extensions for this mode and resubscribe events.
async fn rebind_session(state: &Arc<PrintModeState>) -> Result<(), String> {
	let session = state.core.host.session();
	*state.session.lock() = Arc::clone(&session);

	let core = ModeCore { host: Arc::clone(&state.core.host), io: Arc::clone(&state.core.io) };
	let mode = match state.mode {
		PrintOutputMode::Json => ExtensionMode::Json,
		PrintOutputMode::Text => ExtensionMode::Print,
	};
	let io = Arc::clone(&state.core.io);
	session
		.bind_extensions(ExtensionBindings {
			ui_context: None,
			mode: Some(mode),
			command_context_actions: Some(command_context_actions(&core, &session)),
			on_error: Some(Arc::new(move |error: &ExtensionError| {
				io.write_stderr(&format!("Extension error ({}): {}", error.extension_path, error.error));
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

	if state.mode == PrintOutputMode::Json {
		let io = Arc::clone(&state.core.io);
		*state.unsubscribe.lock() = Some(session.subscribe(move |event: AgentSessionEvent| {
			let line = match to_json_event(&event) {
				Ok(line) => line,
				Err(message) => panic!("{message}"),
			};
			io.write_raw_stdout(&format!("{}\n", line));
		}));
		let io = Arc::clone(&state.core.io);
		*state.unsubscribe_backpressure.lock() = Some(session.agent().subscribe(move |_, _| {
			let io = Arc::clone(&io);
			Box::pin(async move { io.wait_for_raw_stdout_backpressure().await })
		}));
	}

	Ok(())
}


#[cfg(test)]
mod tests {
	//! Port of `test/print-mode.test.ts` (3 cases). The upstream fake session
	//! objects become a real [`AgentSession`] wired with test doubles
	//! (mechanical adaptation: TS structural-typing fakes → live components),
	//! and the `console.error` assertion reads the [`ModeStdio`] stderr
	//! capture.

	use std::sync::Arc;

	use futures_util::future::BoxFuture;
	use parking_lot::Mutex;

	use pi_agent::types::AgentMessage;
	use pi_ai::types::{ImageContent, Message, UserContentField};

	use crate::core::agent_session::{AgentSession, AgentSessionConfig, ScopedModelEntry};
	use crate::core::extensions::runner::emit_session_shutdown_event;
	use crate::core::extensions::types::{ExtensionFactory, SessionShutdownEvent, SessionShutdownReason};
	use crate::core::session_manager::SessionManager;
	use crate::core::settings_manager::{Settings, SettingsManager, SettingsManagerCreateOptions};

	use crate::modes::test_support::{
		assistant_message, delayed_stream_fn, failing_stream_fn, loader_with_factories, temp_dir, test_model_runtime,
		text_model,
	};
	use crate::modes::{ModeStdio, RebindSessionHook, RuntimeHost};
	use super::{run_print_mode_with_io, PrintModeOptions, PrintOutputMode};

	/// Stdout/stderr capture implementing the mode seam (upstream `vi.mock`
	/// of `core/output-guard`).
	#[derive(Default)]
	struct TestStdio {
		stdout: Mutex<Vec<String>>,
		stderr: Mutex<Vec<String>>,
	}

	impl ModeStdio for TestStdio {
		fn write_raw_stdout(&self, text: &str) {
			self.stdout.lock().push(text.to_string());
		}

		fn write_stderr(&self, text: &str) {
			let mut text = text.to_string();
			if !text.ends_with('\n') {
				text.push('\n');
			}
			self.stderr.lock().push(text);
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
			unreachable!("print mode signal exit({code}) is not exercised")
		}
	}

	/// `FakeRuntimeHost` upstream: fails session switches like the
	/// `vi.fn(async () => ({ cancelled: true }))` doubles and emits
	/// `session_shutdown` through the real extension runner on dispose.
	struct FakeRuntimeHost {
		session: Arc<AgentSession>,
		/// Transcript snapshot taken when `dispose` fires — the assertions
		/// run after the mode tears everything down (upstream fakes keep the
		/// messages alive because they are plain objects).
		messages_at_dispose: Arc<Mutex<Vec<AgentMessage>>>,
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
			let session = Arc::clone(&self.session);
			let messages_at_dispose = Arc::clone(&self.messages_at_dispose);
			Box::pin(async move {
				*messages_at_dispose.lock() = session.state().messages;
				emit_session_shutdown_event(
					session.extension_runner().as_ref(),
					&SessionShutdownEvent { reason: SessionShutdownReason::Quit, target_session_file: None },
				)
				.await;
			})
		}

		fn set_rebind_session(&self, _rebind_session: Option<RebindSessionHook>) {}
	}

	/// One `session_shutdown` recorder extension (`hasHandlers` double
	/// upstream records the emit through a real handler).
	fn shutdown_recorder_factory(events: Arc<Mutex<Vec<String>>>) -> ExtensionFactory {
		Arc::new(move |pi| {
			let events = Arc::clone(&events);
			let _ = pi.on(
				"session_shutdown",
				Arc::new(move |_event, _context| {
					let events = Arc::clone(&events);
					Box::pin(async move {
						events.lock().push("session_shutdown:quit".to_string());
						Ok(None) as Result<Option<serde_json::Value>, String>
					}) as BoxFuture<'static, Result<Option<serde_json::Value>, String>>
				}),
			);
			Box::pin(async move { Ok(()) })
		})
	}

	async fn build_session(stream_fn: pi_agent::types::StreamFn, factories: Vec<ExtensionFactory>) -> Arc<AgentSession> {
		let dir = temp_dir("print");
		let model = text_model(200_000);
		let loader = loader_with_factories(&dir, factories).await;
		let model_runtime = test_model_runtime(model.clone(), true).await;
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
		AgentSession::new(AgentSessionConfig {
			agent,
			session_manager: Arc::new(Mutex::new(SessionManager::in_memory(None, None, None).unwrap())),
			settings_manager: Arc::new(Mutex::new(SettingsManager::in_memory(
				&Settings::new(),
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
		})
	}

	/// `createRuntimeHost`: session plus shutdown-event recorder.
	async fn create_runtime_host(
		response: pi_ai::types::AssistantMessage,
	) -> (Arc<FakeRuntimeHost>, Arc<Mutex<Vec<String>>>) {
		let shutdown_events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
		let session = build_session(
			delayed_stream_fn(0, response),
			vec![shutdown_recorder_factory(Arc::clone(&shutdown_events))],
		)
		.await;
		let messages_at_dispose = Arc::new(Mutex::new(Vec::new()));
		(Arc::new(FakeRuntimeHost { session, messages_at_dispose }), shutdown_events)
	}

	/// Text of the last user message in the session (`session.prompt` was
	/// called with the text upstream).
	fn host_messages(host: &Arc<FakeRuntimeHost>) -> Vec<AgentMessage> {
		host.messages_at_dispose.lock().clone()
	}

	/// Text of the last user message in the dispose-time transcript
	/// (`session.prompt` was called with the text upstream).
	fn last_user_text(messages: &[AgentMessage]) -> Option<String> {
		messages.iter().rev().find_map(|message| match message {
			AgentMessage::Llm(Message::User(user)) => match &user.content {
				UserContentField::Text(text) => Some(text.clone()),
				UserContentField::Blocks(blocks) => blocks.iter().rev().find_map(|part| match part {
					pi_ai::types::UserContent::Text(text) => Some(text.text.clone()),
					_ => None,
				}),
			},
			_ => None,
		})
	}


	#[tokio::test(flavor = "multi_thread")]
	async fn emits_session_shutdown_in_text_mode() {
		let (host, shutdown_events) = create_runtime_host(assistant_message("done")).await;
		let stdio = Arc::new(TestStdio::default());
		let io: Arc<dyn ModeStdio> = Arc::<TestStdio>::clone(&stdio);
		let images = vec![ImageContent { data: "abc".to_string(), mime_type: "image/png".to_string() }];

		let exit_code = run_print_mode_with_io(
			Arc::clone(&host) as Arc<dyn RuntimeHost>,
			Arc::clone(&io),
			PrintModeOptions {
				mode: PrintOutputMode::Text,
				initial_message: Some("Say done".to_string()),
				initial_images: Some(images.clone()),
				..Default::default()
			},
		)
		.await;

		assert_eq!(exit_code, 0, "stderr: {}", stdio.stderr.lock().join(""));
		// `session.prompt` was called with ("Say done", { images }).
		assert_eq!(exit_code, 0, "stderr: {}", stdio.stderr.lock().join(""));
		let messages = host_messages(&host);
		// `session.prompt("Say done", { images })`: the prompt pipeline folds
		// the attached image into the user text (the runtime appends an
		// omission note when the image cannot be inlined).
		let text = last_user_text(&messages).unwrap();
		assert!(
			text.starts_with("Say done") && text.contains("[Image omitted"),
			"prompt text: {text}"
		);
		assert_eq!(*shutdown_events.lock(), vec!["session_shutdown:quit".to_string()]);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn emits_session_shutdown_in_json_mode() {
		let (host, shutdown_events) = create_runtime_host(assistant_message("done")).await;
		let stdio = Arc::new(TestStdio::default());
		let io: Arc<dyn ModeStdio> = Arc::<TestStdio>::clone(&stdio);

		let exit_code = run_print_mode_with_io(
			Arc::clone(&host) as Arc<dyn RuntimeHost>,
			Arc::clone(&io),
			PrintModeOptions {
				mode: PrintOutputMode::Json,
				messages: vec!["hello".to_string()],
				..Default::default()
			},
		)
		.await;

		assert_eq!(exit_code, 0, "stderr: {}", stdio.stderr.lock().join(""));
		let messages = host_messages(&host);
		assert_eq!(last_user_text(&messages).as_deref(), Some("hello"));
		assert_eq!(*shutdown_events.lock(), vec!["session_shutdown:quit".to_string()]);
		// JSON mode streamed the event lines of the prompt turn.
		let stdout = stdio.stdout.lock().join("");
		assert!(stdout.contains("\"type\":\"message_start\""), "stdout: {stdout}");
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn emits_session_shutdown_and_returns_non_zero_on_assistant_error() {
		let shutdown_events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
		let session = build_session(
			failing_stream_fn("provider failure"),
			vec![shutdown_recorder_factory(Arc::clone(&shutdown_events))],
		)
		.await;
		let messages_at_dispose = Arc::new(Mutex::new(Vec::new()));
		let host = Arc::new(FakeRuntimeHost { session, messages_at_dispose });
		let stdio = Arc::new(TestStdio::default());
		let io: Arc<dyn ModeStdio> = Arc::<TestStdio>::clone(&stdio);

		let exit_code = run_print_mode_with_io(
			Arc::clone(&host) as Arc<dyn RuntimeHost>,
			Arc::clone(&io) as Arc<dyn ModeStdio>,
			PrintModeOptions {
				mode: PrintOutputMode::Text,
				initial_message: Some("respond".to_string()),
				..Default::default()
			},
		)
		.await;

		assert_eq!(exit_code, 1, "stderr: {}", stdio.stderr.lock().join(""));
		// `console.error(assistantMsg.errorMessage)` upstream.
		assert_eq!(stdio.stderr.lock().join(""), "provider failure\n");
		assert_eq!(*shutdown_events.lock(), vec!["session_shutdown:quit".to_string()]);
	}
}
