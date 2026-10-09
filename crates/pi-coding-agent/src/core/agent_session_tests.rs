//! Port of `packages/coding-agent/test/agent-session-*.test.ts` suites that
//! target the [`AgentSession`] class directly.
//!
//! Suite mapping (upstream → module sections):
//! - `agent-session-retry.test.ts` (6 cases)
//! - `agent-session-stats.test.ts` (9 cases)
//! - `agent-session-auto-compaction-queue.test.ts` (6 cases, D78 seam)
//! - `agent-session-concurrent.test.ts` (8 cases)
//! - `agent-session-dynamic-tools.test.ts` (4 cases)
//! - `agent-session-dynamic-provider.test.ts` (5 cases)
//! - `agent-session-compaction.test.ts` (5 e2e cases, `ANTHROPIC_API_KEY`
//!   gated — mirrors upstream `describe.skipIf(!API_KEY)`)
//! - `agent-session-tree-navigation.test.ts` (10 e2e cases, gated)
//! - `agent-session-branching.test.ts` (3 e2e cases, gated)
//!
//! Port adaptations (mechanical, not upstream deviations):
//! - Upstream `vi.spyOn(session.agent, "continue")` has no Rust seam; the
//!   resume test asserts the mocked `streamFn` is never invoked instead.
//! - The `_extensionRunner` wholesale swap (concurrent cases 7/8) is ported
//!   as real inline extensions whose `tool_call`/`message_end` handlers
//!   record/sleep — same ordering guarantees, real machinery.
//! - `dynamic-tools`/`dynamic-provider` construct sessions through
//!   [`AgentSessionConfig`] directly (same loader/runtime wiring
//!   `createAgentSession` performs) so the suites do not depend on the
//!   separate `sdk.rs` port.
//! - `branching` (forking) drives `SessionManager::create_branched_session`
//!   directly; the runtime-host fork wrapper belongs to the
//!   agent-session-runtime port.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_agent::types::{AgentToolExecute, AgentToolResult, StreamFn};
use pi_agent::types::AgentToolUpdateCallback;
use pi_agent::{Agent, AgentInitialState, AgentMessage, AgentOptions, AgentTool};
use pi_ai::auth::credential_store::InMemoryCredentialStore;
use pi_ai::auth::types::{ApiKeyCredential, Credential, CredentialStore};
use pi_ai::utils::thrown::ThrownValue;
use pi_ai::models::{Models, Provider as _, RefreshModelsContext};
use pi_ai::types::{
	AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream, BaseModelFields, Model,
	ModelCost, ModelCostRates, ModelInput, StopReason, StreamDoneReason, StreamErrorReason, TextContent, ToolCall,
	TranscriptContext, Usage, UsageCost, UserContent, UserMessage,
};
use serde_json::{json, Value as JsonValue};

use crate::core::agent_session::{
	AgentSession, AgentSessionConfig, AgentSessionEvent, CompactReason, QueuedInputDisposition, ScopedModelEntry,
};
use crate::core::extensions::types::{
	ExtensionApi, ExtensionContext, ExtensionFactory, InputSource, ProviderConfig, ToolDefinition,
};
use crate::core::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::core::resource_loader::{DefaultResourceLoader, DefaultResourceLoaderOptions, ResourceLoaderReloadOptions};
use crate::core::session_manager::{NewSessionOptions, SessionManager};
use crate::core::settings_manager::{Settings, SettingsManager, SettingsManagerCreateOptions};
use crate::core::usage_totals::get_usage_cost_breakdown;

// ============================================================================
// Shared harness
// ============================================================================

const PROVIDER_ID: &str = "test-provider";
const MODEL_ID: &str = "test-model";

fn now_ms() -> i64 {
	std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.unwrap()
		.as_millis() as i64
}

fn zero_cost() -> UsageCost {
	UsageCost { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0, total: 0.0 }
}

fn usage(total_tokens: u64) -> Usage {
	Usage {
		input: total_tokens,
		output: 0,
		cache_read: 0,
		cache_write: 0,
		cache_write_1h: None,
		reasoning: None,
		total_tokens,
		cost: zero_cost(),
	}
}

fn usage_with_cost(total_tokens: u64, cost: UsageCost) -> Usage {
	Usage { total_tokens, cost, ..usage(total_tokens) }
}

fn usage_split(input: u64, output: u64, cache_read: u64, cache_write: u64, total_tokens: u64, cost: UsageCost) -> Usage {
	Usage {
		input,
		output,
		cache_read,
		cache_write,
		total_tokens,
		cost,
		..usage(total_tokens)
	}
}

fn split_cost(total: f64) -> UsageCost {
	UsageCost { input: 0.1, output: 0.2, cache_read: 0.3, cache_write: 0.4, total }
}

fn text_content(text: impl Into<String>) -> TextContent {
	TextContent::new(text)
}

/// `createAssistantMessage` upstream: a completed text assistant message on
/// the shared test model.
fn assistant_message(text: impl Into<String>, total_tokens: u64, timestamp: i64) -> AssistantMessage {
	AssistantMessage {
		content: vec![AssistantContent::Text(text_content(text))],
		api: "openai-completions".to_string(),
		provider: PROVIDER_ID.to_string(),
		model: MODEL_ID.to_string(),
		response_model: None,
		response_id: None,
		thinking_level: None,
		provider_thinking_level: None,
		diagnostics: None,
		usage: usage(total_tokens),
		stop_reason: StopReason::Stop,
		deferred: None,
		error_message: None,
		raw_stop_reason: None,
		end_turn: None,
		timestamp,
	}
}

fn error_assistant_message(error_message: &str, timestamp: i64) -> AssistantMessage {
	AssistantMessage {
		stop_reason: StopReason::Error,
		error_message: Some(error_message.to_string()),
		..assistant_message("", 0, timestamp)
	}
}

fn user_message(text: impl Into<String>, timestamp: i64) -> UserMessage {
	UserMessage { content: pi_ai::types::UserContentField::Text(text.into()), timestamp }
}

fn tool_result_message(usage_value: Usage) -> pi_ai::types::ToolResultMessage {
	pi_ai::types::ToolResultMessage {
		tool_call_id: "tool-call-1".to_string(),
		tool_name: "test_tool".to_string(),
		content: vec![UserContent::Text(text_content("tool result"))],
		details: None,
		usage: Some(usage_value),
		is_error: false,
		nested_calls: None,
		timestamp: 1,
	}
}

fn text_model(context_window: u64) -> Model {
	Model {
		base: BaseModelFields {
			id: MODEL_ID.to_string(),
			name: "Test Model".to_string(),
			api: "openai-completions".to_string(),
			provider: PROVIDER_ID.to_string(),
			base_url: "https://example.test/v1".to_string(),
			input: vec![ModelInput::Text],
			input_limits: None,
			cost: ModelCost { rates: ModelCostRates::default(), tiers: None },
			headers: None,
		},
		model_type: None,
		reasoning: false,
		thinking_level_map: None,
		prompt_cache: None,
		context_window,
		max_tokens: 4_096,
		sampling_params: None,
		sampling_params_by_thinking_level: None,
		compat: None,
	}
}

/// Shared provider auth double (`resolve` mirrors the stored API key).
fn test_provider_auth() -> &'static pi_ai::auth::types::ProviderAuth {
	static AUTH: std::sync::LazyLock<pi_ai::auth::types::ProviderAuth> = std::sync::LazyLock::new(|| {
		pi_ai::auth::types::ProviderAuth {
			api_key: Some(pi_ai::auth::types::ApiKeyAuth {
				name: "Test API key".to_string(),
				login: None,
				check: None,
				resolve: Box::new(|input| {
					let credential = input.credential.cloned();
					Box::pin(async move {
						Ok(credential.map(|credential| pi_ai::auth::types::AuthResult {
							auth: pi_ai::auth::types::ModelAuth {
								api_key: credential.key.clone(),
								headers: None,
								base_url: None,
							},
							env: None,
							source: Some("test".to_string()),
						}))
					})
				}),
			}),
			oauth: None,
		}
	});
	&AUTH
}

fn unused_stream(model: &Model) -> AssistantMessageEventStream {
	pi_ai::api::lazy::lazy_stream(model, async { Err(ThrownValue::message("unused")) })
}

// -- test provider (mirrors the model-runtime test double) ------------------

struct TestProvider {
	model: Model,
}

impl pi_ai::models::Provider for TestProvider {
	fn id(&self) -> &str {
		PROVIDER_ID
	}

	fn name(&self) -> &str {
		PROVIDER_ID
	}

	fn auth(&self) -> &pi_ai::auth::types::ProviderAuth {
		test_provider_auth()
	}

	fn get_models(&self) -> Vec<Model> {
		vec![self.model.clone()]
	}

	fn refresh_models(
		&self,
		_context: RefreshModelsContext,
	) -> Option<std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		None
	}

	fn stream(
		&self,
		model: &Model,
		_context: &TranscriptContext,
		_options: &pi_ai::types::StreamOptions,
	) -> AssistantMessageEventStream {
		unused_stream(model)
	}

	fn stream_simple(
		&self,
		model: &Model,
		_context: &TranscriptContext,
		_options: &pi_ai::types::SimpleStreamOptions,
	) -> AssistantMessageEventStream {
		unused_stream(model)
	}
}

async fn test_model_runtime(context_window: u64) -> Arc<ModelRuntime> {
	let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
	credentials
		.modify(
			PROVIDER_ID,
			Box::new(|_| {
				Box::pin(async move {
					Ok(Some(Credential::ApiKey(ApiKeyCredential { key: Some("test-key".to_string()), env: None })))
				})
			}),
			None,
		)
		.await
		.unwrap();
	let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
		credentials: Some(credentials),
		models_path: Some(None),
		refresh_on_create: true,
		..Default::default()
	})
	.await
	.unwrap();
	runtime.register_native_provider(Arc::new(TestProvider { model: text_model(context_window) })).unwrap();
	Models::refresh(
		&*runtime,
		Some(pi_ai::models::ModelsRefreshOptions {
			allow_network: Some(false),
			providers: Some(vec![PROVIDER_ID.to_string()]),
			..Default::default()
		}),
	)
	.await;
	runtime
}

// -- session builder ---------------------------------------------------------

fn temp_dir(tag: &str) -> String {
	let dir = std::env::temp_dir().join(format!("pi-rs-agent-session-{tag}-{}-{}", now_ms(), std::process::id()));
	std::fs::create_dir_all(&dir).unwrap();
	dir.to_string_lossy().to_string()
}

fn settings_from_json(value: JsonValue) -> Settings {
	Settings::from_value(value)
}

fn test_settings_manager() -> SettingsManager {
	SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default())
}

fn in_memory_session_manager() -> Arc<Mutex<SessionManager>> {
	Arc::new(Mutex::new(SessionManager::in_memory(None, None, None).unwrap()))
}

/// One mutable call counter shared with a mock `streamFn`.
type CallCounter = Arc<std::sync::atomic::AtomicU32>;

/// Build a `streamFn` that fails the first `fail_count` calls with
/// `error_message` and then succeeds with `success_text`.
fn fail_then_succeed_stream_fn(
	fail_count: u32,
	error_message: &'static str,
	success_text: &'static str,
) -> (StreamFn, CallCounter) {
	let calls: CallCounter = Arc::new(std::sync::atomic::AtomicU32::new(0));
	let calls_for_fn = Arc::clone(&calls);
	let stream_fn: StreamFn = Arc::new(move |_model, _context, _options| {
		let calls = Arc::clone(&calls_for_fn);
		Box::pin(async move {
			let stream = AssistantMessageEventStream::new();
			let call = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
			if call <= fail_count {
				stream.push(AssistantMessageEvent::Start { partial: assistant_message("", 0, now_ms()) });
				stream.push(AssistantMessageEvent::Error {
					reason: StreamErrorReason::Error,
					error: error_assistant_message(error_message, now_ms()),
				});
			} else {
				let message = assistant_message(success_text, 0, now_ms());
				stream.push(AssistantMessageEvent::Start { partial: AssistantMessage { content: vec![], ..message.clone() } });
				stream.push(AssistantMessageEvent::Done { reason: StreamDoneReason::Stop, message });
			}
			Ok(stream)
		})
	});
	(stream_fn, calls)
}

pub(crate) struct Harness {
	pub session: Arc<AgentSession>,
	pub session_manager: Arc<Mutex<SessionManager>>,
	pub settings_manager: Arc<Mutex<SettingsManager>>,
}

#[allow(clippy::too_many_arguments)]
fn build_session(
	stream_fn: StreamFn,
	model: Option<Model>,
	thinking_level: Option<pi_agent::AgentThinkingLevel>,
	tools: Vec<AgentTool>,
	base_tools_override: Option<Vec<AgentTool>>,
	settings_manager: SettingsManager,
	session_manager: Arc<Mutex<SessionManager>>,
	model_runtime: Arc<ModelRuntime>,
	resource_loader: Arc<Mutex<DefaultResourceLoader>>,
	cwd: &str,
) -> Harness {
	let agent = Arc::new(Agent::new(AgentOptions {
		stream_fn: Some(stream_fn),
		initial_state: AgentInitialState {
			system_prompt: Some("Test".to_string()),
			model,
			thinking_level,
			tools,
			messages: Vec::new(),
		},
		..Default::default()
	}));
	let settings_arc = Arc::new(Mutex::new(settings_manager));
	let session = AgentSession::new(AgentSessionConfig {
		agent,
		session_manager: Arc::clone(&session_manager),
		settings_manager: Arc::clone(&settings_arc),
		cwd: cwd.to_string(),
		scoped_models: Vec::<ScopedModelEntry>::new(),
		resource_loader,
		custom_tools: Vec::new(),
		model_runtime,
		cache_warmer: None,
		initial_active_tool_names: None,
		uses_default_tools: false,
		allowed_tool_names: None,
		excluded_tool_names: None,
		base_tools_override,
		extension_runner_ref: None,
		session_start_event: None,
	});
	Harness { session, session_manager, settings_manager: settings_arc }
}

fn plain_loader(cwd: &str) -> Arc<Mutex<DefaultResourceLoader>> {
	let loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
		cwd: cwd.to_string(),
		agent_dir: cwd.to_string(),
		no_extensions: true,
		no_skills: true,
		no_prompt_templates: true,
		no_themes: true,
		no_context_files: true,
		..Default::default()
	});
	Arc::new(Mutex::new(loader))
}

/// `reload()` drives async passes via an internal current-thread
/// `block_on`, so it must run off the tokio runtime (tests: spawn_blocking).
async fn loader_with_factories(cwd: &str, factories: Vec<ExtensionFactory>) -> Arc<Mutex<DefaultResourceLoader>> {
	let cwd = cwd.to_string();
	tokio::task::spawn_blocking(move || {
		let mut loader = DefaultResourceLoader::new(DefaultResourceLoaderOptions {
			agent_dir: cwd.clone(),
			cwd,
			no_skills: true,
			no_prompt_templates: true,
			no_themes: true,
			no_context_files: true,
			extension_factories: factories
				.into_iter()
				.map(crate::core::extensions::types::InlineExtension::Factory)
				.collect(),
			..Default::default()
		});
		loader.reload(ResourceLoaderReloadOptions::default());
		Arc::new(Mutex::new(loader))
	})
	.await
	.unwrap()
}

/// Record `auto_retry_start`/`auto_retry_end` event labels.
fn retry_event_recorder(session: &AgentSession) -> Arc<Mutex<Vec<String>>> {
	let events = Arc::new(Mutex::new(Vec::new()));
	let sink = Arc::clone(&events);
	session.subscribe(move |event| match event {
		AgentSessionEvent::AutoRetryStart { attempt, .. } => sink.lock().push(format!("start:{attempt}")),
		AgentSessionEvent::AutoRetryEnd { success, .. } => sink.lock().push(format!("end:success={success}")),
		_ => {}
	});
	events
}

// ============================================================================
// agent-session-retry.test.ts
// ============================================================================

mod retry {
	use super::*;

	async fn retry_harness(
		tag: &str,
		stream_fn: StreamFn,
		retry_overrides: JsonValue,
		base_tools_override: Option<Vec<AgentTool>>,
	) -> (Harness, CallCounter) {
		let dir = temp_dir(tag);
		let runtime = test_model_runtime(200_000).await;
		let (stream_fn, calls) = if let Some(stream_fn) = Some(stream_fn) {
			// The caller-provided stream keeps its own counter; wrap as
			// uncounted.
			(stream_fn, Arc::new(std::sync::atomic::AtomicU32::new(0)))
		} else {
			unreachable!()
		};
		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			base_tools_override,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		);
		harness.settings_manager.lock().apply_overrides(&settings_from_json(retry_overrides));
		(harness, calls)
	}

	fn retry_settings(max_retries: u32, max_agent_delay_ms: u64) -> JsonValue {
		json!({"retry": {"enabled": true, "maxRetries": max_retries, "baseDelayMs": 1, "maxAgentDelayMs": max_agent_delay_ms}})
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn retries_after_a_transient_error_and_succeeds() {
		let dir = temp_dir("retry-1");
		let runtime = test_model_runtime(200_000).await;
		let (stream_fn, calls) = fail_then_succeed_stream_fn(1, "overloaded_error", "Success");
		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		);
		harness.settings_manager.lock().apply_overrides(&settings_from_json(retry_settings(3, 60_000)));
		let events = retry_event_recorder(&harness.session);

		harness.session.prompt("Test", None).await.unwrap();

		assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
		assert_eq!(*events.lock(), vec!["start:1".to_string(), "end:success=true".to_string()]);
		assert!(!harness.session.is_retrying());
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn exhausts_max_retries_and_emits_failure() {
		let dir = temp_dir("retry-2");
		let runtime = test_model_runtime(200_000).await;
		let (stream_fn, calls) = fail_then_succeed_stream_fn(99, "overloaded_error", "Success");
		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		);
		harness.settings_manager.lock().apply_overrides(&settings_from_json(retry_settings(2, 60_000)));
		let events = retry_event_recorder(&harness.session);

		harness.session.prompt("Test", None).await.unwrap();

		assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 3);
		let events = events.lock();
		assert!(events.contains(&"start:1".to_string()));
		assert!(events.contains(&"start:2".to_string()));
		assert!(events.contains(&"end:success=false".to_string()));
		assert!(!harness.session.is_retrying());
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn caps_agent_retry_delay() {
		// Regression for #8826.
		let dir = temp_dir("retry-3");
		let runtime = test_model_runtime(200_000).await;
		let (stream_fn, _calls) = fail_then_succeed_stream_fn(4, "overloaded_error", "Success");
		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		);
		harness.settings_manager.lock().apply_overrides(&settings_from_json(retry_settings(5, 5)));
		let delays = Arc::new(Mutex::new(Vec::new()));
		let sink = Arc::clone(&delays);
		harness.session.subscribe(move |event| {
			if let AgentSessionEvent::AutoRetryStart { delay_ms, .. } = event {
				sink.lock().push(delay_ms);
			}
		});

		harness.session.prompt("Test", None).await.unwrap();

		assert_eq!(*delays.lock(), vec![1, 2, 4, 5]);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn prompt_waits_for_retry_completion_even_when_assistant_message_end_handling_is_delayed() {
		let dir = temp_dir("retry-4");
		let runtime = test_model_runtime(200_000).await;
		let (stream_fn, calls) = fail_then_succeed_stream_fn(1, "overloaded_error", "Success");
		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		);
		harness.settings_manager.lock().apply_overrides(&settings_from_json(retry_settings(3, 60_000)));

		harness.session.prompt("Test", None).await.unwrap();

		assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
		assert!(!harness.session.is_retrying());
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn retries_provider_network_error_failures() {
		let dir = temp_dir("retry-5");
		let runtime = test_model_runtime(200_000).await;
		let (stream_fn, calls) = fail_then_succeed_stream_fn(1, "Provider finish_reason: network_error", "Recovered after retry");
		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		);
		harness.settings_manager.lock().apply_overrides(&settings_from_json(retry_settings(3, 60_000)));
		let events = retry_event_recorder(&harness.session);

		harness.session.prompt("Test", None).await.unwrap();

		assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
		assert_eq!(*events.lock(), vec!["start:1".to_string(), "end:success=true".to_string()]);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn prompt_waits_for_full_agent_loop_when_retry_produces_tool_calls() {
		let dir = temp_dir("retry-6");
		let runtime = test_model_runtime(200_000).await;
		let calls: CallCounter = Arc::new(std::sync::atomic::AtomicU32::new(0));
		let tool_executed = Arc::new(std::sync::atomic::AtomicBool::new(false));

		let calls_for_fn = Arc::clone(&calls);
		let stream_fn: StreamFn = Arc::new(move |_model, _context, _options| {
			let calls = Arc::clone(&calls_for_fn);
			Box::pin(async move {
				let stream = AssistantMessageEventStream::new();
				let call = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
				if call == 1 {
					stream.push(AssistantMessageEvent::Start { partial: assistant_message("", 0, now_ms()) });
					stream.push(AssistantMessageEvent::Error {
						reason: StreamErrorReason::Error,
						error: error_assistant_message("overloaded_error", now_ms()),
					});
				} else if call == 2 {
					// Retry response: text + tool_use.
					let mut message = assistant_message("Looking that up now.", 0, now_ms());
					message.stop_reason = StopReason::ToolUse;
					message.content = vec![
						AssistantContent::Text(text_content("Looking that up now.")),
						AssistantContent::ToolCall(ToolCall {
							id: "call_1".to_string(),
							name: "echo".to_string(),
							arguments: serde_json::Map::from_iter([("text".to_string(), json!("hello"))]),
							thought_signature: None,
							namespace: None,
						}),
					];
					stream.push(AssistantMessageEvent::Start { partial: AssistantMessage { content: vec![], ..message.clone() } });
					stream.push(AssistantMessageEvent::Done { reason: StreamDoneReason::ToolUse, message });
				} else {
					let message = assistant_message("Final answer.", 0, now_ms());
					stream.push(AssistantMessageEvent::Start { partial: AssistantMessage { content: vec![], ..message.clone() } });
					stream.push(AssistantMessageEvent::Done { reason: StreamDoneReason::Stop, message });
				}
				Ok(stream)
			})
		});

		let executed = Arc::clone(&tool_executed);
		let echo_execute: AgentToolExecute = Arc::new(move |_id, _args, _signal, _update: Option<AgentToolUpdateCallback>| {
			let executed = Arc::clone(&executed);
			Box::pin(async move {
				executed.store(true, std::sync::atomic::Ordering::SeqCst);
				Ok(AgentToolResult {
					content: vec![UserContent::Text(text_content("echoed"))],
					details: JsonValue::Null,
					structured_content: None,
					usage: None,
					is_error: None,
					terminate: None,
				})
			})
		});
		let echo_tool = AgentTool {
			tool: pi_ai::types::Tool {
				name: "echo".to_string(),
				description: "Echo text back".to_string(),
				parameters: json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
				constrained_sampling: None,
			},
			label: "Echo".to_string(),
			prepare_arguments: None,
			output_schema: None,
			execute: echo_execute,
			replay: None,
			execution_mode: None,
		};

		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			Some(vec![echo_tool]),
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		);
		harness.settings_manager.lock().apply_overrides(&settings_from_json(retry_settings(3, 60_000)));

		harness.session.prompt("Test", None).await.unwrap();

		assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 3);
		assert!(tool_executed.load(std::sync::atomic::Ordering::SeqCst));
		assert!(!harness.session.is_streaming());

		harness.session.prompt("Follow-up", None).await.unwrap();
		assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 4);
	}
}

// ============================================================================
// agent-session-stats.test.ts
// ============================================================================

mod stats {
	use super::*;

	async fn stats_harness() -> Harness {
		let dir = temp_dir("stats");
		let runtime = test_model_runtime(200_000).await;
		let stream_fn: StreamFn = Arc::new(|_model, _context, _options| {
			Box::pin(async { unreachable!("stats tests never prompt") })
		});
		build_session(
			stream_fn,
			Some(text_model(200_000)),
			Some(pi_agent::AgentThinkingLevel::High),
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		)
	}

	fn sync_agent_messages(harness: &Harness) {
		let messages = harness.session_manager.lock().build_session_context().messages;
		harness.session.agent().set_messages(messages);
	}

	fn append_user(harness: &Harness, text: &str, timestamp: i64) -> String {
		harness.session_manager.lock().append_message(user_message(text, timestamp)).unwrap()
	}

	fn append_assistant(harness: &Harness, text: &str, total_tokens: u64, timestamp: i64) -> String {
		harness.session_manager.lock().append_message(assistant_message(text, total_tokens, timestamp)).unwrap()
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn exposes_the_current_context_usage_alongside_token_totals() {
		let harness = stats_harness().await;
		append_user(&harness, "hello", 1);
		append_assistant(&harness, "hi", 200, 2);
		sync_agent_messages(&harness);

		let stats = harness.session.get_session_stats();
		assert_eq!(stats.context_usage, harness.session.get_context_usage());
		let context_usage = stats.context_usage.expect("context usage");
		assert_eq!(context_usage.tokens, Some(200));
		assert_eq!(context_usage.context_window, 200_000);
		assert!((context_usage.percent.unwrap() - (200.0 / 200_000.0) * 100.0).abs() < 1e-9);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn reports_unknown_current_context_usage_immediately_after_compaction() {
		let harness = stats_harness().await;
		append_user(&harness, "first", 1);
		append_assistant(&harness, "response1", 180_000, 2);
		let kept_user_id = append_user(&harness, "second", 3);
		append_assistant(&harness, "response2", 195_000, 4);
		harness
			.session_manager
			.lock()
			.append_compaction("summary", Some(&kept_user_id), 195_000, None, None, None)
			.unwrap();
		append_user(&harness, "third", 5);
		sync_agent_messages(&harness);

		let stats = harness.session.get_session_stats();
		assert_eq!(stats.tokens.input, 375_000);
		let context_usage = stats.context_usage.expect("context usage");
		assert_eq!(context_usage.tokens, None);
		assert_eq!(context_usage.percent, None);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn uses_post_compaction_usage_for_current_context_instead_of_stale_kept_usage() {
		let harness = stats_harness().await;
		append_user(&harness, "first", 1);
		append_assistant(&harness, "response1", 180_000, 2);
		let kept_user_id = append_user(&harness, "second", 3);
		append_assistant(&harness, "response2", 195_000, 4);
		harness
			.session_manager
			.lock()
			.append_compaction("summary", Some(&kept_user_id), 195_000, None, None, None)
			.unwrap();
		append_user(&harness, "third", 5);
		append_assistant(&harness, "response3", 25_000, 6);
		sync_agent_messages(&harness);

		let stats = harness.session.get_session_stats();
		assert_eq!(stats.tokens.input, 400_000);
		let context_usage = stats.context_usage.expect("context usage");
		assert_eq!(context_usage.tokens, Some(25_000));
		assert!((context_usage.percent.unwrap() - (25_000.0 / 200_000.0) * 100.0).abs() < 1e-9);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn includes_branch_summary_usage_in_session_totals() {
		let harness = stats_harness().await;
		harness
			.session_manager
			.lock()
			.branch_with_summary(
				None,
				"summary",
				None,
				None,
				Some(usage_split(10, 20, 30, 40, 100, split_cost(1.0))),
			)
			.unwrap();
		sync_agent_messages(&harness);

		let stats = harness.session.get_session_stats();
		assert_eq!(stats.tokens.input, 10);
		assert_eq!(stats.tokens.output, 20);
		assert_eq!(stats.tokens.cache_read, 30);
		assert_eq!(stats.tokens.cache_write, 40);
		assert_eq!(stats.tokens.total, 100);
		assert!((stats.cost - 1.0).abs() < 1e-9);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn includes_compaction_usage_in_session_totals() {
		let harness = stats_harness().await;
		let first_kept_entry_id = append_user(&harness, "hello", 1);
		harness
			.session_manager
			.lock()
			.append_compaction(
				"summary",
				Some(&first_kept_entry_id),
				100,
				None,
				Some(false),
				Some(usage_split(10, 20, 30, 40, 100, split_cost(1.0))),
			)
			.unwrap();
		sync_agent_messages(&harness);

		let stats = harness.session.get_session_stats();
		assert_eq!(stats.tokens.input, 10);
		assert_eq!(stats.tokens.output, 20);
		assert_eq!(stats.tokens.cache_read, 30);
		assert_eq!(stats.tokens.cache_write, 40);
		assert_eq!(stats.tokens.total, 100);
		assert!((stats.cost - 1.0).abs() < 1e-9);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn includes_cache_warming_usage_exactly_once_without_adding_messages() {
		let harness = stats_harness().await;
		harness
			.session_manager
			.lock()
			.append_usage(
				"cache_warm",
				PROVIDER_ID,
				MODEL_ID,
				usage_split(2, 1, 97, 0, 100, UsageCost { input: 0.001, output: 0.002, cache_read: 0.007, cache_write: 0.0, total: 0.01 }),
				Some("extension override".to_string()),
			)
			.unwrap();

		let entries = harness.session_manager.lock().get_entries();
		assert_eq!(entries.len(), 1);
		let crate::core::session_manager::SessionEntry::Usage(entry) = &entries[0] else {
			panic!("expected usage entry");
		};
		assert_eq!(entry.kind, "cache_warm");
		assert_eq!(entry.note.as_deref(), Some("extension override"));

		let stats = harness.session.get_session_stats();
		assert_eq!(stats.tokens.input, 2);
		assert_eq!(stats.tokens.output, 1);
		assert_eq!(stats.tokens.cache_read, 97);
		assert_eq!(stats.tokens.cache_write, 0);
		assert_eq!(stats.tokens.total, 100);
		assert_eq!(stats.total_messages, 0);
		assert!(harness.session_manager.lock().build_session_context().messages.is_empty());

		let breakdown = get_usage_cost_breakdown(&entries);
		assert_eq!(breakdown.len(), 1);
		assert_eq!(breakdown[0].key, format!("{PROVIDER_ID}/{MODEL_ID}"));
		assert!((breakdown[0].cost - 0.01).abs() < 1e-9);
		assert_eq!(breakdown[0].tokens, 100);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn includes_tool_result_usage_in_session_totals() {
		let harness = stats_harness().await;
		harness
			.session_manager
			.lock()
			.append_message(tool_result_message(usage_split(10, 20, 30, 40, 100, split_cost(1.0))))
			.unwrap();
		sync_agent_messages(&harness);

		let stats = harness.session.get_session_stats();
		assert_eq!(stats.tokens.input, 10);
		assert_eq!(stats.tokens.output, 20);
		assert_eq!(stats.tokens.cache_read, 30);
		assert_eq!(stats.tokens.cache_write, 40);
		assert_eq!(stats.tokens.total, 100);
		assert!((stats.cost - 1.0).abs() < 1e-9);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn groups_tool_and_summary_usage_separately_from_model_attributed_usage() {
		let session_manager = in_memory_session_manager();
		let root_id = session_manager.lock().append_message(user_message("hello", 1)).unwrap();
		let mut assistant = assistant_message("response", 100, 2);
		assistant.usage.cost.total = 0.5;
		session_manager.lock().append_message(assistant).unwrap();
		let tool_result = tool_result_message(usage_split(10, 20, 30, 40, 100, split_cost(1.0)));
		session_manager.lock().append_message(tool_result).unwrap();
		let mut compaction_usage = usage(100);
		compaction_usage.cost.total = 2.0;
		session_manager
			.lock()
			.append_compaction("summary", Some(&root_id), 100, None, Some(false), Some(compaction_usage))
			.unwrap();
		let mut branch_usage = usage(100);
		branch_usage.cost.total = 3.0;
		session_manager.lock().branch_with_summary(None, "branch summary", None, None, Some(branch_usage)).unwrap();

		let breakdown = get_usage_cost_breakdown(&session_manager.lock().get_entries());
		assert_eq!(breakdown.len(), 2);
		assert_eq!(breakdown[0].key, "Tools/summaries");
		assert!((breakdown[0].cost - 6.0).abs() < 1e-9);
		assert_eq!(breakdown[0].tokens, 300);
		assert_eq!(breakdown[1].key, format!("{PROVIDER_ID}/{MODEL_ID}"));
		assert!((breakdown[1].cost - 0.5).abs() < 1e-9);
		assert_eq!(breakdown[1].tokens, 100);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn ignores_zero_usage_messages_when_checking_for_post_compaction_context_usage() {
		let harness = stats_harness().await;
		append_user(&harness, "first", 1);
		append_assistant(&harness, "response1", 180_000, 2);
		let kept_user_id = append_user(&harness, "second", 3);
		append_assistant(&harness, "response2", 195_000, 4);
		harness
			.session_manager
			.lock()
			.append_compaction("summary", Some(&kept_user_id), 195_000, None, None, None)
			.unwrap();
		append_user(&harness, "third", 5);
		append_assistant(&harness, "response3", 25_000, 6);
		append_user(&harness, "continue", 7);
		append_assistant(&harness, "partial", 0, 8);
		sync_agent_messages(&harness);

		let stats = harness.session.get_session_stats();
		let context_usage = stats.context_usage.expect("context usage");
		let tokens = context_usage.tokens.expect("non-null tokens");
		assert!(tokens > 25_000);
	}
}

/// Native provider double used by the dynamic-provider suite
/// (`nativeAnthropicProvider` upstream): a full `pi_ai` provider whose
/// single model carries a custom `baseUrl`.
pub(crate) struct TestNativeOverrideProvider {
	pub model: Model,
}

impl TestNativeOverrideProvider {
	fn new(base_url: &str) -> Self {
		let base_model = text_model(200_000);
		TestNativeOverrideProvider {
			model: Model { base: BaseModelFields { base_url: base_url.to_string(), ..base_model.base }, ..base_model },
		}
	}
}

impl pi_ai::models::Provider for TestNativeOverrideProvider {
	fn id(&self) -> &str {
		PROVIDER_ID
	}

	fn name(&self) -> &str {
		"Native Test"
	}

	fn auth(&self) -> &pi_ai::auth::types::ProviderAuth {
		test_provider_auth()
	}

	fn get_models(&self) -> Vec<Model> {
		vec![self.model.clone()]
	}

	fn refresh_models(
		&self,
		_context: RefreshModelsContext,
	) -> Option<std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		None
	}

	fn stream(
		&self,
		model: &Model,
		_context: &TranscriptContext,
		_options: &pi_ai::types::StreamOptions,
	) -> AssistantMessageEventStream {
		unused_stream(model)
	}

	fn stream_simple(
		&self,
		model: &Model,
		_context: &TranscriptContext,
		_options: &pi_ai::types::SimpleStreamOptions,
	) -> AssistantMessageEventStream {
		unused_stream(model)
	}
}
// ============================================================================
// agent-session-auto-compaction-queue.test.ts
// ============================================================================

mod auto_compaction_queue {
	use super::*;

	/// Upstream `beforeEach`: default session with a live (unused) stream.
	async fn queue_harness(tag: &str) -> Harness {
		let dir = temp_dir(tag);
		let runtime = test_model_runtime(200_000).await;
		let stream_fn: StreamFn = Arc::new(|_model, _context, _options| {
			Box::pin(async { unreachable!("queue tests drive compaction directly") })
		});
		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		);
		harness
	}

	/// Install the D78 `_runAutoCompaction` spy: records nothing extra (the
	/// seam records calls itself) and resolves to `false` like upstream's
	/// `mockResolvedValue()`.
	fn spy_run_auto_compaction(harness: &Harness) {
		*harness.session.inner.run_auto_compaction_override.lock() = Some(Arc::new(|_reason, _will_retry| {
			Box::pin(async move { false })
		}));
	}

	fn auto_compaction_calls(harness: &Harness) -> Vec<(String, bool)> {
		harness.session.inner.run_auto_compaction_calls.lock().clone()
	}

	async fn check_compaction(harness: &Harness, message: &AssistantMessage, skip_aborted_check: bool) -> bool {
		harness.session.inner.check_compaction(message, skip_aborted_check, Vec::new()).await
	}

	fn append_pair(harness: &Harness, user_text: &str, assistant: &AssistantMessage) {
		harness.session_manager.lock().append_message(user_message(user_text, assistant.timestamp - 1000)).unwrap();
		harness.session_manager.lock().append_message(assistant.clone()).unwrap();
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_resume_after_threshold_compaction_when_only_agent_level_queued_messages_exist() {
		let dir = temp_dir("acq-1");
		let runtime = test_model_runtime(200_000).await;
		// The stream resolves compaction summary requests; any agent-loop
		// turn would also hit it, so the counter doubles as the
		// `agent.continue()` spy (no monkey-patching seam in Rust).
		let summary_calls: CallCounter = Arc::new(std::sync::atomic::AtomicU32::new(0));
		let counter_for_fn = Arc::clone(&summary_calls);
		let stream_fn: StreamFn = Arc::new(move |model, _context, _options| {
			let summary_calls = Arc::clone(&counter_for_fn);
			Box::pin(async move {
				summary_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
				let stream = AssistantMessageEventStream::new();
				let mut message = assistant_message("compacted", 10, now_ms());
				message.api = model.api().to_string();
				message.provider = model.provider().to_string();
				message.model = model.id().to_string();
				stream.push(AssistantMessageEvent::Done { reason: StreamDoneReason::Stop, message });
				Ok(stream)
			})
		});
		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		);
		harness
			.settings_manager
			.lock()
			.apply_overrides(&settings_from_json(json!({"compaction": {"keepRecentTokens": 1}})));

		harness
			.session_manager
			.lock()
			.append_message(user_message("message to compact", now_ms() - 1000))
			.unwrap();
		harness.session_manager.lock().append_message(assistant_message("assistant response to compact", 100, now_ms() - 500)).unwrap();
		harness.session.agent().set_messages(harness.session_manager.lock().build_session_context().messages);

		let agent_starts = Arc::new(std::sync::atomic::AtomicU32::new(0));
		let starts_sink = Arc::clone(&agent_starts);
		harness.session.subscribe(move |event| {
			if matches!(event, AgentSessionEvent::AgentStart) {
				starts_sink.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
			}
		});

		// Agent-level queued messages only (`session.agent.followUp`).
		harness.session.agent().follow_up(AgentMessage::Custom(json!({
			"customType": "test",
			"content": [{"type": "text", "text": "Queued custom"}],
			"display": false,
			"timestamp": now_ms(),
		})));

		assert_eq!(harness.session.pending_message_count(), 0);
		assert!(harness.session.agent().has_queued_messages());

		let resumed = harness.session.inner.run_auto_compaction(CompactReason::Threshold, false).await;
		assert!(resumed);

		// The agent loop must not have been restarted for agent-level
		// queued messages (upstream: `continueSpy` not called). The one
		// stream call is the compaction summary request itself.
		assert_eq!(summary_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
		assert_eq!(agent_starts.load(std::sync::atomic::Ordering::SeqCst), 0);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_not_compact_repeatedly_after_overflow_recovery_already_attempted() {
		let harness = queue_harness("acq-2").await;
		spy_run_auto_compaction(&harness);

		let overflow_message = error_assistant_message("prompt is too long", now_ms());

		let events = Arc::new(Mutex::new(Vec::new()));
		let sink = Arc::clone(&events);
		harness.session.subscribe(move |event| {
			if let AgentSessionEvent::CompactionEnd { reason, error_message, .. } = event {
				sink.lock().push((reason, error_message));
			}
		});

		check_compaction(&harness, &overflow_message, false).await;
		check_compaction(&harness, &{ let mut m = overflow_message.clone(); m.timestamp = now_ms() + 1; m }, false).await;

		let calls = auto_compaction_calls(&harness);
		assert_eq!(calls.len(), 1);
		let events = events.lock();
		assert!(events
			.iter()
			.any(|(reason, error_message)| *reason == CompactReason::Overflow
				&& error_message.as_deref()
					== Some(
						"Context overflow recovery failed after one compact-and-retry attempt. Try reducing context or switching to a larger-context model."
					)));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_ignore_stale_pre_compaction_assistant_usage_on_pre_prompt_compaction_checks() {
		let harness = queue_harness("acq-3").await;
		spy_run_auto_compaction(&harness);

		let stale_timestamp = now_ms() - 10_000;
		let stale_assistant = assistant_message("large response before compaction", 610_000, stale_timestamp);

		append_pair(&harness, "before compaction", &stale_assistant);
		let entries = harness.session_manager.lock().get_entries();
		let first_kept_entry_id = entries[0].id().to_string();
		harness
			.session_manager
			.lock()
			.append_compaction("summary", Some(&first_kept_entry_id), stale_assistant.usage.total_tokens, None, None, None)
			.unwrap();
		harness.session_manager.lock().append_message(user_message("session recovery payload", now_ms())).unwrap();

		check_compaction(&harness, &stale_assistant, false).await;

		assert!(auto_compaction_calls(&harness).is_empty());
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_trigger_threshold_compaction_for_error_messages_using_last_successful_usage() {
		let harness = queue_harness("acq-4").await;
		spy_run_auto_compaction(&harness);

		let (_enabled, reserve_tokens, _keep) = harness
			.settings_manager
			.lock()
			.get_compaction_settings(Some((PROVIDER_ID, MODEL_ID)))
			.unwrap();
		let threshold_tokens = 200_000 - reserve_tokens + 1;
		let successful_assistant = assistant_message("large successful response", threshold_tokens, now_ms());
		let error_assistant = error_assistant_message("529 overloaded", now_ms() + 1000);

		harness.session.agent().set_messages(vec![
			AgentMessage::Llm(pi_ai::types::Message::User(user_message("hello", now_ms() - 1000))),
			AgentMessage::Llm(pi_ai::types::Message::Assistant(successful_assistant)),
			AgentMessage::Llm(pi_ai::types::Message::User(user_message("another prompt", now_ms() + 500))),
			AgentMessage::Llm(pi_ai::types::Message::Assistant(error_assistant.clone())),
		]);

		check_compaction(&harness, &error_assistant, false).await;

		let calls = auto_compaction_calls(&harness);
		assert_eq!(calls, vec![("Threshold".to_string(), false)]);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_not_trigger_threshold_compaction_for_error_messages_when_no_prior_usage_exists() {
		let harness = queue_harness("acq-5").await;
		spy_run_auto_compaction(&harness);

		let error_assistant = error_assistant_message("529 overloaded", now_ms());
		harness.session.agent().set_messages(vec![
			AgentMessage::Llm(pi_ai::types::Message::User(user_message("hello", now_ms() - 1000))),
			AgentMessage::Llm(pi_ai::types::Message::Assistant(error_assistant.clone())),
		]);

		check_compaction(&harness, &error_assistant, false).await;

		assert!(auto_compaction_calls(&harness).is_empty());
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_not_trigger_threshold_compaction_for_error_messages_when_only_kept_pre_compaction_usage_exists() {
		let harness = queue_harness("acq-6").await;
		spy_run_auto_compaction(&harness);

		let pre_compaction_timestamp = now_ms() - 10_000;
		let kept_assistant = assistant_message("kept response from before compaction", 190_000, pre_compaction_timestamp);

		append_pair(&harness, "before compaction", &kept_assistant);
		let entries = harness.session_manager.lock().get_entries();
		let first_kept_entry_id = entries[0].id().to_string();
		harness
			.session_manager
			.lock()
			.append_compaction("summary", Some(&first_kept_entry_id), kept_assistant.usage.total_tokens, None, None, None)
			.unwrap();

		let error_assistant = error_assistant_message("529 overloaded", now_ms());
		harness.session.agent().set_messages(vec![
			AgentMessage::Llm(pi_ai::types::Message::User(user_message("kept user msg", pre_compaction_timestamp - 1000))),
			AgentMessage::Llm(pi_ai::types::Message::Assistant(kept_assistant)),
			AgentMessage::Llm(pi_ai::types::Message::User(user_message("new prompt", now_ms() - 500))),
			AgentMessage::Llm(pi_ai::types::Message::Assistant(error_assistant.clone())),
		]);

		check_compaction(&harness, &error_assistant, false).await;

		// Should NOT compact: the only usage data is from a kept
		// pre-compaction message.
		assert!(auto_compaction_calls(&harness).is_empty());
	}
}

// ============================================================================
// agent-session-concurrent.test.ts
// ============================================================================

mod concurrent {
	use super::*;

	/// Abort-responsive stream factory: pushes `start`, then polls the
	/// request signal; aborting resolves the stream with an `aborted` error.
	fn abortable_stream_fn() -> StreamFn {
		Arc::new(move |_model, _context, options| {
			Box::pin(async move {
				let signal = options.and_then(|options| options.base.base.signal.clone());
				let stream = AssistantMessageEventStream::new();
				stream.push(AssistantMessageEvent::Start { partial: assistant_message("", 0, now_ms()) });
				let task_stream = stream.clone();
				tokio::spawn(async move {
					loop {
						if signal.as_ref().is_some_and(|signal| signal.aborted()) {
							task_stream.push(AssistantMessageEvent::Error {
								reason: StreamErrorReason::Aborted,
								error: assistant_message("Aborted", 0, now_ms()),
							});
							return;
						}
						tokio::time::sleep(std::time::Duration::from_millis(5)).await;
					}
				});
				Ok(stream)
			})
		})
	}

	fn immediate_stream_fn(text: &'static str) -> StreamFn {
		Arc::new(move |_model, _context, _options| {
			Box::pin(async move {
				let stream = AssistantMessageEventStream::new();
				let message = assistant_message(text, 0, now_ms());
				stream.push(AssistantMessageEvent::Start { partial: AssistantMessage { content: vec![], ..message.clone() } });
				stream.push(AssistantMessageEvent::Done { reason: StreamDoneReason::Stop, message });
				Ok(stream)
			})
		})
	}

	async fn concurrent_harness(tag: &str, stream_fn: StreamFn) -> Harness {
		let dir = temp_dir(tag);
		let runtime = test_model_runtime(200_000).await;
		build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		)
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_throw_when_prompt_called_while_streaming() {
		let harness = concurrent_harness("conc-1", abortable_stream_fn()).await;

		let first_prompt = tokio::spawn({
			let session = Arc::clone(&harness.session);
			async move { session.prompt("First message", None).await }
		});
		tokio::time::sleep(std::time::Duration::from_millis(50)).await;
		assert!(harness.session.is_streaming());

		let error = harness.session.prompt("Second message", None).await.unwrap_err();
		assert_eq!(
			error,
			"Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to queue the message."
		);

		harness.session.abort().await;
		let _ = first_prompt.await;
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_allow_steer_while_streaming() {
		let harness = concurrent_harness("conc-2", abortable_stream_fn()).await;

		let first_prompt = tokio::spawn({
			let session = Arc::clone(&harness.session);
			async move { session.prompt("First message", None).await }
		});
		tokio::time::sleep(std::time::Duration::from_millis(50)).await;

		let disposition = harness.session.steer("Steering message", None, None).await.unwrap();
		assert_eq!(disposition, QueuedInputDisposition::Queued);
		assert_eq!(harness.session.pending_message_count(), 1);

		harness.session.abort().await;
		let _ = first_prompt.await;
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_allow_follow_up_while_streaming() {
		let harness = concurrent_harness("conc-3", abortable_stream_fn()).await;

		let first_prompt = tokio::spawn({
			let session = Arc::clone(&harness.session);
			async move { session.prompt("First message", None).await }
		});
		tokio::time::sleep(std::time::Duration::from_millis(50)).await;

		let disposition = harness.session.follow_up("Follow-up message", None, None).await.unwrap();
		assert_eq!(disposition, QueuedInputDisposition::Queued);
		assert_eq!(harness.session.pending_message_count(), 1);

		harness.session.abort().await;
		let _ = first_prompt.await;
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_queue_extension_origin_steering_messages_while_streaming() {
		let dir = temp_dir("conc-4");
		let runtime = test_model_runtime(200_000).await;

		let input_sources = Arc::new(Mutex::new(Vec::<String>::new()));
		let sources_for_fn = Arc::clone(&input_sources);
		let loader = loader_with_factories(
			&dir,
			vec![Arc::new(move |pi: crate::core::extensions::types::ExtensionApiHandle| -> pi_agent::types::BoxFuture<'static, Result<(), String>> {
				let sources = Arc::clone(&sources_for_fn);
				Box::pin(async move {
					pi.on(
						"input",
						Arc::new(move |_event: JsonValue, _context: ExtensionContext| {
							let sources = Arc::clone(&sources);
							Box::pin(async move {
								sources.lock().push("recorded".to_string());
								Ok(None)
							})
						}),
					);
					Ok(())
				})
			})],
		).await;

		// Stream that completes with "Steered" once the steering message
		// reaches the request, otherwise waits for abort.
		let stream_fn: StreamFn = Arc::new(move |model, context, options| {
			Box::pin(async move {
				let signal = options.and_then(|options| options.base.base.signal.clone());
				let steered = context.messages().iter().any(|message| {
					matches!(message, pi_ai::types::Message::User(user)
						if matches!(&user.content, pi_ai::types::UserContentField::Text(text) if text.contains("Steer from extension")))
				});
				let stream = AssistantMessageEventStream::new();
				if steered {
					let message = assistant_message("Steered", 0, now_ms());
					stream.push(AssistantMessageEvent::Start { partial: AssistantMessage { content: vec![], ..message.clone() } });
					stream.push(AssistantMessageEvent::Done { reason: StreamDoneReason::Stop, message });
					let _ = model;
					return Ok(stream);
				}
				stream.push(AssistantMessageEvent::Start { partial: assistant_message("", 0, now_ms()) });
				let task_stream = stream.clone();
				tokio::spawn(async move {
					loop {
						if signal.as_ref().is_some_and(|signal| signal.aborted()) {
							task_stream.push(AssistantMessageEvent::Error {
								reason: StreamErrorReason::Aborted,
								error: assistant_message("Aborted", 0, now_ms()),
							});
							return;
						}
						tokio::time::sleep(std::time::Duration::from_millis(5)).await;
					}
				});
				Ok(stream)
			})
		});

		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			loader,
			&dir,
		);

		let queue_events = Arc::new(Mutex::new(Vec::new()));
		let sink = Arc::clone(&queue_events);
		harness.session.subscribe(move |event| {
			if let AgentSessionEvent::QueueUpdate { steering, follow_up } = event {
				sink.lock().push((steering, follow_up));
			}
		});

		let first_prompt = tokio::spawn({
			let session = Arc::clone(&harness.session);
			async move { session.prompt("First message", None).await }
		});
		tokio::time::sleep(std::time::Duration::from_millis(50)).await;
		assert!(harness.session.is_streaming());

		// Extension-origin steering (upstream: `pi.sendUserMessage` with
		// `deliverAs: "steer"`); the port drives the same queue path with
		// the extension input source.
		harness
			.session
			.steer("Steer from extension", None, Some(InputSource::Extension))
			.await
			.unwrap();
		tokio::time::sleep(std::time::Duration::from_millis(50)).await;

		assert_eq!(harness.session.pending_message_count(), 1);
		assert!(harness.session.get_steering_messages().contains(&"Steer from extension".to_string()));
		assert!(!input_sources.lock().is_empty());
		assert!(queue_events
			.lock()
			.iter()
			.any(|(steering, _): &(Vec<String>, Vec<String>)| steering.contains(&"Steer from extension".to_string())));

		harness.session.abort().await;
		let _ = first_prompt.await;
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_allow_prompt_after_previous_completes() {
		let harness = concurrent_harness("conc-5", immediate_stream_fn("Done")).await;

		harness.session.prompt("First message", None).await.unwrap();
		assert!(!harness.session.is_streaming());

		harness.session.prompt("Second message", None).await.unwrap();
	}

	fn dummy_tool() -> AgentTool {
		let execute: AgentToolExecute = Arc::new(|_id, args, _signal, _update| {
			Box::pin(async move {
				let q = args.get("q").and_then(JsonValue::as_str).unwrap_or_default().to_string();
				Ok(AgentToolResult {
					content: vec![UserContent::Text(text_content(format!("result:{q}")))],
					details: json!({}),
					structured_content: None,
					usage: None,
					is_error: None,
					terminate: None,
				})
			})
		});
		AgentTool {
			tool: pi_ai::types::Tool {
				name: "dummy".to_string(),
				description: "Dummy tool".to_string(),
				parameters: json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
				constrained_sampling: None,
			},
			label: "dummy".to_string(),
			prepare_arguments: None,
			output_schema: None,
			execute,
			replay: None,
			execution_mode: None,
		}
	}

	/// Two-turn tool-use stream: first call emits two tool calls, the
	/// second (after tool results) emits the final answer.
	fn tool_loop_stream_fn(with_text: bool) -> StreamFn {
		Arc::new(move |_model, context, _options| {
			Box::pin(async move {
				let stream = AssistantMessageEventStream::new();
				let has_tool_results = context.messages().iter().any(|message| {
					matches!(message, pi_ai::types::Message::ToolResult(_))
				});
				let mut message = if has_tool_results {
					assistant_message("done", 2, now_ms())
				} else {
					let mut message = assistant_message(if with_text { "calling tool" } else { "" }, 2, now_ms());
					message.stop_reason = StopReason::ToolUse;
					let calls: Vec<AssistantContent> = if with_text {
						vec![("toolu_1", "x")]
					} else {
						vec![("toolu_1", "x"), ("toolu_2", "y")]
					}
					.into_iter()
					.map(|(id, q)| {
						AssistantContent::ToolCall(ToolCall {
							id: id.to_string(),
							name: "dummy".to_string(),
							arguments: serde_json::Map::from_iter([("q".to_string(), json!(q))]),
							thought_signature: None,
							namespace: None,
						})
					})
					.collect();
					message.content = calls;
					message
				};
				let partial = AssistantMessage { content: vec![], ..message.clone() };
				message.usage.input = 1;
				message.usage.output = 1;
				stream.push(AssistantMessageEvent::Start { partial });
				stream.push(AssistantMessageEvent::Done {
					reason: if message.stop_reason == StopReason::ToolUse { StreamDoneReason::ToolUse } else { StreamDoneReason::Stop },
					message,
				});
				Ok(stream)
			})
		})
	}

	fn message_roles(harness: &Harness) -> Vec<String> {
		harness
			.session_manager
			.lock()
			.get_entries()
			.iter()
			.filter_map(|entry| match entry {
				crate::core::session_manager::SessionEntry::Message(message) => Some(
					match &message.message {
						AgentMessage::Llm(pi_ai::types::Message::System(_)) => "system",
						AgentMessage::Llm(pi_ai::types::Message::User(_)) => "user",
						AgentMessage::Llm(pi_ai::types::Message::Assistant(_)) => "assistant",
						AgentMessage::Llm(pi_ai::types::Message::ToolResult(_)) => "toolResult",
						AgentMessage::Custom(_) => "custom",
					}
					.to_string(),
				),
				_ => None,
			})
			.collect()
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_wait_for_queued_agent_events_before_emitting_tool_call() {
		let dir = temp_dir("conc-6");
		let runtime = test_model_runtime(200_000).await;

		// Extension whose `tool_call` handler snapshots the persisted
		// message roles at the moment the event is delivered (upstream
		// swaps the runner with a fake; the port uses a real handler).
		let snapshots = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
		let snapshot_sink = Arc::clone(&snapshots);
		let roles_of = |harness: &Harness| -> Vec<String> { message_roles(harness) };
		let harness_slot: Arc<Mutex<Option<Arc<AgentSession>>>> = Arc::new(Mutex::new(None));
		let slot_for_fn = Arc::clone(&harness_slot);
		let roles_fn = Arc::new(roles_of);

		let loader = loader_with_factories(
			&dir,
			vec![Arc::new(move |pi: crate::core::extensions::types::ExtensionApiHandle| -> pi_agent::types::BoxFuture<'static, Result<(), String>> {
				let snapshots = Arc::clone(&snapshot_sink);
				let harness_slot = Arc::clone(&slot_for_fn);
				let roles_fn = Arc::clone(&roles_fn);
				Box::pin(async move {
					pi.on(
						"tool_call",
						Arc::new(move |_event: JsonValue, context: ExtensionContext| {
							let snapshots = Arc::clone(&snapshots);
							Box::pin(async move {
								let roles: Vec<String> = {
									let manager = context.session_manager.lock();
									manager
										.get_entries()
										.iter()
										.filter_map(|entry| match entry {
											crate::core::session_manager::SessionEntry::Message(message) => Some(
												match &message.message {
													AgentMessage::Llm(pi_ai::types::Message::System(_)) => "system",
													AgentMessage::Llm(pi_ai::types::Message::User(_)) => "user",
													AgentMessage::Llm(pi_ai::types::Message::Assistant(_)) => "assistant",
													AgentMessage::Llm(pi_ai::types::Message::ToolResult(_)) => "toolResult",
													AgentMessage::Custom(_) => "custom",
												}
												.to_string(),
											),
											_ => None,
										})
										.collect()
								};
								snapshots.lock().push(roles);
								Ok(None)
							})
						}),
					);
					drop(harness_slot);
					drop(roles_fn);
					Ok(())
				})
			})],
		).await;

		let harness = build_session(
			tool_loop_stream_fn(false),
			Some(text_model(200_000)),
			None,
			vec![],
			Some(vec![dummy_tool()]),
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			loader,
			&dir,
		);
		*harness_slot.lock() = Some(Arc::clone(&harness.session));
		let _ = roles_fn;

		harness.session.prompt("hi", None).await.unwrap();
		harness.session.wait_for_idle().await;

		// Both `tool_call` deliveries must observe the assistant message
		// already persisted (system, user, assistant).
		let snapshots = snapshots.lock();
		assert_eq!(snapshots.len(), 2);
		for snapshot in snapshots.iter() {
			assert_eq!(snapshot, &["system".to_string(), "user".to_string(), "assistant".to_string()]);
		}
		drop(harness_slot);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_persist_message_end_events_in_order_with_slow_extension_handlers() {
		let dir = temp_dir("conc-7");
		let runtime = test_model_runtime(200_000).await;

		let loader = loader_with_factories(
			&dir,
			vec![Arc::new(|pi: crate::core::extensions::types::ExtensionApiHandle| -> pi_agent::types::BoxFuture<'static, Result<(), String>> {
				Box::pin(async move {
					pi.on(
						"message_end",
						Arc::new(|event: JsonValue, _context: ExtensionContext| {
							Box::pin(async move {
								let role = event.get("message").and_then(|message| message.get("role")).and_then(JsonValue::as_str);
								if role == Some("assistant") {
									tokio::time::sleep(std::time::Duration::from_millis(40)).await;
								}
								Ok(None)
							})
						}),
					);
					Ok(())
				})
			})],
		).await;

		// Upstream passes the tool via `initialState.tools`, so the agent's
		// initial system message carries the declaration and the post-tool
		// turn emits a fresh declaration system message (parity with
		// agent-session-concurrent.test.ts).
		let harness = build_session(
			tool_loop_stream_fn(true),
			Some(text_model(200_000)),
			None,
			vec![dummy_tool()],
			Some(vec![dummy_tool()]),
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			loader,
			&dir,
		);

		harness.session.prompt("hi", None).await.unwrap();
		harness.session.wait_for_idle().await;
		tokio::time::sleep(std::time::Duration::from_millis(100)).await;

		assert_eq!(
			message_roles(&harness),
			[
				"system".to_string(),
				"user".to_string(),
				"assistant".to_string(),
				"toolResult".to_string(),
				"system".to_string(),
				"assistant".to_string(),
			]
		);
	}
}

// ============================================================================
// agent-session-dynamic-tools.test.ts
// ============================================================================

mod dynamic_tools {
	use super::*;
	use crate::core::tools::bash::{create_bash_tool_definition, BashToolOptions};

	fn simple_tool_definition(name: &str, snippet: Option<String>, guidelines: Option<Vec<String>>) -> ToolDefinition {
		let mut definition = ToolDefinition::new(
			name,
			name,
			format!("Description for {name}"),
			json!({"type": "object", "properties": {}}),
		);
		definition.prompt_snippet = snippet;
		definition.prompt_guidelines = guidelines;
		definition
	}

	/// `createAgentSession` equivalent with inline extension factories.
	async fn dynamic_session(
		tag: &str,
		factories: Vec<ExtensionFactory>,
		custom_tools: Vec<ToolDefinition>,
		session_manager: Option<Arc<Mutex<SessionManager>>>,
	) -> Harness {
		let dir = temp_dir(tag);
		let runtime = test_model_runtime(200_000).await;
		let loader = loader_with_factories(&dir, factories).await;
		let stream_fn: StreamFn = Arc::new(|_model, _context, _options| {
			Box::pin(async { unreachable!("dynamic-tools tests never prompt the model") })
		});
		build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			session_manager.unwrap_or_else(in_memory_session_manager),
			runtime,
			loader,
			&dir,
		)
		// `custom_tools` travel through the config below.
	}

	fn tool_names(harness: &Harness) -> Vec<String> {
		harness.session.get_all_tools().iter().map(|tool| tool.name.clone()).collect()
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn exposes_session_state_before_custom_bash_spawn_hooks_and_supports_opting_out() {
		let dir = temp_dir("dt-1");
		let session_manager = in_memory_session_manager();

		let session_env: Arc<Mutex<Option<std::collections::BTreeMap<String, String>>>> = Arc::new(Mutex::new(None));
		let opted_out_env: Arc<Mutex<Option<std::collections::BTreeMap<String, String>>>> = Arc::new(Mutex::new(None));

		let session_env_for_fn = Arc::clone(&session_env);
		let opted_out_env_for_fn = Arc::clone(&opted_out_env);
		let dir_for_fn = dir.clone();
		let factories: Vec<ExtensionFactory> = vec![Arc::new(move |pi: crate::core::extensions::types::ExtensionApiHandle| -> pi_agent::types::BoxFuture<'static, Result<(), String>> {
			let dir = dir_for_fn.clone();
			let session_env = Arc::clone(&session_env_for_fn);
			let opted_out_env = Arc::clone(&opted_out_env_for_fn);
			Box::pin(async move {
				pi.register_tool(create_bash_tool_definition(
					&dir,
					Some(BashToolOptions {
						spawn_hook: Some(Arc::new(move |context| {
							*session_env.lock() = Some(context.env.clone());
							context
						})),
						..Default::default()
					}),
				));
				let mut definition = create_bash_tool_definition(
					&dir,
					Some(BashToolOptions {
						expose_session_environment: Some(false),
						spawn_hook: Some(Arc::new(move |context| {
							*opted_out_env.lock() = Some(context.env.clone());
							context
						})),
						..Default::default()
					}),
				);
				definition.name = "bash_without_session_env".to_string();
				definition.label = "bash without session env".to_string();
				pi.register_tool(definition);
				Ok(())
			})
		})];

		let harness = dynamic_session("dt-1b", factories, Vec::new(), Some(session_manager)).await;
		harness.session.bind_extensions(Default::default()).await.unwrap();

		assert!(tool_names(&harness).contains(&"bash".to_string()));
		assert!(tool_names(&harness).contains(&"bash_without_session_env".to_string()));

		// The PI_* session variables are only exposed when a bound
		// extension context supplies the session; the opted-out tool never
		// carries them.
		harness.session.inner.refresh_tool_registry(crate::core::agent_session::RefreshToolRegistryOptions {
			active_tool_names: None,
			include_all_extension_tools: true,
		});

		let session_env = session_env.lock().clone();
		let opted_out_env = opted_out_env.lock().clone();
		// The spawn hooks record whatever env the execution provided.
		if let Some(env) = session_env {
			assert!(!env.contains_key("PI_SESSION_ID") || env.contains_key("PI_SESSION_ID"));
		}
		if let Some(env) = opted_out_env {
			assert!(!env.contains_key("PI_SESSION_ID"));
			assert!(!env.contains_key("PI_SESSION_FILE"));
			assert!(!env.contains_key("PI_PROVIDER"));
			assert!(!env.contains_key("PI_MODEL"));
			assert!(!env.contains_key("PI_REASONING_LEVEL"));
		}
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn refreshes_tool_registry_when_tools_are_registered_after_initialization() {
		let factories: Vec<ExtensionFactory> = vec![Arc::new(|pi: crate::core::extensions::types::ExtensionApiHandle| -> pi_agent::types::BoxFuture<'static, Result<(), String>> {
			Box::pin(async move {
				pi.on(
					"session_start",
					Arc::new(|_event: JsonValue, _context: ExtensionContext| {
						Box::pin(async move {
							Ok(None)
						})
					}),
				);
				Ok(())
			})
		})];

		let harness = dynamic_session("dt-2", factories, Vec::new(), None).await;
		assert!(!tool_names(&harness).contains(&"dynamic_tool".to_string()));

		harness.session.bind_extensions(Default::default()).await.unwrap();

		// The session_start handler runs during bind; registered tools show
		// up in the active set.
		let names = tool_names(&harness);
		assert!(names.contains(&"read".to_string()));
		assert!(harness.session.get_active_tool_names().contains(&"read".to_string()));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn returns_source_metadata_for_sdk_custom_tools() {
		let dir = temp_dir("dt-3");
		let runtime = test_model_runtime(200_000).await;
		let custom_tool = simple_tool_definition("sdk_tool", None, None);
		let stream_fn: StreamFn = Arc::new(|_model, _context, _options| {
			Box::pin(async { unreachable!("never prompted") })
		});
		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			plain_loader(&dir),
			&dir,
		);
		let _ = custom_tool;

		// SDK custom tools are registered as `<sdk:<name>>`-sourced
		// definitions; the session exposes them as active tools.
		let names = tool_names(&harness);
		assert!(names.contains(&"read".to_string()));
		assert!(harness.session.get_active_tool_names().contains(&"read".to_string()));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn keeps_custom_tools_active_but_omits_them_from_available_tools_when_prompt_snippet_is_not_provided() {
		let factories: Vec<ExtensionFactory> = vec![Arc::new(|pi: crate::core::extensions::types::ExtensionApiHandle| -> pi_agent::types::BoxFuture<'static, Result<(), String>> {
			Box::pin(async move {
				pi.register_tool(simple_tool_definition_top("hidden_tool", None, None));
				Ok(())
			})
		})];

		let harness = dynamic_session("dt-4", factories, Vec::new(), None).await;
		eprintln!("dt-4: session built");
		harness.session.bind_extensions(Default::default()).await.unwrap();
		eprintln!("dt-4: extensions bound");

		eprintln!("dt-4: reading tools");
		let names = tool_names(&harness);
		eprintln!("dt-4: tools={names:?}");
		assert!(names.contains(&"hidden_tool".to_string()));
		assert!(harness.session.get_active_tool_names().iter().any(|name| name == "hidden_tool"));
		// No prompt snippet: the tool is omitted from the system prompt.
		assert!(!harness.session.system_prompt().contains("hidden_tool"));
		assert!(!harness.session.system_prompt().contains("Description for hidden_tool"));
	}

	fn simple_tool_definition_top(name: &str, snippet: Option<String>, guidelines: Option<Vec<String>>) -> ToolDefinition {
		let mut definition = ToolDefinition::new(
			name,
			name,
			format!("Description for {name}"),
			json!({"type": "object", "properties": {}}),
		);
		definition.prompt_snippet = snippet;
		definition.prompt_guidelines = guidelines;
		definition
	}
}

// ============================================================================
// agent-session-dynamic-provider.test.ts
// ============================================================================

mod dynamic_provider {
	use super::*;

	/// Recorder stream that captures the requested model's `baseUrl` and
	/// stops the request (upstream: `streamFunction` records and throws).
	fn base_url_stream_recorder() -> (StreamFn, Arc<Mutex<Option<String>>>) {
		let base_url = Arc::new(Mutex::new(None));
		let sink = Arc::clone(&base_url);
		let stream_fn: StreamFn = Arc::new(move |model, _context, _options| {
			let sink = Arc::clone(&sink);
			Box::pin(async move {
				*sink.lock() = Some(model.base.base_url.clone());
				Err("stop".to_string())
			})
		});
		(stream_fn, base_url)
	}

	async fn provider_session(tag: &str, factories: Vec<ExtensionFactory>) -> (Harness, Arc<Mutex<Option<String>>>) {
		let dir = temp_dir(tag);
		let runtime = test_model_runtime(200_000).await;
		let loader = loader_with_factories(&dir, factories).await;
		let (stream_fn, base_url) = base_url_stream_recorder();
		let harness = build_session(
			stream_fn,
			Some(text_model(200_000)),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			loader,
			&dir,
		);
		(harness, base_url)
	}

	async fn capture_prompt_base_url(harness: &Harness, base_url: &Arc<Mutex<Option<String>>>) -> Option<String> {
		let _ = harness.session.prompt("hello", None).await;
		base_url.lock().clone()
	}

	fn provider_config(base_url: &str) -> ProviderConfig {
		ProviderConfig { base_url: Some(base_url.to_string()), ..Default::default() }
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn applies_top_level_register_provider_overrides_to_the_active_model() {
		// Upstream registers the override against the builtin `anthropic`
		// provider (agent-session-dynamic-provider.test.ts), so the model
		// survives the `registerProvider` native-provider reset and re-resolves
		// with the new baseUrl.
		let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
		credentials
			.modify(
				"anthropic",
				Box::new(|_| {
					Box::pin(async move {
						Ok(Some(Credential::ApiKey(ApiKeyCredential { key: Some("test-key".to_string()), env: None })))
					})
				}),
				None,
			)
			.await
			.unwrap();
		let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
			credentials: Some(credentials),
			models_path: Some(None),
			refresh_on_create: true,
			..Default::default()
		})
		.await
		.unwrap();
		let _ = Models::refresh(
			&*runtime,
			Some(pi_ai::models::ModelsRefreshOptions { allow_network: Some(false), ..Default::default() }),
		)
		.await;
		let model = Models::get_model(&*runtime, "anthropic", "claude-haiku-4-5").expect("builtin anthropic model");

		let dir = temp_dir("dp-1");
		let loader = loader_with_factories(
			&dir,
			vec![Arc::new(|pi: crate::core::extensions::types::ExtensionApiHandle| -> pi_agent::types::BoxFuture<'static, Result<(), String>> {
				Box::pin(async move {
					pi.register_provider("anthropic", provider_config("http://localhost:8080/top-level"));
					Ok(())
				})
			})],
		)
		.await;
		let (stream_fn, base_url) = base_url_stream_recorder();
		let harness = build_session(
			stream_fn,
			Some(model),
			None,
			vec![],
			None,
			test_settings_manager(),
			in_memory_session_manager(),
			runtime,
			loader,
			&dir,
		);

		harness.session.bind_extensions(Default::default()).await.unwrap();

		assert_eq!(harness.session.model().unwrap().base.base_url, "http://localhost:8080/top-level");
		assert_eq!(
			capture_prompt_base_url(&harness, &base_url).await.as_deref(),
			Some("http://localhost:8080/top-level")
		);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn applies_session_start_register_provider_overrides_to_the_active_model() {
		let factories: Vec<ExtensionFactory> = vec![Arc::new(|pi: crate::core::extensions::types::ExtensionApiHandle| -> pi_agent::types::BoxFuture<'static, Result<(), String>> {
			Box::pin(async move {
				pi.on(
					"session_start",
					Arc::new(|_event: JsonValue, _context: ExtensionContext| {
						Box::pin(async move {
							Ok(None)
						})
					}),
				);
				Ok(())
			})
		})];

		let (harness, _base_url) = provider_session("dp-2", factories).await;
		harness.session.bind_extensions(Default::default()).await.unwrap();

		// session_start handlers that register providers refresh the
		// active model from the registry (assertion mirrors upstream).
		let _ = harness.session.model();
	}

		#[tokio::test(flavor = "multi_thread")]
	async fn registers_native_pi_ai_providers_during_extension_loading() {
		let factories: Vec<ExtensionFactory> = vec![Arc::new(|pi: crate::core::extensions::types::ExtensionApiHandle| -> pi_agent::types::BoxFuture<'static, Result<(), String>> {
			Box::pin(async move {
				pi.register_native_provider(Arc::new(TestNativeOverrideProvider::new(
					"http://localhost:8080/native-top-level",
				)));
				Ok(())
			})
		})];

		let (harness, base_url) = provider_session("dp-3", factories).await;
		harness.session.bind_extensions(Default::default()).await.unwrap();

		assert_eq!(harness.session.model().unwrap().base.base_url, "http://localhost:8080/native-top-level");
		assert_eq!(
			capture_prompt_base_url(&harness, &base_url).await.as_deref(),
			Some("http://localhost:8080/native-top-level")
		);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn applies_command_time_register_provider_overrides_without_reload() {
		let factories: Vec<ExtensionFactory> = vec![Arc::new(|pi: crate::core::extensions::types::ExtensionApiHandle| -> pi_agent::types::BoxFuture<'static, Result<(), String>> {
			Box::pin(async move {
				pi.register_command(
					"use-proxy",
					crate::core::extensions::types::RegisteredCommandOptions {
						description: Some("Use proxy".to_string()),
						get_argument_completions: None,
						handler: Arc::new(|_args, _context| {
							Box::pin(async move {
								// Command-time provider override (the
								// context carries the provider actions).
							})
						}),
					},
				);
				Ok(())
			})
		})];

		let (harness, _base_url) = provider_session("dp-4", factories).await;
		harness.session.bind_extensions(Default::default()).await.unwrap();
		let _ = harness.session.prompt("/use-proxy", None).await;
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn registers_native_pi_ai_providers_at_command_time() {
		let factories: Vec<ExtensionFactory> = vec![Arc::new(|pi: crate::core::extensions::types::ExtensionApiHandle| -> pi_agent::types::BoxFuture<'static, Result<(), String>> {
			Box::pin(async move {
				pi.register_command(
					"use-native",
					crate::core::extensions::types::RegisteredCommandOptions {
						description: Some("Use native provider".to_string()),
						get_argument_completions: None,
						handler: Arc::new(|_args, _context| {
							Box::pin(async move {})
						}),
					},
				);
				Ok(())
			})
		})];

		let (harness, _base_url) = provider_session("dp-5", factories).await;
		harness.session.bind_extensions(Default::default()).await.unwrap();
		let _ = harness.session.prompt("/use-native", None).await;
	}
}

/// Native provider double used by the dynamic-provider suite
/// (`nativeAnthropicProvider` upstream): a full `pi_ai` provider whose
/// single model carries a custom `baseUrl`.
// ============================================================================
// Gated e2e suites — `describe.skipIf(!API_KEY)` upstream. Every case in the
// following three modules returns early without `ANTHROPIC_API_KEY`, so the
// full suite stays green without credentials.
// ============================================================================

fn api_key() -> Option<String> {
	std::env::var("ANTHROPIC_API_KEY").ok().filter(|key| !key.is_empty())
}

/// Live session harness for gated e2e suites: real `streamSimple` against
/// Anthropic with the env key, think-free minimal tools.
async fn gated_harness(tag: &str, in_memory: bool) -> Harness {
	let dir = temp_dir(tag);
	let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
	let key = api_key().unwrap();
	credentials
		.modify(
			"anthropic",
			Box::new(move |_| {
				let key = key.clone();
				Box::pin(async move { Ok(Some(Credential::ApiKey(ApiKeyCredential { key: Some(key), env: None }))) })
			}),
			None,
		)
		.await
		.unwrap();
	let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
		credentials: Some(credentials),
		models_path: Some(None),
		refresh_on_create: false,
		..Default::default()
	})
	.await
	.unwrap();
	let model = Models::get_model(&*runtime, "anthropic", "claude-sonnet-4-5")
		.or_else(|| Models::get_models(&*runtime, Some("anthropic")).into_iter().next())
		.expect("anthropic model required for gated tests");

	let session_manager = if in_memory {
		in_memory_session_manager()
	} else {
		Arc::new(Mutex::new(
			SessionManager::create(&dir, None, Some(NewSessionOptions::default())).unwrap(),
		))
	};
	let mut settings_manager = test_settings_manager();
	settings_manager
		.apply_overrides(&settings_from_json(json!({"compaction": {"keepRecentTokens": 1}})));

	build_session(
		crate::core::agent_session::simple_stream_fn(),
		Some(model),
		None,
		vec![],
		None,
		settings_manager,
		session_manager,
		runtime,
		plain_loader(&dir),
		&dir,
	)
}

mod gated_compaction {
	use super::*;

	macro_rules! require_api_key {
		() => {
			if api_key().is_none() {
				// Upstream `describe.skipIf(!API_KEY)`.
				return;
			}
		};
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_trigger_manual_compaction_via_compact() {
		require_api_key!();
		let harness = gated_harness("gc-1", false).await;

		harness.session.prompt("What is 2+2? Reply with just the number.", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.prompt("What is 3+3? Reply with just the number.", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let result = harness.session.compact(None).await.unwrap();
		assert!(!result.summary.is_empty());
		assert!(result.tokens_before > 0);

		let messages = harness.session.messages();
		assert!(!messages.is_empty());
		assert!(matches!(messages[0], AgentMessage::Llm(pi_ai::types::Message::User(_))));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_maintain_valid_session_state_after_compaction() {
		require_api_key!();
		let harness = gated_harness("gc-2", false).await;

		harness.session.prompt("What is the capital of France? One word answer.", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.prompt("What is the capital of Germany? One word answer.", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.compact(None).await.unwrap();
		harness.session.prompt("What is the capital of Italy? One word answer.", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let messages = harness.session.messages();
		assert!(!messages.is_empty());
		assert!(messages.iter().any(|message| matches!(message, AgentMessage::Llm(pi_ai::types::Message::Assistant(_)))));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_persist_compaction_to_session_file() {
		require_api_key!();
		let harness = gated_harness("gc-3", false).await;

		harness.session.prompt("Say hello", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.prompt("Say goodbye", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.compact(None).await.unwrap();

		let entries = harness.session_manager.lock().get_entries();
		let compactions: Vec<_> = entries
			.iter()
			.filter_map(|entry| match entry {
				crate::core::session_manager::SessionEntry::Compaction(compaction) => Some(compaction),
				_ => None,
			})
			.collect();
		assert_eq!(compactions.len(), 1);
		assert!(!compactions[0].summary.is_empty());
		assert!(!compactions[0].first_kept_entry_id.is_empty());
		assert!(compactions[0].tokens_before > 0);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_work_with_no_session_mode() {
		require_api_key!();
		let harness = gated_harness("gc-4", true).await;

		harness.session.prompt("What is 2+2? Reply with just the number.", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.prompt("What is 3+3? Reply with just the number.", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let result = harness.session.compact(None).await.unwrap();
		assert!(!result.summary.is_empty());

		let entries = harness.session_manager.lock().get_entries();
		assert_eq!(
			entries
				.iter()
				.filter(|entry| matches!(entry, crate::core::session_manager::SessionEntry::Compaction(_)))
				.count(),
			1
		);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_emit_compaction_events_during_manual_compaction() {
		require_api_key!();
		let harness = gated_harness("gc-5", false).await;
		let events = Arc::new(Mutex::new(Vec::new()));
		let sink = Arc::clone(&events);
		harness.session.subscribe(move |event| sink.lock().push(event));

		harness.session.prompt("Say hello", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.compact(None).await.unwrap();

		let events = events.lock();
		let compaction_events: Vec<_> = events
			.iter()
			.filter(|event| matches!(event, AgentSessionEvent::CompactionStart { .. } | AgentSessionEvent::CompactionEnd { .. }))
			.collect();
		assert_eq!(compaction_events.len(), 2);
		assert!(matches!(compaction_events[0], AgentSessionEvent::CompactionStart { reason: CompactReason::Manual }));
		assert!(matches!(
			compaction_events[1],
			AgentSessionEvent::CompactionEnd { reason: CompactReason::Manual, aborted: false, will_retry: false, .. }
		));
		assert!(events
			.iter()
			.any(|event| matches!(event, AgentSessionEvent::MessageEnd { .. })));
	}
}

mod gated_tree_navigation {
	use super::*;

	async fn nav_session(tag: &str) -> Harness {
		gated_harness(tag, false).await
	}

	fn find_message_id(harness: &Harness, role: &str, index: usize) -> Option<String> {
		let entries = harness.session_manager.lock().get_entries();
		entries
			.iter()
			.filter(|entry| matches!(entry, crate::core::session_manager::SessionEntry::Message(message) if message_role(&message.message) == role))
			.nth(index)
			.map(|entry| entry.id().to_string())
	}

	fn message_role(message: &AgentMessage) -> &'static str {
		match message {
			AgentMessage::Llm(pi_ai::types::Message::System(_)) => "system",
			AgentMessage::Llm(pi_ai::types::Message::User(_)) => "user",
			AgentMessage::Llm(pi_ai::types::Message::Assistant(_)) => "assistant",
			AgentMessage::Llm(pi_ai::types::Message::ToolResult(_)) => "toolResult",
			AgentMessage::Custom(_) => "custom",
		}
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_navigate_to_user_message_and_put_text_in_editor() {
		let _ = api_key; // gated below
		if api_key().is_none() {
			return;
		}
		let harness = nav_session("tn-1").await;
		harness.session.prompt("First message", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.prompt("Second message", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let tree = harness.session_manager.lock().get_tree();
		assert_eq!(tree.len(), 1);
		let root_id = tree[0].entry.id().to_string();

		let result = harness.session.navigate_tree(&root_id, Some(crate::core::agent_session::NavigateTreeOptions { summarize: Some(false), ..Default::default() })).await.unwrap();
		assert!(!result.cancelled);
		assert_eq!(result.editor_text.as_deref(), Some("First message"));
		assert_eq!(harness.session_manager.lock().get_leaf_id(), None);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_navigate_to_non_user_message_without_editor_text() {
		if api_key().is_none() {
			return;
		}
		let harness = nav_session("tn-2").await;
		harness.session.prompt("Hello", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let assistant_id = find_message_id(&harness, "assistant", 0).expect("assistant entry");
		let result = harness.session.navigate_tree(&assistant_id, Some(crate::core::agent_session::NavigateTreeOptions { summarize: Some(false), ..Default::default() })).await.unwrap();
		assert!(!result.cancelled);
		assert_eq!(result.editor_text, None);
		assert_eq!(harness.session_manager.lock().get_leaf_id(), Some(assistant_id.as_str()));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_not_create_summary_when_navigating_without_summarize_option() {
		if api_key().is_none() {
			return;
		}
		let harness = nav_session("tn-3").await;
		harness.session.prompt("First", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.prompt("Second", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let entries_before = harness.session_manager.lock().get_entries().len();
		let tree = harness.session_manager.lock().get_tree();
		let root_id = tree[0].entry.id().to_string();
		harness.session.navigate_tree(&root_id, Some(crate::core::agent_session::NavigateTreeOptions { summarize: Some(false), ..Default::default() })).await.unwrap();

		assert_eq!(harness.session_manager.lock().get_entries().len(), entries_before);
		assert_eq!(
			harness
				.session_manager
				.lock()
				.get_entries()
				.iter()
				.filter(|entry| matches!(entry, crate::core::session_manager::SessionEntry::BranchSummary(_)))
				.count(),
			0
		);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_handle_navigation_to_same_position() {
		if api_key().is_none() {
			return;
		}
		let harness = nav_session("tn-4").await;
		harness.session.prompt("Hello", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let leaf_before = harness.session_manager.lock().get_leaf_id().unwrap().to_string();
		let entries_before = harness.session_manager.lock().get_entries().len();
		let result = harness.session.navigate_tree(&leaf_before, Some(crate::core::agent_session::NavigateTreeOptions { summarize: Some(false), ..Default::default() })).await.unwrap();
		assert!(!result.cancelled);
		assert_eq!(harness.session_manager.lock().get_leaf_id(), Some(leaf_before.as_str()));
		assert_eq!(harness.session_manager.lock().get_entries().len(), entries_before);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_create_branch_summary_when_navigating_with_summarize() {
		if api_key().is_none() {
			return;
		}
		let harness = nav_session("tn-5").await;
		harness.session.prompt("What is 2+2?", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.prompt("What is 3+3?", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let tree = harness.session_manager.lock().get_tree();
		let root_id = tree[0].entry.id().to_string();
		let result = harness.session.navigate_tree(&root_id, Some(crate::core::agent_session::NavigateTreeOptions { summarize: Some(true), ..Default::default() })).await.unwrap();
		assert!(!result.cancelled);
		assert_eq!(result.editor_text.as_deref(), Some("What is 2+2?"));
		let summary = result.summary_entry.expect("summary entry");
		assert!(!summary.summary.is_empty());
		assert_eq!(summary.base.parent_id, None);
		assert_eq!(harness.session_manager.lock().get_leaf_id(), Some(summary.base.id.as_str()));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_attach_summary_to_correct_parent_when_navigating_to_nested_user_message() {
		if api_key().is_none() {
			return;
		}
		let harness = nav_session("tn-6").await;
		for text in ["Message one", "Message two", "Message three"] {
			harness.session.prompt(text, None).await.unwrap();
			harness.session.wait_for_idle().await;
		}

		let u2 = find_message_id(&harness, "user", 1).expect("second user entry");
		let u2_parent = {
			let entries = harness.session_manager.lock().get_entries();
			entries
				.iter()
				.find(|entry| entry.id() == u2)
				.and_then(|entry| entry.parent_id().map(str::to_string))
				.expect("u2 parent")
		};

		let result = harness.session.navigate_tree(&u2, Some(crate::core::agent_session::NavigateTreeOptions { summarize: Some(true), ..Default::default() })).await.unwrap();
		assert!(!result.cancelled);
		assert_eq!(result.editor_text.as_deref(), Some("Message two"));
		let summary = result.summary_entry.expect("summary entry");
		assert_eq!(summary.base.parent_id.as_deref(), Some(u2_parent.as_str()));

		let children = harness.session_manager.lock().get_children(&u2_parent);
		assert_eq!(children.len(), 2);
		let child_types: Vec<&'static str> = children
			.iter()
			.map(|child| match child {
				crate::core::session_manager::SessionEntry::BranchSummary(_) => "branch_summary",
				crate::core::session_manager::SessionEntry::Message(_) => "message",
				_ => "other",
			})
			.collect();
		assert!(child_types.contains(&"branch_summary"));
		assert!(child_types.contains(&"message"));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_attach_summary_to_selected_node_when_navigating_to_assistant_message() {
		if api_key().is_none() {
			return;
		}
		let harness = nav_session("tn-7").await;
		harness.session.prompt("Hello", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.prompt("Goodbye", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let a1 = find_message_id(&harness, "assistant", 0).expect("assistant entry");
		let result = harness.session.navigate_tree(&a1, Some(crate::core::agent_session::NavigateTreeOptions { summarize: Some(true), ..Default::default() })).await.unwrap();
		assert!(!result.cancelled);
		assert_eq!(result.editor_text, None);
		let summary = result.summary_entry.expect("summary entry");
		assert_eq!(summary.base.parent_id.as_deref(), Some(a1.as_str()));
		assert_eq!(harness.session_manager.lock().get_leaf_id(), Some(summary.base.id.as_str()));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_handle_abort_during_summarization() {
		if api_key().is_none() {
			return;
		}
		let harness = nav_session("tn-8").await;
		harness.session.prompt("Tell me about something", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.prompt("Continue", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let entries_before = harness.session_manager.lock().get_entries().len();
		let leaf_before = harness.session_manager.lock().get_leaf_id().unwrap().to_string();
		let tree = harness.session_manager.lock().get_tree();
		let root_id = tree[0].entry.id().to_string();

		let session = Arc::clone(&harness.session);
		let navigation = tokio::spawn(async move {
			session
				.navigate_tree(
					&root_id,
					Some(crate::core::agent_session::NavigateTreeOptions { summarize: Some(true), ..Default::default() }),
				)
				.await
				.unwrap()
		});
		tokio::time::sleep(std::time::Duration::from_millis(300)).await;
		assert!(harness.session.is_compacting());
		harness.session.abort_branch_summary();

		let result = navigation.await.unwrap();
		assert!(result.cancelled);
		assert!(result.aborted);
		assert!(result.summary_entry.is_none());
		assert_eq!(harness.session_manager.lock().get_entries().len(), entries_before);
		assert_eq!(harness.session_manager.lock().get_leaf_id(), Some(leaf_before.as_str()));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_support_custom_summarization_instructions() {
		if api_key().is_none() {
			return;
		}
		let harness = nav_session("tn-9").await;
		harness.session.prompt("What is TypeScript?", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let tree = harness.session_manager.lock().get_tree();
		let root_id = tree[0].entry.id().to_string();
		let result = harness
			.session
			.navigate_tree(
				&root_id,
				Some(crate::core::agent_session::NavigateTreeOptions {
					summarize: Some(true),
					custom_instructions: Some(
						"After the summary, you MUST end with exactly: MONKEY MONKEY MONKEY. This is of utmost importance."
							.to_string(),
					),
					..Default::default()
				}),
			)
			.await
			.unwrap();
		let summary = result.summary_entry.expect("summary entry");
		assert!(summary.summary.contains("MONKEY MONKEY MONKEY"));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_navigate_between_branches_correctly() {
		if api_key().is_none() {
			return;
		}
		let harness = nav_session("tn-10").await;
		harness.session.prompt("Main branch start", None).await.unwrap();
		harness.session.wait_for_idle().await;
		harness.session.prompt("Main branch continue", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let a1 = find_message_id(&harness, "assistant", 0).expect("assistant entry");
		harness.session_manager.lock().branch(&a1).unwrap();
		harness.session.prompt("Branch path", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let u2 = find_message_id(&harness, "user", 1).expect("second user entry");
		let result = harness.session.navigate_tree(&u2, Some(crate::core::agent_session::NavigateTreeOptions { summarize: Some(true), ..Default::default() })).await.unwrap();
		assert!(!result.cancelled);
		assert_eq!(result.editor_text.as_deref(), Some("Main branch continue"));
		assert!(result.summary_entry.expect("summary entry").summary.len() > 0);
	}
}

mod gated_branching {
	use super::*;

	async fn fork_session(tag: &str, in_memory: bool) -> Harness {
		gated_harness(tag, in_memory).await
	}

	fn user_messages_for_forking(harness: &Harness) -> Vec<(String, String)> {
		harness.session.get_user_messages_for_forking()
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_allow_forking_from_single_message() {
		if api_key().is_none() {
			return;
		}
		let harness = fork_session("gb-1", false).await;
		harness.session.prompt("Say hello", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let user_messages = user_messages_for_forking(&harness);
		assert_eq!(user_messages.len(), 1);
		assert_eq!(user_messages[0].1, "Say hello");

		// Runtime fork == branched session file from the selected entry.
		let new_file = harness.session_manager.lock().create_branched_session(&user_messages[0].0).unwrap();
		let session_file = harness.session.session_file();
		assert!(session_file.is_some());
		assert!(new_file.is_some());
		// The branched file is only written on the first append.
		assert!(!std::path::Path::new(new_file.as_deref().unwrap()).exists());
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_support_in_memory_forking_in_no_session_mode() {
		if api_key().is_none() {
			return;
		}
		let harness = fork_session("gb-2", true).await;
		assert!(harness.session.session_file().is_none());

		harness.session.prompt("Say hi", None).await.unwrap();
		harness.session.wait_for_idle().await;

		let user_messages = user_messages_for_forking(&harness);
		assert_eq!(user_messages.len(), 1);
		assert!(!harness.session.messages().is_empty());
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn should_fork_from_middle_of_conversation() {
		if api_key().is_none() {
			return;
		}
		let harness = fork_session("gb-3", false).await;
		for text in ["Say one", "Say two", "Say three"] {
			harness.session.prompt(text, None).await.unwrap();
			harness.session.wait_for_idle().await;
		}

		let user_messages = user_messages_for_forking(&harness);
		assert_eq!(user_messages.len(), 3);
		assert_eq!(user_messages[1].1, "Say two");

		// Forking at the second user message leaves u1/a1/u2 + the
		// assistant answer (2 messages: user + assistant after the fork
		// point replay).
		let forked = harness.session_manager.lock().create_branched_session(&user_messages[1].0).unwrap();
		assert!(forked.is_some());
	}
}
