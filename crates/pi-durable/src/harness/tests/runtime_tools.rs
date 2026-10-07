use pi_ai::models::MutableModels;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use pi_chord::context::Context;
use pi_ai::providers::faux::{
	faux_assistant_message, faux_provider, faux_tool_call, FauxContent, FauxContentBlock, FauxDeferredOptions, FauxMessageOptions,
	FauxProviderHandle, FauxResponseStep, RegisterFauxProviderOptions,
};
use pi_ai::types::{
	Message, StopReason, TextContent, ToolResultMessage, UserContent, UserContentField, UserMessage,
};
use serde_json::{json, Value};
use tokio::sync::Notify;

use super::support::{context, open, root};
use crate::env::{ExecutionEnv, NodeEnvOptions, NodeExecutionEnv};
use crate::EnvTarget;

use crate::harness::api::{
	OutputRetain, ToolControl, ToolDiagnostic, ToolDiagnosticSeverity, ToolExecutionApi, ToolExecutionMode,
	ToolExecutionResult, ToolOutputLimits, ToolRegistration, ToolReplay,
};
use crate::harness::define::define_tool_of;
use crate::harness::generation::generation_task;
use crate::harness::harness::{Conversation, Harness};
use crate::harness::live::live_doc;
use crate::harness::registry::{create_registry, RegistryImpl};
use crate::harness::tool::tool_task;
use crate::harness::types::{
	AnyTask, AgentChange, BeforeRequestResult, BeforeToolDecision, ConversationCreateOptions, EntryId,
	ConversationStreamOptions, Extension, ExtensionsChange, GenerationHooks, HarnessOptions, HarnessSettings,
	HarnessSettingsProvider, HookRegistration, InputSubmissionDraft, ModelRef, OnYieldResult, PromptSection, Registry,
	RegistryReader, SetOrClear, Submission, SubmissionDraft, SubmissionId, SubmissionStatus, ToolHooks, ToolsChange,
};
use crate::storage::memory::MemoryStorage;
use crate::tasks::{BoxFuture, NextTaskState, PhaseHandler, TaskDefinition, TaskRuntime, commit_change};
use crate::types::{ConversationOwnership, TaskOptions, TaskOutcomeStatus, TaskOwnership};

const DONE: &str = "done";

type ExecuteFuture = BoxFuture<ToolExecutionResult>;

struct Fixture {
	storage: Arc<MemoryStorage>,
	registry: Arc<RegistryImpl>,
	faux: FauxProviderHandle,
	harness: Arc<Harness>,
	root: Arc<Conversation>,
	reports: Arc<Mutex<Vec<String>>>,
	settings: HarnessSettings,
	env: Option<crate::harness::types::EnvBuilder>,
}

fn tool<F, Fut>(name: &str, execute: F) -> ToolRegistration
where
	F: Fn(Value, Arc<dyn ToolExecutionApi>, Context) -> Fut + Send + Sync + 'static,
	Fut: Future<Output = ToolExecutionResult> + Send + 'static,
{
	define_tool_of(ToolRegistration {
		tool: pi_ai::types::Tool {
			name: name.to_owned(),
			description: format!("The {name} tool"),
			parameters: json!({"type":"object","properties":{"text":{"type":"string"}}}),
			constrained_sampling: None,
		},
		replay: None,
		execution_mode: None,
		prepare_arguments: None,
		output_limits: None,
		execute: Arc::new(move |arguments, api, context| -> ExecuteFuture {
			Box::pin(execute(arguments, api, context))
		}),
	})
}

fn tool_call(name: &str, args: Value, id: &str) -> FauxContentBlock {
	FauxContentBlock::ToolCall(faux_tool_call(name, args.as_object().cloned().unwrap_or_default(), Some(id)))
}

fn calls(list: &[(&str, Value, &str)]) -> FauxResponseStep {
	FauxResponseStep::Message(faux_assistant_message(
		FauxContent::Blocks(list.iter().map(|(name, args, id)| tool_call(name, args.clone(), id)).collect()),
		FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
	))
}

fn answer(text: &str) -> FauxResponseStep {
	FauxResponseStep::Message(faux_assistant_message(text, FauxMessageOptions::default()))
}

fn steps(list: &[(&str, Value, &str)]) -> Vec<FauxResponseStep> {
	vec![calls(list), answer(DONE)]
}

fn content_text(text: impl Into<String>) -> Value {
	serde_json::to_value(vec![UserContent::Text(TextContent::new(text))]).expect("tool content serializes")
}

fn result_text(result: &ToolResultMessage) -> String {
	result.content
		.iter()
		.map(|content| match content {
			UserContent::Text(text) => text.text.clone(),
			UserContent::Image(_) => "[image]".to_owned(),
		})
		.collect::<Vec<_>>()
		.join("|")
}

fn results(entries: &[crate::types::EntryRecord]) -> Vec<ToolResultMessage> {
	entries
		.iter()
		.filter(|entry| entry.kind == "pi.tool-result")
		.filter_map(|entry| {
			entry.model.as_ref()?.iter().find_map(|message| match message {
				Message::ToolResult(result) => Some(result.clone()),
				_ => None,
			})
		})
		.collect()
}

fn result_for<'a>(results: &'a [ToolResultMessage], call_id: &str) -> &'a ToolResultMessage {
	results.iter().find(|result| result.tool_call_id == call_id).expect("tool result for call")
}

async fn all_entries(root: &Conversation) -> Vec<crate::types::EntryRecord> {
	let mut cursor = None;
	let mut entries = Vec::new();
	loop {
		let page = root
			.entries(crate::harness::types::ConversationEntryQuery::default(), 128, cursor.as_ref(), &context())
			.await
			.expect("scan conversation entries");
		entries.extend(page.items);
		cursor = page.next;
		if cursor.is_none() {
			// The storage scan is newest-first; upstream `allEntries()` is
			// oldest-first.
			entries.reverse();
			return entries;
		}
	}
}

async fn open_fixture(
	storage: Arc<MemoryStorage>,
	registry: Arc<RegistryImpl>,
	faux: FauxProviderHandle,
	reports: Arc<Mutex<Vec<String>>>,
	settings: HarnessSettings,
	env: Option<crate::harness::types::EnvBuilder>,
) -> Fixture {
	open_fixture_with_provider(storage, registry, faux, reports, settings, env, None).await
}

async fn open_fixture_with_provider(
	storage: Arc<MemoryStorage>,
	registry: Arc<RegistryImpl>,
	faux: FauxProviderHandle,
	reports: Arc<Mutex<Vec<String>>>,
	settings: HarnessSettings,
	env: Option<crate::harness::types::EnvBuilder>,
	settings_provider: Option<HarnessSettingsProvider>,
) -> Fixture {
	let models = pi_ai::create_models(None);
	models.set_provider(faux.provider.clone());
	let report_sink = reports.clone();
	let harness = open(
		storage.clone(),
		HarnessOptions {
			models,
			registry: registry.clone(),
			settings: Some(settings.clone()),
			settings_provider,
			env: env.clone(),
			on_report: Some(Arc::new(move |error| report_sink.lock().push(error))),
			..Default::default()
		},
	)
	.await;
	harness.resume();
	let root = Arc::new(root(&harness).await);
	root.configure(
		AgentChange {
			model: Some(SetOrClear::Set(ModelRef { provider: "faux".into(), model_id: "faux-1".into() })),
			..Default::default()
		},
		&context(),
	)
	.await
	.expect("configure faux model");
	Fixture { storage, registry, faux, harness, root, reports, settings, env }
}

fn clone_faux(faux: &FauxProviderHandle) -> FauxProviderHandle {
	FauxProviderHandle { provider: faux.provider.clone(), core: faux.core.clone() }
}


async fn submit(fixture: &Fixture) -> (SubmissionStatus, Vec<crate::types::EntryRecord>) {
	let receipt = fixture
		.root
		.submit(
			SubmissionDraft::Input(InputSubmissionDraft {
				request_id: None,
				content: UserContentField::Text("go".into()),
				when_busy: None,
			}),
			&context(),
		)
		.await
		.expect("submit input");
	let settled = receipt.wait(&context()).await.expect("wait for run");
	(settled.status, all_entries(&fixture.root).await)
}

async fn close(fixture: &Fixture) {
	fixture.harness.close(&context()).await.expect("close harness");
}

fn install_tools(registry: &Arc<RegistryImpl>, extension: &str, tools: Vec<ToolRegistration>) -> Extension {
	let extension = Extension::new(extension).with_tools(tools.into_iter().map(Arc::new).collect());
	registry.install(extension.clone());
	extension
}

fn install_tool_hooks(registry: &Arc<RegistryImpl>, extension: &str, hooks: ToolHooks) {
	registry.install(Extension::new(extension).with_hooks(vec![HookRegistration {
		task: tool_task().definition.name.clone(),
		handlers: Arc::new(hooks),
	}]));
}

fn install_generation_hooks(registry: &Arc<RegistryImpl>, extension: &str, hooks: GenerationHooks) {
	registry.install(Extension::new(extension).with_hooks(vec![HookRegistration {
		task: generation_task().definition.name.clone(),
		handlers: Arc::new(hooks),
	}]));
}

async fn rerun(fixture: &Fixture) -> Fixture {
	open_fixture(
		fixture.storage.clone(),
		fixture.registry.clone(),
		clone_faux(&fixture.faux),
		fixture.reports.clone(),
		fixture.settings.clone(),
		fixture.env.clone(),
	)
	.await
}

fn interrupted_tool(name: &str, replay: ToolReplay, started: Arc<Notify>, runs: Arc<Mutex<usize>>) -> ToolRegistration {
	let mut registration = tool(name, {
		let started = started.clone();
		let runs = runs.clone();
		move |_args, api, call_context| {
			let started = started.clone();
			let runs = runs.clone();
			let call_context = call_context.clone();
			async move {
				let run = {
					let mut runs = runs.lock();
					*runs += 1;
					*runs
				};
				api.output(format!("run {run}\n").as_bytes(), None);
				api.details(json!({"run": run}), &call_context).await.expect("durably commit progress");
				started.notify_one();
				std::future::pending::<()>().await;
				ToolExecutionResult::default()
			}
		}
	});
	registration.replay = Some(replay);
	registration
}

#[tokio::test]
async fn unavailable_and_late_unregistered_tools_get_results_without_running() {
	let ran = Arc::new(Mutex::new(0));
	let registry = Arc::new(create_registry());
	let ran_tool = {
		let ran = ran.clone();
		tool("present", move |_args, _api, _ctx| {
			let ran = ran.clone();
			async move {
				*ran.lock() += 1;
				ToolExecutionResult::default()
			}
		})
	};
	install_tools(&registry, "available", vec![ran_tool]);
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("ghost", json!({}), "ghost"), ("present", json!({}), "present")]));
	let fixture = open_fixture(
		Arc::new(MemoryStorage::new()), registry.clone(), faux, Arc::new(Mutex::new(Vec::new())),
		HarnessSettings::default(), None,
	).await;
	let (status, entries) = submit(&fixture).await;
	assert_eq!(status, SubmissionStatus::Done);
	let tool_results = results(&entries);
	assert_eq!(tool_results.len(), 2);
	let ghost = result_for(&tool_results, "ghost");
	let present = result_for(&tool_results, "present");
	assert!(ghost.is_error);
	assert_eq!(result_text(ghost), "<harness>\n[error] Tool ghost is not available\n</harness>");
	assert!(!present.is_error);
	assert_eq!(*ran.lock(), 1);
	close(&fixture).await;

	// A sequential first call can remove the active implementation after the
	// round was prepared; its pending sibling must resolve against the new registry.
	let registry = Arc::new(create_registry());
	let second_extension = install_tools(&registry, "second", vec![tool("second", |_a, _api, _ctx| async {
		ToolExecutionResult { content: Some(content_text("must not execute")), ..Default::default() }
	})]);
	let registry_in_call = registry.clone();
	let first = tool("first", move |_a, _api, _ctx| {
		let registry = registry_in_call.clone();
		let second_extension = second_extension.clone();
		async move {
			registry.uninstall(&second_extension);
			ToolExecutionResult::default()
		}
	});
	let mut first = first;
	first.execution_mode = Some(ToolExecutionMode::Sequential);
	install_tools(&registry, "first", vec![first]);
	let fixture = fixture_with_registry(
		registry,
		steps(&[("first", json!({}), "first"), ("second", json!({}), "second")]),
	).await;
	let (_, entries) = submit(&fixture).await;
	let tool_results = results(&entries);
	assert_eq!(result_text(result_for(&tool_results, "second")), "<harness>\n[error] Tool second is not available\n</harness>");
	close(&fixture).await;
}

#[tokio::test]
async fn removing_and_reregistering_an_active_tool_changes_future_offers_and_resolution() {
	let registry = Arc::new(create_registry());
	let runs = Arc::new(Mutex::new(0usize));
	let run_count = runs.clone();
	let echo = tool("echo", move |_args, _api, _ctx| {
		let runs = run_count.clone();
		async move { *runs.lock() += 1; ToolExecutionResult::default() }
	});
	let extension = install_tools(&registry, "echo-extension", vec![echo]);
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(vec![answer("ready")]);
	let fixture = open_fixture(
		Arc::new(MemoryStorage::new()), registry.clone(), faux, Arc::new(Mutex::new(Vec::new())),
		HarnessSettings::default(), None,
	).await;
	submit(&fixture).await;
	registry.uninstall(&extension);
	assert!(fixture.root.agent(&context()).await.unwrap().tools.is_empty());
	fixture.faux.set_responses(steps(&[("echo", json!({}), "removed")]));
	let (_, removed_entries) = submit(&fixture).await;
	let removed_result = results(&removed_entries).into_iter().find(|result| result.tool_call_id == "removed").unwrap();
	assert!(removed_result.is_error);
	assert_eq!(result_text(&removed_result), "<harness>\n[error] Tool echo is not available\n</harness>");
	assert_eq!(*runs.lock(), 0);
	registry.install(extension);
	assert_eq!(fixture.root.agent(&context()).await.unwrap().tools.iter().map(|tool| tool.tool.name.as_str()).collect::<Vec<_>>(), ["echo"]);
	fixture.faux.set_responses(steps(&[("echo", json!({}), "restored")]));
	let (_, restored_entries) = submit(&fixture).await;
	let restored_result = results(&restored_entries).into_iter().find(|result| result.tool_call_id == "restored").unwrap();
	assert!(!restored_result.is_error);
	assert_eq!(*runs.lock(), 1);
	close(&fixture).await;
}

#[tokio::test]
async fn deselecting_a_tool_after_round_preparation_makes_its_pending_call_unavailable() {
	let registry = Arc::new(create_registry());
	let second = tool("second", |_args, _api, _ctx| async { ToolExecutionResult::default() });
	let second_extension = install_tools(&registry, "second", vec![second]);
	let selected_root = Arc::new(Mutex::new(None::<Arc<Conversation>>));
	let root_for_tool = selected_root.clone();
	let mut first = tool("first", move |_args, _api, call_context| {
		let root = root_for_tool.lock().clone().expect("tool root was installed");
		async move {
			root.configure(
				AgentChange { tools: Some(SetOrClear::Set(ToolsChange::List(Vec::new()))), ..Default::default() },
				&call_context,
			).await.expect("deselect tools while first sequential call runs");
			ToolExecutionResult::default()
		}
	});
	first.execution_mode = Some(ToolExecutionMode::Sequential);
	install_tools(&registry, "first", vec![first]);
	let fixture = fixture_with_registry(registry, steps(&[("first", json!({}), "first"), ("second", json!({}), "second")])).await;
	*selected_root.lock() = Some(fixture.root.clone());
	let (_, entries) = submit(&fixture).await;
	let results = results(&entries);
	assert_eq!(result_text(result_for(&results, "second")), "<harness>\n[error] Tool second is not available\n</harness>");
	assert!(fixture.root.agent(&context()).await.unwrap().tools.is_empty());
	close(&fixture).await;
	let _ = second_extension;
}

async fn fixture_with_registry(registry: Arc<RegistryImpl>, responses: Vec<FauxResponseStep>) -> Fixture {
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(responses);
	open_fixture(
		Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None,
	).await
}

#[tokio::test]
async fn tool_rounds_are_parallel_by_default_and_sequential_by_policy_or_tool() {
	let events = Arc::new(Mutex::new(Vec::<String>::new()));
	let second_started = Arc::new(Notify::new());
	let registry = Arc::new(create_registry());
	let first_events = events.clone();
	let first_second_started = second_started.clone();
	let a = tool("a", move |_args, _api, _ctx| {
		let events = first_events.clone();
		let second_started = first_second_started.clone();
		async move {
			events.lock().push("start a".into());
			second_started.notified().await;
			events.lock().push("end a".into());
			ToolExecutionResult::default()
		}
	});
	let b_events = events.clone();
	let b_started = second_started.clone();
	let b = tool("b", move |_args, _api, _ctx| {
		let events = b_events.clone();
		let started = b_started.clone();
		async move {
			events.lock().push("start b".into());
			started.notify_one();
			events.lock().push("end b".into());
			ToolExecutionResult::default()
		}
	});
	install_tools(&registry, "parallel", vec![a, b]);
	let fixture = fixture_with_registry(
		registry,
		steps(&[("a", json!({}), "a"), ("b", json!({}), "b")]),
	).await;
	let (_, entries) = submit(&fixture).await;
	let observed = events.lock().clone();
	assert!(observed.iter().position(|event| event == "start b").unwrap() < observed.iter().position(|event| event == "end a").unwrap());
	assert!(observed.iter().position(|event| event == "start a").unwrap() < observed.iter().position(|event| event == "end a").unwrap());
	assert_eq!(results(&entries).len(), 2);
	close(&fixture).await;

	for (mode, per_tool) in [(ToolExecutionMode::Sequential, false), (ToolExecutionMode::Parallel, true)] {
		let events = Arc::new(Mutex::new(Vec::<String>::new()));
		let registry = Arc::new(create_registry());
		let mut tools = Vec::new();
		for name in ["a", "b"] {
			let events = events.clone();
			let mut registered = tool(name, move |_args, _api, _ctx| {
				let events = events.clone();
				async move {
					events.lock().push(format!("start {name}"));
					events.lock().push(format!("end {name}"));
					ToolExecutionResult::default()
				}
			});
			if per_tool && name == "a" { registered.execution_mode = Some(ToolExecutionMode::Sequential); }
			tools.push(registered);
		}
		install_tools(&registry, "mode", tools);
		let settings = HarnessSettings { tool_execution: Some(mode), ..Default::default() };
		let faux = faux_provider(RegisterFauxProviderOptions::default());
		faux.set_responses(steps(&[("a", json!({}), "a"), ("b", json!({}), "b")]));
		let fixture = open_fixture(
			Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), settings, None,
		).await;
		submit(&fixture).await;
		assert_eq!(*events.lock(), ["start a", "end a", "start b", "end b"]);
		close(&fixture).await;
	}
}

#[tokio::test]
async fn tool_round_pins_the_live_execution_mode_after_each_resolution() {
	let settings = Arc::new(RwLock::new(HarnessSettings {
		tool_execution: Some(ToolExecutionMode::Parallel),
		..Default::default()
	}));
	let provider_state = settings.clone();
	let settings_provider: HarnessSettingsProvider = Arc::new(move || provider_state.read().clone());
	let events = Arc::new(Mutex::new(Vec::<String>::new()));
	let registry = Arc::new(create_registry());
	let mut tools = Vec::new();
	for name in ["a", "b"] {
		let events = events.clone();
		let settings = settings.clone();
		tools.push(tool(name, move |_args, _api, _ctx| {
			let events = events.clone();
			let settings = settings.clone();
			async move {
				events.lock().push(format!("start {name}"));
				settings.write().tool_execution = Some(ToolExecutionMode::Parallel);
				tokio::time::sleep(std::time::Duration::from_millis(20)).await;
				events.lock().push(format!("end {name}"));
				ToolExecutionResult::default()
			}
		}));
	}
	install_tools(&registry, "live-mode", tools);
	let mode_during_request = settings.clone();
	let request = FauxResponseStep::sync(move |_transcript, _options, _state, _model| {
		mode_during_request.write().tool_execution = Some(ToolExecutionMode::Sequential);
		faux_assistant_message(
			FauxContent::Blocks(vec![tool_call("a", json!({}), "a"), tool_call("b", json!({}), "b")]),
			FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
		)
	});
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(vec![request, answer(DONE)]);
	let fixture = open_fixture_with_provider(
		Arc::new(MemoryStorage::new()),
		registry,
		faux,
		Arc::new(Mutex::new(Vec::new())),
		HarnessSettings::default(),
		None,
		Some(settings_provider),
	)
	.await;
	submit(&fixture).await;
	assert_eq!(*events.lock(), ["start a", "end a", "start b", "end b"]);
	close(&fixture).await;
}

#[tokio::test]
async fn output_fallback_diagnostics_and_tail_bounds_are_preserved_in_results() {
	let windows = Arc::new(Mutex::new(Vec::new()));
	let registry = Arc::new(create_registry());
	let windows_capture = windows.clone();
	let mut tail = tool("tail", move |_args, api, ctx| {
		let windows = windows_capture.clone();
		async move {
			windows.lock().push(api.output_window());
			api.output(b"dropped\n", None);
			api.output(b"x\ny\n", Some(crate::env::ShellOutputSkip { bytes: 8, newlines: 1, ends_with_newline: true }));
			ToolExecutionResult::default()
		}
	});
	tail.output_limits = Some(ToolOutputLimits { max_lines: Some(1), max_bytes: None, retain: Some(OutputRetain::Tail) });
	let windows_capture = windows.clone();
	let mut head = tool("head", move |_args, api, _ctx| {
		let windows = windows_capture.clone();
		async move {
			windows.lock().push(api.output_window());
			api.output(b"line 1\n", None);
			api.output(b"line 2\nline 3\n", None);
			api.diagnostic(ToolDiagnostic { severity: ToolDiagnosticSeverity::Info, message: "from api".into(), code: None });
			api.details(json!({"step":1}), &context()).await.expect("first details");
			api.details(json!({"step":2}), &context()).await.expect("second details");
			ToolExecutionResult {
				diagnostics: Some(vec![ToolDiagnostic { severity: ToolDiagnosticSeverity::Warn, message: "from result".into(), code: None }]),
				..Default::default()
			}
		}
	});
	head.output_limits = Some(ToolOutputLimits { max_lines: Some(2), max_bytes: None, retain: None });
	install_tools(&registry, "bounds", vec![tail, head]);
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("tail", json!({}), "tail"), ("head", json!({}), "head")]));
	let fixture = open_fixture(
		Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())),
		HarnessSettings { progress: Some(crate::harness::types::ProgressPolicyOverrides { output_interval_ms: Some(250), ..Default::default() }), ..Default::default() },
		None,
	).await;
	let (_, entries) = submit(&fixture).await;
	let results = results(&entries);
	let tail_result = results.iter().find(|result| result.tool_call_id == "tail").unwrap();
	assert_eq!(result_text(tail_result), "y\n|<harness>\n[warn] Output truncated to its end: 3 lines, 18 bytes dropped\n</harness>");
	let head_result = results.iter().find(|result| result.tool_call_id == "head").unwrap();
	assert_eq!(head_result.details, Some(json!({"step":2})));
	assert_eq!(result_text(head_result), "line 1\nline 2\n|<harness>\n[info] from api\n[warn] from result\n[warn] Output truncated to its beginning: 1 lines, 7 bytes dropped\n</harness>");
	assert!(windows.lock().iter().any(|window| window.as_ref().is_some_and(|window| window.max_lines == 1 && window.min_interval_ms == 250)));
	assert!(windows.lock().iter().any(Option::is_none));
	close(&fixture).await;
}

#[tokio::test]
async fn explicit_text_is_bounded_around_images_and_tool_errors_keep_partial_output() {
	let registry = Arc::new(create_registry());
	let mut explicit = tool("explicit", |_args, _api, _ctx| async {
		ToolExecutionResult {
			content: Some(serde_json::to_value(vec![
				UserContent::Text(TextContent::new("a\nb\n")),
				UserContent::Image(pi_ai::types::ImageContent::new("AAAA", "image/png")),
				UserContent::Text(TextContent::new("c\nd\n")),
			]).unwrap()),
			..Default::default()
		}
	});
	explicit.output_limits = Some(ToolOutputLimits { max_lines: Some(2), max_bytes: None, retain: Some(OutputRetain::Tail) });
	let failure = tool("failure", |_args, api, _ctx| async move {
		api.output(b"partial\n", None);
		panic!("boom");
	});
	install_tools(&registry, "explicit", vec![explicit, failure]);
	let fixture = fixture_with_registry(
		registry,
		steps(&[("explicit", json!({}), "explicit"), ("failure", json!({}), "failure")]),
	).await;
	let (_, entries) = submit(&fixture).await;
	let results = results(&entries);
	assert_eq!(result_text(result_for(&results, "explicit")), "[image]|c\nd\n|<harness>\n[warn] Output truncated to its end: 2 lines, 4 bytes dropped\n</harness>");
	let failure_result = result_for(&results, "failure");
	assert!(failure_result.is_error);
	assert_eq!(result_text(failure_result), "partial\n|<harness>\n[error] boom\n</harness>");
	close(&fixture).await;
}

#[tokio::test]
async fn tool_arguments_are_validated_repaired_and_blocked_hooks_run_in_order() {
	let registry = Arc::new(create_registry());
	let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
	let seen_tool = seen.clone();
	let echo = tool("echo", move |args, _api, _ctx| {
		let seen = seen_tool.clone();
		async move {
			seen.lock().push(args);
			ToolExecutionResult::default()
		}
	});
	install_tools(&registry, "echo", vec![echo]);
	let hook_order = Arc::new(Mutex::new(Vec::<String>::new()));
	let order_first = hook_order.clone();
	install_tool_hooks(&registry, "first", ToolHooks {
		before_tool: Some(Arc::new(move |mut call, _api, _ctx| {
			let order = order_first.clone();
			Box::pin(async move {
				order.lock().push("first".into());
				if call.id == "block" {
					return Ok(Some(BeforeToolDecision { block: Some("first says no".into()), arguments: None }));
				}
				if call.id == "throw" { return Err("hook failed".into()); }
				if call.id == "bad" { call.arguments.insert("text".into(), json!({"not":"a string"})); }
				else if let Some(Value::String(text)) = call.arguments.get("text").cloned() {
					call.arguments.insert("text".into(), json!(format!("{text}!")));
				}
				Ok(Some(BeforeToolDecision { arguments: Some(call.arguments), block: None }))
			}) as BoxFuture<Result<Option<BeforeToolDecision>, String>>
		})),
		after_tool: None,
	});
	let order_second = hook_order.clone();
	install_tool_hooks(&registry, "second", ToolHooks {
		before_tool: Some(Arc::new(move |call, _api, _ctx| {
			let order = order_second.clone();
			Box::pin(async move {
				if call.id == "block" {
					order.lock().push("second".into());
					return Ok(Some(BeforeToolDecision { block: Some("second says no".into()), arguments: None }));
				}
				Ok(None)
			})
		})),
		after_tool: None,
	});
	let fixture = fixture_with_registry(registry, steps(&[
		("echo", json!({"text":"x"}), "ok"),
		("echo", json!({"text":"bad"}), "bad"),
		("echo", json!({}), "block"),
		("echo", json!({"text":{"not":"a string"}}), "invalid"),
		("echo", json!({}), "throw"),
		("echo", json!({"text":1}), "coerced"),
	])).await;
	let (_, entries) = submit(&fixture).await;
	let results = results(&entries);
	let by_id: std::collections::BTreeMap<_, _> = results.iter().map(|result| (result.tool_call_id.as_str(), result)).collect();
	assert_eq!(result_text(by_id["block"]), "<harness>\n[error] Tool call blocked: first says no\n</harness>");
	assert_eq!(result_text(by_id["throw"]), "<harness>\n[error] Tool call blocked: hook failed\n</harness>");
	assert!(by_id["bad"].is_error && result_text(by_id["bad"]).contains("Validation failed"));
	assert!(by_id["invalid"].is_error && result_text(by_id["invalid"]).contains("Validation failed"));
	assert!(by_id["ok"].is_error == false);
	assert_eq!(hook_order.lock().iter().filter(|hook| hook.as_str() == "second").count(), 0);
	assert_eq!(seen.lock().len(), 2);
	assert!(seen.lock().contains(&json!({"text":"x!"})));
	assert_eq!(result_text(by_id["coerced"]), "");
	close(&fixture).await;
}

#[tokio::test]
async fn prepare_arguments_runs_before_validation_and_after_tool_replacements_chain() {
	let registry = Arc::new(create_registry());
	let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
	let seen_copy = seen.clone();
	let mut echo = tool("echo", move |args, _api, _ctx| {
		let seen = seen_copy.clone();
		async move {
			seen.lock().push(args);
			ToolExecutionResult { content: Some(content_text("raw")), ..Default::default() }
		}
	});
	echo.prepare_arguments = Some(Arc::new(|args| {
		let text = args.get("text").cloned().unwrap_or(Value::Null);
		if text == json!("throw") { panic!("cannot repair"); }
		if let Some(number) = text.as_i64() { json!({"text": format!("#{number}")}) } else { args.clone() }
	}));
	install_tools(&registry, "echo", vec![echo]);
	install_tool_hooks(&registry, "rewrite", ToolHooks {
		before_tool: Some(Arc::new(|mut call, _api, _ctx| Box::pin(async move {
			if let Some(Value::String(value)) = call.arguments.get("text").cloned() {
				call.arguments.insert("text".into(), json!(format!("{value}!")));
			}
			Ok(Some(BeforeToolDecision { arguments: Some(call.arguments), block: None }))
		}))),
		after_tool: Some(Arc::new(|_call, mut result, _api, _ctx| Box::pin(async move {
			result.content = Some(content_text("replaced"));
			Ok(Some(result))
		}))),
	});
	install_tool_hooks(&registry, "details", ToolHooks {
		before_tool: None,
		after_tool: Some(Arc::new(|_call, mut result, _api, _ctx| Box::pin(async move {
			result.details = Some(json!({"text": "replaced"}));
			Ok(Some(result))
		}))),
	});
	let observed = Arc::new(Mutex::new(Vec::<(EntryId, Vec<crate::types::EntryId>)>::new()));
	let observed_copy = observed.clone();
	install_generation_hooks(&registry, "after-tools", GenerationHooks {
		after_tools: Some(Arc::new(move |assistant, entries, _api, _ctx| {
			let observed = observed_copy.clone();
			Box::pin(async move { observed.lock().push((assistant, entries)); Ok(()) })
		})),
		..Default::default()
	});
	let fixture = fixture_with_registry(registry, steps(&[("echo", json!({"text":7}), "fixed"), ("echo", json!({"text":"throw"}), "broken")])).await;
	let (_, entries) = submit(&fixture).await;
	let results = results(&entries);
	assert_eq!(results.len(), 2);
	let broken = result_for(&results, "broken");
	assert!(broken.is_error);
	assert_eq!(result_text(broken), "<harness>\n[error] cannot repair\n</harness>");
	assert_eq!(broken.details, None);
	let result = result_for(&results, "fixed");
	assert_eq!(result_text(result), "replaced");
	assert_eq!(result.details, Some(json!({"text":"replaced"})));
	assert_eq!(*seen.lock(), [json!({"text":"#7!"})]);
	let call = entries.iter().filter(|entry| entry.kind == "pi.assistant").find_map(|entry| {
		entry.model.as_ref()?.iter().find_map(|message| match message {
			Message::Assistant(assistant) => assistant.content.iter().find_map(|content| match content {
				pi_ai::types::AssistantContent::ToolCall(call) => Some(call.arguments.clone()),
				_ => None,
			}),
			_ => None,
		})
	}).unwrap();
	assert_eq!(call.get("text"), Some(&json!(7)));
	assert_eq!(observed.lock().len(), 1);
	assert_eq!(observed.lock()[0].1.len(), 2);
	close(&fixture).await;
}

#[tokio::test]
async fn generation_hooks_change_request_observe_answers_and_continue_on_yield() {
	let registry = Arc::new(create_registry());
	let requested = Arc::new(Mutex::new(Vec::<String>::new()));
	let seen_requests = requested.clone();
	let before = GenerationHooks {
		before_request: Some(Arc::new(|mut payload, _api, _ctx| Box::pin(async move {
			payload.messages.push(Message::User(UserMessage { content: UserContentField::Text("injected".into()), timestamp: 0 }));
			Ok(Some(BeforeRequestResult { messages: payload.messages }))
		}))),
		..Default::default()
	};
	install_generation_hooks(&registry, "request", before);
	let observed = Arc::new(Mutex::new(Vec::<String>::new()));
	let observed_response = observed.clone();
	let yield_count = Arc::new(Mutex::new(0usize));
	let yield_count_copy = yield_count.clone();
	install_generation_hooks(&registry, "response", GenerationHooks {
		after_response: Some(Arc::new(move |message, _api, _ctx| {
			let observed = observed_response.clone();
			Box::pin(async move {
				let text = message.content.iter().filter_map(|block| match block { pi_ai::types::AssistantContent::Text(text) => Some(text.text.clone()), _ => None }).collect::<String>();
				observed.lock().push(text);
				Ok(())
			})
		})),
		on_yield: Some(Arc::new(move |_message, _api, _ctx| {
			let count = yield_count_copy.clone();
			Box::pin(async move {
				let mut count = count.lock();
				if *count == 0 {
					*count += 1;
					Ok(Some(OnYieldResult { continue_: UserContentField::Text("keep going".into()) }))
				} else { Ok(None) }
			})
		})),
		..Default::default()
	});
	let requests = requested.clone();
	let record = FauxResponseStep::sync(move |transcript, _options, _state, _model| {
		let last_users = transcript.messages().iter().filter_map(|message| match message {
			Message::User(user) => match &user.content { UserContentField::Text(text) => Some(text.clone()), _ => None },
			_ => None,
		}).collect::<Vec<_>>();
		requests.lock().push(last_users.join("|"));
		faux_assistant_message("answer", FauxMessageOptions::default())
	});
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(vec![record.clone(), record]);
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None).await;
	let (status, entries) = submit(&fixture).await;
	assert_eq!(status, SubmissionStatus::Done);
	assert!(requested.lock()[0].ends_with("go|injected"));
	assert_eq!(observed.lock().as_slice(), ["answer", "answer"]);
	assert_eq!(entries.iter().filter(|entry| entry.kind == "pi.user").count(), 2);
	assert!(!entries.iter().any(|entry| entry.model.as_ref().is_some_and(|messages| messages.iter().any(|message| matches!(message, Message::User(user) if matches!(&user.content, UserContentField::Text(text) if text == "injected"))))));
	close(&fixture).await;
}

#[tokio::test]
async fn yielded_generation_keeps_its_submission_placed_until_the_final_answer() {
	let registry = Arc::new(create_registry());
	let harness = Arc::new(Mutex::new(None::<Arc<Harness>>));
	let submission_id = Arc::new(Mutex::new(None::<SubmissionId>));
	let observed_status = Arc::new(Mutex::new(Vec::<SubmissionStatus>::new()));
	let harness_copy = harness.clone();
	let id_copy = submission_id.clone();
	let observed_copy = observed_status.clone();
	let release = Arc::new(Notify::new());
	let release_copy = release.clone();
	let yielded = Arc::new(Mutex::new(false));
	let yielded_copy = yielded.clone();
	install_generation_hooks(&registry, "open-submission", GenerationHooks {
		before_request: Some(Arc::new(move |_payload, _api, ctx| {
			let harness = harness_copy.lock().clone();
			let id = *id_copy.lock();
			let observed = observed_copy.clone();
			Box::pin(async move {
				if let (Some(harness), Some(id)) = (harness, id) {
					let submission = harness.submission(id, &ctx).await?.expect("placed submission");
					let status = submission.status(&ctx).await?.status;
					observed.lock().push(status);
				}
				Ok(None)
			})
		})),
		on_yield: Some(Arc::new(move |_message, _api, _ctx| {
			let first = {
				let mut yielded = yielded_copy.lock();
				let first = !*yielded;
				*yielded = true;
				first
			};
			let release = release_copy.clone();
			Box::pin(async move {
				if first {
					release.notified().await;
					Ok(Some(OnYieldResult { continue_: UserContentField::Text("again".into()) }))
				} else {
					Ok(None)
				}
			})
		})),
		..Default::default()
	});
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(vec![answer("first"), answer("second")]);
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None).await;
	*harness.lock() = Some(fixture.harness.clone());
	let receipt = fixture.root.submit(SubmissionDraft::Input(InputSubmissionDraft {
		request_id: None, content: UserContentField::Text("go".into()), when_busy: None,
	}), &context()).await.expect("submit input");
	*submission_id.lock() = Some(receipt.id());
	release.notify_one();
	let settled = receipt.wait(&context()).await.expect("settle continued run");
	let entries = all_entries(&fixture.root).await;
	let answers: Vec<_> = entries.iter().filter(|entry| entry.kind == "pi.assistant").collect();
	assert_eq!(*observed_status.lock(), [SubmissionStatus::Placed]);
	assert_eq!(settled.status, SubmissionStatus::Done);
	assert_eq!(settled.answer, Some(answers[1].id));
	close(&fixture).await;
}

#[tokio::test]
async fn after_response_observes_a_deferred_answer_once_it_is_polled() {
	let registry = Arc::new(create_registry());
	let observed = Arc::new(Mutex::new(Vec::<String>::new()));
	let observed_copy = observed.clone();
	install_generation_hooks(&registry, "deferred-response", GenerationHooks {
		after_response: Some(Arc::new(move |message, _api, _ctx| {
			let observed = observed_copy.clone();
			Box::pin(async move {
				let text = message.content.iter().filter_map(|content| match content {
					pi_ai::types::AssistantContent::Text(text) => Some(text.text.as_str()),
					_ => None,
				}).collect::<String>();
				let stop = if message.stop_reason == StopReason::Stop { "stop" } else { "other" };
				observed.lock().push(format!("{stop}:{text}"));
				Ok(())
			})
		})),
		..Default::default()
	});
	let faux = faux_provider(RegisterFauxProviderOptions {
		deferred: Some(FauxDeferredOptions { pending_fetches: Some(1), poll_after_ms: Some(1) }),
		..Default::default()
	});
	faux.set_responses(vec![answer("late")]);
	let settings = HarnessSettings {
		stream: Some(ConversationStreamOptions { deferred: Some(json!(true)), ..Default::default() }),
		..Default::default()
	};
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), settings, None).await;
	let (status, _) = submit(&fixture).await;
	assert_eq!(status, SubmissionStatus::Done);
	assert_eq!(*observed.lock(), ["stop:late"]);
	close(&fixture).await;
}

#[tokio::test]
async fn generation_hook_errors_do_not_stop_later_observers_and_first_yield_wins() {
	let registry = Arc::new(create_registry());
	let called = Arc::new(Mutex::new(Vec::<String>::new()));
	let throwing = called.clone();
	install_generation_hooks(&registry, "throw-observer", GenerationHooks {
		after_response: Some(Arc::new(move |_message, _api, _ctx| {
			let called = throwing.clone();
			Box::pin(async move { called.lock().push("throw".into()); Err("observer failed".into()) })
		})),
		..Default::default()
	});
	let later = called.clone();
	install_generation_hooks(&registry, "later-observer", GenerationHooks {
		after_response: Some(Arc::new(move |_message, _api, _ctx| {
			let called = later.clone();
			Box::pin(async move { called.lock().push("later".into()); Ok(()) })
		})),
		..Default::default()
	});
	let first_count = Arc::new(Mutex::new(0usize));
	let first_count_copy = first_count.clone();
	install_generation_hooks(&registry, "first-yield", GenerationHooks {
		on_yield: Some(Arc::new(move |_message, _api, _ctx| {
			let count = first_count_copy.clone();
			Box::pin(async move {
				let mut count = count.lock();
				if *count == 0 { *count += 1; Ok(Some(OnYieldResult { continue_: UserContentField::Text("first".into()) })) } else { Ok(None) }
			})
		})),
		..Default::default()
	});
	let second_count = Arc::new(Mutex::new(0usize));
	let second_count_copy = second_count.clone();
	install_generation_hooks(&registry, "second-yield", GenerationHooks {
		on_yield: Some(Arc::new(move |_message, _api, _ctx| {
			let count = second_count_copy.clone();
			Box::pin(async move {
				let mut count = count.lock();
				*count += 1;
				if *count == 1 { Ok(Some(OnYieldResult { continue_: UserContentField::Text("second".into()) })) } else { Ok(None) }
			})
		})),
		..Default::default()
	});
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(vec![answer("a"), answer("b"), answer("c")]);
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None).await;
	let (status, entries) = submit(&fixture).await;
	assert_eq!(status, SubmissionStatus::Done);
	let inputs: Vec<_> = entries.iter().filter(|entry| entry.kind == "pi.user").map(|entry| {
		entry.model.as_ref().and_then(|messages| messages.first()).and_then(|message| match message { Message::User(user) => match &user.content { UserContentField::Text(text) => Some(text.clone()), _ => None }, _ => None }).unwrap()
	}).collect();
	assert_eq!(inputs, ["go", "first", "second"]);
	assert_eq!(*second_count.lock(), 2);
	assert_eq!(called.lock().iter().filter(|name| name.as_str() == "later").count(), 3);
	assert!(fixture.reports.lock().iter().any(|report| report.contains("observer failed")));
	close(&fixture).await;
}

#[tokio::test]
async fn tool_failure_environment_and_diagnostic_content_are_consumer_visible() {
	let registry = Arc::new(create_registry());
	let fail = tool("fail", |_args, api, _ctx| async move {
		api.output(b"partial\n", None);
		ToolExecutionResult { is_error: Some(true), diagnostics: Some(vec![ToolDiagnostic { severity: ToolDiagnosticSeverity::Error, message: "boom".into(), code: None }]), ..Default::default() }
	});
	install_tools(&registry, "fail", vec![fail]);
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("fail", json!({}), "c1")]));
	let failure = open_fixture(Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None).await;
	let (status, entries) = submit(&failure).await;
	assert_eq!(status, SubmissionStatus::Done);
	assert_eq!(result_text(&results(&entries)[0]), "partial\n|<harness>\n[error] boom\n</harness>");
	close(&failure).await;

	let registry = Arc::new(create_registry());
	install_tools(&registry, "probe", vec![tool("probe", |_args, _api, _ctx| async { ToolExecutionResult::default() })]);
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("probe", json!({}), "probe")]));
	let reports = Arc::new(Mutex::new(Vec::new()));
	let failure = open_fixture(
		Arc::new(MemoryStorage::new()), registry, faux, reports.clone(), HarnessSettings::default(),
		Some(Arc::new(|_target: EnvTarget, _ctx: Context| Box::pin(async { Err("no sandbox".to_owned()) }))),
	).await;
	let (_, entries) = submit(&failure).await;
	assert_eq!(result_text(&results(&entries)[0]), "<harness>\n[error] no sandbox\n</harness>");
	assert_eq!(reports.lock().iter().filter(|error| error.as_str() == "no sandbox").count(), 2);
	close(&failure).await;
}

#[tokio::test]
async fn selected_extensions_gate_hooks_and_tool_controls_apply_to_whole_round() {
	let registry = Arc::new(create_registry());
	let echo_extension = install_tools(&registry, "echo", vec![tool("echo", |_a, _api, _ctx| async { ToolExecutionResult::default() })]);
	let hook_seen = Arc::new(Mutex::new(0usize));
	let hook_seen_copy = hook_seen.clone();
	let audit = Extension::new("audit").with_hooks(vec![HookRegistration {
		task: tool_task().definition.name.clone(),
		handlers: Arc::new(ToolHooks {
			before_tool: Some(Arc::new(move |_call, _api, _ctx| {
				let seen = hook_seen_copy.clone();
				Box::pin(async move { *seen.lock() += 1; Ok(None) })
			})),
			after_tool: None,
		}),
	}]);
	registry.install(audit.clone());
	let settings = HarnessSettings { extensions: Some(vec![Arc::new(echo_extension.clone())]), ..Default::default() };
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("echo", json!({}), "first")]));
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry.clone(), faux, Arc::new(Mutex::new(Vec::new())), settings, None).await;
	submit(&fixture).await;
	assert_eq!(*hook_seen.lock(), 0);
	fixture.root.configure(
		AgentChange { extensions: Some(SetOrClear::Set(crate::harness::types::ExtensionsChange::Edit { add: vec![audit], remove: vec![] })), ..Default::default() },
		&context(),
	).await.expect("select audit");
	fixture.faux.set_responses(steps(&[("echo", json!({}), "second")]));
	submit(&fixture).await;
	assert_eq!(*hook_seen.lock(), 1);
	close(&fixture).await;

	let registry = Arc::new(create_registry());
	let stop = tool("stop", |_a, _api, _ctx| async { ToolExecutionResult { control: Some(ToolControl { terminate: Some(true), ..Default::default() }), ..Default::default() } });
	let grow = tool("grow", |_a, _api, _ctx| async { ToolExecutionResult { control: Some(ToolControl { add_tools: Some(vec!["extra".into()]), ..Default::default() }), ..Default::default() } });
	let extra = tool("extra", |_a, _api, _ctx| async { ToolExecutionResult::default() });
	install_tools(&registry, "controls", vec![stop.clone(), grow.clone(), extra]);
	let fixture = fixture_with_registry(registry, vec![calls(&[("stop", json!({}), "stop")])]).await;
	fixture.root.configure(
		AgentChange { tools: Some(SetOrClear::Set(ToolsChange::List(vec![stop, grow]))), ..Default::default() },
		&context(),
	).await.expect("select control tools");
	let (status, entries) = submit(&fixture).await;
	assert_eq!(status, SubmissionStatus::Done);
	assert_eq!(entries.last().unwrap().kind, "pi.tool-result");
	// One terminating call is not enough when its sibling did not also request termination.
	fixture.faux.set_responses(steps(&[("stop", json!({}), "stop2"), ("grow", json!({}), "grow")]));
	let (second_status, second_entries) = submit(&fixture).await;
	assert_eq!(second_status, SubmissionStatus::Done);
	assert_eq!(second_entries.last().unwrap().kind, "pi.assistant");
	let agent = fixture.root.agent(&context()).await.expect("updated tools");
	assert_eq!(agent.tools.iter().map(|tool| tool.tool.name.as_str()).collect::<Vec<_>>(), ["stop", "grow", "extra"]);
	close(&fixture).await;
}

#[tokio::test]
async fn add_tools_removes_names_from_an_existing_exclusion_filter() {
	let registry = Arc::new(create_registry());
	let extra_runs = Arc::new(Mutex::new(0usize));
	let extra_runs_copy = extra_runs.clone();
	let extra = tool("extra", move |_args, _api, _ctx| {
		let runs = extra_runs_copy.clone();
		async move {
			*runs.lock() += 1;
			ToolExecutionResult::default()
		}
	});
	let grow = tool("grow", |_args, _api, _ctx| async {
		ToolExecutionResult {
			control: Some(ToolControl { add_tools: Some(vec!["extra".into()]), ..Default::default() }),
			..Default::default()
		}
	});
	install_tools(&registry, "controls", vec![grow, extra.clone()]);
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("grow", json!({}), "grow")]));
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None).await;
	fixture.root.configure(
		AgentChange { tools: Some(SetOrClear::Set(ToolsChange::Remove(vec![extra]))), ..Default::default() },
		&context(),
	).await.expect("exclude extra");
	let before = fixture.root.agent(&context()).await.expect("filtered tools");
	assert_eq!(before.tools.iter().map(|tool| tool.tool.name.as_str()).collect::<Vec<_>>(), ["grow"]);
	let _ = submit(&fixture).await;
	let after = fixture.root.agent(&context()).await.expect("tools after addTools");
	assert_eq!(after.tools.iter().map(|tool| tool.tool.name.as_str()).collect::<Vec<_>>(), ["grow", "extra"]);
	fixture.faux.set_responses(steps(&[("extra", json!({}), "extra")]));
	let _ = submit(&fixture).await;
	assert_eq!(*extra_runs.lock(), 1);
	close(&fixture).await;
}

#[tokio::test]
async fn task_owned_conversations_inherit_selected_tool_hooks_but_ownerless_do_not() {
	let registry = Arc::new(create_registry());
	let echo_extension = install_tools(&registry, "echo", vec![tool("echo", |_args, _api, _ctx| async { ToolExecutionResult::default() })]);
	let seen = Arc::new(Mutex::new(Vec::<u64>::new()));
	let seen_copy = seen.clone();
	registry.install(Extension::new("audit").with_hooks(vec![HookRegistration {
		task: tool_task().definition.name.clone(),
		handlers: Arc::new(ToolHooks {
			before_tool: Some(Arc::new(move |_call, api, _ctx| {
				let seen = seen_copy.clone();
				Box::pin(async move { seen.lock().push(api.conversation_id().0); Ok(None) })
			})),
			after_tool: None,
		}),
	}]));
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(vec![
		answer("root ready"),
		calls(&[("echo", json!({}), "root")]), answer("root done"),
		calls(&[("echo", json!({}), "owned")]), answer("owned done"),
		calls(&[("echo", json!({}), "ownerless")]), answer("ownerless done"),
	]);
	let settings = HarnessSettings { extensions: Some(vec![Arc::new(echo_extension)]), ..Default::default() };
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), settings, None).await;
	let _ = submit(&fixture).await;
	let audit = fixture.registry.snapshot().extension("audit").expect("audit extension");
	fixture.root.configure(
		AgentChange { extensions: Some(SetOrClear::Set(ExtensionsChange::Edit { add: vec![audit.as_ref().clone()], remove: vec![] })), ..Default::default() },
		&context(),
	).await.expect("select audit");
	let _ = submit(&fixture).await;

	let no_op: PhaseHandler = Arc::new(|_, _, _| Box::pin(async { Ok(()) }));
	let owner = AnyTask::new(TaskDefinition {
		name: "test.port-owner".into(),
		version: 1,
		initial: Arc::new(|_| json!({"phase":"blocked"})),
		phases: Arc::new(BTreeMap::from([("blocked".into(), no_op.clone())])),
		abort: no_op,
		migrate: None,
		hooks: None,
	});
	let owner_id = fixture.root.commit(move |tx| {
		tx.create_task(&owner, Value::Null, TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None })
			.map_err(|error| error.to_string())
	}, &context()).await.expect("create owner task");
	let owned = fixture.harness.create_conversation(ConversationCreateOptions {
		ownership: ConversationOwnership::Task { task_id: owner_id }, agent: None, init: None,
	}, &context()).await.expect("create task-owned conversation");
	let ownerless = fixture.harness.create_conversation(ConversationCreateOptions {
		ownership: ConversationOwnership::Ownerless, agent: Some(AgentChange { model: Some(SetOrClear::Set(ModelRef { provider: "faux".into(), model_id: "faux-1".into() })), ..Default::default() }), init: None,
	}, &context()).await.expect("create ownerless conversation");

	for conversation in [&owned, &ownerless] {
		let receipt = conversation.submit(SubmissionDraft::Input(InputSubmissionDraft {
			request_id: None, content: UserContentField::Text("go".into()), when_busy: None,
		}), &context()).await.expect("submit child conversation");
		receipt.wait(&context()).await.expect("wait for child run");
	}
	assert_eq!(*seen.lock(), vec![fixture.root.id.0, owned.id.0]);
	close(&fixture).await;
}


#[tokio::test]
async fn prompt_sections_and_hooks_use_an_extension_reloaded_during_tool_execution() {
	let registry = Arc::new(create_registry());
	let requests = Arc::new(Mutex::new(Vec::<String>::new()));
	let prompt = |version: &str| {
		let section_version = version.to_owned();
		let hook_version = version.to_owned();
		let requests = requests.clone();
		Extension::new("prompt")
			.with_sections(vec![PromptSection {
				key: "mode".into(),
				tag: None,
				render: Arc::new(move |_input, _ctx| {
					let version = section_version.clone();
					Box::pin(async move { Ok(Some(version)) })
				}),
			}])
			.with_hooks(vec![HookRegistration {
				task: generation_task().definition.name.clone(),
				handlers: Arc::new(GenerationHooks {
					before_request: Some(Arc::new(move |_payload, _api, _ctx| {
						let requests = requests.clone();
						let version = hook_version.clone();
						Box::pin(async move {
							requests.lock().push(version);
							Ok(None)
						})
					})),
					..Default::default()
				}),
			}])
	};
	registry.install(prompt("v1"));
	let started = Arc::new(Notify::new());
	let release = Arc::new(Notify::new());
	let work = tool("work", {
		let started = started.clone();
		let release = release.clone();
		move |_args, _api, _ctx| {
			let started = started.clone();
			let release = release.clone();
			async move {
				started.notify_one();
				release.notified().await;
				ToolExecutionResult::default()
			}
		}
	});
	install_tools(&registry, "work", vec![work]);
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("work", json!({}), "work")]));
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry.clone(), faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None).await;
	let receipt = fixture.root.submit(SubmissionDraft::Input(InputSubmissionDraft {
		request_id: None, content: UserContentField::Text("go".into()), when_busy: None,
	}), &context()).await.expect("submit tool call");
	started.notified().await;
	registry.install(prompt("v2"));
	release.notify_one();
	receipt.wait(&context()).await.expect("finish reloaded run");
	let entries = all_entries(&fixture.root).await;
	let sections: Vec<_> = entries.iter().filter_map(|entry| entry.model.as_ref()).flatten().filter_map(|message| {
		match message {
			Message::System(system) => system.sections.as_ref()?.get("mode").and_then(Option::as_deref).map(str::to_owned),
			_ => None,
		}
	}).collect();
	assert_eq!(sections, ["<mode>\nv1\n</mode>", "<mode>\nv2\n</mode>"]);
	assert_eq!(*requests.lock(), ["v1", "v2"]);
	close(&fixture).await;
}

#[tokio::test]
async fn default_output_limits_sanitize_progress_but_not_explicit_content_and_keep_null_details() {
	let registry = Arc::new(create_registry());
	let output_slot = Arc::new(Mutex::new(None::<String>));
	let output_slot_copy = output_slot.clone();
	let noisy = tool("noisy", move |_args, api, ctx| {
		let output_slot = output_slot_copy.clone();
		async move {
			api.output(b"a\x07b\r\n", None);
			api.details(json!({"ready":true}), &ctx).await.expect("progress details");
			let live = api.snapshot(&live_doc(), Some(api.conversation_id().0), None, &ctx).await.unwrap();
			*output_slot.lock() = live.and_then(|live| live.get("tools").cloned()).and_then(|tools| tools.as_array().and_then(|tools| tools.first().and_then(|slot| slot.get("output").and_then(Value::as_str).map(str::to_owned))));
			ToolExecutionResult::default()
		}
	});
	let explicit = tool("explicit", |_args, _api, _ctx| async {
		ToolExecutionResult { content: Some(content_text("c\u{1b}d")), ..Default::default() }
	});
	let mut null_details = tool("null", |_args, api, ctx| async move {
		api.details(json!({"old":1}), &ctx).await.expect("old details");
		ToolExecutionResult { details: Some(Value::Null), ..Default::default() }
	});
	null_details.output_limits = Some(ToolOutputLimits { max_lines: Some(1), max_bytes: None, retain: None });
	install_tools(&registry, "lifetime", vec![noisy, explicit, null_details]);
	let fixture = fixture_with_registry(registry, steps(&[
		("noisy", json!({}), "noisy"), ("explicit", json!({}), "explicit"), ("null", json!({}), "null")
	])).await;
	let (_, entries) = submit(&fixture).await;
	let results = results(&entries);
	assert_eq!(output_slot.lock().as_deref(), Some("ab\n"));
	assert_eq!(result_text(result_for(&results, "noisy")), "ab\n");
	assert_eq!(result_text(result_for(&results, "explicit")), "c\u{1b}d");
	assert_eq!(result_for(&results, "null").details, Some(Value::Null));
	close(&fixture).await;
}

#[tokio::test]
async fn tool_registry_replacement_does_not_change_an_in_flight_resolved_implementation() {
	let registry = Arc::new(create_registry());
	let started = Arc::new(Notify::new());
	let release = Arc::new(Notify::new());
	let started_copy = started.clone();
	let release_copy = release.clone();
	let old = tool("work", move |_args, _api, _ctx| {
		let started = started_copy.clone();
		let release = release_copy.clone();
		async move {
			started.notify_one();
			release.notified().await;
			ToolExecutionResult { content: Some(content_text("v1")), ..Default::default() }
		}
	});
	install_tools(&registry, "work", vec![old]);
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("work", json!({}), "work")]));
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry.clone(), faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None).await;
	let receipt = fixture.root.submit(SubmissionDraft::Input(InputSubmissionDraft { request_id: None, content: UserContentField::Text("go".into()), when_busy: None }), &context()).await.unwrap();
	started.notified().await;
	install_tools(&registry, "work", vec![tool("work", |_args, _api, _ctx| async { ToolExecutionResult { content: Some(content_text("v2")), ..Default::default() } })]);
	release.notify_one();
	receipt.wait(&context()).await.expect("run settles");
	assert_eq!(result_text(&results(&all_entries(&fixture.root).await)[0]), "v1");
	close(&fixture).await;
}

#[tokio::test]
async fn aborting_during_after_tool_rejects_details_waiting_for_progress() {
	let registry = Arc::new(create_registry());
	let after_tool = Arc::new(Notify::new());
	let details_started = Arc::new(Notify::new());
	let pending_details = Arc::new(Mutex::new(None::<tokio::task::JoinHandle<Result<(), String>>>));
	let pending_copy = pending_details.clone();
	let details_started_copy = details_started.clone();
	let slow = tool("slow", move |_args, api, ctx| {
		let pending = pending_copy.clone();
		let started = details_started_copy.clone();
		async move {
			api.output(b"first\n", None);
			let details_api = api.clone();
			let details_context = ctx.clone();
			*pending.lock() = Some(tokio::spawn(async move {
				let details = details_api.details(json!({"step":1}), &details_context);
				started.notify_one();
				details.await
			}));
			ToolExecutionResult::default()
		}
	});
	install_tools(&registry, "slow", vec![slow]);
	let after_tool_copy = after_tool.clone();
	install_tool_hooks(&registry, "block-after-tool", ToolHooks {
		before_tool: None,
		after_tool: Some(Arc::new(move |_call, _result, _api, _ctx| {
			let after_tool = after_tool_copy.clone();
			Box::pin(async move {
				after_tool.notify_one();
				std::future::pending::<()>().await;
				Ok(None)
			})
		})),
	});
	let fixture = fixture_with_registry(registry, steps(&[("slow", json!({}), "slow")])).await;
	let receipt = fixture.root.submit(SubmissionDraft::Input(InputSubmissionDraft {
		request_id: None, content: UserContentField::Text("go".into()), when_busy: None,
	}), &context()).await.unwrap();
	details_started.notified().await;
	after_tool.notified().await;
	let live = fixture.harness.session.snapshot(&live_doc(), Some(fixture.root.id.0), None, &context()).await.unwrap().unwrap();
	let task_id = live.get("tools").and_then(Value::as_array).and_then(|slots| slots.first())
		.and_then(|slot| slot.get("taskId")).and_then(Value::as_u64).unwrap();
	assert_eq!(fixture.harness.abort_task(crate::types::TaskId(task_id), &context()).await.unwrap(), "marked");
	let pending = pending_details.lock().take().expect("details operation started");
	let outcome = tokio::time::timeout(std::time::Duration::from_secs(1), pending).await.expect("details waiter rejects");
	assert!(outcome.expect("details task").is_err());
	assert_eq!(receipt.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	close(&fixture).await;
}

#[tokio::test]
async fn tool_abort_writes_only_durable_output_and_settles_the_run() {
	let registry = Arc::new(create_registry());
	let started = Arc::new(Notify::new());
	let started_copy = started.clone();
	let slow = tool("slow", move |_args, api, ctx| {
		let started = started_copy.clone();
		async move {
			eprintln!("[tw] tool entered");
			api.output(b"durable\n", None);
			eprintln!("[tw] output done");
			api.details(json!({"started":true}), &ctx).await.expect("persist durable progress");
			eprintln!("[tw] details done");
			api.output(b"buffered\n", None);
			eprintln!("[tw] output2 done");
			started.notify_one();
			std::future::pending::<()>().await;
			ToolExecutionResult::default()
		}
	});
	install_tools(&registry, "slow", vec![slow]);
	let fixture = fixture_with_registry(registry, steps(&[("slow", json!({}), "slow")])).await;
	let receipt = fixture.root.submit(SubmissionDraft::Input(InputSubmissionDraft { request_id: None, content: UserContentField::Text("go".into()), when_busy: None }), &context()).await.unwrap();
	started.notified().await;
	let live = fixture.harness.session.snapshot(&live_doc(), Some(fixture.root.id.0), None, &context()).await.unwrap().unwrap();
	let task_id = live.get("tools").and_then(Value::as_array).and_then(|slots| slots.first()).and_then(|slot| slot.get("taskId")).and_then(Value::as_u64).unwrap();
	assert_eq!(fixture.harness.abort_task(crate::types::TaskId(task_id), &context()).await.unwrap(), "marked");
	eprintln!("[ta] abort marked");
	receipt.wait(&context()).await.expect("aborted tool result continues run");
	eprintln!("[ta] receipt settled");
	assert_eq!(result_text(&results(&all_entries(&fixture.root).await)[0]), "durable\n|<harness>\n[error] Tool slow was aborted\n</harness>");
	close(&fixture).await;
}

#[tokio::test]
async fn interrupted_unsafe_tool_recovers_with_durable_partial_output_without_reexecution() {
	let registry = Arc::new(create_registry());
	let started = Arc::new(Notify::new());
	let runs = Arc::new(Mutex::new(0));
	install_tools(&registry, "unsafe", vec![interrupted_tool("work", ToolReplay::Unsafe, started.clone(), runs.clone())]);
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("work", json!({}), "work")]));
	let storage = Arc::new(MemoryStorage::new());
	let reports = Arc::new(Mutex::new(Vec::new()));
	let first = open_fixture(storage.clone(), registry.clone(), clone_faux(&faux), reports.clone(), HarnessSettings::default(), None).await;
	let receipt = first.root.submit(SubmissionDraft::Input(InputSubmissionDraft { request_id: None, content: UserContentField::Text("go".into()), when_busy: None }), &context()).await.unwrap();
	started.notified().await;
	first.harness.close(&context()).await.expect("close interrupted runtime");
	let reopened = rerun(&first).await;
	let resumed = reopened.harness.submission(receipt.id, &context()).await.unwrap().unwrap();
	assert_eq!(resumed.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(*runs.lock(), 1);
	let result = results(&all_entries(&reopened.root).await)[0].clone();
	assert!(result.is_error);
	assert_eq!(result.details, Some(json!({"run":1})));
	assert_eq!(result_text(&result), "run 1\n|<harness>\n[error] Tool work was interrupted and may have partially run\n</harness>");
	close(&reopened).await;
}

#[tokio::test]
async fn real_bash_output_survives_close_and_reopen_as_an_interrupted_tool_result() {
	let directory = std::env::temp_dir().join(format!("pi-durable-bash-recovery-{}", std::process::id()));
	std::fs::create_dir_all(&directory).expect("create temp directory");
	let registry = Arc::new(create_registry());
	install_tools(&registry, "bash", vec![crate::tools::create_bash_tool(None)]);
	let env: crate::harness::types::EnvBuilder = Arc::new(|target, _ctx| Box::pin(async move {
		let cwd = target.cwd.unwrap_or_else(|| std::env::temp_dir().to_string_lossy().into_owned());
		Ok(Some(Arc::new(NodeExecutionEnv::new(NodeEnvOptions { cwd, ..NodeEnvOptions::new() })) as Arc<dyn ExecutionEnv>))
	}));
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("bash", json!({"command":"echo started; sleep 30"}), "bash")]));
	let storage = Arc::new(MemoryStorage::new());
	let first = open_fixture(storage, registry, clone_faux(&faux), Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), Some(env)).await;
	let cwd = directory.to_string_lossy().to_string();
	first.root.configure(AgentChange { cwd: Some(SetOrClear::Set(cwd)), ..Default::default() }, &context()).await.unwrap();
	let receipt = first.root.submit(SubmissionDraft::Input(InputSubmissionDraft {
		request_id: None, content: UserContentField::Text("go".into()), when_busy: None,
	}), &context()).await.expect("submit shell");
	tokio::time::timeout(std::time::Duration::from_secs(5), async {
		loop {
			let live = first.harness.session.snapshot(&live_doc(), Some(first.root.id.0), None, &context()).await.unwrap().unwrap();
			if live.get("tools").and_then(Value::as_array).and_then(|slots| slots.first())
				.and_then(|slot| slot.get("output")).and_then(Value::as_str) == Some("started\n")
			{
				break;
			}
			tokio::time::sleep(std::time::Duration::from_millis(10)).await;
		}
	}).await.expect("shell output becomes durable");
	first.harness.close(&context()).await.expect("close running shell");
	let reopened = rerun(&first).await;
	let resumed = reopened.harness.submission(receipt.id(), &context()).await.unwrap().unwrap();
	assert_eq!(resumed.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	let result = results(&all_entries(&reopened.root).await)[0].clone();
	assert_eq!(result_text(&result), "started\n|<harness>\n[error] Tool bash was interrupted and may have partially run\n</harness>");
	close(&reopened).await;
	std::fs::remove_dir_all(directory).expect("remove temp directory");
}

#[tokio::test]
async fn safe_tool_replay_requires_both_persisted_and_current_policies() {
	for (stored, current, expected_runs) in [
		(ToolReplay::Safe, ToolReplay::Safe, 2),
		(ToolReplay::Safe, ToolReplay::Unsafe, 1),
		(ToolReplay::Unsafe, ToolReplay::Safe, 1),
	] {
		let registry = Arc::new(create_registry());
		let started = Arc::new(Notify::new());
		let runs = Arc::new(Mutex::new(0));
		let initial = interrupted_tool("work", stored, started.clone(), runs.clone());
		let old = install_tools(&registry, "work", vec![initial.clone()]);
		let faux = faux_provider(RegisterFauxProviderOptions::default());
		faux.set_responses(steps(&[("work", json!({}), "work")]));
		let storage = Arc::new(MemoryStorage::new());
		let first = open_fixture(storage.clone(), registry.clone(), clone_faux(&faux), Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None).await;
		let receipt = first.root.submit(SubmissionDraft::Input(InputSubmissionDraft { request_id: None, content: UserContentField::Text("go".into()), when_busy: None }), &context()).await.unwrap();
		started.notified().await;
		first.harness.close(&context()).await.unwrap();
		let replacement = interrupted_tool("work", current, Arc::new(Notify::new()), runs.clone());
		registry.uninstall(&old);
		install_tools(&registry, "work", vec![replacement]);
		let reopened = rerun(&first).await;
		let resumed = reopened.harness.submission(receipt.id, &context()).await.unwrap().unwrap();
		assert_eq!(resumed.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
		assert_eq!(*runs.lock(), expected_runs);
		let result = results(&all_entries(&reopened.root).await)[0].clone();
		assert_eq!(result.is_error, expected_runs == 1);
		if expected_runs == 2 { assert_eq!(result_text(&result), "run 2\n"); }
		close(&reopened).await;
	}
}

#[tokio::test]
async fn safe_recovery_uses_current_cwd_but_deselection_prevents_rerun() {
	for deselect in [false, true] {
		let cwd_seen = Arc::new(Mutex::new(Vec::<Option<String>>::new()));
		let first_started = Arc::new(Notify::new());
		let registry = Arc::new(create_registry());
		let cwd_capture = cwd_seen.clone();
		let started_capture = first_started.clone();
		let mut work = tool("work", move |_args, api, _ctx| {
			let cwd_seen = cwd_capture.clone();
			let started = started_capture.clone();
			async move {
				cwd_seen.lock().push(api.env().map(|env| env.cwd().to_owned()));
				if cwd_seen.lock().len() == 1 { started.notify_one(); std::future::pending::<()>().await; }
				ToolExecutionResult::default()
			}
		});
		work.replay = Some(ToolReplay::Safe);
		let extension = install_tools(&registry, "safe", vec![work]);
		let env: crate::harness::types::EnvBuilder = Arc::new(|target, _ctx| Box::pin(async move {
			let cwd = target.cwd.unwrap_or_else(|| "/".into());
			Ok(Some(Arc::new(NodeExecutionEnv::new(NodeEnvOptions { cwd, ..NodeEnvOptions::new() })) as Arc<dyn ExecutionEnv>))
		}));
		let faux = faux_provider(RegisterFauxProviderOptions::default());
		faux.set_responses(steps(&[("work", json!({}), "work")]));
		let storage = Arc::new(MemoryStorage::new());
		let first = open_fixture(storage.clone(), registry.clone(), clone_faux(&faux), Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), Some(env)).await;
		first.root.configure(AgentChange { cwd: Some(SetOrClear::Set("/one".into())), ..Default::default() }, &context()).await.unwrap();
		let receipt = first.root.submit(SubmissionDraft::Input(InputSubmissionDraft { request_id: None, content: UserContentField::Text("go".into()), when_busy: None }), &context()).await.unwrap();
		first_started.notified().await;
		first.root.configure(if deselect { AgentChange { extensions: Some(SetOrClear::Set(crate::harness::types::ExtensionsChange::List(vec![]))), ..Default::default() } } else { AgentChange { cwd: Some(SetOrClear::Set("/two".into())), ..Default::default() } }, &context()).await.unwrap();
		first.harness.close(&context()).await.unwrap();
		let reopened = rerun(&first).await;
		let resumed = reopened.harness.submission(receipt.id, &context()).await.unwrap().unwrap();
		assert_eq!(resumed.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
		if deselect {
			assert_eq!(cwd_seen.lock().len(), 1);
			assert!(result_text(&results(&all_entries(&reopened.root).await)[0]).contains("was interrupted"));
		} else {
			assert_eq!(*cwd_seen.lock(), [Some("/one".into()), Some("/two".into())]);
			assert!(!results(&all_entries(&reopened.root).await)[0].is_error);
		}
		close(&reopened).await;
		let _ = extension;
	}
}

#[tokio::test]
async fn safe_replay_clears_prior_progress_diagnostics_and_details() {
	let registry = Arc::new(create_registry());
	let started = [Arc::new(Notify::new()), Arc::new(Notify::new())];
	let runs = Arc::new(Mutex::new(0usize));
	let runs_copy = runs.clone();
	let started_copy = started.clone();
	let mut work = tool("work", move |_args, api, ctx| {
		let started = started_copy.clone();
		let runs = runs_copy.clone();
		async move {
			let run = { let mut runs = runs.lock(); let n=*runs; *runs += 1; n };
			if run == 0 {
				api.diagnostic(ToolDiagnostic { severity: ToolDiagnosticSeverity::Info, message: "first a".into(), code: None });
				api.diagnostic(ToolDiagnostic { severity: ToolDiagnosticSeverity::Info, message: "first b".into(), code: None });
				api.details(json!({"run":1,"extra":true}), &ctx).await.unwrap();
			} else {
				api.diagnostic(ToolDiagnostic { severity: ToolDiagnosticSeverity::Info, message: "second".into(), code: None });
				api.details(json!({"run":2}), &ctx).await.unwrap();
			}
			started[run].notify_one();
			std::future::pending::<()>().await;
			ToolExecutionResult::default()
		}
	});
	work.replay = Some(ToolReplay::Safe);
	install_tools(&registry, "work", vec![work]);
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("work", json!({}), "work")]));
	let storage = Arc::new(MemoryStorage::new());
	let first = open_fixture(storage, registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None).await;
	first.root.submit(SubmissionDraft::Input(InputSubmissionDraft { request_id: None, content: UserContentField::Text("go".into()), when_busy: None }), &context()).await.unwrap();
	started[0].notified().await;
	first.harness.close(&context()).await.unwrap();
	let reopened = rerun(&first).await;
	started[1].notified().await;
	let live = reopened.harness.session.snapshot(&live_doc(), Some(reopened.root.id.0), None, &context()).await.unwrap().unwrap();
	let slot = &live["tools"][0];
	assert_eq!(slot["diagnostics"], json!([{"severity":"info","message":"second"}]));
	assert_eq!(slot["details"], json!({"run":2}));
	close(&reopened).await;
}

#[tokio::test]
async fn interrupted_before_tool_intent_repeats_hooks_but_keeps_first_durable_memo() {
	let registry = Arc::new(create_registry());
	let ran = Arc::new(Mutex::new(0usize));
	let ran_copy = ran.clone();
	install_tools(&registry, "work", vec![tool("work", move |_args, _api, _ctx| {
		let ran = ran_copy.clone();
		async move { *ran.lock() += 1; ToolExecutionResult::default() }
	})]);
	let asked = Arc::new(Mutex::new(0usize));
	let decisions = Arc::new(Mutex::new(Vec::<Value>::new()));
	let asked_copy = asked.clone();
	let decisions_copy = decisions.clone();
	let reached = Arc::new(Notify::new());
	let reached_copy = reached.clone();
	install_tool_hooks(&registry, "approval", ToolHooks {
		before_tool: Some(Arc::new(move |_call, api, ctx| {
			let asked = asked_copy.clone();
			let decisions = decisions_copy.clone();
			let reached = reached_copy.clone();
			Box::pin(async move {
				let attempt = { let mut asked = asked.lock(); *asked += 1; *asked };
				let decision = api.memo_put("decision", json!(format!("attempt {attempt}")), &ctx).await?;
				decisions.lock().push(decision);
				if attempt == 1 { reached.notify_one(); std::future::pending::<()>().await; }
				Ok(None)
			})
		})),
		after_tool: None,
	});
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("work", json!({}), "work")]));
	let storage = Arc::new(MemoryStorage::new());
	let first = open_fixture(storage, registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None).await;
	let receipt = first.root.submit(SubmissionDraft::Input(InputSubmissionDraft { request_id: None, content: UserContentField::Text("go".into()), when_busy: None }), &context()).await.unwrap();
	reached.notified().await;
	first.harness.close(&context()).await.unwrap();
	let reopened = rerun(&first).await;
	let resumed = reopened.harness.submission(receipt.id, &context()).await.unwrap().unwrap();
	assert_eq!(resumed.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!((*asked.lock(), *ran.lock()), (2, 1));
	assert_eq!(*decisions.lock(), [json!("attempt 1"), json!("attempt 1")]);
	close(&reopened).await;
}

#[tokio::test]
async fn generation_after_tools_hook_recovers_without_duplicating_tool_result() {
	let registry = Arc::new(create_registry());
	install_tools(&registry, "work", vec![tool("work", |_args, _api, _ctx| async { ToolExecutionResult::default() })]);
	let reached = Arc::new(Notify::new());
	let seen = Arc::new(Mutex::new(0usize));
	let reached_copy = reached.clone();
	let seen_copy = seen.clone();
	install_generation_hooks(&registry, "after-tools", GenerationHooks {
		after_tools: Some(Arc::new(move |_assistant, _results, _api, _ctx| {
			let reached = reached_copy.clone();
			let seen = seen_copy.clone();
			Box::pin(async move {
				let count = { let mut seen = seen.lock(); *seen += 1; *seen };
				if count == 1 { reached.notify_one(); std::future::pending::<()>().await; }
				Ok(())
			})
		})),
		..Default::default()
	});
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("work", json!({}), "work")]));
	let storage = Arc::new(MemoryStorage::new());
	let first = open_fixture(storage, registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), None).await;
	let receipt = first.root.submit(SubmissionDraft::Input(InputSubmissionDraft { request_id: None, content: UserContentField::Text("go".into()), when_busy: None }), &context()).await.unwrap();
	reached.notified().await;
	first.harness.close(&context()).await.unwrap();
	let reopened = rerun(&first).await;
	let resumed = reopened.harness.submission(receipt.id, &context()).await.unwrap().unwrap();
	assert_eq!(resumed.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(*seen.lock(), 2);
	assert_eq!(results(&all_entries(&reopened.root).await).len(), 1);
	close(&reopened).await;
}

#[tokio::test]
async fn tool_task_abort_is_distinct_from_recovery_interruption() {
	let registry = Arc::new(create_registry());
	let started = Arc::new(Notify::new());
	let runs = Arc::new(Mutex::new(0));
	install_tools(&registry, "work", vec![interrupted_tool("work", ToolReplay::Unsafe, started.clone(), runs)]);
	let fixture = fixture_with_registry(registry, steps(&[("work", json!({}), "work")])).await;
	let receipt = fixture.root.submit(SubmissionDraft::Input(InputSubmissionDraft { request_id: None, content: UserContentField::Text("go".into()), when_busy: None }), &context()).await.unwrap();
	started.notified().await;
	let live = fixture.harness.session.snapshot(&live_doc(), Some(fixture.root.id.0), None, &context()).await.unwrap().unwrap();
	let task_id = live["tools"][0]["taskId"].as_u64().unwrap();
	assert_eq!(fixture.harness.abort_task(crate::types::TaskId(task_id), &context()).await.unwrap(), "marked");
	let task = fixture.harness.wait_for_task(crate::types::TaskId(task_id), &context()).await.unwrap();
	assert_eq!(task.state.outcome.as_ref().unwrap().status, TaskOutcomeStatus::Aborted);
	assert_eq!(receipt.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(result_text(&results(&all_entries(&fixture.root).await)[0]), "run 1\n|<harness>\n[error] Tool work was aborted\n</harness>");
	close(&fixture).await;
}

#[tokio::test]
async fn failing_bash_command_keeps_the_retained_tail_and_ordered_diagnostics() {
	let directory = std::env::temp_dir().join(format!("pi-durable-failing-tool-port-{}", std::process::id()));
	std::fs::create_dir_all(&directory).expect("create temp directory");
	let registry = Arc::new(create_registry());
	install_tools(&registry, "bash", vec![crate::tools::create_bash_tool(None)]);
	let cwd = directory.to_string_lossy().to_string();
	let env: crate::harness::types::EnvBuilder = Arc::new(|target, _ctx| Box::pin(async move {
		let cwd = target.cwd.unwrap_or_else(|| std::env::temp_dir().to_string_lossy().into_owned());
		Ok(Some(Arc::new(NodeExecutionEnv::new(NodeEnvOptions { cwd, ..NodeEnvOptions::new() })) as Arc<dyn ExecutionEnv>))
	}));
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[(
		"bash",
		json!({"command":"i=1; while [ \"$i\" -le 3000 ]; do printf 'line-%s\\n' \"$i\"; i=$((i + 1)); done; exit 7"}),
		"bash",
	)]));
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), Some(env)).await;
	fixture.root.configure(AgentChange { cwd: Some(SetOrClear::Set(cwd)), ..Default::default() }, &context()).await.unwrap();
	let (status, entries) = submit(&fixture).await;
	assert_eq!(status, SubmissionStatus::Done);
	let entry = entries.iter().find(|entry| entry.kind == "pi.tool-result").expect("bash result");
	let result = &results(&entries)[0];
	let text = result_text(result);
	assert!(result.is_error);
	assert!(text.starts_with("line-1001\n"));
	assert!(text.contains("line-3000\n|<harness>\n[info] Full output: "));
	assert!(text.contains("\n[error] Command exited with code 7\n[warn] Output truncated to its end: 1000 lines, "));
	let codes: Vec<_> = entry.data.as_ref().and_then(|data| data.get("diagnostics")).and_then(Value::as_array)
		.expect("diagnostics")
		.iter()
		.map(|diagnostic| diagnostic.get("code").and_then(Value::as_str).map(str::to_owned))
		.collect();
	assert_eq!(codes, [Some("full_output".into()), Some("tool_error".into()), Some("truncated".into())]);
	close(&fixture).await;
	std::fs::remove_dir_all(directory).expect("remove temp directory");
}

#[tokio::test]
async fn coding_tools_read_edit_and_execute_changes_in_one_run() {
	let directory = std::env::temp_dir().join(format!("pi-durable-tool-port-{}", std::process::id()));
	std::fs::create_dir_all(&directory).expect("create temp directory");
	std::fs::write(directory.join("notes.txt"), "hello world\n").expect("seed notes");
	let registry = Arc::new(create_registry());
	let extension = Extension::new("coding").with_tools(vec![
		Arc::new(crate::tools::create_read_tool()),
		Arc::new(crate::tools::create_edit_tool()),
		Arc::new(crate::tools::create_bash_tool(None)),
	]);
	registry.install(extension);
	let cwd = directory.to_string_lossy().to_string();
	let env: crate::harness::types::EnvBuilder = Arc::new(|target, _ctx| Box::pin(async move {
		let cwd = target.cwd.unwrap_or_else(|| std::env::temp_dir().to_string_lossy().into_owned());
		Ok(Some(Arc::new(NodeExecutionEnv::new(NodeEnvOptions { cwd, ..NodeEnvOptions::new() })) as Arc<dyn ExecutionEnv>))
	}));
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(vec![
		calls(&[("read", json!({"path":"notes.txt"}), "read")]),
		calls(&[("edit", json!({"path":"notes.txt","edits":[{"oldText":"world","newText":"durable"}]}), "edit")]),
		calls(&[("bash", json!({"command":"cat notes.txt"}), "bash")]),
		answer(DONE),
	]);
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings::default(), Some(env)).await;
	fixture.root.configure(AgentChange { cwd: Some(SetOrClear::Set(cwd)), ..Default::default() }, &context()).await.unwrap();
	let (status, entries) = submit(&fixture).await;
	assert_eq!(status, SubmissionStatus::Done);
	let tool_results = results(&entries);
	let outputs: Vec<_> = tool_results.iter().map(|result| (result.tool_name.as_str(), result.is_error, result_text(result))).collect();
	assert_eq!(outputs, [("read", false, "hello world\n".into()), ("edit", false, "Successfully replaced 1 block(s) in notes.txt.".into()), ("bash", false, "hello durable\n".into())]);
	assert_eq!(std::fs::read_to_string(directory.join("notes.txt")).unwrap(), "hello durable\n");
	close(&fixture).await;
	std::fs::remove_dir_all(directory).expect("remove temp directory");
}

#[tokio::test]
async fn api_builds_environment_for_each_call_and_provides_agent_registry_commits_and_memos() {
	let registry = Arc::new(create_registry());
	let child_phase: PhaseHandler = Arc::new(|task, runtime, ctx| Box::pin(async move {
		let result = task.input.get("n").and_then(Value::as_i64).unwrap_or_default() * 2;
		runtime.commit(
			commit_change(move |_tx, _running| Ok(Some(NextTaskState::completed(json!(result))))),
			&ctx,
		).await?;
		Ok(())
	}));
	let child = AnyTask::new(TaskDefinition {
		name: "test.api-child".into(),
		version: 1,
		initial: Arc::new(|_| json!({"phase":"run"})),
		phases: Arc::new(BTreeMap::from([("run".into(), child_phase)])),
		abort: Arc::new(|_, _, _| Box::pin(async { Ok(()) }) as BoxFuture<Result<(), String>>),
		migrate: None,
		hooks: None,
	});
	registry.install(Extension::new("child-runtime").with_tasks(vec![child.clone()]));
	let env_targets = Arc::new(Mutex::new(Vec::<(u64, Option<String>)>::new()));
	let targets = env_targets.clone();
	let env: crate::harness::types::EnvBuilder = Arc::new(move |target, _ctx| {
		let targets = targets.clone();
		Box::pin(async move {
			targets.lock().push((target.conversation_id.0, target.cwd.clone()));
			let cwd = target.cwd.unwrap_or_else(|| "/".into());
			Ok(Some(Arc::new(NodeExecutionEnv::new(NodeEnvOptions { cwd, ..NodeEnvOptions::new() })) as Arc<dyn ExecutionEnv>))
		})
	});
	let observed = Arc::new(Mutex::new(Vec::<Value>::new()));
	let observed_copy = observed.clone();
	let probe = tool("probe", {
		let child = child.clone();
		move |_args, api, ctx| {
			let observed = observed_copy.clone();
			let child = child.clone();
			async move {
				let env_cwd = api.env().map(|env| env.cwd().to_owned()).unwrap_or_default();
				let agent_tools = api.agent(&ctx).await.unwrap().tools.iter().map(|tool| tool.tool.name.clone()).collect::<Vec<_>>();
				let has_extension = api.registry().extension("probe-extension").is_some();
				let task_id = api.task_id().0;
				let conversation_id = api.conversation_id();
				let change_cwd = api.call_id() == "probe-1";
				let mut commit = move |tx: &mut crate::session::Tx<'_>| {
					let entry = tx.append_entry(conversation_id, crate::types::EntryDraft { kind: Some("test.note".into()), data: Some(json!({"task":task_id})), ..Default::default() }).map_err(|error| error.to_string())?;
					if change_cwd {
						crate::harness::agent::configure(tx, conversation_id, &AgentChange {
							cwd: Some(SetOrClear::Set("/".into())), ..Default::default()
						})?;
					}
					Ok(json!(entry.by_task_id == Some(crate::types::TaskId(task_id))))
				};
				let committed = api.commit(&mut commit, &ctx).await.unwrap();
				let first = api.memo_put("memo", json!(1), &ctx).await.unwrap();
				let second = api.memo_put("memo", json!(2), &ctx).await.unwrap();
				let child_id = api.create_task(&child, json!({"n":21}), TaskOptions {
					ownership: TaskOwnership::Conversation, conversation_id: None, background: None,
				}, &ctx).await.unwrap();
				let child = api.wait_for_task(child_id, &ctx).await.unwrap();
				let child_result = child.state.outcome.as_ref().unwrap().result.as_ref().unwrap().clone();
				observed.lock().push(json!([env_cwd, agent_tools, has_extension, committed, first, second, child_result]));
				ToolExecutionResult::default()
			}
		}
	});
	install_tools(&registry, "probe-extension", vec![probe]);
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(steps(&[("probe", json!({}), "probe-1"), ("probe", json!({}), "probe-2")]));
	let fixture = open_fixture(Arc::new(MemoryStorage::new()), registry, faux, Arc::new(Mutex::new(Vec::new())), HarnessSettings { tool_execution: Some(ToolExecutionMode::Sequential), ..Default::default() }, Some(env)).await;
	fixture.root.configure(AgentChange { cwd: Some(SetOrClear::Set("/tmp".into())), ..Default::default() }, &context()).await.unwrap();
	let _ = submit(&fixture).await;
	assert_eq!(*observed.lock(), vec![
		json!(["/tmp", ["probe"], true, true, 1, 1, 42]),
		json!(["/", ["probe"], true, true, 1, 1, 42]),
	]);
	assert!(env_targets.lock().iter().any(|(_, cwd)| cwd.as_deref() == Some("/tmp")));
	assert!(env_targets.lock().iter().any(|(_, cwd)| cwd.as_deref() == Some("/")));
	close(&fixture).await;
}

#[tokio::test]
async fn invocation_bound_task_wait_rejects_after_its_tool_call_finishes() {
	let registry = Arc::new(create_registry());
	let no_op: PhaseHandler = Arc::new(|_, _, _| Box::pin(async { Ok(()) }));
	let pending = AnyTask::new(TaskDefinition {
		name: "test.invocation-pending".into(),
		version: 1,
		initial: Arc::new(|_| json!({"phase":"never"})),
		phases: Arc::new(BTreeMap::from([("never".into(), no_op.clone())])),
		abort: no_op,
		migrate: None,
		hooks: None,
	});
	let wait_slot = Arc::new(Mutex::new(None::<tokio::task::JoinHandle<Result<crate::harness::types::SettledTask, String>>>));
	let wait_slot_copy = wait_slot.clone();
	let detach = tool("detach", move |_args, api, ctx| {
		let pending = pending.clone();
		let wait_slot = wait_slot_copy.clone();
		async move {
			let child = api.create_task(&pending, Value::Null, TaskOptions {
				ownership: TaskOwnership::Conversation, conversation_id: None, background: None,
			}, &ctx).await.unwrap();
			let wait_api = api.clone();
			let wait_context = ctx.clone();
			*wait_slot.lock() = Some(tokio::spawn(async move { wait_api.wait_for_task(child, &wait_context).await }));
			ToolExecutionResult::default()
		}
	});
	install_tools(&registry, "detach", vec![detach]);
	let fixture = fixture_with_registry(registry, steps(&[("detach", json!({}), "detach")])).await;
	let _ = submit(&fixture).await;
	let wait = wait_slot.lock().take().expect("invocation wait started");
	let result = tokio::time::timeout(std::time::Duration::from_secs(1), wait).await.expect("invocation wait rejects");
	let result = result.expect("invocation wait task");
	match result {
		Ok(_) => panic!("wait_for_task escaped its tool invocation"),
		Err(error) => assert!(error.contains("invocation has ended")),
	}
	close(&fixture).await;
}

#[tokio::test]
async fn tool_document_watch_closes_when_its_invocation_ends() {
	let registry = Arc::new(create_registry());
	let watch_slot = Arc::new(Mutex::new(None::<tokio::task::JoinHandle<crate::types::WatchEnd>>));
	let watch_slot_copy = watch_slot.clone();
	let watch_tool = tool("watch", move |_args, api, ctx| {
		let watch_slot = watch_slot_copy.clone();
		async move {
			let watch = api.watch_doc(&live_doc(), Some(api.conversation_id().0), None, &ctx).await
				.expect("tool watch is available")
				.expect("watch handle");
			*watch_slot.lock() = Some(tokio::spawn(async move { watch.closed().await }));
			ToolExecutionResult::default()
		}
	});
	install_tools(&registry, "watch", vec![watch_tool]);
	let fixture = fixture_with_registry(registry, steps(&[("watch", json!({}), "watch")])).await;
	let _ = submit(&fixture).await;
	let closed = watch_slot.lock().take().expect("watch started");
	let _ = tokio::time::timeout(std::time::Duration::from_secs(1), closed).await.expect("watch closes after call");
	close(&fixture).await;
}

#[tokio::test]
async fn pending_details_settle_on_the_terminal_commit_after_progress_coalesces() {
	let registry = Arc::new(create_registry());
	let completed = Arc::new(Mutex::new(Vec::<&'static str>::new()));
	let completed_copy = completed.clone();
	let third_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
	let third_done_copy = third_done.clone();
	let details = tool("details", move |_args, api, ctx| {
		let completed = completed_copy.clone();
		let third_done = third_done_copy.clone();
		async move {
			let (first, second) = tokio::join!(
				api.details(json!({"n":1}), &ctx),
				api.details(json!({"n":2}), &ctx),
			);
			first.expect("first progress commit");
			second.expect("coalesced progress commit");
			completed.lock().extend(["first", "second"]);

			let third_api = api.clone();
			let third_context = ctx.clone();
			let third_completed = completed.clone();
			let started = Arc::new(Notify::new());
			let started_copy = started.clone();
			let third_done_for_task = third_done.clone();
			tokio::spawn(async move {
				let pending = third_api.details(json!({"n":3}), &third_context);
				started_copy.notify_one();
				pending.await.expect("terminal commit settles pending details");
				third_completed.lock().push("third");
				third_done_for_task.store(true, std::sync::atomic::Ordering::SeqCst);
			});
			started.notified().await;
			tokio::task::yield_now().await;
			assert!(!third_done.load(std::sync::atomic::Ordering::SeqCst));
			ToolExecutionResult::default()
		}
	});
	install_tools(&registry, "details", vec![details]);
	let fixture = fixture_with_registry(registry, steps(&[("details", json!({}), "details")])).await;
	let (_, entries) = submit(&fixture).await;
	tokio::time::timeout(std::time::Duration::from_secs(1), async {
		while !third_done.load(std::sync::atomic::Ordering::SeqCst) {
			tokio::time::sleep(std::time::Duration::from_millis(1)).await;
		}
	}).await.expect("terminal progress waiter settles");
	assert_eq!(*completed.lock(), ["first", "second", "third"]);
	assert_eq!(results(&entries)[0].details, Some(json!({"n":3})));
	close(&fixture).await;
}

#[tokio::test]
async fn retained_output_defaults_to_the_head_and_null_details_override_progress() {
	let registry = Arc::new(create_registry());
	let lines = tool("lines", |_args, api, _ctx| async move {
		for index in 1..=2500 { api.output(format!("{index}\n").as_bytes(), None); }
		ToolExecutionResult::default()
	});
	let null = tool("null", |_args, api, ctx| async move {
		api.details(json!({"old":1}), &ctx).await.unwrap();
		ToolExecutionResult { details: Some(Value::Null), ..Default::default() }
	});
	install_tools(&registry, "defaults", vec![lines, null]);
	let fixture = fixture_with_registry(registry, steps(&[("lines", json!({}), "lines"), ("null", json!({}), "null")])).await;
	let (_, entries) = submit(&fixture).await;
	let results = results(&entries);
	let first = result_text(&results[0]);
	assert!(first.starts_with("1\n2\n"));
	assert!(first.ends_with("\n2000\n|<harness>\n[warn] Output truncated to its beginning: 500 lines, 2500 bytes dropped\n</harness>"));
	assert_eq!(results[1].details, Some(Value::Null));
	close(&fixture).await;
}
