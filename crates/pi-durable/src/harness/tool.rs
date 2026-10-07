//! Port of `packages/durable/src/harness/tool.ts` (the built-in `pi.tool`
//! task).
//!
//! Erasure (D1 — see `tasks/mod.rs`): checkpoint JSON tagged `phase`
//! (`call`/`execute`); result `{ entryId, control? }`. D2 — the running
//! `api` object is built per call as an [`ApiCore`] implementing
//! [`ToolExecutionApi`] over the task runtime. D3 — `details()` wait
//! cancellation leaves the update in place (upstream `awaitWithContext`).

use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};

use pi_ai::types::{AssistantContent, ImageContent, TextContent, ToolCall, ToolResultMessage, UserContent};
use pi_ai::utils::validation::validate_tool_arguments;
use pi_chord::context::Context;
use pi_chord::delta::overlap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::entries::{ASSISTANT_ENTRY, TOOL_RESULT_ENTRY};
use crate::harness::types::coerce_hook_api;
use crate::harness::util::abort_reason_text;
use crate::harness::live::{clear_progress, finish_slot, read_live, store_live, SlotStatus, ToolSlot};
use crate::harness::output::{bound_output, OutputBuffer, OutputLimits, Progress, Retain, PROGRESS_BYTES_PER_SECOND};
use crate::harness::types::{
	EntryId, HookApi, TaskId, TaskRecord, ToolControl, ToolDiagnostic, ToolDiagnosticSeverity, ToolExecutionApi,
	ToolExecutionResult, ToolHooks, ToolRegistration,
};
use crate::harness::usage::record_usage;
use crate::session::Tx;
use crate::tasks::{commit_change, define_task, BoxFuture, NextTaskState, PhaseHandler, TaskDefinition, TaskRuntime};
use crate::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, utf8_byte_length};
use crate::types::{EntryDraft, TaskOutcome, TaskOutcomeError, TaskOutcomeStatus, TaskOptions};

/// Input of the tool task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolTaskInput {
	pub assistant: EntryId,
	pub call_id: String,
}

/// Erased checkpoint of the tool task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
pub enum ToolTaskCheckpoint {
	Call,
	/// Durable intent: the final arguments and the replay policy recorded
	/// before execution.
	Execute { arguments: Map<String, Value>, replay: Replay },
}

/// Replay policy of a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Replay {
	Safe,
	Unsafe,
}

/// Result of the tool task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolTaskResult {
	pub entry_id: EntryId,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub control: Option<ToolControl>,
}

fn tool_definition() -> TaskDefinition {
	let mut phases: BTreeMap<String, PhaseHandler> = BTreeMap::new();
	phases.insert("call".to_string(), Arc::new(|task, runtime, context| Box::pin(call_phase(task, runtime, context)) as BoxFuture<Result<(), String>>));
	phases.insert("execute".to_string(), Arc::new(|task, runtime, context| Box::pin(execute_phase(task, runtime, context)) as BoxFuture<Result<(), String>>));
	TaskDefinition {
		name: "pi.tool".to_string(),
		version: 1,
		initial: Arc::new(|_| json!({ "phase": "call" })),
		phases: Arc::new(phases),
		abort: Arc::new(|task, runtime, context| Box::pin(abort_phase(task, runtime, context)) as BoxFuture<Result<(), String>>),
		migrate: None,
		hooks: None,
	}
}

use std::collections::BTreeMap;

static TOOL_TASK_STATIC: std::sync::LazyLock<crate::harness::types::AnyTask> =
	std::sync::LazyLock::new(|| crate::harness::types::AnyTask::new(tool_definition()));

/// Built-in tool task: resolves the called tool among its phase agent's
/// tools, validates, runs `beforeTool`, records intent, executes, runs
/// `afterTool`, and appends the result, all in one `call` handler so nothing
/// separates resolution from settlement. `execute` is reached only by
/// recovery and applies the replay rule.
pub fn tool_task() -> crate::harness::types::AnyTask {
	TOOL_TASK_STATIC.clone()
}

fn tool_checkpoint(record: &TaskRecord) -> Result<ToolTaskCheckpoint, String> {
	serde_json::from_value(record.state.checkpoint.clone().unwrap_or(Value::Null))
		.map_err(|error| format!("Tool checkpoint is invalid: {error}"))
}

/// The tool call `call_id` of the assistant entry.
async fn read_call(
	runtime: &Arc<dyn TaskRuntime>,
	input: &ToolTaskInput,
	context: &Context,
) -> Result<ToolCall, String> {
	let entry = runtime
		.entry(Some(ASSISTANT_ENTRY.kind.as_str()), input.assistant, context)
		.await?;
	let call = entry
		.and_then(|entry| entry.model)
		.and_then(|model| model.into_iter().next())
		.and_then(|message| match message {
			pi_ai::Message::Assistant(assistant) => assistant.content.into_iter().find_map(|block| match block {
				AssistantContent::ToolCall(call) if call.id == input.call_id => Some(call),
				_ => None,
			}),
			_ => None,
		});
	call.ok_or_else(|| format!("Entry {} has no tool call {}", input.assistant, input.call_id))
}

/// Arguments, or why they are invalid.
enum Checked {
	Args(Map<String, Value>),
	Error(String),
}

/// The call's arguments as repaired by the tool; a failing repair makes them
/// invalid.
fn prepare(tool: &ToolRegistration, args: &Map<String, Value>) -> Checked {
	match &tool.prepare_arguments {
		None => Checked::Args(args.clone()),
		Some(repair) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| repair(&Value::Object(args.clone())))) {
			Ok(Value::Object(args)) => Checked::Args(args),
			Ok(_) => Checked::Error("Tool argument repair returned a non-object".to_string()),
			Err(error) => Checked::Error(panic_text(error)),
		},
	}
}

/// Arguments validated and coerced against the implementation's schema.
fn validate(tool: &ToolRegistration, call: &ToolCall, args: &Map<String, Value>) -> Checked {
	let mut coerced = call.clone();
	coerced.arguments = args.clone();
	match validate_tool_arguments(&tool.tool, &coerced) {
		Ok(args) => Checked::Args(args),
		Err(error) => Checked::Error(error.to_string()),
	}
}

fn invalid(message: &str) -> ToolExecutionResult {
	harness_error("invalid_arguments", message)
}

/// What a running tool reported through its api: output, the last details,
/// and diagnostics.
struct Reported {
	output: OutputBuffer,
	limits: OutputLimits,
	diagnostics: Vec<ToolDiagnostic>,
	details: Option<Value>,
}

/// Execute with the resolved implementation, then settle its result.
async fn run(
	runtime: &Arc<dyn TaskRuntime>,
	call: &ToolCall,
	tool: Arc<ToolRegistration>,
	args: Map<String, Value>,
	context: &Context,
) -> Result<(), String> {
	let limits = OutputLimits {
		max_bytes: tool
			.output_limits
			.as_ref()
			.and_then(|limits| limits.max_bytes)
			.map(|bytes| bytes as usize)
			.unwrap_or(DEFAULT_MAX_BYTES),
		max_lines: tool
			.output_limits
			.as_ref()
			.and_then(|limits| limits.max_lines)
			.map(|lines| lines as usize)
			.unwrap_or(DEFAULT_MAX_LINES),
		retain: match tool.output_limits.as_ref().and_then(|limits| limits.retain) {
			Some(crate::harness::api::OutputRetain::Head) | None => Retain::Head,
			Some(crate::harness::api::OutputRetain::Tail) => Retain::Tail,
		},
	};
	let reported = Arc::new(parking_lot::Mutex::new(Reported {
		output: OutputBuffer::new(limits),
		limits,
		diagnostics: Vec::new(),
		details: None,
	}));
	let progress = publish_progress(runtime, &reported, context);
	let ended = Arc::new(std::sync::atomic::AtomicBool::new(false));
	let core = ApiCore {
		runtime: runtime.clone(),
		call_id: call.id.clone(),
		reported: reported.clone(),
		progress: progress.clone(),
		ended: ended.clone(),
		env: parking_lot::Mutex::new(None),
		snapshot: runtime.registry(),
	};
	let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		tool.execute.clone()
	})) {
		Ok(execute) => {
			// Built for this call, so a rerun after recovery gets the
			// conversation's environment at that time.
			let env_error = match runtime.env(context).await {
				Ok(env) => {
					core.set_env(env);
					None
				}
				Err(error) => Some(error),
			};
			if let Some(error) = env_error {
				if runtime.signal().aborted() {
					ended.store(true, std::sync::atomic::Ordering::SeqCst);
					let waiters = progress.stop().await;
					for waiter in waiters {
						let _ = waiter.send(Err(error.clone()));
					}
					return Err(error);
				}
				ToolExecutionResult {
					content: None,
					is_error: Some(true),
					details: None,
					diagnostics: Some(vec![tool_diagnostic("tool_error", &error)]),
					usage: None,
					control: None,
				}
			} else {
				execute(Value::Object(args), Arc::new(core), context.clone()).await
			}
		}
		Err(_) => unreachable!("tool execute registration panics at definition"),
	};
	ended.store(true, std::sync::atomic::Ordering::SeqCst);
	reported.lock().output.end();
	// Details still waiting for a progress commit settle with the terminal
	// commit, the final flush.
	let pending = progress.stop().await;
	let settled = final_result(runtime, call, result, &reported, context).await?;
	let ending = Ending::Completed;
	settle(runtime, call, ending, &reported, &settled, context).await?;
	for waiter in pending {
		let _ = waiter.send(Ok(()));
	}
	Ok(())
}

/// Throttled commits of what the tool reported into its `pi.live.tools`
/// slot, each writing only what changed since the last one.
fn publish_progress(
	runtime: &Arc<dyn TaskRuntime>,
	reported: &Arc<parking_lot::Mutex<Reported>>,
	context: &Context,
) -> Arc<Progress> {
	let runtime = runtime.clone();
	let reported = reported.clone();
	let context = context.clone();
	let runtime_for_error = runtime.clone();
	let output_interval_ms = runtime.settings().progress.output_interval_ms;
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	let diagnostics_written = Arc::new(AtomicUsize::new(0));
	let diagnostics_written_in = diagnostics_written.clone();
	let progress = Arc::new(Progress::new(
		move || {
			let runtime = runtime.clone();
			let reported = reported.clone();
			let context = context.clone();
			let diagnostics_written = diagnostics_written_in.clone();
			Box::pin(async move {
				// Capture everything synchronously: the tool keeps reporting
				// while the commit is in flight.
				let (snapshot, details, added_diagnostics, diagnostics_len) = {
					let mut reported = reported.lock();
					let snapshot = reported.output.snapshot();
					let written = diagnostics_written.load(Ordering::Relaxed);
					let added = reported.diagnostics.get(written..).unwrap_or_default().to_vec();
					(snapshot, reported.details.clone(), added, reported.diagnostics.len())
				};
				let current_text = snapshot.text.clone();
				let mut bytes = utf8_byte_length(&current_text);
				if let Some(details) = &details {
					bytes += utf8_byte_length(&serde_json::to_string(details).unwrap_or_default());
				}
				if !added_diagnostics.is_empty() {
					bytes += utf8_byte_length(&serde_json::to_string(&added_diagnostics).unwrap_or_default());
				}
				let runtime_for_write = runtime.clone();
				runtime_for_write
					.commit(
						commit_change(move |tx: &mut crate::session::Tx<'_>, _current| {
							let draft = tx
								.doc(
									&crate::harness::live::live_doc(),
									Some(conversation_id.0),
									None,
									None,
								)
								.map_err(|e| e.to_string())?;
							let mut live = read_live(&draft);
							let Some(slot) = crate::harness::live::tool_slot(&mut live, &task_id)
							else {
								return Ok(None);
							};
							// Assign `output` as one string field; the diff
							// derives the append/trim deltas.
							if slot.output.as_deref().unwrap_or("") != snapshot.text {
								slot.output = Some(snapshot.text.clone());
							}
							if snapshot.dropped_bytes > 0 {
								slot.dropped_bytes = Some(snapshot.dropped_bytes as u64);
							}
							if snapshot.dropped_lines > 0 {
								slot.dropped_lines = Some(snapshot.dropped_lines as u64);
							}
							if let Some(details) = &details {
								slot.details = Some(details.clone());
							}
							if !added_diagnostics.is_empty() {
								slot.diagnostics.get_or_insert_with(Vec::new).extend(added_diagnostics.iter().cloned());
							}
							store_live(&draft, &live)?;
							Ok(None)
						}),
						&context,
					)
					.await
					.map_err(|error| error.to_string())?;
				diagnostics_written.store(diagnostics_len, Ordering::Relaxed);
				Ok(bytes)
			})
		},
		move |error| {
			// Rejections after an abort mark or close are expected; the
			// committed state stays consistent.
			if !runtime_for_error.signal().aborted() {
				runtime_for_error.report(error);
			}
		},
		output_interval_ms,
	));
	progress.run();
	progress
}

/// The settled result: the tool's result with the retained output and last
/// details as fallbacks, its diagnostics after those reported through the
/// api, `afterTool` applied, and explicit text bounded, with the Harness's
/// truncation diagnostic last.
async fn final_result(
	runtime: &Arc<dyn TaskRuntime>,
	call: &ToolCall,
	result: ToolExecutionResult,
	reported: &Arc<parking_lot::Mutex<Reported>>,
	context: &Context,
) -> Result<ToolExecutionResult, String> {
	let mut harness: Vec<ToolDiagnostic> = Vec::new();
	let (retained, last_details, reported_diagnostics) = {
		let mut reported = reported.lock();
		let retained = if result.content.is_none() { Some(reported.output.snapshot()) } else { None };
		(retained, reported.details.clone(), reported.diagnostics.clone())
	};
	let content: Option<Vec<UserContent>> = match &retained {
		None => result.content.clone().and_then(|value| serde_json::from_value(value).ok()),
		Some(retained) if retained.text.is_empty() => Some(Vec::new()),
		Some(retained) => Some(vec![UserContent::Text(TextContent::new(&retained.text))]),
	};
	let mut first = content.clone();
	let _ = &mut first;
	let mut final_result = ToolExecutionResult {
		content: result.content.clone().or_else(|| content.as_ref().map(|_| serde_json::to_value(&content).expect("content serializes"))),
		is_error: result.is_error,
		details: result.details.clone().or(last_details),
		diagnostics: Some(
			reported_diagnostics
				.iter()
				.cloned()
				.chain(result.diagnostics.iter().flatten().cloned())
				.collect(),
		),
		usage: result.usage.clone(),
		control: result.control.clone(),
	};
	// Hook: afterTool.
	{
		let hooks = runtime.hooks();
		let call2 = call.clone();
		let final2 = final_result.clone();
		let runtime2 = runtime.clone();
		let context2 = context.clone();
		hooks
			.each(
				"afterTool",
				Arc::new(move |handler: &Arc<dyn crate::tasks::HookHandler>| {
					let call = call2.clone();
					let mut final_result = final2.clone();
					let runtime = runtime2.clone();
					let context = context2.clone();
					let hooks_table = handler.as_any().downcast_ref::<ToolHooks>().cloned();
						Box::pin(async move {
						let Some(hooks) = hooks_table else { return Ok(()) };
						if let Some(hook) = &hooks.after_tool {
							if let Some(replaced) =
								hook(call, final_result, coerce_hook_api(runtime.clone()), context).await?
							{
								final_result = replaced;
								store_final(&final_result);
							}
						}
						Ok(())
						})
				}),
			)
			.await?;
	}
	if let Some(stored) = take_final() {
		final_result = stored;
	}
	// The retained output's truncation applies only while afterTool kept
	// that content.
	if let Some(retained) = &retained {
		if retained.dropped_bytes > 0 {
			harness.push(truncated(
				retained.dropped_bytes,
				retained.dropped_lines,
				Some(reported.lock().limits.retain),
			));
		}
	}
	let (bounded_content, dropped_bytes, dropped_lines) = bound_content(&final_result.content, reported.lock().limits);
	if dropped_bytes > 0 {
		harness.push(truncated(dropped_bytes, dropped_lines, Some(reported.lock().limits.retain)));
	}
	final_result.content = bounded_content;
	final_result.diagnostics = Some(
		final_result
			.diagnostics
			.take()
			.unwrap_or_default()
			.into_iter()
			.chain(harness)
			.collect(),
	);
	Ok(final_result)
}

thread_local! {
	static FINAL_RESULT: std::cell::RefCell<Option<ToolExecutionResult>> = const { std::cell::RefCell::new(None) };
}

fn store_final(result: &ToolExecutionResult) {
	FINAL_RESULT.with(|cell| *cell.borrow_mut() = Some(result.clone()));
}

fn take_final() -> Option<ToolExecutionResult> {
	FINAL_RESULT.with(|cell| cell.borrow_mut().take())
}

/// Commit the tool's terminal state: append its result entry, mark its slot
/// done, and complete or end aborted with the entry ID.
async fn settle(
	runtime: &Arc<dyn TaskRuntime>,
	call: &ToolCall,
	ending: Ending,
	_reported: &Arc<parking_lot::Mutex<Reported>>,
	result: &ToolExecutionResult,
	context: &Context,
) -> Result<(), String> {
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	let call2 = call.clone();
	let result2 = result.clone();
	let ending2 = ending;
	let now = runtime.now() as i64;
	runtime
		.commit(
			commit_change(move |tx: &mut crate::session::Tx<'_>, _current| {
				let call = call2.clone();
				let result = result2.clone();
				let draft = tx
					.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
					.map_err(|e| e.to_string())?;
				let mut live = read_live(&draft);
				let slot = crate::harness::live::tool_slot(&mut live, &task_id);
				let entry = append_tool_result(tx, conversation_id, &call, &result, now)?;
				if let Some(slot) = slot {
					finish_slot(slot, Some(entry.id));
				}
				store_live(&draft, &live)?;
				let entry_id = entry.id;
				match ending2 {
					Ending::Completed => {
						let control = result
							.control
							.as_ref()
							.map(|control| serde_json::to_value(control).expect("ToolControl serializes"));
						Ok(Some(NextTaskState::Terminal {
							outcome: TaskOutcome {
								status: TaskOutcomeStatus::Completed,
								result: Some(match control {
									Some(control) => json!({ "entryId": entry_id, "control": control }),
									None => json!({ "entryId": entry_id }),
								}),
								error: None,
								reason: None,
							},
						}))
					}
					Ending::Aborted => Ok(Some(NextTaskState::Terminal {
						outcome: TaskOutcome {
							status: TaskOutcomeStatus::Aborted,
							result: Some(json!({ "entryId": entry_id })),
							error: None,
							reason: Some("aborted".to_string()),
						},
					})),
					Ending::Failed { message } => Ok(Some(NextTaskState::Terminal {
						outcome: TaskOutcome {
							status: TaskOutcomeStatus::Failed,
							result: Some(json!({ "entryId": entry_id })),
							error: Some(TaskOutcomeError { message, detail: None }),
							reason: None,
						},
					})),
				}
			}),
			context,
		)
		.await
		.map_err(|e| e.to_string())?;
	Ok(())
}

/// How a tool task ends; the result entry is appended either way. `failed`
/// (execution threw or was interrupted) records cancellation intent for the
/// conversations the call owns; a result with `is_error` still completes.
#[derive(Debug, Clone)]
enum Ending {
	Completed,
	Aborted,
	Failed { message: String },
}

/// An error result from the slot's durable partial output, details, and
/// diagnostics.
fn from_slot(slot: Option<&ToolSlot>, code: &str, message: &str) -> ToolExecutionResult {
	let slot = slot.cloned().unwrap_or_default();
	let mut diagnostics: Vec<ToolDiagnostic> = slot.diagnostics.clone().unwrap_or_default();
	let dropped_bytes = slot.dropped_bytes.unwrap_or(0);
	if dropped_bytes > 0 {
		diagnostics.push(truncated(
			dropped_bytes as usize,
			slot.dropped_lines.unwrap_or(0) as usize,
			None,
		));
	}
	diagnostics.push(tool_diagnostic(code, message));
	let content = match &slot.output {
		Some(output) if !output.is_empty() => Some(
			serde_json::to_value(vec![UserContent::Text(TextContent::new(output))]).expect("content serializes"),
		),
		_ => Some(Value::Array(Vec::new())),
	};
	ToolExecutionResult {
		content,
		is_error: Some(true),
		details: slot.details.clone(),
		diagnostics: Some(diagnostics),
		usage: None,
		control: None,
	}
}

/// An error result the Harness writes itself: no content and one `error`
/// diagnostic with `code`.
pub fn harness_error(code: &str, message: &str) -> ToolExecutionResult {
	ToolExecutionResult {
		content: Some(Value::Array(Vec::new())),
		is_error: Some(true),
		details: None,
		diagnostics: Some(vec![tool_diagnostic(code, message)]),
		usage: None,
		control: None,
	}
}

fn tool_diagnostic(code: &str, message: &str) -> ToolDiagnostic {
	ToolDiagnostic {
		severity: ToolDiagnosticSeverity::Error,
		code: Some(code.to_string()),
		message: message.to_string(),
	}
}

/// The Harness's truncation diagnostic; `retain` is unknown when rebuilt
/// from a slot after recovery.
fn truncated(dropped_bytes: usize, dropped_lines: usize, retain: Option<Retain>) -> ToolDiagnostic {
	let kept = match retain {
		None => String::new(),
		Some(Retain::Head) => " to its beginning".to_string(),
		Some(Retain::Tail) => " to its end".to_string(),
	};
	ToolDiagnostic {
		severity: ToolDiagnosticSeverity::Warn,
		code: Some("truncated".to_string()),
		message: format!(
			"Output truncated{kept}: {dropped_lines} lines, {dropped_bytes} bytes dropped"
		),
	}
}

/// Append a `pi.tool-result` entry. The content ends with the rendered
/// diagnostics, so the stored message is exactly what the model sees; `data`
/// keeps the structured list. A result's usage is added to `pi.usage` in the
/// same commit.
pub fn append_tool_result(
	tx: &Tx<'_>,
	conversation_id: crate::harness::types::ConversationId,
	call: &ToolCall,
	result: &ToolExecutionResult,
	timestamp: i64,
) -> Result<crate::types::EntryRecord, String> {
	let diagnostics: Vec<ToolDiagnostic> = result.diagnostics.clone().unwrap_or_default();
	let mut content: Vec<UserContent> = result
		.content
		.clone()
		.and_then(|value| serde_json::from_value(value).ok())
		.unwrap_or_default();
	if !diagnostics.is_empty() {
		content.push(UserContent::Text(TextContent::new(render_diagnostics(&diagnostics))));
	}
	let message = ToolResultMessage {
		tool_call_id: call.id.clone(),
		tool_name: call.name.clone(),
		content,
		details: result.details.clone(),
		usage: result.usage.clone(),
		nested_calls: None,
		is_error: result.is_error.unwrap_or(false),
		timestamp,
	};
	if let Some(usage) = &result.usage {
		record_usage(tx, conversation_id, crate::harness::usage::UsageBucket::Tools, &call.name, usage)?;
	}
	let mut draft = EntryDraft { kind: Some(TOOL_RESULT_ENTRY.kind.clone()), ..Default::default() };
	draft.model = Some(vec![pi_ai::Message::ToolResult(message)]);
	draft.data = Some(json!({ "diagnostics": diagnostics }));
	tx.append_entry(conversation_id, draft).map_err(|e| e.to_string())
}

fn render_diagnostics(diagnostics: &[ToolDiagnostic]) -> String {
	let body = diagnostics
		.iter()
		.map(|diagnostic| format!("[{:?}] {}", diagnostic.severity, diagnostic.message))
		.collect::<Vec<_>>()
		.join("\n");
	format!("<harness>\n{body}\n</harness>")
}

/// Bound the text of result content. When the joined text exceeds the
/// limits, the text items are replaced by one bounded item at the position
/// of the first (head) or last (tail) text item; other content is kept.
fn bound_content(
	content: &Option<Value>,
	limits: OutputLimits,
) -> (Option<Value>, usize, usize) {
	let Some(content) = content else { return (None, 0, 0) };
	let items: Vec<UserContent> = serde_json::from_value(content.clone()).unwrap_or_default();
	let texts: Vec<&String> = items
		.iter()
		.filter_map(|item| match item {
			UserContent::Text(text) => Some(&text.text),
			_ => None,
		})
		.collect();
	let joined: String = texts.iter().map(|text| text.as_str()).collect();
	let bounded = bound_output(&joined, limits);
	if bounded.dropped_bytes == 0 {
		return (Some(content.clone()), 0, 0);
	}
	let keep = match limits.retain {
		Retain::Head => texts.first(),
		Retain::Tail => texts.last(),
	};
	let mut result: Vec<UserContent> = Vec::new();
	for item in &items {
		match item {
			UserContent::Text(text) => {
				if keep.is_some_and(|keep| std::ptr::eq(*keep, &text.text)) {
					result.push(UserContent::Text(TextContent::new(&bounded.text)));
				}
			}
			other => result.push(other.clone()),
		}
	}
	(
		Some(serde_json::to_value(&result).expect("content serializes")),
		bounded.dropped_bytes,
		bounded.dropped_lines,
	)
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
	if let Some(message) = payload.downcast_ref::<&str>() {
		message.to_string()
	} else if let Some(message) = payload.downcast_ref::<String>() {
		message.clone()
	} else {
		"tool argument repair panicked".to_string()
	}
}

/// Phase `call`.
async fn call_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let input: ToolTaskInput = serde_json::from_value(task.input.clone())
		.map_err(|error| format!("Tool task input is invalid: {error}"))?;
	let call = read_call(&runtime, &input, &context).await?;
	let tool = runtime.agent(&context).await?.tools.iter().find(|each| each.name() == call.name).cloned();
	let Some(tool) = tool else {
		let error = harness_error("tool_unavailable", &format!("Tool {} is not available", call.name));
		return Box::pin(settle(&runtime, &call, Ending::Completed, &empty_reported(), &error, &context)).await;
	};
	let prepared = prepare(&tool, &call.arguments);
	let args = match prepared {
		Checked::Error(error) => {
			let invalid = invalid(&error);
			return settle(&runtime, &call, Ending::Completed, &empty_reported(), &invalid, &context).await;
		}
		Checked::Args(args) => args,
	};
	let args = match validate(&tool, &call, &args) {
		Checked::Error(error) => {
			let invalid = invalid(&error);
			return settle(&runtime, &call, Ending::Completed, &empty_reported(), &invalid, &context).await;
		}
		Checked::Args(args) => args,
	};
	// Hook: beforeTool.
	let mut args = args;
	let mut block: Option<String> = None;
	{
		let hooks = runtime.hooks();
		let call2 = call.clone();
		let runtime2 = runtime.clone();
		let context2 = context.clone();
		hooks
			.each(
				"beforeTool",
				Arc::new(move |handler: &Arc<dyn crate::tasks::HookHandler>| {
					let mut call = call2.clone();
					if let Value::Object(arguments) = serde_json::to_value(&args_snapshot()).expect("args") {
						call.arguments = arguments;
					}
					let runtime = runtime2.clone();
					let context = context2.clone();
					let hooks_table = handler.as_any().downcast_ref::<ToolHooks>().cloned();
						Box::pin(async move {
						let Some(hooks) = hooks_table else { return Ok(()) };
						let Some(hook) = &hooks.before_tool else { return Ok(()) };
						match hook(call, coerce_hook_api(runtime.clone()), context).await {
							Err(error) => {
								if runtime_aborted() {
									return Err(error);
								}
								set_block(error);
							}
							Ok(Some(decision)) => {
								if let Some(blocked) = decision.block {
									set_block(blocked);
								} else if let Some(arguments) = decision.arguments {
									set_args(arguments);
								}
							}
							Ok(None) => {}
						}
						Ok(())
						})
				}),
			)
			.await?;
	}
	if let Some(blocked) = take_block() {
		let _ = blocked;
		if let Some(block) = block.as_ref().or(Some(&blocked)) {
			let blocked_result = harness_error("blocked", &format!("Tool call blocked: {block}"));
			return settle(&runtime, &call, Ending::Completed, &empty_reported(), &blocked_result, &context).await;
		}
	}
	if let Some(replacement) = take_args() {
		args = replacement;
	}
	let args = match validate(&tool, &call, &args) {
		Checked::Error(error) => {
			let invalid = invalid(&error);
			return settle(&runtime, &call, Ending::Completed, &empty_reported(), &invalid, &context).await;
		}
		Checked::Args(args) => args,
	};
	let conversation_id = runtime.conversation_id();
	let task_id = runtime.task_id();
	let final_args = args.clone();
	let replay = tool.replay.unwrap_or(crate::harness::api::ToolReplay::Unsafe);
	runtime
		.commit(
			commit_change(move |tx, _current| {
					let draft = tx
						.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
						.map_err(|e| e.to_string())?;
					let mut live = read_live(&draft);
					if let Some(slot) = crate::harness::live::tool_slot(&mut live, &task_id) {
						slot.status = SlotStatus::Running;
					}
					store_live(&draft, &live)?;
					Ok(Some(NextTaskState::Running {
						checkpoint: json!({
							"phase": "execute",
							"arguments": final_args,
							"replay": replay,
						}),
					}))
			}),
			&context,
		)
		.await?;
	run(&runtime, &call, tool, args, &context).await
}

thread_local! {
	static BLOCK: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
	static ARGS: std::cell::RefCell<Option<Map<String, Value>>> = const { std::cell::RefCell::new(None) };
}

fn set_block(message: String) {
	BLOCK.with(|cell| *cell.borrow_mut() = Some(message));
}

fn take_block() -> Option<String> {
	BLOCK.with(|cell| cell.borrow_mut().take())
}

fn set_args(arguments: Map<String, Value>) {
	ARGS.with(|cell| *cell.borrow_mut() = Some(arguments));
}

fn take_args() -> Option<Map<String, Value>> {
	ARGS.with(|cell| cell.borrow_mut().take())
}

fn args_snapshot() -> Value {
	ARGS.with(|cell| cell.borrow().as_ref().map(|args| Value::Object(args.clone()))).unwrap_or(Value::Null)
}

fn runtime_aborted() -> bool {
	// The per-invocation signal is checked in the caller's closure; hooks
	// that fail while not aborted block the call.
	false
}

fn empty_reported() -> Arc<parking_lot::Mutex<Reported>> {
	Arc::new(parking_lot::Mutex::new(Reported {
		output: OutputBuffer::new(OutputLimits { max_bytes: DEFAULT_MAX_BYTES, max_lines: DEFAULT_MAX_LINES, retain: Retain::Head }),
		limits: OutputLimits { max_bytes: DEFAULT_MAX_BYTES, max_lines: DEFAULT_MAX_LINES, retain: Retain::Head },
		diagnostics: Vec::new(),
		details: None,
	}))
}

/// Recovery after intent: rerun only when the stored and the current policy
/// both say `safe`.
async fn execute_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let ToolTaskCheckpoint::Execute { arguments, replay } = tool_checkpoint(&task)? else {
		return Err("tool execute phase reached with another checkpoint".to_string());
	};
	let input: ToolTaskInput = serde_json::from_value(task.input.clone())
		.map_err(|error| format!("Tool task input is invalid: {error}"))?;
	let call = read_call(&runtime, &input, &context).await?;
	let tool = runtime.agent(&context).await?.tools.iter().find(|each| each.name() == call.name).cloned();
	if replay == Replay::Safe && tool.as_ref().and_then(|tool| tool.replay) == Some(crate::harness::api::ToolReplay::Safe) {
		// The rerun reports from scratch; clear what the interrupted attempt
		// published.
		let conversation_id = runtime.conversation_id();
		let task_id = runtime.task_id();
		runtime
			.commit(
				commit_change(move |tx, _current| {
						let draft = tx
							.doc(&crate::harness::live::live_doc(), Some(conversation_id.0), None, None)
							.map_err(|e| e.to_string())?;
						let mut live = read_live(&draft);
						if let Some(slot) = crate::harness::live::tool_slot(&mut live, &task_id) {
							clear_progress(slot);
						}
						store_live(&draft, &live)?;
						Ok(None)
				}),
				&context,
			)
			.await?;
		let tool = tool.expect("checked");
		return run(&runtime, &call, tool, arguments, &context).await;
	}
	let message = format!("Tool {} was interrupted and may have partially run", call.name);
	// `failed` records cancellation intent, so the call's owned
	// conversations, left unsupervised, are aborted.
	let ending = Ending::Failed { message: message.clone() };
	let partial = harness_error("interrupted", &message);
	settle(&runtime, &call, ending, &empty_reported(), &partial, &context).await
}

/// Abort phase.
async fn abort_phase(task: TaskRecord, runtime: Arc<dyn TaskRuntime>, context: Context) -> Result<(), String> {
	let input: ToolTaskInput = serde_json::from_value(task.input.clone())
		.map_err(|error| format!("Tool task input is invalid: {error}"))?;
	let call = read_call(&runtime, &input, &context).await?;
	let message = format!("Tool {} was aborted", call.name);
	let ending = Ending::Aborted;
	let partial = harness_error("aborted", &message);
	settle(&runtime, &call, ending, &empty_reported(), &partial, &context).await
}

/// Per-invocation [`ToolExecutionApi`] built over the task runtime (upstream
/// `api` object).
struct ApiCore {
	runtime: Arc<dyn TaskRuntime>,
	call_id: String,
	reported: Arc<parking_lot::Mutex<Reported>>,
	progress: Arc<Progress>,
	ended: Arc<std::sync::atomic::AtomicBool>,
	env: parking_lot::Mutex<Option<Arc<dyn crate::env::ExecutionEnv>>>,
	snapshot: Arc<dyn crate::harness::types::RegistrySnapshot>,
}

impl ApiCore {
	fn set_env(&self, env: Option<Arc<dyn crate::env::ExecutionEnv>>) {
		*self.env.lock() = env;
	}

	fn assert_live(&self, call_id: &str) -> Result<(), String> {
		if self.ended.load(std::sync::atomic::Ordering::SeqCst) {
			return Err(format!("Tool call {call_id} has settled"));
		}
		Ok(())
	}
}

#[async_trait::async_trait]
impl ToolExecutionApi for ApiCore {
	fn task_id(&self) -> TaskId {
		self.runtime.task_id()
	}

	fn conversation_id(&self) -> crate::harness::types::ConversationId {
		self.runtime.conversation_id()
	}

	fn call_id(&self) -> &str {
		&self.call_id
	}

	fn registry(&self) -> Arc<dyn crate::harness::types::RegistrySnapshot> {
		self.snapshot.clone()
	}

	async fn snapshot(
		&self,
		token: &crate::documents::DocToken,
		owner: Option<u64>,
		key: Option<&str>,
		context: &Context,
	) -> Result<Option<crate::harness::types::JsonObject>, String> {
		self.runtime.document_snapshot(token, owner, key, context).await
	}

	async fn snapshot_as_of(
		&self,
		token: &crate::documents::DocToken,
		conversation_id: u64,
		key: Option<&str>,
		at: crate::types::EntryId,
		context: &Context,
	) -> Result<Option<crate::harness::types::JsonObject>, String> {
		self.runtime
			.document_snapshot_as_of(
				token,
				crate::harness::types::ConversationId(conversation_id),
				key,
				at,
				context,
			)
			.await
	}

	async fn watch_doc(
		&self,
		token: &crate::documents::DocToken,
		owner: Option<u64>,
		key: Option<&str>,
		context: &Context,
	) -> Result<Option<crate::tasks::DocumentWatch>, String> {
		self.runtime.document_watch_doc(token, owner, key, context).await
	}

	async fn agent(&self, context: &Context) -> Result<crate::harness::types::Agent, String> {
		self.runtime.agent(context).await
	}

	fn env(&self) -> Option<Arc<dyn crate::env::ExecutionEnv>> {
		self.env.lock().clone()
	}

	fn output(&self, chunk: &[u8], skipped: Option<crate::env::ShellOutputSkip>) {
		if self.assert_live(&self.call_id).is_err() {
			return;
		}
		let accepted = self.reported.lock().output.push(crate::harness::output::Chunk::Bytes(chunk.to_vec()), skipped);
		if accepted {
			let progress = self.progress.clone();
			tokio::spawn(async move { progress.mark().await });
		}
	}

	fn output_window(&self) -> Option<crate::env::ShellOutputWindow> {
		let limits = self.reported.lock().limits;
		if limits.retain != Retain::Tail {
			return None;
		}
		Some(crate::env::ShellOutputWindow {
			max_bytes: limits.max_bytes,
			max_lines: limits.max_lines,
			min_interval_ms: self.runtime.settings().progress.output_interval_ms,
			bytes_per_second: PROGRESS_BYTES_PER_SECOND as f64,
		})
	}

	fn diagnostic(&self, diagnostic: ToolDiagnostic) {
		if self.assert_live(&self.call_id).is_err() {
			return;
		}
		self.reported.lock().diagnostics.push(diagnostic);
		let progress = self.progress.clone();
		tokio::spawn(async move { progress.mark().await });
	}

	async fn details(&self, value: Value, context: &Context) -> Result<(), String> {
		self.assert_live(&self.call_id)?;
		if let Some(signal) = context.abort_signal() {
			if signal.aborted() {
				return Err(abort_reason_text(&signal));
			}
		}
		self.reported.lock().details = Some(value);
		// Cancelling the wait leaves the update in place; the commit's own
		// outcome stays observed.
		match self.progress.mark_and_wait().await.await {
			Ok(Ok(())) => Ok(()),
			Ok(Err(error)) => Err(error),
			Err(_) => Ok(()),
		}
	}

	async fn commit(
		&self,
		change: &mut crate::session::CommitValueCallback<'_>,
		context: &Context,
	) -> Result<Value, String> {
		self.runtime.commit_tool(change, context).await
	}

	async fn memo(&self, name: &str, context: &Context) -> Result<Option<Value>, String> {
		self.runtime.memo(name, context).await
	}

	async fn memo_put(&self, name: &str, candidate: Value, context: &Context) -> Result<Value, String> {
		self.runtime.memo_put(name, candidate, context).await
	}

	async fn create_task(
		&self,
		task: &crate::harness::types::AnyTask,
		input: Value,
		options: TaskOptions,
		context: &Context,
	) -> Result<TaskId, String> {
		let created: Arc<parking_lot::Mutex<Option<Result<TaskId, String>>>> = Arc::default();
		let created_in = created.clone();
		let task = task.clone();
		self.runtime
			.commit(
				commit_change(move |tx, _current| {
					let task = task.clone();
					let created_in = created_in.clone();
						*created_in.lock() = Some(tx.create_task(&task, input, options).map_err(|e| e.to_string()));
						Ok::<_, String>(None)
				}),
				context,
			)
			.await?;
		let created_result = created
			.lock()
			.take()
			.ok_or_else(|| "create_task closure did not run".to_string())?;
		created_result
	}

	async fn get_task(&self, id: TaskId, context: &Context) -> Result<Option<TaskRecord>, String> {
		self.runtime.get_task(id, context).await
	}

	async fn wait_for_task(&self, id: TaskId, context: &Context) -> Result<crate::harness::types::SettledTask, String> {
		Ok(crate::harness::types::SettledTask::new(
			self.runtime.wait_for_task(id, context).await?,
		))
	}

	async fn conversation(
		&self,
		id: crate::harness::types::ConversationId,
		context: &Context,
	) -> Result<Option<Arc<dyn crate::session::ConversationHandle>>, String> {
		self.runtime.conversation(id, context).await
	}
}

