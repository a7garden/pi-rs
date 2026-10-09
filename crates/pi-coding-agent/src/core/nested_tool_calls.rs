//! Port of `packages/coding-agent/src/core/nested-tool-calls.ts` — tool calls
//! that a tool makes while it runs (`ctx.executeTool()`), for example from
//! codemode scripts. The agent loop does not know about them: the session
//! runs each one through the agent's tool pipeline (`runToolCall`) with its
//! own hooks, emits `tool_execution_*` events with `parentToolCallId`, and
//! records the calls and their usage on the model-issued call's tool result
//! message.
//!
//! Nothing here runs until a tool calls [`NestedToolCallRunner::execute`].
//!
//! Deviations (crate docs D68):
//! - **D68a — `emit` is synchronous.** Upstream `NestedToolCallHost.emit`
//!   returns a promise the runner awaits; the ported extension event paths
//!   are synchronous (D57a precedent), so emission is ordered inline and the
//!   update forwarder keeps the crate's sync `AgentToolUpdateCallback`
//!   convention.
//! - **D68b — the exclusive-call queue is a FIFO `tokio::sync::Mutex`.**
//!   Upstream chains promises through a `queueTail`; the mutex gives the same
//!   FIFO serialization for calls that must not run concurrently, and
//!   `holds_queue` scopes keep nested calls of the queue holder reentrant.
//! - **D68c — `runToolCall` cannot reject.** Upstream returns a promise that
//!   may reject; the trait returns the outcome directly (tool failures are
//!   `is_error` results).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use futures_util::future::BoxFuture;
use parking_lot::Mutex;
use pi_agent::types::{
	AgentTool, AgentToolCall, AgentToolCallOutcome, AgentToolResult, AgentToolUpdateCallback,
	ToolExecutionMode,
};
use pi_ai::types::{JsonObject, NestedToolCallRecord, NestedToolCalls, Usage};
use serde::{Deserialize, Serialize};

use crate::core::usage_totals::combine_usage;
use crate::utils::abort::AbortSignal;

/// Limits of the nested-call record on a tool result: arguments over the
/// per-call or total size are omitted, calls beyond the count are dropped,
/// and the record is marked incomplete when any of that happens
/// (`NESTED_CALL_LIMITS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NestedCallLimits {
	pub max_calls: usize,
	pub max_argument_bytes_per_call: usize,
	pub max_argument_bytes_total: usize,
	pub max_error_chars: usize,
}

/// `NESTED_CALL_LIMITS`.
pub const NESTED_CALL_LIMITS: NestedCallLimits = NestedCallLimits {
	max_calls: 256,
	max_argument_bytes_per_call: 8 * 1024,
	max_argument_bytes_total: 32 * 1024,
	max_error_chars: 500,
};

/// What the nested calls of one model-issued tool call leave on its tool
/// result message (`NestedCallSummary`).
#[derive(Debug, Clone, PartialEq)]
pub struct NestedCallSummary {
	/// Becomes `nestedCalls`. `None` when no nested call was made.
	pub calls: Option<NestedToolCalls>,
	/// Summed `usage` of the nested results, added to the message's `usage`.
	pub usage: Option<Usage>,
}

/// Handle to one started record inside a [`NestedCallRecorder`] (upstream
/// hands out the record object itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NestedCallRecordHandle(usize);

/// Collects the nested calls of one model-issued tool call, including calls
/// made by nested tools. The snapshot becomes `nestedCalls` on the tool
/// result message (`NestedCallRecorder`).
#[derive(Debug, Default)]
pub struct NestedCallRecorder {
	calls: Vec<NestedToolCallRecord>,
	started_at: HashMap<usize, Instant>,
	complete: bool,
	argument_bytes: usize,
	/// Summed usage of every nested result, including calls dropped from the
	/// record.
	usage: Option<Usage>,
}

impl NestedCallRecorder {
	pub fn new() -> NestedCallRecorder {
		// Upstream starts with `complete: true`; only drops (call/argument
		// limits) mark the record incomplete.
		NestedCallRecorder { complete: true, ..NestedCallRecorder::default() }
	}

	/// Record a call as it starts. Returns `None` when the call is dropped
	/// (`start`).
	pub fn start(&mut self, tool_call: &AgentToolCall) -> Option<NestedCallRecordHandle> {
		if self.calls.len() >= NESTED_CALL_LIMITS.max_calls {
			self.complete = false;
			return None;
		}
		let mut record = NestedToolCallRecord {
			id: tool_call.id.clone(),
			name: tool_call.name.clone(),
			arguments: None,
			arguments_bytes: None,
			status: pi_ai::types::NestedToolCallStatus::Unfinished,
			duration_ms: None,
			error: None,
		};
		// `JSON.stringify(arguments ?? {})` — arguments are always an object
		// on the Rust `ToolCall`, and the empty object serializes to `{}`.
		let json = serde_json::to_string(&tool_call.arguments).expect("arguments serialize");
		let bytes = json.len();
		if bytes > NESTED_CALL_LIMITS.max_argument_bytes_per_call
			|| self.argument_bytes + bytes > NESTED_CALL_LIMITS.max_argument_bytes_total
		{
			record.arguments_bytes = Some(bytes as u64);
			self.complete = false;
		} else {
			// `JSON.parse(json)`: same value, normalized through the wire form.
			record.arguments = Some(serde_json::from_str::<JsonObject>(&json).expect("serialized arguments reparse"));
			self.argument_bytes += bytes;
		}
		let handle = NestedCallRecordHandle(self.calls.len());
		self.calls.push(record);
		self.started_at.insert(handle.0, Instant::now());
		Some(handle)
	}

	/// `finish`: settle a started record with its outcome and error text.
	pub fn finish(&mut self, record: Option<NestedCallRecordHandle>, is_error: bool, error_text: &str) {
		let Some(record) = record else { return };
		let started = self.started_at.remove(&record.0).unwrap_or_else(Instant::now);
		let Some(entry) = self.calls.get_mut(record.0) else { return };
		entry.status = if is_error {
			pi_ai::types::NestedToolCallStatus::Error
		} else {
			pi_ai::types::NestedToolCallStatus::Ok
		};
		entry.duration_ms = Some(started.elapsed().as_millis() as u64);
		if is_error && !error_text.is_empty() {
			// `.slice(0, NESTED_CALL_LIMITS.maxErrorChars)` — JS slices UTF-16
			// code units; here chars, so the cap is character-preserving.
			entry.error = Some(error_text.chars().take(NESTED_CALL_LIMITS.max_error_chars).collect());
		}
	}

	/// `addUsage`.
	pub fn add_usage(&mut self, usage: &Usage) {
		self.usage = Some(match &self.usage {
			Some(existing) => combine_usage(existing, usage),
			None => usage.clone(),
		});
	}

	/// `totalUsage`.
	pub fn total_usage(&self) -> Option<Usage> {
		self.usage.clone()
	}

	/// Copy of the record so far, or `None` when no nested call was made
	/// (`snapshot`).
	pub fn snapshot(&self) -> Option<NestedToolCalls> {
		if self.calls.is_empty() && self.complete {
			return None;
		}
		let complete = self
			.complete
			&& self.calls.iter().all(|call| call.status != pi_ai::types::NestedToolCallStatus::Unfinished);
		Some(NestedToolCalls { calls: self.calls.clone(), complete })
	}
}

/// `NestedToolCallOptions`.
#[derive(Default)]
pub struct NestedToolCallOptions {
	/// Defaults to the calling tool's signal.
	pub signal: Option<AbortSignal>,
	/// Receives partial results of the nested tool, in addition to
	/// `tool_execution_update` events.
	pub on_update: Option<AgentToolUpdateCallback>,
}

/// `tool_execution_*` events of nested calls (`NestedToolExecutionEvent`).
/// Serialize-only: `AgentToolResult` has no `Deserialize` (pi-agent), and the
/// events flow outward to extension handlers and listeners.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum NestedToolExecutionEvent {
	#[serde(rename = "tool_execution_start")]
	Start {
		#[serde(rename = "toolCallId")]
		tool_call_id: String,
		#[serde(rename = "toolName")]
		tool_name: String,
		args: serde_json::Value,
		#[serde(rename = "parentToolCallId")]
		parent_tool_call_id: String,
	},
	#[serde(rename = "tool_execution_update")]
	Update {
		#[serde(rename = "toolCallId")]
		tool_call_id: String,
		#[serde(rename = "toolName")]
		tool_name: String,
		args: serde_json::Value,
		#[serde(rename = "partialResult")]
		partial_result: AgentToolResult,
		#[serde(rename = "parentToolCallId")]
		parent_tool_call_id: String,
	},
	#[serde(rename = "tool_execution_end")]
	End {
		#[serde(rename = "toolCallId")]
		tool_call_id: String,
		#[serde(rename = "toolName")]
		tool_name: String,
		result: AgentToolResult,
		#[serde(rename = "isError")]
		is_error: bool,
		#[serde(rename = "parentToolCallId")]
		parent_tool_call_id: String,
	},
}

/// Update callback handed to [`NestedToolCallHost::run_tool_call`]: forwards
/// the partial result after the host tool has processed it. Ready-made in
/// practice (D68a), so hosts may bridge it to the sync tool callback.
pub type NestedUpdateForwarder = Arc<dyn Fn(AgentToolResult) -> BoxFuture<'static, ()> + Send + Sync>;

/// The agent-loop surface nested calls run through (`NestedToolCallHost`).
pub trait NestedToolCallHost: Send + Sync {
	/// Tools nested calls resolve against (`getTools`).
	fn get_tools(&self) -> Vec<AgentTool>;
	/// Whether every nested call runs exclusively, as when the agent executes
	/// tool calls sequentially (`isSequential`).
	fn is_sequential(&self) -> bool;
	/// Run the call through the tool pipeline, with hooks that report
	/// `parentToolCallId` (`runToolCall`).
	fn run_tool_call(
		&self,
		tool_call: AgentToolCall,
		parent_tool_call_id: &str,
		signal: Option<AbortSignal>,
		on_update: NestedUpdateForwarder,
	) -> BoxFuture<'static, AgentToolCallOutcome>;
	/// Emit a `tool_execution_*` event (`emit`; D68a).
	fn emit(&self, event: NestedToolExecutionEvent);
}

/// Calls below one model-issued call share its recorder (`CallScope`).
struct CallScope {
	recorder: Arc<Mutex<NestedCallRecorder>>,
	next_id: AtomicU32,
	/// Set inside a call that holds the exclusive queue, so its own nested
	/// calls do not wait on it.
	holds_queue: bool,
}

/// Runs nested tool calls on behalf of a host pipeline
/// (`NestedToolCallRunner`).
pub struct NestedToolCallRunner {
	host: Arc<dyn NestedToolCallHost>,
	/// Scopes by the id of the calling tool call.
	scopes: Mutex<HashMap<String, CallScope>>,
	/// Serializes nested calls that must not run concurrently (D68b).
	queue: tokio::sync::Mutex<()>,
}

/// Text of a tool result: text content blocks joined with `\n` (`textOf`).
fn text_of(result: &AgentToolResult) -> String {
	result
		.content
		.iter()
		.filter_map(|block| match block {
			pi_ai::types::UserContent::Text(text) => Some(text.text.as_str()),
			_ => None,
		})
		.collect::<Vec<_>>()
		.join("\n")
}

impl NestedToolCallRunner {
	pub fn new(host: Arc<dyn NestedToolCallHost>) -> NestedToolCallRunner {
		NestedToolCallRunner { host, scopes: Mutex::new(HashMap::new()), queue: tokio::sync::Mutex::new(()) }
	}

	/// Run `name` on behalf of the call `callerId`. The nested call gets the
	/// id `<callerId>/<n>`. Never fails for tool failures: they come back as
	/// `is_error` outcomes (`execute`).
	pub async fn execute(
		&self,
		caller_id: &str,
		name: &str,
		args: Option<JsonObject>,
		options: NestedToolCallOptions,
	) -> AgentToolCallOutcome {
		let (recorder, parent_holds_queue, tool_call, record) = {
			let mut scopes = self.scopes.lock();
			let scope = scopes.entry(caller_id.to_string()).or_insert_with(|| CallScope {
				recorder: Arc::new(Mutex::new(NestedCallRecorder::new())),
				next_id: AtomicU32::new(1),
				holds_queue: false,
			});
			let n = scope.next_id.fetch_add(1, Ordering::Relaxed);
			let tool_call = AgentToolCall {
				id: format!("{caller_id}/{n}"),
				name: name.to_string(),
				arguments: args.unwrap_or_default(),
				thought_signature: None,
				namespace: None,
			};
			let record = scope.recorder.lock().start(&tool_call);
			(Arc::clone(&scope.recorder), scope.holds_queue, tool_call, record)
		};
		self.host.emit(NestedToolExecutionEvent::Start {
			tool_call_id: tool_call.id.clone(),
			tool_name: name.to_string(),
			args: serde_json::Value::Object(tool_call.arguments.clone()),
			parent_tool_call_id: caller_id.to_string(),
		});

		let sequential_tool = self
			.host
			.get_tools()
			.iter()
			.find(|tool| tool.name() == name)
			.and_then(|tool| tool.execution_mode)
			.is_some_and(|mode| mode == ToolExecutionMode::Sequential);
		let exclusive = !parent_holds_queue && (self.host.is_sequential() || sequential_tool);
		// The guard lives until `run_tool_call` settles — upstream's
		// `release()` in `finally`.
		let queue_guard = if exclusive { Some(self.queue.lock().await) } else { None };
		{
			let mut scopes = self.scopes.lock();
			scopes.insert(
				tool_call.id.clone(),
				CallScope {
					recorder: Arc::clone(&recorder),
					next_id: AtomicU32::new(1),
					holds_queue: parent_holds_queue || exclusive,
				},
			);
		}

		let update_forwarder: NestedUpdateForwarder = {
			let host = Arc::clone(&self.host);
			let on_update = options.on_update.clone();
			let tool_call = tool_call.clone();
			let caller_id = caller_id.to_string();
			Arc::new(move |partial_result: AgentToolResult| {
				if let Some(on_update) = &on_update {
					on_update(partial_result.clone());
				}
				host.emit(NestedToolExecutionEvent::Update {
					tool_call_id: tool_call.id.clone(),
					tool_name: tool_call.name.clone(),
					args: serde_json::Value::Object(tool_call.arguments.clone()),
					partial_result,
					parent_tool_call_id: caller_id.clone(),
				});
				Box::pin(async {})
			})
		};
		let outcome = self
			.host
			.run_tool_call(tool_call.clone(), caller_id, options.signal, update_forwarder)
			.await;

		// `finally { this.scopes.delete(...); release?.(); }`
		self.scopes.lock().remove(&tool_call.id);
		drop(queue_guard);

		recorder.lock().finish(record, outcome.is_error, &text_of(&outcome.result));
		// Nested results are not persisted, so their usage is only counted
		// through the recorder.
		if let Some(usage) = &outcome.result.usage {
			recorder.lock().add_usage(usage);
		}
		self.host.emit(NestedToolExecutionEvent::End {
			tool_call_id: tool_call.id.clone(),
			tool_name: name.to_string(),
			result: outcome.result.clone(),
			is_error: outcome.is_error,
			parent_tool_call_id: caller_id.to_string(),
		});
		outcome
	}

	/// Remove and return the record of the nested calls a model-issued call
	/// made (`takeRecord`).
	pub fn take_record(&self, tool_call_id: &str) -> Option<NestedCallSummary> {
		let scope = self.scopes.lock().remove(tool_call_id)?;
		let recorder = scope.recorder.lock();
		Some(NestedCallSummary { calls: recorder.snapshot(), usage: recorder.total_usage() })
	}

	/// `clear`.
	pub fn clear(&self) {
		self.scopes.lock().clear();
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use pi_agent::types::AgentToolExecute;
	use pi_ai::types::Tool;
	use std::sync::atomic::AtomicUsize;

	fn usage(input: u64, cost: f64) -> Usage {
		Usage {
			input,
			output: 0,
			cache_read: 0,
			cache_write: 0,
			cache_write_1h: None,
			reasoning: None,
			total_tokens: input,
			cost: pi_ai::types::UsageCost {
				input: cost,
				output: 0.0,
				cache_read: 0.0,
				cache_write: 0.0,
				total: cost,
			},
		}
	}

	fn text_content(text: &str) -> pi_ai::types::TextContent {
		pi_ai::types::TextContent { text: text.to_string(), text_signature: None }
	}

	fn empty_result() -> AgentToolResult {
		AgentToolResult {
			content: vec![],
			details: serde_json::json!({}),
			structured_content: None,
			usage: None,
			is_error: None,
			terminate: None,
		}
	}

	fn text_result(text: &str) -> AgentToolResult {
		AgentToolResult { content: vec![pi_ai::types::UserContent::Text(text_content(text))], ..empty_result() }
	}

	fn simple_tool(name: &str, execute: AgentToolExecute) -> AgentTool {
		AgentTool {
			tool: Tool {
				name: name.to_string(),
				description: name.to_string(),
				parameters: serde_json::json!({ "type": "object", "properties": {} }),
				constrained_sampling: None,
			},
			label: name.to_string(),
			prepare_arguments: None,
			output_schema: None,
			execute,
			replay: None,
			execution_mode: None,
		}
	}

	struct TestHost {
		tools: Mutex<Vec<AgentTool>>,
		sequential: bool,
		events: Arc<Mutex<Vec<(String, String, String)>>>,
	}

	impl TestHost {
		fn push_tool(&self, tool: AgentTool) {
			self.tools.lock().push(tool);
		}
	}

	impl NestedToolCallHost for TestHost {
		fn get_tools(&self) -> Vec<AgentTool> {
			self.tools.lock().clone()
		}

		fn is_sequential(&self) -> bool {
			self.sequential
		}

		fn run_tool_call(
			&self,
			tool_call: AgentToolCall,
			_parent_tool_call_id: &str,
			signal: Option<AbortSignal>,
			on_update: NestedUpdateForwarder,
		) -> BoxFuture<'static, AgentToolCallOutcome> {
			let tools = self.tools.lock().clone();
			Box::pin(async move {
				let Some(tool) = tools.iter().find(|tool| tool.name() == tool_call.name) else {
					let not_found = tool_call.name.clone();
					return AgentToolCallOutcome {
						tool_call,
						result: text_result(&format!("Tool {not_found} not found")),
						is_error: true,
					};
				};
				// `(partial) => void onUpdate(partial)` — fire and forget; the
				// forwarded future is trivially ready (D68a), so poll it
				// inline.
				let forwarder = Arc::clone(&on_update);
				let tool = tool.clone();
				let result = (tool.execute)(
					tool_call.id.clone(),
					tool_call.arguments.clone(),
					crate::core::tools::tool_definition_wrapper::bridge_signal_to_ai(signal),
					Some(Arc::new(move |partial: AgentToolResult| {
						let forwarder = Arc::clone(&forwarder);
						// D68a: the forwarded future is trivially ready, so
						// poll it inline — `tokio::spawn` would race the
						// assertion on a current-thread runtime.
						drop(futures_util::FutureExt::now_or_never(forwarder(partial)));
					})),
				)
				.await;
				let is_error = result.as_ref().map(|result| result.is_error == Some(true)).unwrap_or(false);
				AgentToolCallOutcome {
						tool_call,
						result: result.unwrap_or_else(|message| {
							let mut error = text_result(&message);
							error.is_error = Some(true);
							error
						}),
						is_error,
					}
			})
		}

		fn emit(&self, event: NestedToolExecutionEvent) {
			let value = serde_json::to_value(&event).unwrap();
			self.events.lock().push((
				value["type"].as_str().unwrap().to_string(),
				value["toolCallId"].as_str().unwrap().to_string(),
				value["parentToolCallId"].as_str().unwrap().to_string(),
			));
		}
	}

	fn create_runner(
		tools: Vec<AgentTool>,
		sequential: bool,
	) -> (Arc<NestedToolCallRunner>, Arc<TestHost>, Arc<Mutex<Vec<(String, String, String)>>>) {
		let events = Arc::new(Mutex::new(Vec::new()));
		let host = Arc::new(TestHost {
			tools: Mutex::new(tools),
			sequential,
			events: Arc::clone(&events),
		});
		(Arc::new(NestedToolCallRunner::new(host.clone())), host, events)
	}

	fn echo_tool() -> AgentTool {
		simple_tool(
			"echo",
			Arc::new(|_id, _args, _signal, on_update| {
				Box::pin(async move {
					if let Some(on_update) = &on_update {
						on_update(text_result("partial"));
					}
					Ok(text_result("ok"))
				})
			}),
		)
	}

	fn call(id: &str, args: serde_json::Value) -> AgentToolCall {
		AgentToolCall {
			id: id.to_string(),
			name: "t".to_string(),
			arguments: args.as_object().cloned().unwrap_or_default(),
			thought_signature: None,
			namespace: None,
		}
	}

	// Upstream `assigns ids below the caller, emits events with the parent
	// id, and records the calls`.
	#[tokio::test]
	async fn assigns_ids_below_the_caller_and_records_the_calls() {
		let (runner, _host, events) = create_runner(vec![echo_tool()], false);
		let updates = Arc::new(Mutex::new(Vec::new()));
		let updates_sink = Arc::clone(&updates);

		let first = runner
			.execute(
				"call",
				"echo",
				Some(serde_json::json!({ "a": 1 }).as_object().unwrap().clone()),
				NestedToolCallOptions {
					on_update: Some(Arc::new(move |partial: AgentToolResult| {
						updates_sink.lock().push(partial);
					})),
					..Default::default()
				},
			)
			.await;
		let missing = runner.execute("call", "missing", None, NestedToolCallOptions::default()).await;

		assert_eq!(first.tool_call.id, "call/1");
		assert_eq!(missing.tool_call.id, "call/2");
		assert!(missing.is_error);
		assert_eq!(updates.lock().len(), 1);
		assert_eq!(
			*events.lock(),
			vec![
				("tool_execution_start".to_string(), "call/1".to_string(), "call".to_string()),
				("tool_execution_update".to_string(), "call/1".to_string(), "call".to_string()),
				("tool_execution_end".to_string(), "call/1".to_string(), "call".to_string()),
				("tool_execution_start".to_string(), "call/2".to_string(), "call".to_string()),
				("tool_execution_end".to_string(), "call/2".to_string(), "call".to_string()),
			]
		);
		let record = runner.take_record("call").unwrap().calls.unwrap();
		assert_eq!(record.calls.len(), 2);
		assert_eq!(record.calls[0].id, "call/1");
		assert_eq!(record.calls[0].name, "echo");
		assert_eq!(
			record.calls[0].arguments.as_ref().unwrap(),
			serde_json::json!({ "a": 1 }).as_object().unwrap()
		);
		assert_eq!(record.calls[0].status, pi_ai::types::NestedToolCallStatus::Ok);
		assert!(record.calls[0].duration_ms.is_some());
		assert_eq!(record.calls[1].id, "call/2");
		assert_eq!(record.calls[1].status, pi_ai::types::NestedToolCallStatus::Error);
		assert_eq!(record.calls[1].error.as_deref(), Some("Tool missing not found"));
		assert_eq!(record.calls[1].arguments.as_ref().unwrap().len(), 0);
		assert!(record.complete);
		// The record is taken once.
		assert!(runner.take_record("call").is_none());
		assert!(runner.take_record("other").is_none());
	}

	// Upstream `records calls of nested tools on the model-issued call`.
	#[tokio::test]
	async fn records_calls_of_nested_tools_on_the_model_issued_call() {
		let (runner, host, _events) = create_runner(vec![simple_tool("leaf", empty_execute())], false);
		let runner_for_tool = Arc::downgrade(&runner);
		host.push_tool(simple_tool(
			"middle",
			Arc::new(move |tool_call_id, _args, _signal, _on_update| {
				let runner = runner_for_tool.clone();
				Box::pin(async move {
					if let Some(runner) = runner.upgrade() {
						runner.execute(&tool_call_id, "leaf", None, NestedToolCallOptions::default()).await;
					}
					Ok(empty_result())
				})
			}),
		));

		runner.execute("call", "middle", None, NestedToolCallOptions::default()).await;

		let record = runner.take_record("call").unwrap().calls.unwrap();
		let ids: Vec<&str> = record.calls.iter().map(|call| call.id.as_str()).collect();
		assert_eq!(ids, vec!["call/1", "call/1/1"]);
	}

	// Upstream `sums the usage of nested results at every depth`.
	#[tokio::test]
	async fn sums_the_usage_of_nested_results_at_every_depth() {
		let leaf = simple_tool(
			"leaf",
			Arc::new(|_id, _args, _signal, _on_update| {
				Box::pin(async {
					let mut result = empty_result();
					result.usage = Some(usage(10, 0.01));
					Ok(result)
				})
			}),
		);
		let plain = simple_tool("plain", empty_execute());
		let (runner, host, _events) = create_runner(vec![leaf, plain], false);
		let runner_for_tool = Arc::downgrade(&runner);
		host.push_tool(simple_tool(
			"middle",
			Arc::new(move |tool_call_id, _args, _signal, _on_update| {
				let runner = runner_for_tool.clone();
				Box::pin(async move {
					if let Some(runner) = runner.upgrade() {
						runner.execute(&tool_call_id, "leaf", None, NestedToolCallOptions::default()).await;
					}
					// Its own usage only: the leaf's usage is counted once, by
					// the recorder.
					let mut result = empty_result();
					result.usage = Some(usage(5, 0.005));
					Ok(result)
				})
			}),
		));

		runner.execute("call", "middle", None, NestedToolCallOptions::default()).await;
		runner.execute("call", "leaf", None, NestedToolCallOptions::default()).await;
		runner.execute("call", "plain", None, NestedToolCallOptions::default()).await;
		runner.execute("free", "plain", None, NestedToolCallOptions::default()).await;

		let summary = runner.take_record("call").unwrap();
		assert_eq!(summary.usage.as_ref().unwrap().input, 25);
		assert!((summary.usage.as_ref().unwrap().cost.total - 0.025).abs() < 1e-10);
		let free = runner.take_record("free").unwrap();
		assert!(free.calls.as_ref().unwrap().complete);
		assert!(free.usage.is_none());
	}

	// Upstream `serializes concurrent calls to sequential tools`.
	#[tokio::test]
	async fn serializes_concurrent_calls_to_sequential_tools() {
		let active = Arc::new(AtomicUsize::new(0));
		let max_sequential = Arc::new(AtomicUsize::new(0));
		let max_parallel = Arc::new(AtomicUsize::new(0));

		let make_tool = |name: &'static str,
		                 execution_mode: Option<ToolExecutionMode>,
		                 active: Arc<AtomicUsize>,
		                 max: Arc<AtomicUsize>|
		 -> AgentTool {
			let mut tool = simple_tool(
				name,
				Arc::new(move |_id, _args, _signal, _on_update| {
					let active = Arc::clone(&active);
					let max = Arc::clone(&max);
					Box::pin(async move {
						let now = active.fetch_add(1, Ordering::SeqCst) + 1;
						max.fetch_max(now, Ordering::SeqCst);
						tokio::time::sleep(std::time::Duration::from_millis(5)).await;
						active.fetch_sub(1, Ordering::SeqCst);
						Ok(empty_result())
					})
				}),
			);
			tool.execution_mode = execution_mode;
			tool
		};

		let (runner, _host, _events) = create_runner(
			vec![
				make_tool(
					"sequential",
					Some(ToolExecutionMode::Sequential),
					Arc::clone(&active),
					Arc::clone(&max_sequential),
				),
				make_tool("parallel", None, active, Arc::clone(&max_parallel)),
			],
			false,
		);

		let mut sequential_calls = Vec::new();
		for _ in 0..3 {
			sequential_calls.push(runner.execute("call", "sequential", None, NestedToolCallOptions::default()));
		}
		futures_util::future::join3(sequential_calls.remove(0), sequential_calls.remove(0), sequential_calls.remove(0)).await;

		let mut parallel_calls = Vec::new();
		for _ in 0..3 {
			parallel_calls.push(runner.execute("call", "parallel", None, NestedToolCallOptions::default()));
		}
		futures_util::future::join3(parallel_calls.remove(0), parallel_calls.remove(0), parallel_calls.remove(0)).await;

		assert_eq!(max_sequential.load(Ordering::SeqCst), 1);
		assert_eq!(max_parallel.load(Ordering::SeqCst), 3);
	}

	fn empty_execute() -> AgentToolExecute {
		Arc::new(|_id, _args, _signal, _on_update| Box::pin(async { Ok(empty_result()) }))
	}

	// -- NestedCallRecorder ----------------------------------------------------

	// Upstream `omits oversized arguments and drops calls beyond the limit`.
	#[test]
	fn omits_oversized_arguments_and_drops_calls_beyond_the_limit() {
		let mut recorder = NestedCallRecorder::new();
		assert!(recorder.snapshot().is_none());
		let small = recorder.start(&call("a", serde_json::json!({ "x": 1 })));
		recorder.finish(small, false, "");
		let snapshot = recorder.snapshot().unwrap();
		assert_eq!(snapshot.calls.len(), 1);
		assert_eq!(snapshot.calls[0].id, "a");
		assert_eq!(snapshot.calls[0].name, "t");
		assert_eq!(
			snapshot.calls[0].arguments.as_ref().unwrap(),
			serde_json::json!({ "x": 1 }).as_object().unwrap()
		);
		assert_eq!(snapshot.calls[0].status, pi_ai::types::NestedToolCallStatus::Ok);
		assert!(snapshot.calls[0].duration_ms.is_some());
		assert!(snapshot.complete);

		let big = recorder.start(&call(
			"b",
			serde_json::json!({ "text": "x".repeat(NESTED_CALL_LIMITS.max_argument_bytes_per_call) }),
		));
		recorder.finish(big, true, &"e".repeat(1000));
		let snapshot = recorder.snapshot().unwrap();
		assert!(!snapshot.complete);
		assert_eq!(snapshot.calls[1].id, "b");
		assert_eq!(snapshot.calls[1].status, pi_ai::types::NestedToolCallStatus::Error);
		assert!(snapshot.calls[1].arguments.is_none());
		assert!(
			snapshot.calls[1].arguments_bytes.unwrap() as usize > NESTED_CALL_LIMITS.max_argument_bytes_per_call
		);
		assert_eq!(snapshot.calls[1].error.as_ref().unwrap().chars().count(), NESTED_CALL_LIMITS.max_error_chars);

		for i in 0..NESTED_CALL_LIMITS.max_calls {
			let handle = recorder.start(&call(&format!("c{i}"), serde_json::json!({})));
			recorder.finish(handle, false, "");
		}
		assert_eq!(recorder.snapshot().unwrap().calls.len(), NESTED_CALL_LIMITS.max_calls);
	}

	// Upstream `caps the total argument size and marks unfinished calls
	// incomplete`.
	#[test]
	fn caps_the_total_argument_size_and_marks_unfinished_calls_incomplete() {
		let mut recorder = NestedCallRecorder::new();
		let chunk = serde_json::json!({ "text": "x".repeat(7000) });
		let mut records = Vec::new();
		for i in 0..6 {
			records.push(recorder.start(&call(&format!("c{i}"), chunk.clone())));
		}
		let snapshot = recorder.snapshot().unwrap();
		// 32 KiB fits four 7000-byte argument objects.
		assert_eq!(snapshot.calls.iter().filter(|entry| entry.arguments.is_some()).count(), 4);
		assert!(snapshot.calls.iter().all(|entry| entry.status == pi_ai::types::NestedToolCallStatus::Unfinished));
		assert!(!snapshot.complete);
		assert_eq!(records.len(), 6);
	}
}
