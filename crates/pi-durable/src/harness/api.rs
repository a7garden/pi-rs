//! Tool-facing harness surface (upstream `harness/types.ts` lines 120–235 +
//! `harness/define.ts` `defineTool`).
//!
//! INTEGRATOR-OWNED seam: the `tools` slice and the `harness` slice both
//! compile against these definitions; the harness slice re-exports them but
//! does not redefine them. Generic parameters collapse to erased
//! `serde_json::Value` forms (D1, matches `types.rs` D-notes): tool
//! `parameters` are JSON Schemas (`pi_ai::Tool`), tool args are validated
//! `serde_json::Value`s, details are `Value`.

use std::future::Future;
use std::sync::Arc;

use pi_chord::context::Context;
use serde_json::Value;

use crate::types::{ConversationId, JsonObject, TaskId, TaskRecord};

/// Post-tools controls requested by a tool result.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolControl {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub add_tools: Option<Vec<String>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub terminate: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub handoff: Option<String>,
}

/// Remark about a call for the model and the UI, such as truncation or a spill path; never part
/// of the tool's data.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDiagnostic {
	pub severity: ToolDiagnosticSeverity,
	pub message: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub code: Option<String>,
}

/// Severity of a [`ToolDiagnostic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ToolDiagnosticSeverity {
	Info,
	Warn,
	Error,
}

/// Result of one tool execution (erased details).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionResult {
	/// Omitted: the retained `output()` text becomes the content.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub content: Option<Value>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub is_error: Option<bool>,
	/// Omitted: the last `details()` value becomes the details.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub details: Option<Value>,
	/// Added after those recorded through `ToolExecutionApi::diagnostic`.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub diagnostics: Option<Vec<ToolDiagnostic>>,
	/// Spend of the execution itself, such as a model call; stored on the result and in `pi.usage.tools`.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub usage: Option<pi_ai::Usage>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub control: Option<ToolControl>,
}

/// Whether the tools of one round run at once or one after another in call order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ToolExecutionMode {
	Parallel,
	Sequential,
}

/// How many queued items of one mode a boundary places: the first, or all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum QueueMode {
	All,
	OneAtATime,
}

/// Operations available to one tool invocation. Every operation rejects after the invocation ends.
///
/// Upstream extends `DocumentObserver`/`DocumentReader`; the document operations travel through
/// [`Session`](crate::session::Session) accessors here (D2 — see `session` module ledger).
#[async_trait::async_trait]
pub trait ToolExecutionApi: Send + Sync {
	fn task_id(&self) -> TaskId;
	fn conversation_id(&self) -> ConversationId;
	fn call_id(&self) -> &str;
	/// The tool task's phase snapshot.
	fn registry(&self) -> std::sync::Arc<dyn crate::harness::types::RegistrySnapshot>;
	/// `DocumentReader.snapshot`: current materialized value of one document address.
	async fn snapshot(
		&self,
		token: &crate::documents::DocToken,
		owner: Option<u64>,
		key: Option<&str>,
		context: &Context,
	) -> Result<Option<JsonObject>, String>;
	/// `DocumentReader.snapshotAsOf`: rewindable conversation read at one visible entry.
	async fn snapshot_as_of(
		&self,
		token: &crate::documents::DocToken,
		conversation_id: u64,
		key: Option<&str>,
		at: crate::types::EntryId,
		context: &Context,
	) -> Result<Option<JsonObject>, String>;
	/// `DocumentObserver.watchDoc`: non-creating watch acquisition.
	async fn watch_doc(
		&self,
		token: &crate::documents::DocToken,
		owner: Option<u64>,
		key: Option<&str>,
		context: &Context,
	) -> Result<Option<crate::tasks::DocumentWatch>, String>;
	/// The calling conversation's agent, as the tool task's phase resolved it.
	async fn agent(&self, context: &Context) -> Result<crate::harness::types::Agent, String>;
	/// Built by `HarnessOptions.env` for this call; `None` without an environment.
	fn env(&self) -> Option<Arc<dyn crate::env::ExecutionEnv>>;
	/// Append running output; it becomes the result content when the result omits `content`.
	fn output(&self, chunk: &[u8], skipped: Option<crate::env::ShellOutputSkip>);
	/// The tail this call's output keeps and the pace of its progress commits; `None` when the
	/// tool keeps the head of its output, which cannot accept skips.
	fn output_window(&self) -> Option<crate::env::ShellOutputWindow>;
	/// Record a model-visible remark about this call.
	fn diagnostic(&self, diagnostic: ToolDiagnostic);
	/// Replace running details; the last value becomes the result details when the result omits `details`.
	async fn details(&self, value: Value, context: &Context) -> Result<(), String>;
	/// Commit on the session line (upstream `api.commit(change, context)`).
	///
	/// The closure carries `+ Send` so the returned future stays `Send` (spawn-safe from
	/// tool tasks); session exposes the owned-closure `commit`/`commit_async` entries for it.
	async fn commit(
		&self,
		change: &mut crate::session::CommitValueCallback<'_>,
		context: &Context,
	) -> Result<Value, String>;
	async fn memo(&self, name: &str, context: &Context) -> Result<Option<Value>, String>;
	/// Store `candidate` unless a memo already exists; return the durable winner.
	async fn memo_put(&self, name: &str, candidate: Value, context: &Context) -> Result<Value, String>;
	async fn create_task(
		&self,
		task: &crate::harness::types::AnyTask,
		input: Value,
		options: crate::types::TaskOptions,
		context: &Context,
	) -> Result<TaskId, String>;
	async fn get_task(&self, id: TaskId, context: &Context) -> Result<Option<TaskRecord>, String>;
	async fn wait_for_task(
		&self,
		id: TaskId,
		context: &Context,
	) -> Result<crate::harness::types::SettledTask, String>;
	/// Invocation-bound handle of an existing conversation, such as one this tool created in `commit()`.
	async fn conversation(
		&self,
		id: ConversationId,
		context: &Context,
	) -> Result<Option<std::sync::Arc<dyn crate::session::ConversationHandle>>, String>;
}

/// Executable tool registered in a registry (erased `ToolRegistration`).
///
/// Only pi-ai `Tool` fields enter the transcript; args are validated against `parameters` before
/// `execute` runs.
#[derive(Clone)]
pub struct ToolRegistration {
	/// pi-ai transcript fields: name, label, description, parameters (JSON Schema).
	pub tool: pi_ai::Tool,
	/// Whether an interrupted execution may rerun on recovery. Default `unsafe`.
	pub replay: Option<ToolReplay>,
	/// Default: the settings' `toolExecution`. One sequential call makes its whole round sequential.
	pub execution_mode: Option<ToolExecutionMode>,
	/// Repair arguments models commonly get wrong before validation. Must be pure.
	pub prepare_arguments: Option<Arc<dyn Fn(&Value) -> Value + Send + Sync>>,
	pub output_limits: Option<ToolOutputLimits>,
	pub execute: ToolExecute,
}

/// Replay safety of an interrupted tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ToolReplay {
	Safe,
	Unsafe,
}

/// Output bounds of one tool execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolOutputLimits {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub max_bytes: Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub max_lines: Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub retain: Option<OutputRetain>,
}

/// Which end of the output a tool keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OutputRetain {
	Head,
	Tail,
}

/// Async boxed tool execution.
pub type ToolExecute = Arc<
	dyn Fn(Value, Arc<dyn ToolExecutionApi>, Context) -> std::pin::Pin<Box<dyn Future<Output = ToolExecutionResult> + Send>>
		+ Send
		+ Sync,
>;

impl std::fmt::Debug for ToolRegistration {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ToolRegistration").field("name", &self.tool.name).finish_non_exhaustive()
	}
}

/// Identity function that types a tool: `args` from `parameters`, details from what it reports
/// (upstream `defineTool`; a no-op in Rust, kept for surface parity).
pub fn define_tool(tool: ToolRegistration) -> ToolRegistration {
	tool
}
