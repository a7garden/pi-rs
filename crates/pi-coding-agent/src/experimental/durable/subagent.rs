//! Port of `experimental/durable/subagent.ts` — a foreground subagent: each
//! call creates a child conversation owned by the call's task, runs the task
//! there, and returns the child's answer.
//!
//! Deviations:
//! - **D172 — typebox parameter schemas become JSON-Schema `serde_json`
//!   values** on [`pi_ai::Tool`] (crate-wide erased-tool deviation); the
//!   upstream `Type.Object`/`Type.String` shapes are carried over literally.
//!   Tool failures take the crate's erased error-result convention
//!   (`is_error` + a `tool_error` diagnostic), like the pi-durable coding
//!   tools.
//! - Entry kinds read the erased `EntryRecord.kind` strings
//!   (`pi.assistant`, `pi.user`).
//! - The extension constant becomes [`subagent_extension()`]; identity
//!   comparisons (the self-removal in the child configure) match by
//!   extension name.

use std::sync::Arc;

use pi_chord::context::Context;
use pi_durable::harness::api::{
	define_tool, ToolDiagnostic, ToolDiagnosticSeverity, ToolExecute, ToolExecutionApi, ToolExecutionResult,
	ToolRegistration, ToolReplay,
};
use pi_durable::harness::types::{
	AgentChange, ConversationOwnership, EntryRecord, ExtensionsChange, SetOrClear, TaskId,
};
use pi_durable::session::Tx;
use pi_durable::types::{ConversationId, ConversationQuery};
use serde_json::{json, Value};

/// The upstream `Subagent` extension name; also the self-removal target.
pub const SUBAGENT_EXTENSION_NAME: &str = "subagent";

/// The upstream `Subagent` extension. Identity comparisons match by name, so
/// the child configure removes it with a name-only stub.
pub fn subagent_extension() -> pi_durable::harness::types::Extension {
	pi_durable::harness::types::Extension::new(SUBAGENT_EXTENSION_NAME)
		.with_tools(vec![Arc::new(subagent_tool())])
}

/// The assistant entry's concatenated text blocks (upstream `answerText`).
async fn answer_text(
	api: &Arc<dyn ToolExecutionApi>,
	answer: Option<pi_durable::types::EntryId>,
	context: Context,
) -> Result<String, String> {
	let Some(answer) = answer else {
		return Ok(String::new());
	};
	let value = api
		.commit(
			&mut Box::new(move |tx: &mut Tx<'_>| {
				let text = tx
					.entry(answer)
					.map_err(|error| error.to_string())?
					.as_ref()
					.map(assistant_text)
					.unwrap_or_default();
				Ok(Value::String(text))
			}),
			&context,
		)
		.await?;
	Ok(value.as_str().map(str::to_string).unwrap_or_default())
}

/// The first assistant message's text content of an entry (upstream
/// `entry?.model?.[0]` text projection; shared with the vacation research
/// task's report reader).
pub(crate) fn assistant_text(entry: &EntryRecord) -> String {
	let Some(pi_ai::Message::Assistant(message)) = entry.model.as_ref().and_then(|model| model.first()) else {
		return String::new();
	};
	message
		.content
		.iter()
		.filter_map(|content| match content {
			pi_ai::types::AssistantContent::Text(text) => Some(text.text.as_str()),
			_ => None,
		})
		.collect()
}

fn subagent_tool() -> ToolRegistration {
	let execute: ToolExecute = Arc::new(|args: Value, api: Arc<dyn ToolExecutionApi>, context: Context| {
		Box::pin(async move {
			match run_subagent(args, Arc::clone(&api), context).await {
				Ok(result) => result,
				Err(message) => ToolExecutionResult {
					is_error: Some(true),
					diagnostics: Some(vec![ToolDiagnostic {
						severity: ToolDiagnosticSeverity::Error,
						code: Some("tool_error".to_string()),
						message,
					}]),
					..Default::default()
				},
			}
		})
	});
	define_tool(ToolRegistration {
		tool: pi_ai::Tool {
			name: "subagent".into(),
			description: "Delegate a self-contained task to a subagent with the same tools and get its answer back. \
				Give it everything it needs to know; it does not see this conversation."
				.into(),
			parameters: json!({
				"type": "object",
				"properties": {
					"task": { "type": "string", "description": "What the subagent should do" },
				},
				"required": ["task"],
			}),
			constrained_sampling: None,
		},
		// A rerun after a crash finds the child it created and the submission it made.
		replay: Some(ToolReplay::Safe),
		execution_mode: None,
		prepare_arguments: None,
		output_limits: None,
		execute,
	})
}

async fn run_subagent(
	args: Value,
	api: Arc<dyn ToolExecutionApi>,
	context: Context,
) -> Result<ToolExecutionResult, String> {
	let task = args.get("task").and_then(Value::as_str).unwrap_or_default().to_string();
	let task_id = api.task_id();
	let child = create_child(&api, task_id, &context).await?;
	api.details(json!({ "conversationId": child }), &context).await?;
	let handle = api
		.conversation(child, &context)
		.await?
		.ok_or_else(|| format!("Subagent conversation {child} does not exist"))?;
	let request = pi_durable::harness::types::InputSubmissionDraft {
		request_id: Some(format!("subagent:{task_id}")),
		content: pi_durable::harness::types::UserInput::Text(task),
		when_busy: None,
	};
	let settled = handle.submit(request, &context).await?.wait(&context).await?;
	let done = settled.status == pi_durable::types::SubmissionStatus::Done
		&& settled.submission_type == pi_durable::types::SubmissionType::Input;
	if !done {
		return Err(format!("Subagent {child} failed: {}", submission_status_text(settled.status)));
	}
	let text = answer_text(&api, settled.answer, context).await?;
	Ok(ToolExecutionResult {
		content: Some(json!([{ "type": "text", "text": text }])),
		details: Some(json!({ "conversationId": child })),
		..Default::default()
	})
}

/// Find or create the call's child conversation in one commit (upstream the
/// `api.commit` closure). A rerun after a crash finds the existing child.
async fn create_child(
	api: &Arc<dyn ToolExecutionApi>,
	task_id: TaskId,
	context: &Context,
) -> Result<ConversationId, String> {
	let self_name = SUBAGENT_EXTENSION_NAME.to_string();
	let value = api
		.commit(
			&mut Box::new(move |tx: &mut Tx<'_>| {
				let existing = tx
					.scan_conversations(
						&ConversationQuery { owner_task_id: Some(task_id), ..Default::default() },
						1,
						None,
					)
					.map_err(|error| error.to_string())?;
				if let Some(existing) = existing.items.into_iter().next() {
					return Ok(serde_json::to_value(existing.id).unwrap_or(Value::Null));
				}
				// Starts as a copy of this conversation's agent; without this extension
				// it cannot delegate further.
				let created = tx
					.create_conversation(ConversationOwnership::Task { task_id })
					.map_err(|error| error.to_string())?;
				pi_durable::harness::agent::configure(
					tx,
					created.id,
					&AgentChange {
						extensions: Some(SetOrClear::Set(ExtensionsChange::Edit {
							add: Vec::new(),
							remove: vec![pi_durable::harness::types::Extension::new(self_name.clone())],
						})),
						..Default::default()
					},
				)?;
				Ok(serde_json::to_value(created.id).unwrap_or(Value::Null))
			}),
			context,
		)
		.await?;
	serde_json::from_value(value).map_err(|error| format!("Subagent child conversation id was invalid: {error}"))
}

fn submission_status_text(status: pi_durable::types::SubmissionStatus) -> &'static str {
	match status {
		pi_durable::types::SubmissionStatus::Queued => "queued",
		pi_durable::types::SubmissionStatus::Placed => "placed",
		pi_durable::types::SubmissionStatus::Done => "done",
		pi_durable::types::SubmissionStatus::Unanswered => "unanswered",
	}
}
