//! Port of `packages/durable/src/tools/write.ts` — write a file through the environment.

use std::sync::Arc;

use pi_chord::context::Context;
use serde_json::{json, Value};

use crate::harness::api::{
	define_tool, ToolExecutionApi, ToolExecutionResult, ToolRegistration,
};

use super::env::require_env;
use super::file_mutation_queue::with_file_mutation_queue;
use super::path_utils::resolve_tool_path;

pub fn write_parameters() -> Value {
	json!({
		"type": "object",
		"properties": {
			"path": { "type": "string", "description": "Path to the file to write (relative or absolute)" },
			"content": { "type": "string", "description": "Content to write to the file" },
		},
		"required": ["path", "content"],
	})
}

/// Upstream `WriteToolInput` (`Static<typeof writeSchema>`); erased to validated JSON (D1).
pub type WriteToolInput = Value;

/// Fallible body of one `write` call (see `bash` module D1 for the throw→error-result mapping).
pub async fn execute(
	args: Value,
	api: Arc<dyn ToolExecutionApi>,
	context: Context,
) -> Result<ToolExecutionResult, String> {
	let path = args.get("path").and_then(Value::as_str).unwrap_or_default().to_string();
	let content = args.get("content").and_then(Value::as_str).unwrap_or_default().to_string();
	let env = require_env(api.as_ref())?;
	let absolute_path = resolve_tool_path(env.as_ref(), &path, &context).await.map_err(|error| error.to_string())?;
	let mutation_env = Arc::clone(&env);
	let mutation_path = absolute_path.clone();
	let mutation_context = context.clone();
	with_file_mutation_queue(env, &absolute_path, async move {
		if mutation_context.abort_signal().is_some_and(|signal| signal.aborted()) {
			return Err("Operation aborted".to_string());
		}
		mutation_env
			.write_file(&mutation_path, content.into(), &mutation_context)
			.await
			.map_err(|error| error.to_string())?;
		if mutation_context.abort_signal().is_some_and(|signal| signal.aborted()) {
			return Err("Operation aborted".to_string());
		}
		Ok(ToolExecutionResult {
			content: Some(json!([{ "type": "text", "text": format!("Successfully wrote to {path}") }])),
			..Default::default()
		})
	}, &context)
	.await
}

pub fn create_write_tool() -> ToolRegistration {
	define_tool(ToolRegistration {
		tool: pi_ai::Tool {
			name: "write".to_string(),
			description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.".to_string(),
			parameters: write_parameters(),
			constrained_sampling: None,
		},
		replay: None,
		execution_mode: None,
		prepare_arguments: None,
		output_limits: None,
		execute: super::wrap_execute(move |args, api, context| execute(args, api, context)),
	})
}
