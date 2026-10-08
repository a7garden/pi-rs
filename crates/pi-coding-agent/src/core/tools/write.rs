//! Port of `packages/coding-agent/src/core/tools/write.ts` — the write tool:
//! create or overwrite a file, creating parent directories.
//!
//! Deviations: the TypeBox parameter schema becomes a `serde_json` literal and
//! tool `details` are erased JSON (D27); the write renderers are deferred with
//! the excluded interactive mode, so `renderCall`/`renderResult` are `None`
//! (D32). `path.dirname` becomes a local helper mirroring Node's `"."` and
//! `"/"` edge cases; the abort checks stay inside the mutation queue exactly
//! like upstream (checked after each step).

use std::path::Path;
use std::sync::Arc;

use pi_agent::{AgentTool, AgentToolResult, AgentToolUpdateCallback, BoxFuture};
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};

use pi_ai::types::{
	ConstrainedSamplingConfig, ConstrainedSamplingSetting, StrictSchemaMode, TextContent, UserContent,
};

use crate::core::extensions::types::{ExtensionToolContext, ToolDefinition, ToolExecute};
use crate::core::tools::file_mutation_queue::with_file_mutation_queue;
use crate::core::tools::path_utils::resolve_to_cwd;
use crate::core::tools::tool_definition_wrapper::wrap_tool_definition;
use crate::utils::abort::AbortSignal;

/// `writeSchema` — the write tool parameter schema.
pub fn write_schema() -> JsonValue {
	json!({
		"type": "object",
		"properties": {
			"path": {
				"type": "string",
				"description": "Path to the file to write (relative or absolute)"
			},
			"content": {
				"type": "string",
				"description": "Content to write to the file"
			}
		},
		"required": ["path", "content"]
	})
}

/// `writeToolSystemPromptContribution`.
pub const WRITE_TOOL_SYSTEM_PROMPT_SNIPPET: &str = "Create or overwrite files";

/// `writeToolSystemPromptContribution.guidelines`.
pub const WRITE_TOOL_SYSTEM_PROMPT_GUIDELINES: [&str; 1] = ["Use write only for new files or complete rewrites."];

/// `WriteToolInput` = `Static<typeof writeSchema>`.
#[derive(Debug, Clone, Deserialize)]
pub struct WriteToolInput {
	pub path: String,
	pub content: String,
}

/// Pluggable operations for the write tool (`WriteOperations`). Override these
/// to delegate file writing to remote systems (for example SSH).
pub trait WriteOperations: Send + Sync {
	/// Write content to a file
	fn write_file(&self, absolute_path: String, content: String) -> BoxFuture<'static, Result<(), String>>;
	/// Create directory recursively
	fn mkdir(&self, dir: String) -> BoxFuture<'static, Result<(), String>>;
}

/// The default local-filesystem operations (`defaultWriteOperations`).
pub fn local_write_operations() -> Arc<dyn WriteOperations> {
	struct LocalWriteOperations;

	impl WriteOperations for LocalWriteOperations {
		fn write_file(&self, absolute_path: String, content: String) -> BoxFuture<'static, Result<(), String>> {
			Box::pin(async move { tokio::fs::write(&absolute_path, content.as_bytes()).await.map_err(|error| error.to_string()) })
		}

		fn mkdir(&self, dir: String) -> BoxFuture<'static, Result<(), String>> {
			Box::pin(async move { tokio::fs::create_dir_all(&dir).await.map_err(|error| error.to_string()) })
		}
	}

	Arc::new(LocalWriteOperations)
}

/// Options for [`create_write_tool_definition`] (`WriteToolOptions`).
#[derive(Clone, Default)]
pub struct WriteToolOptions {
	/// Custom operations for file writing. Default: local filesystem
	pub operations: Option<Arc<dyn WriteOperations>>,
}

/// Node `path.dirname`: the directory portion of a path (`"/"` for the root,
/// `"."` when there is no directory component).
fn dirname(path: &str) -> String {
	let path = Path::new(path);
	match path.parent() {
		Some(parent) if !parent.as_os_str().is_empty() => parent.to_string_lossy().into_owned(),
		None if path.as_os_str() == "/" => "/".to_string(),
		_ => ".".to_string(),
	}
}

/// `createWriteToolDefinition`.
pub fn create_write_tool_definition(cwd: &str, options: Option<WriteToolOptions>) -> ToolDefinition {
	let ops = options.and_then(|options| options.operations).unwrap_or_else(local_write_operations);
	let cwd = cwd.to_string();

	let mut definition = ToolDefinition::new(
		"write",
		"write",
		"Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
		write_schema(),
	);
	definition.prompt_snippet = Some(WRITE_TOOL_SYSTEM_PROMPT_SNIPPET.to_string());
	definition.prompt_guidelines = Some(WRITE_TOOL_SYSTEM_PROMPT_GUIDELINES.iter().map(|g| g.to_string()).collect());
	definition.constrained_sampling = Some(ConstrainedSamplingSetting::Config(
		ConstrainedSamplingConfig::JsonSchema {
			strict: StrictSchemaMode::Prefer,
		},
	));

	let execute: ToolExecute = Arc::new(
		move |_tool_call_id: String,
		      input: JsonValue,
		      signal: Option<AbortSignal>,
		      _on_update: Option<AgentToolUpdateCallback>,
		      ctx: Option<ExtensionToolContext>|
		      -> BoxFuture<'static, Result<AgentToolResult, String>> {
			let ops = ops.clone();
			// `ctx?.cwd || cwd`
			let cwd = ctx
				.as_ref()
				.map(|ctx| ctx.cwd.clone())
				.filter(|cwd| !cwd.is_empty())
				.unwrap_or_else(|| cwd.clone());
			Box::pin(async move {
				let WriteToolInput { path, content } =
					serde_json::from_value(input).map_err(|error| error.to_string())?;
				let absolute_path = resolve_to_cwd(&path, &cwd);
				let dir = dirname(&absolute_path);
				with_file_mutation_queue(&absolute_path, || async {
					// Do not reject from an abort event listener here: that would release the
					// mutation queue while an in-flight filesystem operation may still finish.
					// Checking signal.aborted after each await observes the same aborts while
					// keeping the queue locked until the current operation has settled.
					let throw_if_aborted = || -> Result<(), String> {
						if signal.as_ref().is_some_and(|signal| signal.aborted()) {
							return Err("Operation aborted".to_string());
						}
						Ok(())
					};

					throw_if_aborted()?;

					// Create parent directories if needed.
					ops.mkdir(dir).await?;
					throw_if_aborted()?;

					// Write the file contents.
					ops.write_file(absolute_path.clone(), content).await?;
					throw_if_aborted()?;

					Ok(AgentToolResult::new(
						vec![UserContent::from(TextContent::new(format!("Successfully wrote to {path}")))],
						JsonValue::Null,
					))
				})
				.await
			})
		},
	);
	definition.execute = execute;
	definition
}

/// `createWriteTool`.
pub fn create_write_tool(cwd: &str, options: Option<WriteToolOptions>) -> AgentTool {
	wrap_tool_definition(create_write_tool_definition(cwd, options), None)
}
