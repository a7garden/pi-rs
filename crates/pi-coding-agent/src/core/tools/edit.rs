//! Port of `packages/coding-agent/src/core/tools/edit.ts` — the edit tool:
//! exact-text replacement in a single file with multi-edit support.
//!
//! Deviations: the TypeBox schemas become `serde_json` literals (D27); the
//! `EditRenderState` renderers are deferred with the excluded interactive
//! mode (D32), so `renderCall`/`renderResult` are `None`.

use std::sync::Arc;

use pi_agent::{AgentTool, AgentToolResult, AgentToolUpdateCallback, BoxFuture};
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};

use pi_ai::types::{
	ConstrainedSamplingConfig, ConstrainedSamplingSetting, StrictSchemaMode, TextContent, UserContent,
};

use crate::core::extensions::types::{ExtensionToolContext, RenderShell, ToolDefinition, ToolExecute};
use crate::core::tools::edit_diff::{
	apply_edits_to_normalized_content, detect_line_ending, generate_diff_string, generate_unified_patch,
	normalize_to_lf, restore_line_endings, Edit,
};
use crate::core::tools::file_mutation_queue::with_file_mutation_queue;
use crate::core::tools::path_utils::resolve_to_cwd;
use crate::core::tools::tool_definition_wrapper::wrap_tool_definition;
use crate::utils::abort::AbortSignal;
use crate::utils::text::split_bom;

/// `replaceEditSchema` — the single replacement object schema.
pub fn replace_edit_schema() -> JsonValue {
	json!({
		"type": "object",
		"properties": {
			"oldText": {
				"type": "string",
				"description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."
			},
			"newText": {
				"type": "string",
				"description": "Replacement text for this targeted edit."
			}
		},
		"required": ["oldText", "newText"],
		"additionalProperties": false
	})
}

/// `editSchema` — the edit tool parameter schema.
pub fn edit_schema() -> JsonValue {
	let mut schema = json!({
		"type": "object",
		"properties": {
			"path": {
				"type": "string",
				"description": "Path to the file to edit (relative or absolute)"
			},
			"edits": {
				"type": "array",
				"items": replace_edit_schema(),
				"description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead."
			}
		},
		"required": ["path", "edits"],
		"additionalProperties": false
	});
	schema["properties"]["edits"]["items"] = replace_edit_schema();
	schema
}

/// `editToolSystemPromptContribution`.
pub const EDIT_TOOL_SYSTEM_PROMPT_SNIPPET: &str =
	"Make precise file edits with exact text replacement, including multiple disjoint edits in one call";

/// `editToolSystemPromptContribution.guidelines`.
pub const EDIT_TOOL_SYSTEM_PROMPT_GUIDELINES: [&str; 4] = [
	"Use edit for precise changes (edits[].oldText must match exactly)",
	"When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
	"Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
	"Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
];

/// `EditToolInput` = `Static<typeof editSchema>`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct EditToolInput {
	#[serde(default)]
	pub path: String,
	#[serde(default)]
	pub edits: Vec<Edit>,
}

/// Tool `details` payload (`EditToolDetails`).
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EditToolDetails {
	/// Display-oriented diff of the changes made
	pub diff: String,
	/// Standard unified patch of the changes made
	pub patch: String,
	/// Line number of the first change in the new file (for editor navigation)
	#[serde(skip_serializing_if = "Option::is_none")]
	pub first_changed_line: Option<usize>,
}

/// Pluggable operations for the edit tool (`EditOperations`). Override these
/// to delegate file editing to remote systems (for example SSH).
pub trait EditOperations: Send + Sync {
	/// Read file contents as a Buffer
	fn read_file<'a>(&'a self, absolute_path: String) -> BoxFuture<'a, Result<Vec<u8>, String>>;
	/// Write content to a file
	fn write_file<'a>(&'a self, absolute_path: String, content: String) -> BoxFuture<'a, Result<(), String>>;
	/// Check if file is readable and writable (throw if not)
	fn access<'a>(&'a self, absolute_path: String) -> BoxFuture<'a, Result<(), String>>;
}

/// The default local-filesystem operations (`defaultEditOperations`).
pub fn local_edit_operations() -> Arc<dyn EditOperations> {
	struct LocalEditOperations;

	impl EditOperations for LocalEditOperations {
		fn read_file<'a>(&'a self, absolute_path: String) -> BoxFuture<'a, Result<Vec<u8>, String>> {
			Box::pin(async move { tokio::fs::read(&absolute_path).await.map_err(|error| error.to_string()) })
		}

		fn write_file<'a>(&'a self, absolute_path: String, content: String) -> BoxFuture<'a, Result<(), String>> {
			Box::pin(async move {
				tokio::fs::write(&absolute_path, content)
					.await
					.map_err(|error| error.to_string())
			})
		}

		fn access<'a>(&'a self, absolute_path: String) -> BoxFuture<'a, Result<(), String>> {
			Box::pin(async move {
				// Upstream `fsAccess(path, R_OK | W_OK)`. Errors carry the
				// upstream Node error-code rendering (`Error code: ENOENT`, …).
				let meta = tokio::fs::metadata(&absolute_path)
					.await
					.map_err(|error| access_error_message(&error))?;
				#[cfg(unix)]
				{
					use std::os::unix::fs::PermissionsExt;
					if meta.permissions().mode() & 0o600 != 0o600 {
						return Err(access_error_message(&std::io::Error::from(std::io::ErrorKind::PermissionDenied)));
					}
				}
				Ok(())
			})
		}
	}

	Arc::new(LocalEditOperations)
}

/// Options for [`create_edit_tool_definition`] (`EditToolOptions`).
#[derive(Clone, Default)]
pub struct EditToolOptions {
	/// Custom operations for file editing. Default: local filesystem
	pub operations: Option<Arc<dyn EditOperations>>,
}

/// `isSingleEditInput`.
fn is_single_edit_input(value: &JsonValue) -> bool {
	let Some(object) = value.as_object() else {
		return false;
	};
	object.get("oldText").is_some_and(JsonValue::is_string)
		&& object.get("newText").is_some_and(JsonValue::is_string)
}

/// `prepareEditArguments`: normalize model-sent argument shapes before schema
/// validation — some models (Opus 4.6, GLM-5.1) send `edits` as a JSON string
/// instead of an array; others send a single edit object instead of a
/// one-element `edits` array; legacy callers pass top-level `oldText`/`newText`.
pub fn prepare_edit_arguments(input: JsonValue) -> JsonValue {
	if !input.is_object() {
		return input;
	}

	let mut args = input;

	// Some models (Opus 4.6, GLM-5.1) send edits as a JSON string instead of an array.
	// Others send a single edit object instead of a one-element edits array.
	if let Some(edits) = args.get("edits") {
		if let Some(edits_string) = edits.as_str() {
			let parsed: Result<JsonValue, _> = serde_json::from_str(edits_string);
			if let Ok(parsed) = parsed {
				if parsed.is_array() {
					args["edits"] = parsed;
				} else if is_single_edit_input(&parsed) {
					args["edits"] = json!([parsed]);
				}
			}
		} else if is_single_edit_input(edits) {
			args["edits"] = json!([edits]);
		}
	}

	let old_text = args.get("oldText").and_then(JsonValue::as_str);
	let new_text = args.get("newText").and_then(JsonValue::as_str);
	let (Some(old_text), Some(new_text)) = (old_text, new_text) else {
		return args;
	};

	let mut edits = match args.get("edits") {
		Some(edits) if edits.is_array() => edits.clone(),
		_ => json!([]),
	};
	edits.as_array_mut().expect("array").push(json!({
		"oldText": old_text,
		"newText": new_text,
	}));
	if let Some(object) = args.as_object_mut() {
		object.remove("oldText");
		object.remove("newText");
	}
	args["edits"] = edits;
	args
}

/// `validateEditInput`.
fn validate_edit_input(input: &EditToolInput) -> Result<(String, Vec<Edit>), String> {
	if input.edits.is_empty() {
		return Err("Edit tool input is invalid. edits must contain at least one replacement.".to_string());
	}
	Ok((input.path.clone(), input.edits.clone()))
}

/// Map an `io::Error` onto the upstream Node error-code string used by the
/// `"code" in error` branch (`ENOENT`, `EACCES`, else `String(error)`).
fn access_error_message(error: &std::io::Error) -> String {
	match error.kind() {
		std::io::ErrorKind::NotFound => "Error code: ENOENT".to_string(),
		std::io::ErrorKind::PermissionDenied => "Error code: EACCES".to_string(),
		_ => error.to_string(),
	}
}

/// `createEditToolDefinition`.
pub fn create_edit_tool_definition(cwd: &str, options: Option<EditToolOptions>) -> ToolDefinition {
	let ops = options
		.and_then(|options| options.operations)
		.unwrap_or_else(local_edit_operations);
	let cwd = cwd.to_string();

	let mut definition = ToolDefinition::new(
		"edit",
		"edit",
		"Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
		edit_schema(),
	);
	definition.prompt_snippet = Some(EDIT_TOOL_SYSTEM_PROMPT_SNIPPET.to_string());
	definition.prompt_guidelines = Some(EDIT_TOOL_SYSTEM_PROMPT_GUIDELINES.iter().map(|g| g.to_string()).collect());
	definition.constrained_sampling = Some(ConstrainedSamplingSetting::Config(
		ConstrainedSamplingConfig::JsonSchema {
			strict: StrictSchemaMode::Prefer,
		},
	));
	definition.render_shell = Some(RenderShell::SelfShell);
	definition.prepare_arguments = Some(Arc::new(prepare_edit_arguments));

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
				let input: EditToolInput = serde_json::from_value(input).unwrap_or_default();
				let (path, edits) = validate_edit_input(&input)?;
				let absolute_path = resolve_to_cwd(&path, &cwd);

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

					// Check if file exists.
					if let Err(error) = ops.access(absolute_path.clone()).await {
						throw_if_aborted()?;
						// Upstream: coded errors render `Error code: ENOENT`;
						// anything else goes through `String(error)`, i.e.
						// `Error: disk offline`.
						let error_message =
							if error.starts_with("Error code: ") { error } else { format!("Error: {error}") };
						return Err(format!("Could not edit file: {path}. {error_message}."));
					}
					throw_if_aborted()?;

					// Read the file.
					let buffer = ops.read_file(absolute_path.clone()).await?;
					let raw_content = String::from_utf8_lossy(&buffer).into_owned();
					throw_if_aborted()?;

					// Strip BOM before matching. The model will not include an invisible BOM in oldText.
					let (bom, content) = split_bom(&raw_content);
					let original_ending = detect_line_ending(content);
					let normalized_content = normalize_to_lf(content);
					let applied = apply_edits_to_normalized_content(&normalized_content, &edits, &path)?;
					throw_if_aborted()?;

					let final_content = format!("{}{}", bom, restore_line_endings(&applied.new_content, original_ending));
					ops.write_file(absolute_path.clone(), final_content).await?;
					throw_if_aborted()?;

					let (diff, first_changed_line) = generate_diff_string(&applied.base_content, &applied.new_content, 4);
					let patch = generate_unified_patch(&path, &applied.base_content, &applied.new_content, 4);
					let details = serde_json::to_value(EditToolDetails {
						diff,
						patch,
						first_changed_line,
					})
					.unwrap_or(JsonValue::Null);
					let text = format!("Successfully replaced {} block(s) in {path}.", edits.len());
					Ok(AgentToolResult::new(
						vec![UserContent::from(TextContent::new(text))],
						details,
					))
				})
				.await
			})
		},
	);
	definition.execute = execute;
	definition
}

/// `createEditTool`.
pub fn create_edit_tool(cwd: &str, options: Option<EditToolOptions>) -> AgentTool {
	wrap_tool_definition(create_edit_tool_definition(cwd, options), None)
}
