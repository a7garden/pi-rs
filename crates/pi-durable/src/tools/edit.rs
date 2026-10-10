//! Port of `packages/durable/src/tools/edit.ts` — exact-text replacement editing.
//!
//! D1: upstream throws inside `execute`; the fallible body returns `Result` instead (see
//! `bash` module D1). `prepareArguments` stays a pure `Value → Value` repair step.

use std::sync::Arc;

use pi_chord::context::Context;
use serde_json::{json, Map, Value};

use crate::env::FileError;
use crate::harness::api::{define_tool, ToolExecutionApi, ToolExecutionResult, ToolRegistration};

use super::edit_diff::{
    apply_edits_to_normalized_content, detect_line_ending, generate_diff_string,
    generate_unified_patch, normalize_to_lf, restore_line_endings, strip_bom, AppliedEditsResult,
    Edit,
};
use super::env::require_env;
use super::file_mutation_queue::with_file_mutation_queue;
use super::path_utils::resolve_tool_path;

pub fn edit_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Path to the file to edit (relative or absolute)" },
            "edits": {
                "type": "array",
                "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": { "type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call." },
                        "newText": { "type": "string", "description": "Replacement text for this targeted edit." },
                    },
                    "required": ["oldText", "newText"],
                },
            },
        },
        "required": ["path", "edits"],
    })
}

/// Upstream `EditToolInput` (`Static<typeof editSchema>`); erased to validated JSON (D1).
pub type EditToolInput = Value;

fn is_single_edit_input(value: &Value) -> bool {
    value.as_object().is_some_and(|edit| {
        edit.get("oldText").is_some_and(Value::is_string)
            && edit.get("newText").is_some_and(Value::is_string)
    })
}

/// Repair shapes models commonly send: `edits` as a JSON string or as a single edit object, and
/// a top-level `oldText`/`newText` pair. Works on a copy; the call's arguments stay unchanged.
pub fn prepare_edit_arguments(input: &Value) -> Value {
    let Some(object) = input.as_object() else {
        return input.clone();
    };
    let mut args: Map<String, Value> = object.clone();
    if let Some(edits) = args.get("edits") {
        if let Some(edits_string) = edits.as_str() {
            let parsed: Result<Value, _> = serde_json::from_str(edits_string);
            if let Ok(parsed) = parsed {
                if parsed.is_array() {
                    args.insert("edits".to_string(), parsed);
                } else if is_single_edit_input(&parsed) {
                    args.insert("edits".to_string(), json!([parsed]));
                }
            }
        } else if is_single_edit_input(edits) {
            args.insert("edits".to_string(), json!([edits]));
        }
    }

    let legacy_old = args.get("oldText").and_then(Value::as_str);
    let legacy_new = args.get("newText").and_then(Value::as_str);
    let (Some(old_text), Some(new_text)) = (legacy_old, legacy_new) else {
        return Value::Object(args);
    };
    let mut edits = match args.get("edits") {
        Some(Value::Array(edits)) => edits.clone(),
        _ => Vec::new(),
    };
    edits.push(json!({ "oldText": old_text, "newText": new_text }));
    args.remove("oldText");
    args.remove("newText");
    args.insert("edits".to_string(), Value::Array(edits));
    Value::Object(args)
}

/// Upstream `EditToolDetails`: `{ diff, patch, firstChangedLine? }`, erased to JSON (D1).
pub type EditToolDetails = Value;

struct ValidatedInput {
    path: String,
    edits: Vec<Edit>,
}

fn validate_edit_input(input: &Value) -> Result<ValidatedInput, String> {
    let edits_value = input.get("edits");
    let edits = match edits_value {
        Some(Value::Array(edits)) if !edits.is_empty() => edits
            .iter()
            .map(|edit| serde_json::from_value(edit.clone()))
            .collect::<Result<Vec<Edit>, _>>()
            .map_err(|error| error.to_string())?,
        _ => {
            return Err(
                "Edit tool input is invalid. edits must contain at least one replacement."
                    .to_string(),
            )
        }
    };
    let path = input
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Ok(ValidatedInput { path, edits })
}

fn edit_access_error(path: &str, error: &FileError) -> String {
    format!("Could not edit file: {path}. Error code: {}.", error.code)
}

/// Fallible body of one `edit` call.
pub async fn execute(
    args: Value,
    api: Arc<dyn ToolExecutionApi>,
    context: Context,
) -> Result<ToolExecutionResult, String> {
    let input = validate_edit_input(&args)?;
    let path = input.path;
    let edits = input.edits;
    let env = require_env(api.as_ref())?;
    let absolute_path = resolve_tool_path(env.as_ref(), &path, &context)
        .await
        .map_err(|error| error.to_string())?;
    let mutation_env = Arc::clone(&env);
    let mutation_path = absolute_path.clone();
    let mutation_context = context.clone();
    with_file_mutation_queue(
        env,
        &absolute_path,
        async move {
            if mutation_context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
            {
                return Err("Operation aborted".to_string());
            }
            let info = mutation_env
                .file_info(&mutation_path, &mutation_context)
                .await
                .map_err(|error| edit_access_error(&path, &error))?;
            if info.kind != crate::env::FileKind::File && info.kind != crate::env::FileKind::Symlink
            {
                return Err(format!("Could not edit file: {path}. Path is not a file."));
            }

            let read_result = mutation_env
                .read_text_file(&mutation_path, &mutation_context)
                .await
                .map_err(|error| edit_access_error(&path, &error))?;
            if mutation_context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
            {
                return Err("Operation aborted".to_string());
            }

            let (bom, content) = strip_bom(&read_result);
            let original_ending = detect_line_ending(&content);
            let normalized_content = normalize_to_lf(&content);
            let AppliedEditsResult {
                base_content,
                new_content,
            } = apply_edits_to_normalized_content(&normalized_content, &edits, &path)?;
            if mutation_context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
            {
                return Err("Operation aborted".to_string());
            }

            let final_content = format!(
                "{}{}",
                bom,
                restore_line_endings(&new_content, original_ending)
            );
            mutation_env
                .write_file(&mutation_path, final_content.into(), &mutation_context)
                .await
                .map_err(|error| edit_access_error(&path, &error))?;
            if mutation_context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
            {
                return Err("Operation aborted".to_string());
            }

            let (diff, first_changed_line) = generate_diff_string(&base_content, &new_content, 4);
            let mut details = Map::new();
            details.insert("diff".to_string(), Value::String(diff));
            details.insert(
                "patch".to_string(),
                Value::String(generate_unified_patch(
                    &path,
                    &base_content,
                    &new_content,
                    4,
                )),
            );
            if let Some(first_changed_line) = first_changed_line {
                details.insert("firstChangedLine".to_string(), json!(first_changed_line));
            }
            Ok(ToolExecutionResult {
                content: Some(json!([{
                    "type": "text",
                    "text": format!("Successfully replaced {} block(s) in {path}.", edits.len()),
                }])),
                details: Some(Value::Object(details)),
                ..Default::default()
            })
        },
        &context,
    )
    .await
}

pub fn create_edit_tool() -> ToolRegistration {
    define_tool(ToolRegistration {
		tool: pi_ai::Tool {
			name: "edit".to_string(),
			description: "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.".to_string(),
			parameters: edit_parameters(),
			constrained_sampling: None,
		},
		replay: None,
		execution_mode: None,
		prepare_arguments: Some(Arc::new(prepare_edit_arguments)),
		output_limits: None,
		execute: super::wrap_execute(move |args, api, context| execute(args, api, context)),
	})
}
