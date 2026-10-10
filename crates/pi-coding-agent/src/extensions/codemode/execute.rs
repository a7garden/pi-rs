//! Port of `packages/coding-agent/src/extensions/codemode/execute.ts` — runs
//! one codemode script: the sandbox assembly, the nested-call pipeline
//! through `ctx.executeTool`, the `searchTools`/`describeTool`/
//! `describeNamespace` and `models.*` globals, `store()` persistence over
//! session entries, and the output budget (truncation, spill file, image
//! saving).
//!
//! Upstream splits this from `tool.ts` and loads it through
//! `execute.lazy.ts` so the QuickJS sandbox only loads on the first script.
//! In Rust the executor is a plain module — no dynamic import is needed
//! (deviation D91).
//!
//! The sandbox itself comes from [`pi_codemode::CodemodeSandbox`] with its
//! worker/wasm trait seams ([`pi_codemode::runtime::host::CodeModeWorkerFactory`],
//! [`pi_codemode::wasm::WasmLoader`]); the real QuickJS worker is implemented
//! by pi-runtime and injected through [`CodemodeToolOptions`] (deviation
//! D91). Without an injected worker the execution reports the upstream
//! missing-worker sandbox-error path.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use pi_agent::{AgentTool, AgentToolCallOutcome, AgentToolResult, AgentToolUpdateCallback};
use pi_ai::auth::types::AuthOperationOptions;
use pi_ai::models::{ModelsClassifierOptions, ModelsImagesOptions};
use pi_ai::types::{
    AnyModel, ClassifierContext, ClassifierResult, ClassifierStopReason, ImageContent,
    ImagesContext, ImagesStopReason, ModelType, TextContent, Usage, UserContent,
};
use pi_ai::utils::abort::AbortSignal as AiAbortSignal;
use pi_ai::utils::thrown::ThrownValue;
use pi_codemode::declarations::SampleTool;
use pi_codemode::runtime::host::CodemodeSandboxOptions;
use pi_codemode::{
    parse_codemode_source, render_tool_sample, to_codemode_identifier, CodemodeError,
    CodemodeErrorKind, CodemodeExecuteOptions, CodemodeOutputItem, CodemodeResult, CodemodeSandbox,
    CodemodeTool as SandboxTool, CodemodeToolContext,
};
use serde_json::{json, Map, Value};

use crate::core::extensions::types::{ExecuteToolOptions, ExtensionToolContext, ToolNamespace};
use crate::core::model_registry::ModelRegistry;
use crate::core::session_manager::SessionEntry;
use crate::core::tools::truncate::format_size;
use crate::core::usage_totals::combine_usage;
use crate::extensions::codemode::tool::{
    codemode_docs_path, get_codemode_callable_tools, to_codemode_declaration, CodemodeNestedCall,
    CodemodeNestedCallStatus, CodemodeStoreEntryData, CodemodeToolDetails, CodemodeToolInput,
    CodemodeToolOptions, CODEMODE_STORE_ENTRY_TYPE,
};
use crate::extensions::tool_search::tool::{
    create_tool_search_document, Bm25Ranker, ToolSearchDocument, DEFAULT_TOOL_SEARCH_LIMIT,
};
use crate::utils::abort::AbortSignal;
use crate::utils::output_files::write_output_file;

const ARGS_PREVIEW_CHARS: usize = 200;
const ERROR_PREVIEW_CHARS: usize = 500;
/// `models.classify()` and `models.generateImages()` calls one script may
/// have in flight; `Promise.all` over many items queues the rest.
const MAX_CONCURRENT_MODEL_CALLS: usize = 4;
/// Heap limit for the QuickJS VM. The worker shares pi's process, so without
/// a limit a runaway script can grow to wasm32's 4 GiB and take the session
/// down. Overruns throw `InternalError: out of memory` inside the script.
const CODEMODE_MEMORY_LIMIT_BYTES: u64 = 256 * 1024 * 1024;
/// Default token budget for script output.
const DEFAULT_MAX_OUTPUT_TOKENS: usize = 10_000;
/// Characters per token when estimating.
const CHARS_PER_TOKEN: usize = 4;

/// File extensions of the image types `image()` accepts. Must list every
/// type the sandbox's `image()` detects.
const IMAGE_EXTENSIONS: &[(&str, &str)] = &[
    ("image/png", ".png"),
    ("image/jpeg", ".jpg"),
    ("image/gif", ".gif"),
    ("image/webp", ".webp"),
];

const CLASSIFIER_CONTEXT_SHAPE: &str = "{ state: { ... }, images?: [{ type: \"image\", data: <base64>, mimeType }], questions: { <id>: { type: \"choice\", instructions, criteria: { <label>: <meaning> } } | { type: \"score\", instructions, criteria: [<lowest level>, ..., <highest level>] } | { type: \"bool\", instructions, criteria: { true: <meaning>, false: <meaning> } } } }";

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() > max_chars {
        let head: String = text.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{head}...")
    } else {
        text.to_string()
    }
}

fn preview_args(args: Option<&Value>) -> String {
    match args {
        None => String::new(),
        Some(value) => serde_json::to_string(value)
            .map(|json| truncate_text(&json, ARGS_PREVIEW_CHARS))
            .unwrap_or_default(),
    }
}

/// The text content of a tool result, joined with newlines (`textOf`).
fn text_of(result: &AgentToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|block| match block {
            UserContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<&str>>()
        .join("\n")
}

pub(crate) fn text_item(text: impl Into<String>) -> UserContent {
    UserContent::Text(TextContent::new(text.into()))
}

/// `toModelType`: the script's model-type string.
fn to_model_type(value: &Value) -> Result<ModelType, String> {
    match value.as_str() {
        Some("chat") => Ok(ModelType::Chat),
        Some("image") => Ok(ModelType::Image),
        Some("classifier") => Ok(ModelType::Classifier),
        _ => Err(format!(
            "Unknown model type {}. Use \"chat\", \"image\", or \"classifier\".",
            value
        )),
    }
}

/// Model-type name of a [`ModelType`] (`"chat"`, `"image"`, `"classifier"`).
fn model_type_name(type_: ModelType) -> &'static str {
    match type_ {
        ModelType::Chat => "chat",
        ModelType::Image => "image",
        ModelType::Classifier => "classifier",
    }
}

/// `toProvider`: absent/null → `None`, string → `Some`, anything else errors.
fn to_provider(value: &Value) -> Result<Option<String>, String> {
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_str()
        .map(|provider| Some(provider.to_string()))
        .ok_or_else(|| "provider must be a string".to_string())
}

/// Catalog entry for scripts. `headers` is dropped because models.json
/// headers can carry credentials (`toModelInfo`).
fn to_model_info(model: &AnyModel) -> Value {
    let mut info = model.to_json();
    if let Some(object) = info.as_object_mut() {
        object.remove("headers");
    }
    info
}

/// `an image`, `a classifier` (`withArticle`).
fn with_article(word: &str) -> String {
    let article = if word.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    };
    format!("{article} {word}")
}

/// A script value in an error message: `null`, `a string`, `an array`, or its
/// keys (`{ prompt }`) (`describeValue`). Undefined script values arrive as
/// JSON `null` across the sandbox, exactly like upstream's spread-argument
/// note.
fn describe_value(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Array(items) if items.is_empty() => "an empty array".to_string(),
        Value::Array(_) => "an array".to_string(),
        Value::Object(fields) => {
            if fields.is_empty() {
                return "{}".to_string();
            }
            let mut joined = fields
                .keys()
                .take(6)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            if fields.len() > 6 {
                joined.push_str(", ...");
            }
            format!("{{ {joined} }}")
        }
        Value::String(_) => "a string".to_string(),
        Value::Bool(_) => "a boolean".to_string(),
        Value::Number(_) => "a number".to_string(),
    }
}

/// Question criteria of [`check_classifier_context`], as the upstream
/// strings-array or label-map shape.
fn is_string_values(values: Vec<&Value>) -> bool {
    !values.is_empty() && values.iter().all(|value| value.is_string())
}

/// Check a script's classifier context, so mistakes fail with the expected
/// shape instead of a provider error (`checkClassifierContext`).
fn check_classifier_context(context: &Value) -> Result<Value, String> {
    let fail = |problem: String| -> String {
        format!(
			"models.classify() {problem}. Expected context: {CLASSIFIER_CONTEXT_SHAPE}. See \"Classify\" in {}.",
			codemode_docs_path()
		)
    };
    let Some(fields) = context.as_object() else {
        return Err(fail(format!(
            "expects a context object as its second argument, got {}",
            describe_value(context)
        )));
    };
    let state = fields.get("state").unwrap_or(&Value::Null);
    if !state.is_object() {
        return Err(fail(format!(
            "context.state must be an object, got {}",
            describe_value(state)
        )));
    }
    if let Some(images) = fields.get("images") {
        let Some(image_list) = images.as_array() else {
            return Err(fail(format!(
                "context.images must be an array, got {}",
                describe_value(images)
            )));
        };
        for (index, image) in image_list.iter().enumerate() {
            let valid = image
                .as_object()
                .map(|fields| {
                    fields.get("type").and_then(Value::as_str) == Some("image")
                        && fields.get("data").map(Value::is_string).unwrap_or(false)
                        && fields
                            .get("mimeType")
                            .map(Value::is_string)
                            .unwrap_or(false)
                })
                .unwrap_or(false);
            if !valid {
                return Err(fail(format!(
                    "context.images[{index}] must be an image block, got {}",
                    describe_value(image)
                )));
            }
        }
    }
    let questions = fields.get("questions").unwrap_or(&Value::Null);
    let Some(question_map) = questions.as_object() else {
        return Err(fail(format!(
            "context.questions must map question IDs to questions, got {}",
            describe_value(questions)
        )));
    };
    if question_map.is_empty() {
        return Err(fail(format!(
            "context.questions must map question IDs to questions, got {}",
            describe_value(questions)
        )));
    }
    for (id, question) in question_map {
        let at = format!("context.questions.{id}");
        let Some(question_fields) = question.as_object() else {
            return Err(fail(format!(
                "{at} must be a question object, got {}",
                describe_value(question)
            )));
        };
        if !question_fields
            .get("instructions")
            .map(Value::is_string)
            .unwrap_or(false)
        {
            return Err(fail(format!("{at}.instructions must be a string")));
        }
        let criteria = question_fields.get("criteria").unwrap_or(&Value::Null);
        match question_fields.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let valid = criteria
                    .as_object()
                    .map(|criteria| is_string_values(criteria.values().collect()))
                    .unwrap_or(false);
                if !valid {
                    return Err(fail(format!(
						"{at} is a \"choice\" question, so criteria must map each label to its meaning"
					)));
                }
            }
            Some("score") => {
                let valid = criteria
                    .as_array()
                    .map(|criteria| is_string_values(criteria.iter().collect()))
                    .unwrap_or(false);
                if !valid {
                    return Err(fail(format!(
						"{at} is a \"score\" question, so criteria must list the levels as strings, lowest first"
					)));
                }
            }
            Some("bool") => {
                let valid = criteria
                    .as_object()
                    .map(|criteria| {
                        criteria.get("true").map(Value::is_string).unwrap_or(false)
                            && criteria.get("false").map(Value::is_string).unwrap_or(false)
                    })
                    .unwrap_or(false);
                if !valid {
                    return Err(fail(format!(
						"{at} is a \"bool\" question, so criteria must be {{ true: string, false: string }}"
					)));
                }
            }
            other => {
                let shown = other.map(Value::from).unwrap_or(Value::Null);
                return Err(fail(format!(
                    "{at}.type must be \"choice\", \"score\", or \"bool\", got {}",
                    serde_json::to_string(&shown).unwrap_or_else(|_| "null".to_string())
                )));
            }
        }
    }
    Ok(context.clone())
}

/// Check a script's image context, so mistakes such as `{ prompt }` fail
/// with the expected shape (`checkImagesContext`).
fn check_images_context(context: &Value) -> Result<Value, String> {
    let fail = |problem: String| -> String {
        format!(
			"models.generateImages() {problem}. Expected context: {{ input: [{{ type: \"text\", text: <prompt> }}, ...optional {{ type: \"image\", data: <base64>, mimeType }} references] }}. See \"Generate images\" in {}.",
			codemode_docs_path()
		)
    };
    let Some(fields) = context.as_object() else {
        return Err(fail(format!(
            "expects a context object as its second argument, got {}",
            describe_value(context)
        )));
    };
    let input = fields.get("input").unwrap_or(&Value::Null);
    let Some(input) = input.as_array() else {
        return Err(fail(format!(
            "context.input must be a non-empty array of blocks, got {}",
            describe_value(input)
        )));
    };
    if input.is_empty() {
        return Err(fail(
            "context.input must be a non-empty array of blocks, got an empty array".to_string(),
        ));
    }
    for (index, block) in input.iter().enumerate() {
        let valid = block
            .as_object()
            .map(|fields| match fields.get("type").and_then(Value::as_str) {
                Some("text") => fields.get("text").map(Value::is_string).unwrap_or(false),
                Some("image") => {
                    fields.get("data").map(Value::is_string).unwrap_or(false)
                        && fields
                            .get("mimeType")
                            .map(Value::is_string)
                            .unwrap_or(false)
                }
                _ => false,
            })
            .unwrap_or(false);
        if !valid {
            return Err(fail(format!(
                "context.input[{index}] must be a text or image block, got {}",
                describe_value(block)
            )));
        }
    }
    Ok(context.clone())
}

/// The fields of `ClassifierResult` and `AssistantImages` that a nested call
/// row reports (upstream `ModelCallResult`).
struct ModelCallView {
    stop: bool,
    aborted: bool,
    error_message: Option<String>,
    usage: Option<Usage>,
}

impl ModelCallView {
    fn from_classifier(result: &ClassifierResult) -> ModelCallView {
        ModelCallView {
            stop: result.stop_reason == ClassifierStopReason::Stop,
            aborted: result.stop_reason == ClassifierStopReason::Aborted,
            error_message: result.error_message.clone(),
            usage: result.usage.clone(),
        }
    }

    fn from_images(result: &pi_ai::types::AssistantImages) -> ModelCallView {
        ModelCallView {
            stop: result.stop_reason == ImagesStopReason::Stop,
            aborted: result.stop_reason == ImagesStopReason::Aborted,
            error_message: result.error_message.clone(),
            usage: result.usage.clone(),
        }
    }
}

fn is_store_entry_data(data: &Value) -> bool {
    let Some(fields) = data.as_object() else {
        return false;
    };
    let valid_set = fields.get("set").map(Value::is_object).unwrap_or(false);
    let valid_delete = fields
        .get("delete")
        .and_then(Value::as_array)
        .map(|keys| keys.iter().all(Value::is_string))
        .unwrap_or(false);
    valid_set && valid_delete
}

/// Values of `load()`: the `codemode-store` entries on the branch, applied
/// from the root (`readCodemodeStore`).
pub fn read_codemode_store(branch: &[SessionEntry]) -> Map<String, Value> {
    let mut store = Map::new();
    for entry in branch {
        let SessionEntry::Custom(custom) = entry else {
            continue;
        };
        if custom.custom_type != CODEMODE_STORE_ENTRY_TYPE {
            continue;
        }
        let Some(data) = &custom.data else { continue };
        if !is_store_entry_data(data) {
            continue;
        }
        if let Some(keys) = data.get("delete").and_then(Value::as_array) {
            for key in keys.iter().filter_map(Value::as_str) {
                store.remove(key);
            }
        }
        if let Some(set) = data.get("set").and_then(Value::as_object) {
            for (key, value) in set {
                store.insert(key.clone(), value.clone());
            }
        }
    }
    store
}

/// Like the script's `text()`: strings as is, other values as compact JSON
/// (`valueText`).
fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
    }
}

/// Lay out the script's output so the model can tell items apart: providers
/// join adjacent text blocks with a newline or with nothing. With more than
/// one text item (`text()` or the returned value), each starts with a
/// `==> text N/M <==` line. `console.*` lines follow all other output in one
/// `<console_output>` block (`formatOutput`).
fn format_output(output: &[CodemodeOutputItem]) -> Vec<UserContent> {
    let total = output
        .iter()
        .filter(|item| matches!(item, CodemodeOutputItem::Text { console: None, .. }))
        .count();
    let mut items: Vec<UserContent> = Vec::new();
    let mut console_lines: Vec<String> = Vec::new();
    let mut index = 0usize;
    for item in output {
        match item {
            CodemodeOutputItem::Image { data, mime_type } => {
                items.push(UserContent::Image(ImageContent::new(
                    data.clone(),
                    mime_type.clone(),
                )));
            }
            CodemodeOutputItem::Text { text, console } => {
                if console.is_some() {
                    console_lines.push(text.clone());
                } else {
                    index += 1;
                    let text = if total > 1 {
                        format!("==> text {index}/{total} <==\n{text}")
                    } else {
                        text.clone()
                    };
                    items.push(text_item(text));
                }
            }
        }
    }
    if !console_lines.is_empty() {
        items.push(text_item(format!(
            "<console_output>\n{}\n</console_output>",
            console_lines.join("\n")
        )));
    }
    items
}

/// Join adjacent text items into one, each part starting on its own line
/// (`joinAdjacentText`).
fn join_adjacent_text(items: Vec<UserContent>) -> Vec<UserContent> {
    let mut joined: Vec<UserContent> = Vec::new();
    for item in items {
        let next_text = match &item {
            UserContent::Text(next) => Some(&next.text),
            _ => None,
        };
        let appended = match (joined.last_mut(), next_text) {
            (Some(UserContent::Text(last)), Some(next)) => {
                let separator = if last.text.is_empty() || last.text.ends_with('\n') {
                    ""
                } else {
                    "\n"
                };
                last.text = format!("{}{separator}{}", last.text, next);
                true
            }
            _ => false,
        };
        if !appended {
            joined.push(item);
        }
    }
    joined
}

fn format_call_summary(calls: &[CodemodeNestedCall]) -> String {
    if calls.is_empty() {
        return "No tool calls were made.".to_string();
    }
    let summary = calls
        .iter()
        .map(|call| {
            let status = serde_json::to_value(call.status).unwrap_or(Value::Null);
            format!("{} ({})", call.name, status.as_str().unwrap_or_default())
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("Tool calls made before the failure (they are not undone): {summary}")
}

fn format_error(error: &CodemodeError, calls: &[CodemodeNestedCall]) -> String {
    let head = match error.kind {
        CodemodeErrorKind::Script => error.stack.clone().unwrap_or_else(|| {
            format!(
                "{}: {}",
                error.name.clone().unwrap_or_else(|| "Error".to_string()),
                error.message
            )
        }),
        CodemodeErrorKind::Timeout => format!("Script timed out: {}", error.message),
        CodemodeErrorKind::Aborted => format!("Script aborted: {}", error.message),
        CodemodeErrorKind::Sandbox => format!("Script sandbox failed: {}", error.message),
    };
    format!("{head}\n\n{}", format_call_summary(calls))
}

/// Write the full text output to a temp file, like bash does for truncated
/// output (`spillOutput`).
async fn spill_output(text: &str) -> Result<String, String> {
    write_output_file("pi-codemode", ".txt", text.as_bytes())
        .await
        .map_err(|error| error.to_string())
}

/// Save one image to a temp file; the label names the path. A failed write
/// (disk full, unwritable temp dir) must not discard the result of a script
/// whose tool calls already ran, so it becomes part of the label. The
/// missing-extension error is checked before saving (see [`save_images`]).
async fn image_label(image: &ImageContent) -> String {
    let extension = IMAGE_EXTENSIONS
        .iter()
        .find(|(mime, _)| *mime == image.mime_type)
        .map(|(_, extension)| *extension)
        .expect("caller checked the extension");
    let bytes = crate::utils::image_convert::base64_decode(&image.data);
    let kind = format!("{}, {}", image.mime_type, format_size(bytes.len()));
    match write_output_file("pi-codemode", extension, &bytes).await {
        Ok(path) => format!("[Image saved to {path} ({kind})]"),
        Err(error) => format!("[Image ({kind}) could not be saved: {error}]"),
    }
}

/// Save each image to a temp file and put a text item with its path before
/// it. The model sees the image but has no other way to reach its bytes:
/// scripts cannot write files, and `write` only takes text. Images shown
/// more than once are saved once (`saveImages`).
async fn save_images(items: Vec<UserContent>) -> Result<Vec<UserContent>, String> {
    for item in &items {
        if let UserContent::Image(image) = item {
            if !IMAGE_EXTENSIONS
                .iter()
                .any(|(mime, _)| *mime == image.mime_type)
            {
                return Err(format!(
                    "No file extension for image type {}",
                    image.mime_type
                ));
            }
        }
    }
    let labels: Mutex<HashMap<String, Arc<tokio::sync::OnceCell<String>>>> =
        Mutex::new(HashMap::new());
    let mut result = Vec::new();
    for item in items {
        match item {
            UserContent::Image(image) => {
                let cell = {
                    let mut labels = labels.lock();
                    labels
                        .entry(image.data.clone())
                        .or_insert_with(|| Arc::new(tokio::sync::OnceCell::new()))
                        .clone()
                };
                let label_image = image.clone();
                let label = cell
                    .get_or_init(|| async move { image_label(&label_image).await })
                    .await;
                result.push(text_item(label.clone()));
                result.push(UserContent::Image(image));
            }
            other => result.push(other),
        }
    }
    Ok(result)
}

/// Apply the token budget: when the combined text exceeds it, the text items
/// become one item that keeps the start and end of the text, and images
/// follow it. The full text is written to a temp file (`truncateOutput`).
async fn truncate_output(
    items: Vec<UserContent>,
    max_tokens: usize,
) -> (Vec<UserContent>, Option<String>) {
    let texts: Vec<&str> = items
        .iter()
        .filter_map(|item| match item {
            UserContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    let combined = texts.join("\n");
    let budget = max_tokens * CHARS_PER_TOKEN;
    let total_chars = combined.chars().count();
    if texts.is_empty() || total_chars <= budget {
        return (items, None);
    }
    let head_chars = budget / 2;
    let tail_chars = budget - head_chars;
    let removed = total_chars - head_chars - tail_chars;
    let head: String = combined.chars().take(head_chars).collect();
    let tail: String = combined.chars().skip(total_chars - tail_chars).collect();
    let original_tokens = total_chars.div_ceil(CHARS_PER_TOKEN);
    let removed_tokens = removed.div_ceil(CHARS_PER_TOKEN);
    let total_lines = combined.split('\n').count();
    let mut text = format!(
		"Warning: truncated output (original token count: {original_tokens})\nTotal output lines: {total_lines}\n\n{head}…{removed_tokens} tokens truncated…{tail}"
	);
    let images: Vec<UserContent> = items
        .into_iter()
        .filter(|item| matches!(item, UserContent::Image(_)))
        .collect();
    let mut kept = Vec::with_capacity(images.len() + 1);
    match spill_output(&combined).await {
        Ok(path) => {
            text.push_str(&format!(
                "\n\n[Full output: {path} (read with offset/limit)]"
            ));
            kept.push(text_item(text));
            kept.extend(images);
            return (kept, Some(path));
        }
        Err(error) => {
            text.push_str(&format!("\n\n[Could not save the full output: {error}]"));
        }
    }
    kept.push(text_item(text));
    kept.extend(images);
    (kept, None)
}

/// The value a script receives for a nested call (`toScriptValue`): a tool
/// that declares `outputSchema` resolves to its `structuredContent`, also for
/// error results that carry one (such as MCP results with `isError`); any
/// other tool resolves to its text content. Other failures reject with the
/// tool's error text.
fn to_script_value(tool: &AgentTool, outcome: &AgentToolCallOutcome) -> Result<Value, String> {
    let result = &outcome.result;
    if tool.output_schema.is_some() && result.structured_content.is_some() {
        return Ok(result.structured_content.clone().expect("checked above"));
    }
    let text = text_of(result);
    if outcome.is_error {
        return Err(if text.is_empty() {
            format!("Tool \"{}\" failed", tool.name())
        } else {
            text
        });
    }
    Ok(Value::String(text))
}

/// Whether `query` names the namespace: its name, its script identifier
/// (`mcp__dev-radius` is `mcp__dev_radius`), or the part after its last `__`
/// in either form (`dev-radius`, `dev_radius`) (`isNamespaceName`).
fn is_namespace_name(namespace: &str, query: &str) -> bool {
    fn suffix(name: &str) -> Option<&str> {
        name.rfind("__").map(|at| &name[at + 2..])
    }
    let id = to_codemode_identifier(namespace);
    let query_id = to_codemode_identifier(query);
    namespace == query
        || id == query_id
        || suffix(namespace) == Some(query)
        || suffix(&id) == Some(query_id.as_str())
}

/// `searchTools()`, `describeTool()`, and `describeNamespace()`: ranked
/// search and lookup over the script's nested tools and their namespaces
/// (`createDiscoveryGlobals`).
fn create_discovery_globals(
    tools: &[AgentTool],
    samples: &HashMap<String, String>,
    options: &CodemodeToolOptions,
) -> Vec<SandboxTool> {
    let search_tools_list = tools.to_vec();
    let search_samples = samples.clone();
    let search_options = options.clone();
    let mut search_tools =
        SandboxTool::new("searchTools", move |args, _ctx: CodemodeToolContext| {
            let tools = search_tools_list.clone();
            let samples = search_samples.clone();
            let options = search_options.clone();
            async move {
                let entry = |name: &str| -> Value {
                    json!({
                        "name": to_codemode_identifier(name),
                        "description": samples.get(name).cloned().unwrap_or_default(),
                    })
                };
                let parts = args
                    .as_ref()
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let query = parts.first().cloned().unwrap_or(Value::Null);
                let search_options = parts.get(1).cloned().unwrap_or(Value::Null);
                let Some(query) = query.as_str() else {
                    return Err("searchTools() expects a query string".to_string());
                };
                // Upstream: `searchOptions?.limit ?? DEFAULT_TOOL_SEARCH_LIMIT`,
                // then a positive-integer check.
                let raw_limit = search_options
                    .get("limit")
                    .cloned()
                    .filter(|limit| !limit.is_null())
                    .unwrap_or(json!(DEFAULT_TOOL_SEARCH_LIMIT));
                let limit = match raw_limit.as_f64() {
                    Some(value) if value.fract() == 0.0 && value > 0.0 => value as usize,
                    _ => return Err("searchTools() limit must be a positive integer".to_string()),
                };
                let namespace = search_options
                    .get("namespace")
                    .cloned()
                    .filter(|ns| !ns.is_null());
                if let Some(namespace) = &namespace {
                    if !namespace.is_string() {
                        return Err("searchTools() namespace must be a string".to_string());
                    }
                }
                let documents: Vec<ToolSearchDocument> = tools
                    .iter()
                    .filter_map(|tool| {
                        let tool_namespace = options
                            .get_tool_namespace
                            .as_ref()
                            .and_then(|get| get(tool.name()));
                        if let Some(namespace) = &namespace {
                            let matches = tool_namespace
                                .as_ref()
                                .map(|ns| {
                                    is_namespace_name(
                                        &ns.name,
                                        namespace.as_str().expect("checked string"),
                                    )
                                })
                                .unwrap_or(false);
                            if !matches {
                                return None;
                            }
                        }
                        Some(create_tool_search_document(tool, tool_namespace.as_ref()))
                    })
                    .collect();
                // Upstream builds one ranker per script; ranking is stateless, so
                // a fresh ranker per call behaves the same.
                let ranker = Bm25Ranker::new();
                let matches = ranker.rank(query, &documents, limit);
                Ok(Some(Value::Array(
                    matches.iter().map(|match_| entry(&match_.name)).collect(),
                )))
            }
        });
    search_tools.spread = true;

    let describe_samples = samples.clone();
    let describe_tools_list = tools.to_vec();
    let mut describe_tool =
        SandboxTool::new("describeTool", move |args, _ctx: CodemodeToolContext| {
            let tools = describe_tools_list.clone();
            let samples = describe_samples.clone();
            async move {
                let parts = args
                    .as_ref()
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let name = parts.first().cloned().unwrap_or(Value::Null);
                let Some(name) = name.as_str() else {
                    return Err("describeTool() expects a tool name".to_string());
                };
                let found = tools.iter().find(|candidate| {
                    candidate.name() == name || to_codemode_identifier(candidate.name()) == name
                });
                Ok(found
                    .and_then(|tool| samples.get(tool.name()).cloned())
                    .map(Value::String))
            }
        });
    describe_tool.spread = true;

    let describe_options = options.clone();
    let describe_namespace_tools = tools.to_vec();
    let mut describe_namespace = SandboxTool::new(
        "describeNamespace",
        move |args, _ctx: CodemodeToolContext| {
            let tools = describe_namespace_tools.clone();
            let options = describe_options.clone();
            async move {
                let parts = args
                    .as_ref()
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let name = parts.first().cloned().unwrap_or(Value::Null);
                let Some(name) = name.as_str() else {
                    return Err("describeNamespace() expects a namespace name".to_string());
                };
                let mut namespace: Option<ToolNamespace> = None;
                let mut names: Vec<Value> = Vec::new();
                for tool in &tools {
                    let Some(tool_namespace) = options
                        .get_tool_namespace
                        .as_ref()
                        .and_then(|get| get(tool.name()))
                    else {
                        continue;
                    };
                    if !is_namespace_name(&tool_namespace.name, name) {
                        continue;
                    }
                    namespace = namespace.or(Some(tool_namespace));
                    names.push(Value::String(to_codemode_identifier(tool.name())));
                }
                let Some(namespace) = namespace else {
                    return Ok(None);
                };
                // Upstream spreads fields conditionally, so JSON.stringify drops
                // absent `description`/`instructions` instead of sending null.
                let mut info = serde_json::Map::new();
                info.insert("name".to_string(), Value::String(namespace.name));
                if let Some(description) = namespace.description {
                    info.insert("description".to_string(), Value::String(description));
                }
                if let Some(instructions) = namespace.instructions {
                    info.insert("instructions".to_string(), Value::String(instructions));
                }
                info.insert("tools".to_string(), Value::Array(names));
                Ok(Some(Value::Object(info)))
            }
        },
    );
    describe_namespace.spread = true;

    vec![search_tools, describe_tool, describe_namespace]
}

/// Shared state of one script's `models.*` calls.
struct ModelGlobalsState {
    tool_call_id: String,
    call_count: std::sync::atomic::AtomicUsize,
    calls: Arc<Mutex<Vec<CodemodeNestedCall>>>,
    publish: Arc<dyn Fn() + Send + Sync>,
    usage: Arc<Mutex<Option<Usage>>>,
    generated_images: Arc<Mutex<usize>>,
}

/// The error for a bad first `models.classify()`/`models.generateImages()`
/// argument (`runModelCall`'s model check).
fn model_argument_error(
    name: &str,
    type_word: &str,
    model: Option<&Value>,
    list_hint: &str,
) -> String {
    let undefined_hint = match model {
        None | Some(Value::Null) => {
            " models.getModelOfType() returns undefined for an unknown provider or id."
        }
        _ => "",
    };
    format!(
        "{name}() expects {} model as its first argument, got {}.{} {list_hint}",
        with_article(type_word),
        describe_value(model.unwrap_or(&Value::Null)),
        undefined_hint
    )
}

/// Runs at most [`MAX_CONCURRENT_MODEL_CALLS`] model calls at once, in call
/// order (upstream `createLimiter`; tokio's semaphore is FIFO).
type LimitedRun = Box<
    dyn FnOnce(
            AnyModel,
            Value,
            Option<AiAbortSignal>,
        ) -> Pin<Box<dyn Future<Output = (Value, ModelCallView)> + Send>>
        + Send,
>;

/// Resolve the script's model by provider and id only, check the context,
/// then run the call as a nested call row (`runModelCall`). A script-supplied
/// baseUrl or headers must never receive the credentials.
async fn run_model_call(
    name: &str,
    type_: ModelType,
    args: Option<&Value>,
    models: &ModelRegistry,
    state: &Arc<ModelGlobalsState>,
    limit: &Arc<tokio::sync::Semaphore>,
    signal: &CodemodeToolContext,
    check_context: fn(&Value) -> Result<Value, String>,
    run: LimitedRun,
) -> Result<Value, String> {
    let type_word = model_type_name(type_);
    let list_hint = format!(
        "List the {type_word} models you can use with models.getAvailableOfType(\"{type_word}\")."
    );
    let parts = args.and_then(Value::as_array).cloned().unwrap_or_default();
    let model = parts.first();
    let context = parts.get(1).cloned().unwrap_or(Value::Null);
    let (provider, id) = match model {
        Some(model @ Value::Object(fields)) => {
            let provider = fields.get("provider").and_then(Value::as_str);
            let id = fields.get("id").and_then(Value::as_str);
            match (provider, id) {
                (Some(provider), Some(id)) => (provider.to_string(), id.to_string()),
                _ => {
                    return Err(model_argument_error(
                        name,
                        type_word,
                        Some(model),
                        &list_hint,
                    ))
                }
            }
        }
        other => return Err(model_argument_error(name, type_word, other, &list_hint)),
    };
    let reference = format!("{provider}/{id}");
    let resolved = models.get_model_of_type(type_, &provider, &id);
    let Some(resolved) = resolved else {
        let actual_type = [ModelType::Chat, ModelType::Image, ModelType::Classifier]
            .into_iter()
            .find(|other| {
                *other != type_ && models.get_model_of_type(*other, &provider, &id).is_some()
            });
        return Err(match actual_type {
            Some(actual) => format!(
                "\"{reference}\" is {} model, not {} model. {list_hint}",
                with_article(model_type_name(actual)),
                with_article(type_word)
            ),
            None => format!("Unknown {type_word} model \"{reference}\". {list_hint}"),
        });
    };
    let context = check_context(&context)?;

    let count = state
        .call_count
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        + 1;
    let record_id = format!("{}/{name}/{count}", state.tool_call_id);
    {
        let mut calls = state.calls.lock();
        calls.push(CodemodeNestedCall {
            id: record_id.clone(),
            name: name.to_string(),
            args: format!("{}/{}", resolved.provider(), resolved.id()),
            status: CodemodeNestedCallStatus::Running,
            duration_ms: None,
            error: None,
            cost: None,
        });
    }
    (state.publish)();
    let started_at = Instant::now();
    let ai_signal = bridge_codemode_to_ai_signal(&signal.signal);
    let (value, view) = {
        let _permit = limit
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| "model call limiter closed".to_string())?;
        run(resolved, context, Some(ai_signal)).await
    };
    let duration_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    {
        let mut calls = state.calls.lock();
        let record = calls
            .iter_mut()
            .find(|call| call.id == record_id)
            .expect("model call record was pushed");
        record.duration_ms = Some(duration_ms);
        record.status = if view.stop {
            CodemodeNestedCallStatus::Ok
        } else if view.aborted {
            CodemodeNestedCallStatus::Cancelled
        } else {
            CodemodeNestedCallStatus::Error
        };
        if let Some(error_message) = &view.error_message {
            record.error = Some(truncate_text(error_message, ERROR_PREVIEW_CHARS));
        }
        if let Some(usage) = &view.usage {
            record.cost = Some(usage.cost.total);
            let mut model_usage = state.usage.lock();
            *model_usage = Some(match model_usage.take() {
                Some(existing) => combine_usage(&existing, usage),
                None => usage.clone(),
            });
        }
    }
    (state.publish)();
    Ok(value)
}

/// `models.*` for scripts: the model registry methods documented in
/// docs/codemode.md (`createModelGlobals`). Classifier and image calls appear
/// as nested call rows so the renderer shows them, and their usage goes to
/// the script's usage total. Rows show only the model, never prompts or
/// image data.
fn create_model_globals(
    models: Arc<ModelRegistry>,
    tool_call_id: String,
    calls: Arc<Mutex<Vec<CodemodeNestedCall>>>,
    publish: Arc<dyn Fn() + Send + Sync>,
    usage: Arc<Mutex<Option<Usage>>>,
    generated_images: Arc<Mutex<usize>>,
) -> Vec<SandboxTool> {
    let limit = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_MODEL_CALLS));
    let state = Arc::new(ModelGlobalsState {
        tool_call_id,
        call_count: std::sync::atomic::AtomicUsize::new(0),
        calls,
        publish,
        usage,
        generated_images,
    });

    let registry = models.clone();
    let mut get_models_of_type = SandboxTool::new(
        "models.getModelsOfType",
        move |args, _ctx: CodemodeToolContext| {
            let models = registry.clone();
            async move {
                let parts = args
                    .as_ref()
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let type_ = to_model_type(parts.first().unwrap_or(&Value::Null))?;
                let provider = to_provider(parts.get(1).unwrap_or(&Value::Null))?;
                let infos: Vec<Value> = models
                    .get_models_of_type(type_, provider.as_deref())
                    .iter()
                    .map(to_model_info)
                    .collect();
                Ok(Some(Value::Array(infos)))
            }
        },
    );
    get_models_of_type.spread = true;

    let registry = models.clone();
    let mut get_available_of_type = SandboxTool::new(
        "models.getAvailableOfType",
        move |args, ctx: CodemodeToolContext| {
            let models = registry.clone();
            async move {
                let parts = args
                    .as_ref()
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let type_ = to_model_type(parts.first().unwrap_or(&Value::Null))?;
                let provider = to_provider(parts.get(1).unwrap_or(&Value::Null))?;
                let mut options = AuthOperationOptions::default();
                options.signal = Some(bridge_codemode_to_ai_signal(&ctx.signal));
                let available = models
                    .get_available_of_type(type_, provider.as_deref(), Some(options))
                    .await
                    .map_err(|thrown| match thrown {
                        ThrownValue::Text(text) => text,
                        ThrownValue::Error(error) => error.to_string(),
                    })?;
                Ok(Some(Value::Array(
                    available.iter().map(to_model_info).collect(),
                )))
            }
        },
    );
    get_available_of_type.spread = true;

    let registry = models.clone();
    let mut get_model_of_type = SandboxTool::new(
        "models.getModelOfType",
        move |args, _ctx: CodemodeToolContext| {
            let models = registry.clone();
            async move {
                let parts = args
                    .as_ref()
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let provider = parts.get(1).cloned().unwrap_or(Value::Null);
                let id = parts.get(2).cloned().unwrap_or(Value::Null);
                if !(provider.is_string() && id.is_string()) {
                    let described = parts
                        .iter()
                        .map(describe_value)
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(format!(
					"models.getModelOfType(type, provider, id) expects three strings, got ({described}). The provider and the id are separate arguments, for example models.getModelOfType(\"classifier\", \"typesafe\", \"jev-latest\")."
				));
                }
                let type_ = to_model_type(parts.first().unwrap_or(&Value::Null))?;
                let model = models.get_model_of_type(
                    type_,
                    provider.as_str().expect("checked string"),
                    id.as_str().expect("checked string"),
                );
                Ok(model.as_ref().map(to_model_info))
            }
        },
    );
    get_model_of_type.spread = true;

    let registry = models.clone();
    let state_classify = state.clone();
    let limit_classify = limit.clone();
    let mut classify =
        SandboxTool::new("models.classify", move |args, ctx: CodemodeToolContext| {
            let models = registry.clone();
            let state = state_classify.clone();
            let limit = limit_classify.clone();
            async move {
                let parts = args.clone();
                let registry_for_run = models.clone();
                run_model_call(
                    "models.classify",
                    ModelType::Classifier,
                    parts.as_ref(),
                    &models,
                    &state,
                    &limit,
                    &ctx,
                    check_classifier_context,
                    Box::new(move |resolved, context, signal| {
                        let models = registry_for_run.clone();
                        Box::pin(async move {
                            let AnyModel::Classifier(model) = resolved else {
                                return (
                                    Value::Null,
                                    ModelCallView {
                                        stop: false,
                                        aborted: false,
                                        error_message: Some(
                                            "resolved model is not a classifier".to_string(),
                                        ),
                                        usage: None,
                                    },
                                );
                            };
                            let context = serde_json::from_value::<ClassifierContext>(context);
                            let Ok(context) = context else {
                                return (
                                    Value::Null,
                                    ModelCallView {
                                        stop: false,
                                        aborted: false,
                                        error_message: Some(
                                            "context did not deserialize".to_string(),
                                        ),
                                        usage: None,
                                    },
                                );
                            };
                            let mut options = ModelsClassifierOptions::default();
                            options.base.base.signal = signal;
                            let result: ClassifierResult =
                                models.classify(&model, &context, Some(options)).await;
                            let view = ModelCallView::from_classifier(&result);
                            let value = serde_json::to_value(&result).unwrap_or(Value::Null);
                            (value, view)
                        })
                    }),
                )
                .await
                .map(Some)
            }
        });
    classify.spread = true;

    let registry = models.clone();
    let state_images = state.clone();
    let limit_images = limit.clone();
    let mut generate_images = SandboxTool::new(
        "models.generateImages",
        move |args, ctx: CodemodeToolContext| {
            let models = registry.clone();
            let state = state_images.clone();
            let limit = limit_images.clone();
            async move {
                let parts = args.clone();
                let generated_images = state.generated_images.clone();
                let registry_for_run = models.clone();
                run_model_call(
                    "models.generateImages",
                    ModelType::Image,
                    parts.as_ref(),
                    &models,
                    &state,
                    &limit,
                    &ctx,
                    check_images_context,
                    Box::new(move |resolved, context, signal| {
                        let models = registry_for_run.clone();
                        let generated_images = generated_images.clone();
                        Box::pin(async move {
                            let AnyModel::Image(model) = resolved else {
                                return (
                                    Value::Null,
                                    ModelCallView {
                                        stop: false,
                                        aborted: false,
                                        error_message: Some(
                                            "resolved model is not an image model".to_string(),
                                        ),
                                        usage: None,
                                    },
                                );
                            };
                            let context = serde_json::from_value::<ImagesContext>(context);
                            let Ok(context) = context else {
                                return (
                                    Value::Null,
                                    ModelCallView {
                                        stop: false,
                                        aborted: false,
                                        error_message: Some(
                                            "context did not deserialize".to_string(),
                                        ),
                                        usage: None,
                                    },
                                );
                            };
                            let mut options = ModelsImagesOptions::default();
                            options.base.base.signal = signal;
                            let result = models
                                .generate_images(&model, &context, Some(options))
                                .await;
                            *generated_images.lock() += result
                                .output
                                .iter()
                                .filter(|block| matches!(block, UserContent::Image(_)))
                                .count();
                            let view = ModelCallView::from_images(&result);
                            let value = serde_json::to_value(&result).unwrap_or(Value::Null);
                            (value, view)
                        })
                    }),
                )
                .await
                .map(Some)
            }
        },
    );
    generate_images.spread = true;

    vec![
        get_models_of_type,
        get_available_of_type,
        get_model_of_type,
        classify,
        generate_images,
    ]
}

/// Forward the sandbox per-call signal onto the session execute-tool signal
/// type (crate abort D9).
fn bridge_to_session_signal(ctx: &CodemodeToolContext) -> AbortSignal {
    let controller = crate::utils::abort::AbortController::new();
    let forwarded = controller.signal().clone();
    let source = ctx.signal.clone();
    tokio::spawn(async move {
        source.wait_aborted().await;
        controller.abort();
    });
    forwarded
}

/// Forward the tool call's signal into the sandbox (crate abort D9).
fn bridge_to_sandbox_signal(signal: Option<&AbortSignal>) -> Option<pi_codemode::AbortSignal> {
    let signal = signal?;
    let forwarded = pi_codemode::AbortSignal::new();
    let source = signal.clone();
    let target = forwarded.clone();
    tokio::spawn(async move {
        source.aborted_future().await;
        target.abort("aborted");
    });
    Some(forwarded)
}

/// Forward the sandbox per-call signal onto the pi-ai request signal type.
fn bridge_codemode_to_ai_signal(signal: &pi_codemode::AbortSignal) -> AiAbortSignal {
    let controller = pi_ai::utils::abort::AbortController::new();
    let forwarded = controller.signal();
    let source = signal.clone();
    tokio::spawn(async move {
        source.wait_aborted().await;
        pi_ai::utils::abort::AbortController::abort(&controller);
    });
    forwarded
}

/// Run one script (`executeCodemode`). Without a session context (a plain
/// Agent or a direct call) scripts cannot call tools, `store()` starts empty,
/// and writes are dropped.
#[allow(clippy::too_many_arguments)]
pub async fn execute_codemode(
    tool_call_id: String,
    input: CodemodeToolInput,
    signal: Option<AbortSignal>,
    on_update: Option<AgentToolUpdateCallback>,
    ctx: Option<ExtensionToolContext>,
    options: CodemodeToolOptions,
) -> Result<AgentToolResult, String> {
    let started_at = Instant::now();
    let parsed = parse_codemode_source(&input.code).map_err(|error| error.message)?;
    let calls: Arc<Mutex<Vec<CodemodeNestedCall>>> = Arc::default();
    // Usage of the script's `models.*` calls. Nested tool calls report theirs
    // through the session.
    let model_usage: Arc<Mutex<Option<Usage>>> = Arc::default();
    // Images returned by `models.generateImages()`, to notice a script that
    // never shows them.
    let generated_images: Arc<Mutex<usize>> = Arc::default();

    let snapshot = {
        let calls = calls.clone();
        move || CodemodeToolDetails {
            calls: calls.lock().clone(),
            full_output_path: None,
        }
    };
    let publish: Arc<dyn Fn() + Send + Sync> = {
        let snapshot = snapshot.clone();
        let on_update = on_update.clone();
        Arc::new(move || {
            if let Some(on_update) = &on_update {
                let details = serde_json::to_value(snapshot()).unwrap_or(Value::Null);
                on_update(AgentToolResult::new(Vec::new(), details));
            }
        })
    };

    let callable = ctx
        .as_ref()
        .map(|ctx| get_codemode_callable_tools(&ctx.tools))
        .unwrap_or_default();
    // ALL_TOOLS entries carry the declaration.
    let guidelines = options.get_tool_guidelines.as_ref().map(|get| get());
    let samples: HashMap<String, String> = callable
        .iter()
        .map(|tool| {
            let tool_guidelines = guidelines
                .as_ref()
                .and_then(|map| map.get(tool.name()).cloned())
                .unwrap_or_default();
            let declaration = to_codemode_declaration(tool, &tool_guidelines);
            (
                tool.name().to_string(),
                render_tool_sample(&SampleTool {
                    name: &declaration.name,
                    description: Some(&declaration.description),
                    input_schema: Some(&declaration.input_schema),
                    output_schema: Some(&declaration.output_schema),
                    input_max_chars: None,
                }),
            )
        })
        .collect();

    let mut sandbox_tools: Vec<SandboxTool> = Vec::new();
    for tool in &callable {
        let description = samples.get(tool.name()).cloned();
        let ctx = ctx.clone().expect("only ctx.tools are callable");
        let calls = calls.clone();
        let publish = publish.clone();
        let nested_tool = tool.clone();
        let call_tool_call_id = tool_call_id.clone();
        let mut sandbox_tool = SandboxTool::new(
            tool.name().to_string(),
            move |args, call_ctx: CodemodeToolContext| {
                let ctx = ctx.clone();
                let calls = calls.clone();
                let publish = publish.clone();
                let nested_tool = nested_tool.clone();
                let call_tool_call_id = call_tool_call_id.clone();
                async move {
                    let index = {
                        let mut pending = calls.lock();
                        pending.push(CodemodeNestedCall {
                            id: format!("{call_tool_call_id}/?"),
                            name: nested_tool.name().to_string(),
                            args: preview_args(args.as_ref()),
                            status: CodemodeNestedCallStatus::Running,
                            duration_ms: None,
                            error: None,
                            cost: None,
                        });
                        pending.len() - 1
                    };
                    publish();
                    let call_started_at = Instant::now();
                    let session_signal = bridge_to_session_signal(&call_ctx);
                    let outcome = ctx
                        .execute_tool(
                            nested_tool.name().to_string(),
                            args.unwrap_or(Value::Null),
                            ExecuteToolOptions {
                                signal: Some(session_signal),
                                on_update: None,
                            },
                        )
                        .await;
                    let duration_ms = call_started_at.elapsed().as_secs_f64() * 1000.0;
                    {
                        let mut pending = calls.lock();
                        let record = pending
                            .get_mut(index)
                            .expect("nested call record was pushed");
                        record.id = outcome.tool_call.id.clone();
                        record.duration_ms = Some(duration_ms);
                        if outcome.is_error {
                            record.status = if call_ctx.signal.is_aborted() {
                                CodemodeNestedCallStatus::Cancelled
                            } else {
                                CodemodeNestedCallStatus::Error
                            };
                            let text = text_of(&outcome.result);
                            let error_text = if text.is_empty() {
                                format!("Tool \"{}\" failed", nested_tool.name())
                            } else {
                                text
                            };
                            record.error = Some(truncate_text(&error_text, ERROR_PREVIEW_CHARS));
                        } else {
                            record.status = CodemodeNestedCallStatus::Ok;
                        }
                    }
                    publish();
                    to_script_value(&nested_tool, &outcome).map(Some)
                }
            },
        );
        sandbox_tool.description = description;
        sandbox_tools.push(sandbox_tool);
    }

    let mut globals = create_discovery_globals(&callable, &samples, &options);
    if options.models == Some(true) && ctx.is_some() {
        globals.extend(create_model_globals(
            ctx.as_ref().expect("checked").ctx.model_registry.clone(),
            tool_call_id.clone(),
            calls.clone(),
            publish.clone(),
            model_usage.clone(),
            generated_images.clone(),
        ));
    }

    let timeout_ms = parsed
        .options
        .timeout_ms
        .map(|ms| ms as f64)
        .unwrap_or(f64::INFINITY);
    let sandbox = CodemodeSandbox::new(CodemodeSandboxOptions {
        tools: sandbox_tools,
        globals,
        timeout_ms: Some(timeout_ms),
        memory_limit_bytes: Some(CODEMODE_MEMORY_LIMIT_BYTES),
        wasm_loader: options.wasm_loader.clone(),
        worker_factory: options.worker_factory.clone(),
    })?;

    let store = ctx
        .as_ref()
        .map(|ctx| read_codemode_store(&ctx.ctx.session_manager.lock().get_branch()))
        .unwrap_or_default();
    let executed = sandbox
        .execute(
            parsed.code,
            CodemodeExecuteOptions {
                signal: bridge_to_sandbox_signal(signal.as_ref()),
                timeout_ms: Some(timeout_ms),
                store: Some(store),
            },
        )
        .await;
    sandbox.close().await;
    let result: CodemodeResult = executed?;

    // Calls still marked running were cut off by the script ending, a
    // timeout, or an abort.
    {
        let mut pending = calls.lock();
        for call in pending.iter_mut() {
            if call.status == CodemodeNestedCallStatus::Running {
                call.status = CodemodeNestedCallStatus::Cancelled;
            }
        }
    }

    let mut script_output: Vec<CodemodeOutputItem> = result.output().to_vec();
    if result.is_ok() {
        let writes = result
            .store_writes()
            .expect("ok results carry store writes");
        if !writes.set.is_empty() || !writes.delete.is_empty() {
            if let Some(append_entry) = &options.append_entry {
                append_entry(
                    CODEMODE_STORE_ENTRY_TYPE,
                    Some(
                        serde_json::to_value(CodemodeStoreEntryData {
                            set: writes.set.clone(),
                            delete: writes.delete.clone(),
                        })
                        .unwrap_or(Value::Null),
                    ),
                );
            }
        }
        // pi extension: a returned value is appended like text().
        if let Some(value) = result.value() {
            script_output.push(CodemodeOutputItem::Text {
                text: value_text(value),
                console: None,
            });
        }
    }
    let mut items = format_output(&script_output);
    let mut details = snapshot();
    if !result.is_ok() {
        let error = result.error().expect("error results carry an error");
        let summary = format_error(error, &calls.lock());
        items.push(text_item(format!("Script error:\n{summary}")));
    }

    let generated = *generated_images.lock();
    if generated > 0
        && !items
            .iter()
            .any(|item| matches!(item, UserContent::Image(_)))
    {
        let plural = if generated == 1 { "" } else { "s" };
        items.push(text_item(format!(
			"Note: models.generateImages() returned {generated} image{plural} that the script did not show. Show each image block of result.output with image(block)."
		)));
    }

    let max_tokens = parsed
        .options
        .max_output_tokens
        .map(|tokens| tokens as usize)
        .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
    let (truncated, full_output_path) =
        truncate_output(join_adjacent_text(items), max_tokens).await;
    // After truncation, which joins the text items and moves images after
    // them, so each path stays next to its image and is never cut.
    let output = join_adjacent_text(save_images(truncated).await?);
    let wall_time = format!("{:.1}", started_at.elapsed().as_secs_f64());
    let header = format!(
        "{}\nWall time {wall_time} seconds\nOutput:\n",
        if result.is_ok() {
            "Script completed"
        } else {
            "Script failed"
        }
    );
    details.full_output_path = full_output_path;

    let mut tool_result = AgentToolResult::new(
        std::iter::once(text_item(header)).chain(output).collect(),
        serde_json::to_value(&details).unwrap_or(Value::Null),
    );
    if let Some(usage) = model_usage.lock().clone() {
        tool_result.usage = Some(usage);
    }
    if !result.is_ok() {
        tool_result.is_error = Some(true);
    }
    Ok(tool_result)
}

#[cfg(test)]
mod output_layout {
    use super::*;

    fn text_item_value(text: &str) -> CodemodeOutputItem {
        CodemodeOutputItem::Text {
            text: text.to_string(),
            console: None,
        }
    }

    fn console_item_value(text: &str) -> CodemodeOutputItem {
        CodemodeOutputItem::Text {
            text: text.to_string(),
            console: Some(pi_codemode::types::True),
        }
    }

    fn image_item_value(data: &str) -> CodemodeOutputItem {
        CodemodeOutputItem::Image {
            data: data.to_string(),
            mime_type: "image/png".to_string(),
        }
    }

    #[test]
    fn format_output_numbers_multiple_texts_and_moves_console_last() {
        let output = vec![
            text_item_value("a"),
            console_item_value("log line"),
            image_item_value("Zm9v"),
            text_item_value("b"),
        ];
        let items = format_output(&output);
        let texts: Vec<String> = items
            .iter()
            .filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.clone()),
                UserContent::Image(_) => None,
            })
            .collect();
        assert_eq!(items.len(), 4);
        assert!(matches!(&items[1], UserContent::Image(_)));
        assert_eq!(
            texts,
            vec![
                "==> text 1/2 <==\na".to_string(),
                "==> text 2/2 <==\nb".to_string(),
                "<console_output>\nlog line\n</console_output>".to_string(),
            ]
        );
    }

    #[test]
    fn format_output_leaves_a_single_text_unnumbered() {
        let items = format_output(&[text_item_value("only")]);
        assert_eq!(items.len(), 1);
        match &items[0] {
            UserContent::Text(text) => assert_eq!(text.text, "only"),
            _ => panic!("text"),
        }
    }

    #[test]
    fn join_adjacent_text_merges_neighbors_with_newlines() {
        let joined = join_adjacent_text(vec![
            text_item("x"),
            text_item("y"),
            UserContent::Image(ImageContent::new(
                "Zm9v".to_string(),
                "image/png".to_string(),
            )),
            text_item(""),
            text_item("tail"),
        ]);
        assert_eq!(joined.len(), 3);
        match &joined[0] {
            UserContent::Text(text) => assert_eq!(text.text, "x\ny"),
            _ => panic!("text"),
        }
        match &joined[2] {
            UserContent::Text(text) => assert_eq!(text.text, "tail"),
            _ => panic!("text"),
        }
    }

    #[test]
    fn classifier_context_rejects_malformed_images() {
        let questions =
            json!({ "q": { "type": "choice", "instructions": "i", "criteria": { "a": "b" } } });
        let valid = json!({ "state": {}, "images": [{ "type": "image", "data": "Zm9v", "mimeType": "image/png" }], "questions": questions });
        assert!(check_classifier_context(&valid).is_ok());

        let missing_data = json!({ "state": {}, "images": [{ "type": "image", "mimeType": "image/png" }], "questions": questions });
        let error = check_classifier_context(&missing_data).unwrap_err();
        assert!(
            error.contains("context.images[0] must be an image block"),
            "{error}"
        );

        let not_array = json!({ "state": {}, "images": "nope", "questions": questions });
        let error = check_classifier_context(&not_array).unwrap_err();
        assert!(error.contains("context.images must be an array"), "{error}");
    }
}
