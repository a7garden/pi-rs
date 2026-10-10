//! Port of `packages/coding-agent/src/core/tools/read.ts` — the read tool:
//! text files and images with offset/limit paging and head truncation.
//!
//! Deviations: the TypeBox parameter schema becomes a `serde_json` literal and
//! tool `details` are erased JSON (D27); the read renderers are deferred with
//! the excluded interactive mode, so `renderCall`/`renderResult` are `None`
//! (D32); the image pipeline reuses the ported `processImage`/mime seams, and
//! BMP→PNG conversion goes through the `PhotonEngine` seam (`utils::photon`),
//! unset by default — the central test suite registers a minimal test engine;
//! `ReadToolOptions::resize_options` takes the crate's [`ImageResizeOptions`]
//! (upstream: pi-ai `ModelImageResizeOptions`), the type
//! [`ProcessImageOptions`] accepts, and the model-provided profile is mapped
//! field-for-field, limited to what `pi_ai::ModelInputLimits` carries (D36).
//! Upstream's optional `detectImageMimeType` hook is required on the
//! [`ReadOperations`] trait; [`local_read_operations`] returns the mime-seam
//! result. The upstream `Promise` + `addEventListener` abort race becomes a
//! `tokio::select!` over the work future and `aborted_future()`, preserving
//! first-abort-wins semantics.

use std::sync::Arc;

use pi_agent::{AgentTool, AgentToolResult, AgentToolUpdateCallback, BoxFuture};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

use pi_ai::types::{
    ConstrainedSamplingConfig, ConstrainedSamplingSetting, ImageContent, Model, ModelInput,
    StrictSchemaMode, TextContent, UserContent,
};

use crate::core::extensions::types::{ExtensionToolContext, ToolDefinition, ToolExecute};
use crate::core::tools::path_utils::resolve_read_path_async;
use crate::core::tools::tool_definition_wrapper::wrap_tool_definition;
use crate::core::tools::truncate::{
    format_size, truncate_head, TruncatedBy, TruncationOptions, TruncationResult,
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES,
};
use crate::utils::abort::AbortSignal;
use crate::utils::image_process::{process_image, ProcessImageOptions, ProcessImageResult};
use crate::utils::image_resize::ImageResizeOptions;
use crate::utils::mime::detect_supported_image_mime_type_from_file;

/// `readSchema` — the read tool parameter schema.
pub fn read_schema() -> JsonValue {
    json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (relative or absolute)"
            },
            "offset": {
                "type": "number",
                "description": "Line number to start reading from (1-indexed)"
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of lines to read"
            }
        },
        "required": ["path"]
    })
}

/// `readToolSystemPromptContribution`.
pub const READ_TOOL_SYSTEM_PROMPT_SNIPPET: &str = "Read file contents";

/// `readToolSystemPromptContribution.guidelines`.
pub const READ_TOOL_SYSTEM_PROMPT_GUIDELINES: [&str; 1] =
    ["Use read to examine files instead of cat or sed."];

/// `ReadToolInput` = `Static<typeof readSchema>`.
#[derive(Debug, Clone, Deserialize)]
pub struct ReadToolInput {
    pub path: String,
    pub offset: Option<f64>,
    pub limit: Option<f64>,
}

/// Tool `details` payload (`ReadToolDetails`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadToolDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
}

/// `readOutputSchema` — result for programmatic callers such as codemode
/// scripts: the text for text files, and an image block for images that
/// codemode's `image()` accepts. `note` is the text that goes with the image,
/// such as resize hints. Property descriptions are left out so the type stays
/// on one line in tool descriptions.
pub fn read_output_schema() -> JsonValue {
    json!({
        "anyOf": [
            { "type": "string" },
            {
                "type": "object",
                "properties": {
                    "type": { "const": "image" },
                    "data": { "type": "string" },
                    "mimeType": { "type": "string" },
                    "note": { "type": "string" }
                },
                "required": ["type", "data", "mimeType", "note"],
                "additionalProperties": false
            }
        ]
    })
}

/// `ReadToolOutput` — `Static<typeof readOutputSchema>`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ReadToolOutput {
    Text(String),
    Image {
        #[serde(rename = "type")]
        image_type: String,
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
        note: String,
    },
}

/// `toReadOutput`: the image block and its note, or the text for text files
/// and images that could not be processed.
fn to_read_output(content: &[UserContent]) -> ReadToolOutput {
    let text = content
        .iter()
        .find_map(|block| match block {
            UserContent::Text(text) => Some(text.text.clone()),
            UserContent::Image(_) => None,
        })
        .unwrap_or_default();
    let image = content.iter().find_map(|block| match block {
        UserContent::Image(image) => Some(image),
        UserContent::Text(_) => None,
    });
    match image {
        Some(image) => ReadToolOutput::Image {
            image_type: "image".to_string(),
            data: image.data.clone(),
            mime_type: image.mime_type.clone(),
            note: text,
        },
        None => ReadToolOutput::Text(text),
    }
}

/// Pluggable operations for the read tool (`ReadOperations`). Override these
/// to delegate file reading to remote systems (for example SSH).
pub trait ReadOperations: Send + Sync {
    /// Read file contents as a Buffer
    fn read_file(&self, absolute_path: String) -> BoxFuture<'static, Result<Vec<u8>, String>>;
    /// Check if file is readable (throw if not)
    fn access(&self, absolute_path: String) -> BoxFuture<'static, Result<(), String>>;
    /// Detect image MIME type, return null or undefined for non-images.
    /// Upstream this hook is optional on the interface; the port keeps it
    /// required and the local operations supply the mime-seam result (D36).
    fn detect_image_mime_type(
        &self,
        absolute_path: String,
    ) -> BoxFuture<'static, Result<Option<String>, String>>;
}

/// Map an io error to a Node-style message carrying the error code, matching
/// upstream error surfaces (`ENOENT: no such file or directory, access '...'`).
fn io_error_message(error: &std::io::Error, syscall: &str, path: &str) -> String {
    match error.kind() {
        std::io::ErrorKind::NotFound => {
            format!("ENOENT: no such file or directory, {syscall} '{path}'")
        }
        std::io::ErrorKind::PermissionDenied => {
            format!("EACCES: permission denied, {syscall} '{path}'")
        }
        _ => error.to_string(),
    }
}

/// The default local-filesystem operations (`defaultReadOperations`).
pub fn local_read_operations() -> Arc<dyn ReadOperations> {
    struct LocalReadOperations;

    impl ReadOperations for LocalReadOperations {
        fn read_file(&self, absolute_path: String) -> BoxFuture<'static, Result<Vec<u8>, String>> {
            Box::pin(async move {
                tokio::fs::read(&absolute_path)
                    .await
                    .map_err(|error| io_error_message(&error, "open", &absolute_path))
            })
        }

        fn access(&self, absolute_path: String) -> BoxFuture<'static, Result<(), String>> {
            Box::pin(async move {
                // Upstream `fsAccess(path, constants.R_OK)`: readability via the
                // permission bits — the write-based `readonly()` is not equivalent.
                let metadata = tokio::fs::metadata(&absolute_path)
                    .await
                    .map_err(|error| io_error_message(&error, "access", &absolute_path))?;
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    if metadata.permissions().mode() & 0o0004 == 0 {
                        // EACCES, as Node's `access(R_OK)` reports.
                        return Err("EACCES: permission denied, access".to_string());
                    }
                }
                #[cfg(not(unix))]
                let _ = metadata;
                Ok(())
            })
        }

        fn detect_image_mime_type(
            &self,
            absolute_path: String,
        ) -> BoxFuture<'static, Result<Option<String>, String>> {
            Box::pin(async move {
                detect_supported_image_mime_type_from_file(&absolute_path)
                    .await
                    .map(|mime_type| mime_type.map(str::to_string))
                    .map_err(|error| error.to_string())
            })
        }
    }

    Arc::new(LocalReadOperations)
}

/// Options for [`create_read_tool_definition`] (`ReadToolOptions`).
#[derive(Clone, Default)]
pub struct ReadToolOptions {
    /// Whether to auto-resize images. Default: `true`
    pub auto_resize_images: Option<bool>,
    /// Fallback resize profile when the execution context has no model
    /// metadata (`resizeOptions`). Upstream type is pi-ai
    /// `ModelImageResizeOptions`; the port takes the crate's
    /// [`ImageResizeOptions`], the type [`ProcessImageOptions`] accepts (D36).
    pub resize_options: Option<ImageResizeOptions>,
    /// Custom operations for file reading. Default: local filesystem
    pub operations: Option<Arc<dyn ReadOperations>>,
}

/// `getNonVisionImageNote`: warn once when the current model cannot see images.
fn get_non_vision_image_note(model: Option<&Model>) -> Option<String> {
    let model = model?;
    if model.base.input.contains(&ModelInput::Image) {
        return None;
    }
    Some(
        "[Current model does not support images. The image will be omitted from this request.]"
            .to_string(),
    )
}

/// `createReadToolDefinition`.
pub fn create_read_tool_definition(cwd: &str, options: Option<ReadToolOptions>) -> ToolDefinition {
    let options = options.unwrap_or_default();
    let auto_resize_images = options.auto_resize_images.unwrap_or(true);
    let fallback_resize_options = options.resize_options;
    let ops = options.operations.unwrap_or_else(local_read_operations);
    let cwd = cwd.to_string();

    let mut definition = ToolDefinition::new(
		"read",
		"read",
		format!(
			"Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
			DEFAULT_MAX_BYTES / 1024
		),
		read_schema(),
	);
    definition.prompt_snippet = Some(READ_TOOL_SYSTEM_PROMPT_SNIPPET.to_string());
    definition.prompt_guidelines = Some(
        READ_TOOL_SYSTEM_PROMPT_GUIDELINES
            .iter()
            .map(|g| g.to_string())
            .collect(),
    );
    definition.output_schema = Some(read_output_schema());
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
            let cwd = cwd.clone();
            let fallback_resize_options = fallback_resize_options.clone();
            Box::pin(async move {
                let ReadToolInput {
                    path,
                    offset,
                    limit,
                } = serde_json::from_value(input).map_err(|error| error.to_string())?;

                if signal.as_ref().is_some_and(|signal| signal.aborted()) {
                    return Err("Operation aborted".to_string());
                }

                let work = async {
                    // `ctx?.cwd || cwd`
                    let effective_cwd = ctx
                        .as_ref()
                        .map(|ctx| ctx.cwd.clone())
                        .filter(|cwd| !cwd.is_empty())
                        .unwrap_or_else(|| cwd.clone());

                    let absolute_path = resolve_read_path_async(&path, &effective_cwd).await;
                    // Check if file exists and is readable.
                    ops.access(absolute_path.clone()).await?;
                    let mime_type = ops.detect_image_mime_type(absolute_path.clone()).await?;

                    let model = ctx.as_ref().and_then(|ctx| ctx.model.clone());
                    let non_vision_image_note = get_non_vision_image_note(model.as_ref());
                    // `ctx?.model?.inputLimits?.images?.resize ?? fallbackResizeOptions`
                    let model_resize_options = model
                        .as_ref()
                        .and_then(|model| model.base.input_limits.as_ref())
                        .and_then(|limits| limits.images.as_ref())
                        .and_then(|images| images.resize.clone());

                    if let Some(mime_type) = mime_type {
                        // Read image as binary.
                        let buffer = ops.read_file(absolute_path).await?;
                        // Model profiles carry pi-ai `ModelImageResizeOptions`;
                        // map field-for-field onto the pipeline's
                        // `ImageResizeOptions` (D36).
                        let resize_options = model_resize_options
                            .map(|resize| ImageResizeOptions {
                                max_width: resize.max_width.map(|value| value as u32),
                                max_height: resize.max_height.map(|value| value as u32),
                                max_bytes: resize.max_bytes,
                                jpeg_quality: resize.jpeg_quality,
                            })
                            .or(fallback_resize_options);
                        let processed = process_image(
                            &buffer,
                            &mime_type,
                            Some(&ProcessImageOptions {
                                auto_resize_images: Some(auto_resize_images),
                                resize_options,
                            }),
                        )
                        .await;
                        let content: Vec<UserContent> = match processed {
                            ProcessImageResult::Err { message } => {
                                let mut text_note =
                                    format!("Read image file [{mime_type}]\n{message}");
                                if let Some(note) = &non_vision_image_note {
                                    text_note.push_str(&format!("\n{note}"));
                                }
                                vec![UserContent::from(TextContent::new(text_note))]
                            }
                            ProcessImageResult::Ok {
                                data,
                                mime_type,
                                hints,
                            } => {
                                let mut text_note = format!("Read image file [{mime_type}]");
                                if !hints.is_empty() {
                                    text_note.push_str(&format!("\n{}", hints.join("\n")));
                                }
                                if let Some(note) = &non_vision_image_note {
                                    text_note.push_str(&format!("\n{note}"));
                                }
                                vec![
                                    UserContent::from(TextContent::new(text_note)),
                                    UserContent::from(ImageContent::new(data, mime_type)),
                                ]
                            }
                        };
                        Ok((content, None))
                    } else {
                        // Read text content.
                        let buffer = ops.read_file(absolute_path).await?;
                        // `buffer.toString("utf-8")` never fails; the port uses
                        // the equally lossy `from_utf8_lossy`.
                        let text_content = String::from_utf8_lossy(&buffer);
                        let all_lines: Vec<&str> = text_content.split('\n').collect();
                        let total_file_lines = all_lines.len();
                        // Apply offset if specified. Convert from 1-indexed input
                        // to 0-indexed array access. Upstream `offset ? ... : 0`:
                        // `0` is falsy and maps to the first line as well.
                        let start_line = offset
                            .filter(|offset| *offset != 0.0)
                            .map(|offset| (offset as usize).saturating_sub(1))
                            .unwrap_or(0);
                        let start_line_display = start_line + 1;
                        // Check if offset is out of bounds.
                        if start_line >= all_lines.len() {
                            let offset_display = offset
                                .map(|offset| offset.to_string())
                                .unwrap_or_else(|| "undefined".to_string());
                            return Err(format!(
                                "Offset {} is beyond end of file ({} lines total)",
                                offset_display,
                                all_lines.len()
                            ));
                        }
                        let selected_content: String;
                        let user_limited_lines: Option<usize>;
                        // If limit is specified by the user, honor it first.
                        // Otherwise truncateHead decides.
                        if let Some(limit) = limit {
                            let end_line = start_line
                                .saturating_add(limit as usize)
                                .min(all_lines.len());
                            selected_content = all_lines[start_line..end_line].join("\n");
                            user_limited_lines = Some(end_line - start_line);
                        } else {
                            selected_content = all_lines[start_line..].join("\n");
                            user_limited_lines = None;
                        }
                        // Apply truncation, respecting both line and byte limits.
                        let truncation =
                            truncate_head(&selected_content, TruncationOptions::default());
                        let mut output_text: String;
                        let details: Option<ReadToolDetails>;
                        if truncation.first_line_exceeds_limit {
                            // First line alone exceeds the byte limit. Point the model at a bash fallback.
                            let first_line_size = format_size(all_lines[start_line].len());
                            output_text = format!(
								"[Line {start_line_display} is {first_line_size}, exceeds {} limit. Use bash: sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]",
								format_size(DEFAULT_MAX_BYTES)
							);
                            details = Some(ReadToolDetails {
                                truncation: Some(truncation),
                            });
                        } else if truncation.truncated {
                            // Truncation occurred. Build an actionable continuation notice.
                            let end_line_display = start_line_display + truncation.output_lines - 1;
                            let next_offset = end_line_display + 1;
                            output_text = truncation.content.clone();
                            if truncation.truncated_by == Some(TruncatedBy::Lines) {
                                output_text.push_str(&format!(
									"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. Use offset={next_offset} to continue.]"
								));
                            } else {
                                output_text.push_str(&format!(
									"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} ({} limit). Use offset={next_offset} to continue.]",
									format_size(DEFAULT_MAX_BYTES)
								));
                            }
                            details = Some(ReadToolDetails {
                                truncation: Some(truncation),
                            });
                        } else if let Some(user_limited_lines) = user_limited_lines {
                            if start_line + user_limited_lines < all_lines.len() {
                                // User-specified limit stopped early, but the file still has more content.
                                let remaining = all_lines.len() - (start_line + user_limited_lines);
                                let next_offset = start_line + user_limited_lines + 1;
                                output_text = format!(
									"{}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]",
									truncation.content
								);
                            } else {
                                // No truncation and no remaining user-limited content.
                                output_text = truncation.content.clone();
                            }
                            details = None;
                        } else {
                            // No truncation and no remaining user-limited content.
                            output_text = truncation.content.clone();
                            details = None;
                        }
                        Ok((
                            vec![UserContent::from(TextContent::new(output_text))],
                            details,
                        ))
                    }
                };

                let (content, details) = match signal {
                    Some(signal) => {
                        tokio::select! {
                            result = work => {
                                // Upstream re-checks the abort flag after the last
                                // await, before resolving: an abort that landed in
                                // between rejects.
                                if signal.aborted() {
                                    return Err("Operation aborted".to_string());
                                }
                                result?
                            }
                            () = signal.aborted_future() => return Err("Operation aborted".to_string()),
                        }
                    }
                    None => work.await?,
                };

                let details = match details {
                    Some(details) => serde_json::to_value(details).unwrap_or(JsonValue::Null),
                    None => JsonValue::Null,
                };
                let structured_content = serde_json::to_value(to_read_output(&content)).ok();
                let mut result = AgentToolResult::new(content, details);
                result.structured_content = structured_content;
                Ok(result)
            })
        },
    );
    definition.execute = execute;
    definition
}

/// `createReadTool`.
pub fn create_read_tool(cwd: &str, options: Option<ReadToolOptions>) -> AgentTool {
    wrap_tool_definition(create_read_tool_definition(cwd, options), None)
}

#[cfg(test)]
mod read_output_tests {
    use super::*;

    #[test]
    fn to_read_output_returns_text_for_text_content() {
        let content = vec![UserContent::Text(TextContent::new("file body"))];
        assert_eq!(
            to_read_output(&content),
            ReadToolOutput::Text("file body".to_string())
        );
        assert_eq!(to_read_output(&[]), ReadToolOutput::Text(String::new()));
    }

    #[test]
    fn to_read_output_pairs_the_image_with_its_text_note() {
        let content = vec![
            UserContent::Text(TextContent::new(
                "[Image: original 100x100, displayed at 50x50.]",
            )),
            UserContent::Image(ImageContent::new(
                "Zm9v".to_string(),
                "image/png".to_string(),
            )),
        ];
        assert_eq!(
            to_read_output(&content),
            ReadToolOutput::Image {
                image_type: "image".to_string(),
                data: "Zm9v".to_string(),
                mime_type: "image/png".to_string(),
                note: "[Image: original 100x100, displayed at 50x50.]".to_string(),
            }
        );
    }

    #[test]
    fn output_schema_allows_the_string_and_image_shapes() {
        let schema = read_output_schema();
        let serialized = serde_json::to_value(ReadToolOutput::Text("x".to_string())).unwrap();
        assert!(serialized.is_string());
        let image = serde_json::to_value(ReadToolOutput::Image {
            image_type: "image".to_string(),
            data: "Zm9v".to_string(),
            mime_type: "image/png".to_string(),
            note: "note".to_string(),
        })
        .unwrap();
        assert_eq!(image["type"], "image");
        // Both wire shapes validate against the schema's anyOf arms.
        let arms = schema["anyOf"].as_array().expect("anyOf");
        assert!(arms.iter().any(|arm| arm["type"] == "string"));
        assert!(arms.iter().any(|arm| arm["type"] == "object"));
    }
}
