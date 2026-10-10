//! `cli/file-processor.ts` — process @file CLI arguments into text content
//! and image attachments.

use crate::core::tools::path_utils::resolve_read_path;
use crate::utils::mime::detect_supported_image_mime_type_from_file;
use crate::utils::text::strip_bom;

/// Processed files (`ProcessedFiles`).
#[derive(Debug, Default)]
pub struct ProcessedFiles {
    pub text: String,
    pub images: Vec<pi_ai::types::ImageContent>,
}

/// Options for [`process_file_arguments`] (`ProcessFileOptions`).
#[derive(Debug, Clone, Copy)]
pub struct ProcessFileOptions {
    /// Whether to auto-resize images. Default: `true`.
    pub auto_resize_images: bool,
}

impl Default for ProcessFileOptions {
    fn default() -> Self {
        ProcessFileOptions {
            auto_resize_images: true,
        }
    }
}

/// Process @file arguments into text content and image attachments
/// (`processFileArguments`).
///
/// Upstream prints an error and calls `process.exit(1)` on missing/unreadable
/// files; the port returns [`ProcessFileError`] so the caller decides exit
/// behavior.
pub async fn process_file_arguments(
    file_args: &[String],
    options: Option<ProcessFileOptions>,
) -> Result<ProcessedFiles, ProcessFileError> {
    let auto_resize_images = options.map(|o| o.auto_resize_images).unwrap_or(true);
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut result = ProcessedFiles::default();

    for file_arg in file_args {
        // Expand and resolve path (handles ~ expansion and macOS screenshot
        // Unicode spaces)
        let absolute_path = resolve_read_path(file_arg, &cwd.to_string_lossy());

        // Check if file exists
        if !std::path::Path::new(&absolute_path).exists() {
            return Err(ProcessFileError::NotFound(absolute_path));
        }

        // Check if file is empty
        let metadata = match std::fs::metadata(&absolute_path) {
            Ok(metadata) => metadata,
            Err(error) => {
                return Err(ProcessFileError::Read(
                    absolute_path.clone(),
                    error.to_string(),
                ))
            }
        };
        if metadata.len() == 0 {
            // Skip empty files
            continue;
        }

        let mime_type = detect_supported_image_mime_type_from_file(&absolute_path)
            .await
            .ok()
            .flatten();

        if let Some(mime_type) = mime_type {
            // Handle image file
            let content = match std::fs::read(&absolute_path) {
                Ok(content) => content,
                Err(error) => {
                    return Err(ProcessFileError::Read(
                        absolute_path.clone(),
                        error.to_string(),
                    ))
                }
            };
            let processed = crate::utils::image_process::process_image(
                &content,
                mime_type,
                Some(&crate::utils::image_process::ProcessImageOptions {
                    auto_resize_images: Some(auto_resize_images),
                    resize_options: None,
                }),
            )
            .await;

            match processed {
                crate::utils::image_process::ProcessImageResult::Ok {
                    data,
                    mime_type,
                    hints,
                } => {
                    result
                        .images
                        .push(pi_ai::types::ImageContent { mime_type, data });
                    // Add text reference to image with optional processing hints
                    if hints.is_empty() {
                        result
                            .text
                            .push_str(&format!("<file name=\"{absolute_path}\"></file>\n"));
                    } else {
                        result.text.push_str(&format!(
                            "<file name=\"{absolute_path}\">{}</file>\n",
                            hints.join("\n")
                        ));
                    }
                }
                crate::utils::image_process::ProcessImageResult::Err { message } => {
                    result.text.push_str(&format!(
                        "<file name=\"{absolute_path}\">{message}</file>\n"
                    ));
                }
            }
        } else {
            // Handle text file
            match std::fs::read_to_string(&absolute_path) {
                Ok(content) => {
                    let content = strip_bom(&content);
                    result.text.push_str(&format!(
                        "<file name=\"{absolute_path}\">\n{content}\n</file>\n"
                    ));
                }
                Err(error) => {
                    return Err(ProcessFileError::Read(absolute_path, error.to_string()));
                }
            }
        }
    }

    Ok(result)
}

/// File-processing failure (`process.exit(1)` upstream).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessFileError {
    /// `Error: File not found: <path>`
    NotFound(String),
    /// `Error: Could not read file <path>: <message>`
    Read(String, String),
}

impl std::fmt::Display for ProcessFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessFileError::NotFound(path) => write!(f, "Error: File not found: {path}"),
            ProcessFileError::Read(path, message) => {
                write!(f, "Error: Could not read file {path}: {message}")
            }
        }
    }
}

impl std::error::Error for ProcessFileError {}
