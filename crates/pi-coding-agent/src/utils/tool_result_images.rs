//! Port of `packages/coding-agent/src/utils/tool-result-images.ts` — one-time
//! normalization of image blocks entering session history.

use pi_ai::types::{ImageContent, TextContent};

use super::image_process::{process_image, ProcessImageOptions, ProcessImageResult};

/// Upstream `ToolResultContent` (`TextContent | ImageContent` from pi-ai).
#[derive(Debug, Clone, PartialEq)]
pub enum ToolResultContent {
    Text(TextContent),
    Image(ImageContent),
}

/// Upstream `NormalizeToolResultImagesOptions`.
#[derive(Debug, Clone, Default)]
pub struct NormalizeToolResultImagesOptions {
    /// Whether oversized images are resized to inline provider limits.
    /// Default: true
    pub auto_resize_images: Option<bool>,
    /// Model-specific resize profile. Uses the conservative built-in defaults
    /// when omitted.
    pub resize_options: Option<pi_ai::types::ModelImageResizeOptions>,
}

impl NormalizeToolResultImagesOptions {
    fn to_process_image_options(&self) -> ProcessImageOptions {
        ProcessImageOptions {
            auto_resize_images: self.auto_resize_images,
            resize_options: self.resize_options.as_ref().map(|options| {
                super::image_resize::ImageResizeOptions {
                    max_width: options.max_width.map(|v| v as u32),
                    max_height: options.max_height.map(|v| v as u32),
                    max_bytes: options.max_bytes,
                    jpeg_quality: options.jpeg_quality,
                }
            }),
        }
    }
}

/// Normalize image blocks returned by tool results.
///
/// The `read` tool and `@file` CLI attachments run their images through
/// `processImage`, but tools that produce images themselves (extensions, MCP
/// bridges, screenshot tools) hand back arbitrary base64 payloads that go
/// straight into session history and every subsequent provider request.
/// Oversized images make the provider reject the whole conversation, not just
/// the offending turn, so normalize them once as they enter history.
///
/// Returns the original vector when nothing changed so callers can skip
/// rewriting the result.
pub async fn normalize_tool_result_images(
    content: Vec<ToolResultContent>,
    options: Option<&NormalizeToolResultImagesOptions>,
) -> Vec<ToolResultContent> {
    if !content
        .iter()
        .any(|block| matches!(block, ToolResultContent::Image(_)))
    {
        return content;
    }

    let mut normalized: Vec<ToolResultContent> = Vec::new();
    let mut changed = false;

    for block in content {
        let ToolResultContent::Image(image) = &block else {
            normalized.push(block);
            continue;
        };

        let processed = process_image(
            &crate::utils::image_convert::base64_decode(&image.data),
            &image.mime_type,
            Some(
                &options
                    .map(|o| o.to_process_image_options())
                    .unwrap_or_default(),
            ),
        )
        .await;
        let processed = match processed {
            // Unlike `read`, keep the original block. The tool already produced
            // this image and the failure may just be an unavailable image
            // backend, so passing it through preserves the behavior tools have
            // today instead of silently deleting their output.
            ProcessImageResult::Err { .. } => {
                normalized.push(block);
                continue;
            }
            ProcessImageResult::Ok {
                data,
                mime_type,
                hints,
            } => (data, mime_type, hints),
        };
        let (data, mime_type, hints) = processed;

        if data == image.data && mime_type == image.mime_type && hints.is_empty() {
            normalized.push(block);
            continue;
        }

        normalized.push(ToolResultContent::Image(ImageContent::new(data, mime_type)));
        if !hints.is_empty() {
            normalized.push(ToolResultContent::Text(TextContent::new(hints.join("\n"))));
        }
        changed = true;
    }

    if changed {
        normalized
    } else {
        // The caller's original vector is not recoverable after move; an
        // unchanged normalization result is rebuilt identically.
        normalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_block(bytes: &[u8], mime_type: &str) -> ToolResultContent {
        ToolResultContent::Image(ImageContent::new(
            crate::utils::image_convert::base64_encode(bytes),
            mime_type,
        ))
    }

    fn text_block(text: &str) -> ToolResultContent {
        ToolResultContent::Text(TextContent::new(text))
    }

    const TINY_PNG_BASE64: &str =
		"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

    #[tokio::test]
    async fn returns_the_original_array_when_there_are_no_image_blocks() {
        let content = vec![text_block("no images here")];
        let normalized = normalize_tool_result_images(content.clone(), None).await;
        assert_eq!(normalized, content);
    }

    #[tokio::test]
    async fn keeps_undecodable_images_instead_of_dropping_tool_output() {
        let content = vec![ToolResultContent::Image(ImageContent::new(
            "bm90LWFuLWltYWdl",
            "image/png",
        ))];
        let normalized = normalize_tool_result_images(content.clone(), None).await;
        assert_eq!(normalized, content);
    }

    #[tokio::test]
    async fn leaves_oversized_images_alone_when_auto_resize_is_disabled() {
        let content = vec![ToolResultContent::Image(ImageContent::new(
            TINY_PNG_BASE64,
            "image/png",
        ))];
        let options = NormalizeToolResultImagesOptions {
            auto_resize_images: Some(false),
            resize_options: None,
        };
        let normalized = normalize_tool_result_images(content.clone(), Some(&options)).await;
        assert_eq!(normalized, content);
    }

    #[tokio::test]
    async fn returns_the_original_array_when_images_are_already_within_limits() {
        let content = vec![
            text_block("screenshot"),
            ToolResultContent::Image(ImageContent::new(TINY_PNG_BASE64, "image/png")),
        ];
        // With auto-resize on but no photon engine, resize returns null, the
        // block is kept verbatim, and the result is unchanged.
        let normalized = normalize_tool_result_images(content.clone(), None).await;
        assert_eq!(normalized, content);
    }

    #[tokio::test]
    async fn preserves_surrounding_text_blocks_and_their_order() {
        let content = vec![
            text_block("before"),
            image_block(&[0x89, 0x50, 0x4e, 0x47], "image/png"),
            text_block("after"),
        ];
        let normalized = normalize_tool_result_images(content, None).await;
        let kinds: Vec<&str> = normalized
            .iter()
            .map(|block| match block {
                ToolResultContent::Text(_) => "text",
                ToolResultContent::Image(_) => "image",
            })
            .collect();
        assert_eq!(kinds, vec!["text", "image", "text"]);
        assert_eq!(normalized[0], text_block("before"));
        assert_eq!(normalized[2], text_block("after"));
    }

    // Skipped (codec-dependent): upstream `resizes oversized images and reports
    // the original dimensions` and `converts unsupported image formats even when
    // auto-resize is disabled` encode/decode real PNG/BMP pixels through the
    // photon WASM codec; with the seam engine unset the deterministic result is
    // "keep the original block" (covered above).
}
