//! Port of `packages/coding-agent/src/utils/image-resize.ts` — the public
//! resize entry point plus the dimension-note formatter.
//!
//! Upstream runs the resize in a Node worker thread so the WASM codec does not
//! block the TUI event loop, falling back to in-process resizing when the
//! worker cannot load. Rust has no same-shape worker here, so
//! [`resize_image`] runs [`super::image_resize_core::resize_image_in_process`]
//! on the blocking thread pool via `tokio::task::spawn_blocking`; the
//! fallback chain and result contract are unchanged (crate docs, deviation
//! D15).

use super::image_resize_core::resize_image_in_process_with_engine;
pub use super::image_resize_core::{ImageResizeOptions, ResizedImage};

/// Resize an image to fit within the specified max dimensions and encoded file
/// size. Runs the codec off the async runtime thread so decoding, resizing,
/// and encoding do not block the executor (`resizeImage`).
pub async fn resize_image(
    input_bytes: Vec<u8>,
    mime_type: String,
    options: Option<ImageResizeOptions>,
) -> Option<ResizedImage> {
    let engine = super::photon::load_photon().await?;
    tokio::task::spawn_blocking(move || {
        resize_image_in_process_with_engine(&*engine, &input_bytes, &mime_type, options.as_ref())
    })
    .await
    .ok()
    .flatten()
}

/// Format a dimension note for resized images.
/// This helps the model understand the coordinate mapping
/// (`formatDimensionNote`).
pub fn format_dimension_note(result: &ResizedImage) -> Option<String> {
    if !result.was_resized {
        return None;
    }

    let scale = result.original_width as f64 / result.width as f64;
    Some(format!(
		"[Image: original {}x{}, displayed at {}x{}. Multiply coordinates by {:.2} to map to original image.]",
		result.original_width, result.original_height, result.width, result.height, scale
	))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resized(was_resized: bool, original_width: u32, width: u32) -> ResizedImage {
        ResizedImage {
            data: String::new(),
            mime_type: "image/png".to_string(),
            original_width,
            original_height: original_width * 2,
            width,
            height: width * 2,
            was_resized,
        }
    }

    #[test]
    fn no_note_when_not_resized() {
        assert_eq!(format_dimension_note(&resized(false, 100, 100)), None);
    }

    #[test]
    fn note_maps_coordinates_to_original() {
        let note = format_dimension_note(&resized(true, 2400, 2000)).unwrap();
        assert_eq!(
			note,
			"[Image: original 2400x4800, displayed at 2000x4000. Multiply coordinates by 1.20 to map to original image.]"
		);
    }

    #[tokio::test]
    async fn resize_image_returns_none_without_a_photon_engine() {
        assert!(resize_image(Vec::new(), "image/png".to_string(), None)
            .await
            .is_none());
    }
}
