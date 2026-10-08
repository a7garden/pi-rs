//! Port of `packages/coding-agent/src/utils/image-process.ts` — the image
//! processing pipeline (normalize → optional resize → hints).
//!
//! Conversion of unsupported formats goes through the photon seam
//! ([`super::image_convert`]), so with no engine registered the pipeline
//! takes upstream's "could not be converted" failure path (crate docs,
//! deviation D14).

use super::image_convert::{base64_encode, convert_image_bytes_to_png};
use super::image_resize::{format_dimension_note, resize_image, ImageResizeOptions};

/// Upstream `ProcessImageOptions`.
#[derive(Debug, Clone, Default)]
pub struct ProcessImageOptions {
	/// Whether to resize images to inline provider limits. Default: true
	pub auto_resize_images: Option<bool>,
	/// Optional resize overrides. Uses resizeImage defaults when omitted.
	pub resize_options: Option<ImageResizeOptions>,
}

/// Upstream `ProcessImageResult`.
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessImageResult {
	Ok { data: String, mime_type: String, hints: Vec<String> },
	Err { message: String },
}

struct NormalizedImage {
	bytes: Vec<u8>,
	mime_type: String,
	converted_from: Option<String>,
}

fn base_mime_type(mime_type: &str) -> String {
	mime_type.split(';').next().map(str::trim).unwrap_or(mime_type).to_lowercase()
}

fn normalize_supported_image_mime_type(mime_type: &str) -> Option<&'static str> {
	match base_mime_type(mime_type).as_str() {
		"image/png" => Some("image/png"),
		"image/jpeg" | "image/jpg" => Some("image/jpeg"),
		"image/gif" => Some("image/gif"),
		"image/webp" => Some("image/webp"),
		_ => None,
	}
}

async fn normalize_image(bytes: &[u8], mime_type: &str) -> Option<NormalizedImage> {
	if let Some(normalized_mime_type) = normalize_supported_image_mime_type(mime_type) {
		return Some(NormalizedImage {
			bytes: bytes.to_vec(),
			mime_type: normalized_mime_type.to_string(),
			converted_from: None,
		});
	}

	let png_bytes = convert_image_bytes_to_png(bytes).await?;

	Some(NormalizedImage {
		bytes: png_bytes,
		mime_type: "image/png".to_string(),
		converted_from: Some(base_mime_type(mime_type).to_string()),
	})
}

fn conversion_hint(from: Option<&str>, to: &str) -> Option<String> {
	let from = from?;
	if from == to {
		return None;
	}
	Some(format!("[Image converted from {from} to {to}.]"))
}

/// `processImage` — normalize an image for inline provider use.
pub async fn process_image(bytes: &[u8], mime_type: &str, options: Option<&ProcessImageOptions>) -> ProcessImageResult {
	let auto_resize_images = options.and_then(|o| o.auto_resize_images).unwrap_or(true);
	let Some(normalized) = normalize_image(bytes, mime_type).await else {
		return ProcessImageResult::Err {
			message: "[Image omitted: could not be converted to a supported inline image format.]".to_string(),
		};
	};

	if auto_resize_images {
		let Some(resized) = resize_image(normalized.bytes, normalized.mime_type.clone(), options.and_then(|o| o.resize_options.clone())).await
		else {
			return ProcessImageResult::Err {
				message: "[Image omitted: could not be resized below the inline image size limit.]".to_string(),
			};
		};

		let mut hints: Vec<String> = Vec::new();
		if let Some(converted_hint) = conversion_hint(normalized.converted_from.as_deref(), &resized.mime_type) {
			hints.push(converted_hint);
		}
		if let Some(dimension_note) = format_dimension_note(&resized) {
			hints.push(dimension_note);
		}

		return ProcessImageResult::Ok { data: resized.data, mime_type: resized.mime_type, hints };
	}

	let mut hints: Vec<String> = Vec::new();
	if let Some(converted_hint) = conversion_hint(normalized.converted_from.as_deref(), &normalized.mime_type) {
		hints.push(converted_hint);
	}

	ProcessImageResult::Ok {
		data: base64_encode(&normalized.bytes),
		mime_type: normalized.mime_type,
		hints,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// `createTinyBmp1x1Red24bpp` — 58-byte 24-bit BMP, 1x1 red pixel.
	fn create_tiny_bmp_1x1_red_24bpp() -> Vec<u8> {
		let mut buffer = vec![0u8; 58];
		buffer[0] = b'B';
		buffer[1] = b'M';
		buffer[2..6].copy_from_slice(&58u32.to_le_bytes());
		buffer[10..14].copy_from_slice(&54u32.to_le_bytes());
		buffer[14..18].copy_from_slice(&40u32.to_le_bytes());
		buffer[18..22].copy_from_slice(&1i32.to_le_bytes());
		buffer[22..26].copy_from_slice(&1i32.to_le_bytes());
		buffer[26..28].copy_from_slice(&1u16.to_le_bytes());
		buffer[28..30].copy_from_slice(&24u16.to_le_bytes());
		buffer[34..38].copy_from_slice(&4u32.to_le_bytes());
		buffer[56] = 0xff;
		buffer
	}

	#[test]
	fn detects_the_complete_gif_signatures() {
		assert_eq!(base_mime_type_gate(b"GIF87a"), Some("image/gif"));
		assert_eq!(base_mime_type_gate(b"GIF89a"), Some("image/gif"));
	}

	// `detectSupportedImageMimeType` comes from the shared
	// [`crate::utils::mime`] port of upstream mime.ts.
	fn base_mime_type_gate(bytes: &[u8]) -> Option<&'static str> {
		crate::utils::mime::detect_supported_image_mime_type(bytes)
	}

	#[test]
	fn detects_bmp_files_from_magic_bytes() {
		assert_eq!(base_mime_type_gate(&create_tiny_bmp_1x1_red_24bpp()), Some("image/bmp"));
	}

	#[test]
	fn base_mime_type_strips_parameters() {
		assert_eq!(base_mime_type("Image/PNG ; charset=x"), "image/png");
	}

	#[test]
	fn conversion_hint_matches_upstream() {
		assert_eq!(conversion_hint(Some("image/bmp"), "image/png").as_deref(), Some("[Image converted from image/bmp to image/png.]"));
		assert_eq!(conversion_hint(Some("image/png"), "image/png"), None);
		assert_eq!(conversion_hint(None, "image/png"), None);
	}

	#[tokio::test]
	async fn unsupported_format_is_omitted_without_a_photon_engine() {
		let result = process_image(&create_tiny_bmp_1x1_red_24bpp(), "image/bmp", None).await;
		assert_eq!(
			result,
			ProcessImageResult::Err {
				message: "[Image omitted: could not be converted to a supported inline image format.]".to_string()
			}
		);
	}

	#[tokio::test]
	async fn unresizable_png_is_omitted_without_a_photon_engine() {
		let result = process_image(b"bm90LWFuLWltYWdl", "image/png", None).await;
		assert_eq!(
			result,
			ProcessImageResult::Err {
				message: "[Image omitted: could not be resized below the inline image size limit.]".to_string()
			}
		);
	}

	#[tokio::test]
	async fn supported_image_passes_through_when_auto_resize_is_disabled() {
		let options = ProcessImageOptions { auto_resize_images: Some(false), resize_options: None };
		// Upstream returns base64 of the (normalized) input bytes.
		let result = process_image(b"not-an-image", "image/png", Some(&options)).await;
		assert_eq!(
			result,
			ProcessImageResult::Ok {
				data: "bm90LWFuLWltYWdl".to_string(),
				mime_type: "image/png".to_string(),
				hints: vec![]
			}
		);
	}

	// Skipped (codec-dependent, upstream `converts BMP files to PNG attachments
	// when auto-resize is disabled` and `converts BMP files before auto-resizing`):
	// they assert real BMP→PNG pixels through the photon WASM codec; the seam
	// has no engine registered here, and the deterministic seam-behavior
	// counterpart is `unsupported_format_is_omitted_without_a_photon_engine`.
}
