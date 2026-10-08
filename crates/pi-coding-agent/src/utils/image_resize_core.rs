//! Port of `packages/coding-agent/src/utils/image-resize-core.ts` — in-process
//! image resize driven by the [`super::photon::PhotonEngine`] seam (upstream:
//! photon WASM). Returns `None` when no engine is registered or processing
//! fails, mirroring upstream's `loadPhoton() → null` path (crate docs,
//! deviation D14).

use super::exif_orientation::apply_exif_orientation;
use super::image_convert::base64_encode;
use super::photon::load_photon;

/// Upstream `ImageResizeOptions`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageResizeOptions {
	pub max_width: Option<u32>,
	pub max_height: Option<u32>,
	pub max_bytes: Option<u64>,
	pub jpeg_quality: Option<u8>,
}

/// Upstream `ResizedImage`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResizedImage {
	pub data: String, // base64
	pub mime_type: String,
	pub original_width: u32,
	pub original_height: u32,
	pub width: u32,
	pub height: u32,
	pub was_resized: bool,
}

// 4.5MB of base64 payload. Provides headroom below Anthropic's 5MB limit.
const DEFAULT_MAX_BYTES: u64 = 4 * 1024 * 1024 + 512 * 1024;

struct RequiredOptions {
	max_width: u32,
	max_height: u32,
	max_bytes: u64,
	jpeg_quality: u8,
}

fn resolve_options(options: Option<&ImageResizeOptions>) -> RequiredOptions {
	let options = options.cloned().unwrap_or_default();
	RequiredOptions {
		max_width: options.max_width.unwrap_or(2000),
		max_height: options.max_height.unwrap_or(2000),
		max_bytes: options.max_bytes.unwrap_or(DEFAULT_MAX_BYTES),
		jpeg_quality: options.jpeg_quality.unwrap_or(80),
	}
}

struct EncodedCandidate {
	data: String,
	encoded_size: u64,
	mime_type: &'static str,
}

fn encode_candidate(buffer: &[u8], mime_type: &'static str) -> EncodedCandidate {
	let data = base64_encode(buffer);
	EncodedCandidate { encoded_size: data.len() as u64, data, mime_type }
}

/// Resize an image to fit within the specified max dimensions and encoded file
/// size. Returns `None` if the image cannot be resized below `max_bytes`.
///
/// Uses Photon (Rust/WASM upstream) for image processing. If Photon is not
/// available, returns `None`.
///
/// Strategy for staying under maxBytes:
/// 1. First resize to maxWidth/maxHeight
/// 2. Try both PNG and JPEG formats, pick the smaller one
/// 3. If still too large, try JPEG with decreasing quality
/// 4. If still too large, progressively reduce dimensions until 1x1
pub async fn resize_image_in_process(
	input_bytes: &[u8],
	mime_type: &str,
	options: Option<&ImageResizeOptions>,
) -> Option<ResizedImage> {
	resize_image_in_process_with_engine(&*load_photon().await?, input_bytes, mime_type, options)
}

/// Synchronous core, split out so worker/thread-pool callers can drive it
/// without an async runtime.
pub fn resize_image_in_process_with_engine(
	engine: &dyn super::photon::PhotonEngine,
	input_bytes: &[u8],
	mime_type: &str,
	options: Option<&ImageResizeOptions>,
) -> Option<ResizedImage> {
	let opts = resolve_options(options);
	let input_base64_size = (input_bytes.len().div_ceil(3) * 4) as u64;

	let raw_image = engine.new_from_byteslice(input_bytes)?;
	let image = apply_exif_orientation(raw_image, input_bytes);

	let original_width = image.get_width();
	let original_height = image.get_height();
	let format = mime_type.split('/').nth(1).unwrap_or("png");

	// Check if already within all limits (dimensions AND encoded size)
	if original_width <= opts.max_width && original_height <= opts.max_height && input_base64_size < opts.max_bytes {
		return Some(ResizedImage {
			data: base64_encode(input_bytes),
			mime_type: if mime_type.is_empty() { format!("image/{format}") } else { mime_type.to_string() },
			original_width,
			original_height,
			width: original_width,
			height: original_height,
			was_resized: false,
		});
	}

	// Calculate initial dimensions respecting max limits
	let mut target_width = original_width;
	let mut target_height = original_height;

	if target_width > opts.max_width {
		target_height = ((target_height as u64 * opts.max_width as u64 + target_width as u64 / 2) / target_width as u64) as u32;
		target_width = opts.max_width;
	}
	if target_height > opts.max_height {
		target_width = ((target_width as u64 * opts.max_height as u64 + target_height as u64 / 2) / target_height as u64) as u32;
		target_height = opts.max_height;
	}

	let try_encodings = |width: u32, height: u32, jpeg_qualities: &[u8]| -> Option<Vec<EncodedCandidate>> {
		let resized = engine.resize(&image, width, height)?;
		let mut candidates = vec![];
		if let Some(png) = engine.get_bytes(&resized) {
			candidates.push(encode_candidate(&png, "image/png"));
		}
		for quality in jpeg_qualities {
			if let Some(jpeg) = engine.get_bytes_jpeg(&resized, *quality) {
				candidates.push(encode_candidate(&jpeg, "image/jpeg"));
			}
		}
		Some(candidates)
	};

	// Array.from(new Set([opts.jpegQuality, 85, 70, 55, 40])) — insertion
	// order, first occurrence wins.
	let mut quality_steps: Vec<u8> = vec![opts.jpeg_quality, 85, 70, 55, 40];
	let mut seen = Vec::new();
	quality_steps.retain(|quality| {
		if seen.contains(quality) {
			false
		} else {
			seen.push(*quality);
			true
		}
	});

	let mut current_width = target_width;
	let mut current_height = target_height;

	loop {
		let candidates = try_encodings(current_width, current_height, &quality_steps)?;
		for candidate in candidates {
			if candidate.encoded_size < opts.max_bytes {
				return Some(ResizedImage {
					data: candidate.data,
					mime_type: candidate.mime_type.to_string(),
					original_width,
					original_height,
					width: current_width,
					height: current_height,
					was_resized: true,
				});
			}
		}

		if current_width == 1 && current_height == 1 {
			break;
		}

		let next_width = if current_width == 1 { 1 } else { (current_width * 3 / 4).max(1) };
		let next_height = if current_height == 1 { 1 } else { (current_height * 3 / 4).max(1) };
		if next_width == current_width && next_height == current_height {
			break;
		}

		current_width = next_width;
		current_height = next_height;
	}

	None
}

#[cfg(test)]
mod tests {
	use super::*;

	#[tokio::test]
	async fn returns_none_without_a_photon_engine() {
		assert!(resize_image_in_process(&[], "image/png", None).await.is_none());
	}

	#[test]
	fn default_options_match_upstream() {
		let resolved = resolve_options(None);
		assert_eq!(
			(resolved.max_width, resolved.max_height, resolved.max_bytes, resolved.jpeg_quality),
			(2000, 2000, DEFAULT_MAX_BYTES, 80)
		);
	}

	#[test]
	fn option_overrides_apply() {
		let resolved = resolve_options(Some(&ImageResizeOptions {
			max_width: Some(100),
			max_height: None,
			max_bytes: Some(10),
			jpeg_quality: Some(60),
		}));
		assert_eq!(
			(resolved.max_width, resolved.max_height, resolved.max_bytes, resolved.jpeg_quality),
			(100, 2000, 10, 60)
		);
	}

	#[test]
	fn quality_steps_preserve_upstream_order_and_dedup() {
		// Upstream: Array.from(new Set([jpegQuality, 85, 70, 55, 40])).
		let mut steps = vec![55u8, 85, 70, 55, 40];
		let mut seen = vec![];
		steps.retain(|q| if seen.contains(q) { false } else { seen.push(*q); true });
		assert_eq!(steps, vec![55, 85, 70, 40]);
	}
}
