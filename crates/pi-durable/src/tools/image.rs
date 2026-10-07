//! Port of `packages/durable/src/tools/image.ts` — image mime-type sniffing over positional reads.

use async_trait::async_trait;

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
/// Bytes every check except the APNG chunk walk needs: BMP reads up to offset 29.
const HEADER_BYTES: usize = 32;
const BLOCK_BYTES: usize = 64 * 1024;

/// Positional reads of a file of `size` bytes (upstream `ByteSource`).
#[async_trait]
pub trait ByteSource: Send + Sync {
	fn size(&self) -> u64;
	async fn read(&self, offset: u64, length: usize) -> Vec<u8>;
}

/// `detect_supported_image_mime_type` of a whole file, reading only its header and, for PNG, the
/// chunk headers up to the first `acTL` or `IDAT`.
pub async fn detect_supported_image_mime_type_of(source: &dyn ByteSource) -> Option<&'static str> {
	let header = source.read(0, HEADER_BYTES).await;
	if !starts_with(&header, &PNG_SIGNATURE) {
		return detect_supported_image_mime_type(&header);
	}
	let is_png = is_png(&header);
	if is_png && !is_animated_png_of(source).await {
		Some("image/png")
	} else {
		None
	}
}

/// `is_animated_png` over a file read in blocks.
async fn is_animated_png_of(source: &dyn ByteSource) -> bool {
	let mut block: Vec<u8> = Vec::new();
	let mut block_start = 0u64;
	let bytes_at = async |offset: u64, length: usize, block: &mut Vec<u8>, block_start: &mut u64| -> Vec<u8> {
		if offset < *block_start || offset + length as u64 > *block_start + block.len() as u64 {
			*block_start = offset;
			*block = source.read(offset, BLOCK_BYTES).await;
		}
		let start = (offset - *block_start) as usize;
		block[start..(start + length).min(block.len())].to_vec()
	};
	let mut offset = PNG_SIGNATURE.len() as u64;
	while offset + 8 <= source.size() {
		let chunk_header = bytes_at(offset, 8, &mut block, &mut block_start).await;
		if starts_with_ascii(&chunk_header, 4, b"acTL") {
			return true;
		}
		if starts_with_ascii(&chunk_header, 4, b"IDAT") {
			return false;
		}
		let chunk_length = read_u32_be(&chunk_header, 0) as u64;
		let next_offset = offset + 8 + chunk_length + 4;
		if next_offset <= offset || next_offset > source.size() {
			return false;
		}
		offset = next_offset;
	}
	false
}

/// Sniff the supported image mime type of a buffer; `None` when it is not a supported image.
pub fn detect_supported_image_mime_type(buffer: &[u8]) -> Option<&'static str> {
	if starts_with(buffer, &[0xff, 0xd8, 0xff]) {
		return if buffer.get(3) == Some(&0xf7) { None } else { Some("image/jpeg") };
	}
	if starts_with(buffer, &PNG_SIGNATURE) {
		return if is_png(buffer) && !is_animated_png(buffer) { Some("image/png") } else { None };
	}
	if starts_with_ascii(buffer, 0, b"GIF87a") || starts_with_ascii(buffer, 0, b"GIF89a") {
		return Some("image/gif");
	}
	if starts_with_ascii(buffer, 0, b"RIFF") && starts_with_ascii(buffer, 8, b"WEBP") {
		return Some("image/webp");
	}
	if starts_with_ascii(buffer, 0, b"BM") && is_bmp(buffer) {
		return Some("image/bmp");
	}
	None
}

fn is_png(buffer: &[u8]) -> bool {
	buffer.len() >= 16 && read_u32_be(buffer, PNG_SIGNATURE.len()) == 13 && starts_with_ascii(buffer, 12, b"IHDR")
}

fn is_animated_png(buffer: &[u8]) -> bool {
	let mut offset = PNG_SIGNATURE.len();
	while offset + 8 <= buffer.len() {
		let chunk_length = read_u32_be(buffer, offset);
		let chunk_type_offset = offset + 4;
		if starts_with_ascii(buffer, chunk_type_offset, b"acTL") {
			return true;
		}
		if starts_with_ascii(buffer, chunk_type_offset, b"IDAT") {
			return false;
		}
		let next_offset = offset + 8 + chunk_length as usize + 4;
		if next_offset <= offset || next_offset > buffer.len() {
			return false;
		}
		offset = next_offset;
	}
	false
}

fn is_bmp(buffer: &[u8]) -> bool {
	if buffer.len() < 26 {
		return false;
	}
	let declared_file_size = read_u32_le(buffer, 2);
	let pixel_data_offset = read_u32_le(buffer, 10);
	let dib_header_size = read_u32_le(buffer, 14);
	if declared_file_size != 0 && declared_file_size < 26 {
		return false;
	}
	if pixel_data_offset < 14 + dib_header_size {
		return false;
	}
	if declared_file_size != 0 && pixel_data_offset >= declared_file_size {
		return false;
	}

	let (color_planes, bits_per_pixel) = if dib_header_size == 12 {
		if buffer.len() < 24 {
			return false;
		}
		(read_u16_le(buffer, 22), read_u16_le(buffer, 24))
	} else if (40..=124).contains(&dib_header_size) {
		if buffer.len() < 30 {
			return false;
		}
		(read_u16_le(buffer, 26), read_u16_le(buffer, 28))
	} else {
		return false;
	};
	color_planes == 1 && matches!(bits_per_pixel, 1 | 4 | 8 | 16 | 24 | 32)
}

fn read_u16_le(buffer: &[u8], offset: usize) -> u32 {
	buffer.get(offset).copied().unwrap_or(0) as u32
		| ((buffer.get(offset + 1).copied().unwrap_or(0) as u32) << 8)
}

fn read_u32_be(buffer: &[u8], offset: usize) -> u32 {
	(buffer.get(offset).copied().unwrap_or(0) as u32) << 24
		| (buffer.get(offset + 1).copied().unwrap_or(0) as u32) << 16
		| (buffer.get(offset + 2).copied().unwrap_or(0) as u32) << 8
		| buffer.get(offset + 3).copied().unwrap_or(0) as u32
}

fn read_u32_le(buffer: &[u8], offset: usize) -> u32 {
	buffer.get(offset).copied().unwrap_or(0) as u32
		| (buffer.get(offset + 1).copied().unwrap_or(0) as u32) << 8
		| (buffer.get(offset + 2).copied().unwrap_or(0) as u32) << 16
		| (buffer.get(offset + 3).copied().unwrap_or(0) as u32) << 24
}

fn starts_with(buffer: &[u8], bytes: &[u8]) -> bool {
	buffer.len() >= bytes.len() && &buffer[..bytes.len()] == bytes
}

fn starts_with_ascii(buffer: &[u8], offset: usize, text: &[u8]) -> bool {
	if buffer.len() < offset + text.len() {
		return false;
	}
	&buffer[offset..offset + text.len()] == text
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn detects_complete_gif_signatures() {
		assert_eq!(detect_supported_image_mime_type(b"GIF87a"), Some("image/gif"));
		assert_eq!(detect_supported_image_mime_type(b"GIF89a"), Some("image/gif"));
	}

	#[test]
	fn detects_supported_types() {
		let mut jpeg = vec![0xff, 0xd8, 0xff, 0xe0];
		jpeg.extend(std::iter::repeat_n(0u8, 40));
		assert_eq!(detect_supported_image_mime_type(&jpeg), Some("image/jpeg"));
		// Progressive JPEG (0xf7) is not supported.
		let mut progressive = vec![0xff, 0xd8, 0xff, 0xf7];
		progressive.extend(std::iter::repeat_n(0u8, 40));
		assert_eq!(detect_supported_image_mime_type(&progressive), None);

		let mut riff = b"RIFF".to_vec();
		riff.extend([0u8, 0, 0, 0]);
		riff.extend(b"WEBP");
		riff.extend(std::iter::repeat_n(0u8, 24));
		assert_eq!(detect_supported_image_mime_type(&riff), Some("image/webp"));

		assert_eq!(detect_supported_image_mime_type(b"BM"), None);
		let mut bmp = b"BM".to_vec();
		bmp.extend(46u32.to_le_bytes()); // file size
		bmp.extend([0u8; 4]); // reserved
		bmp.extend(26u32.to_le_bytes()); // pixel data offset
		bmp.extend(12u32.to_le_bytes()); // DIB header size (BITMAPCOREHEADER)
		bmp.extend(8u16.to_le_bytes()); // width
		bmp.extend(2u16.to_le_bytes()); // height
		bmp.extend(1u16.to_le_bytes()); // color planes
		bmp.extend(24u16.to_le_bytes()); // bits per pixel
		assert_eq!(detect_supported_image_mime_type(&bmp), Some("image/bmp"));

		assert_eq!(detect_supported_image_mime_type(b"plain text file"), None);
		assert_eq!(detect_supported_image_mime_type(&[]), None);
	}

	#[test]
	fn detects_animated_png_by_actl_anywhere() {
		// A minimal still PNG: signature + IHDR chunk.
		let mut still = PNG_SIGNATURE.to_vec();
		still.extend(13u32.to_be_bytes());
		still.extend(b"IHDR");
		still.extend([0u8; 17]);
		assert_eq!(detect_supported_image_mime_type(&still), Some("image/png"));

		let mut animated = still.clone();
		animated.extend(8u32.to_be_bytes());
		animated.extend(b"acTL");
		animated.extend([0u8; 12]);
		assert_eq!(detect_supported_image_mime_type(&animated), None);

		// IDAT before acTL means still.
		let mut with_idat = still.clone();
		with_idat.extend(4u32.to_be_bytes());
		with_idat.extend(b"IDAT");
		with_idat.extend([0u8; 8]);
		assert_eq!(detect_supported_image_mime_type(&with_idat), Some("image/png"));
	}

	#[test]
	fn header_only_sniffer_matches_buffer_sniffer() {
		struct Fixed(Vec<u8>);
		#[async_trait]
		impl ByteSource for Fixed {
			fn size(&self) -> u64 {
				self.0.len() as u64
			}
			async fn read(&self, offset: u64, length: usize) -> Vec<u8> {
				let start = offset as usize;
				self.0[start..(start + length).min(self.0.len())].to_vec()
			}
		}
		let mut animated = PNG_SIGNATURE.to_vec();
		animated.extend(13u32.to_be_bytes());
		animated.extend(b"IHDR");
		animated.extend([0u8; 17]);
		animated.extend(200_000u32.to_be_bytes());
		animated.extend(b"iCCP");
		animated.extend(vec![0u8; 200_004]);
		animated.extend(8u32.to_be_bytes());
		animated.extend(b"acTL");
		animated.extend([0u8; 12]);
		animated.extend(10u32.to_be_bytes());
		animated.extend(b"IDAT");
		animated.extend([0u8; 14]);
		let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
		runtime.block_on(async {
			assert_eq!(detect_supported_image_mime_type_of(&Fixed(animated.clone())).await, None);
			assert_eq!(detect_supported_image_mime_type(&animated), None);
		});
	}
}
