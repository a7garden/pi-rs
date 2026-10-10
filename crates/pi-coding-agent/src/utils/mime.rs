//! Port of `packages/coding-agent/src/utils/mime.ts` — sniff the supported
//! image MIME type from magic bytes (JPEG, non-animated PNG, GIF, WebP, BMP).

/// Number of bytes needed to sniff an image type.
const IMAGE_TYPE_SNIFF_BYTES: usize = 4100;

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Detect the supported image MIME type of a byte buffer, or `None` when the
/// bytes are not a supported (non-animated) image.
pub fn detect_supported_image_mime_type(buffer: &[u8]) -> Option<&'static str> {
    if starts_with(buffer, &[0xFF, 0xD8, 0xFF]) {
        return if buffer.get(3) == Some(&0xF7) {
            None
        } else {
            Some("image/jpeg")
        };
    }
    if starts_with(buffer, &PNG_SIGNATURE) {
        return if is_png(buffer) && !is_animated_png(buffer) {
            Some("image/png")
        } else {
            None
        };
    }
    if starts_with_ascii(buffer, 0, "GIF87a") || starts_with_ascii(buffer, 0, "GIF89a") {
        return Some("image/gif");
    }
    if starts_with_ascii(buffer, 0, "RIFF") && starts_with_ascii(buffer, 8, "WEBP") {
        return Some("image/webp");
    }
    if starts_with_ascii(buffer, 0, "BM") && is_bmp(buffer) {
        return Some("image/bmp");
    }
    None
}

/// Sniff the image MIME type of a file by reading its first 4100 bytes.
pub async fn detect_supported_image_mime_type_from_file(
    file_path: &str,
) -> std::io::Result<Option<&'static str>> {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};

    let mut file = tokio::fs::File::open(file_path).await?;
    file.seek(std::io::SeekFrom::Start(0)).await?;
    let mut buffer = vec![0u8; IMAGE_TYPE_SNIFF_BYTES];
    let bytes_read = file.read(&mut buffer).await?;
    Ok(detect_supported_image_mime_type(&buffer[..bytes_read]))
}

fn is_png(buffer: &[u8]) -> bool {
    buffer.len() >= 16
        && read_u32_be(buffer, PNG_SIGNATURE.len()) == 13
        && starts_with_ascii(buffer, 12, "IHDR")
}

fn is_animated_png(buffer: &[u8]) -> bool {
    let mut offset = PNG_SIGNATURE.len();
    while offset + 8 <= buffer.len() {
        let chunk_length = read_u32_be(buffer, offset);
        let chunk_type_offset = offset + 4;
        if starts_with_ascii(buffer, chunk_type_offset, "acTL") {
            return true;
        }
        if starts_with_ascii(buffer, chunk_type_offset, "IDAT") {
            return false;
        }

        let next_offset =
            offset.checked_add(usize::try_from(u64::from(chunk_length) + 12).unwrap_or(usize::MAX));
        let Some(next_offset) = next_offset else {
            return false;
        };
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
        if buffer.len() < 26 {
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

fn read_u16_le(buffer: &[u8], offset: usize) -> u16 {
    let byte = |index: usize| buffer.get(index).copied().unwrap_or(0) as u16;
    byte(offset) | (byte(offset + 1) << 8)
}

fn read_u32_be(buffer: &[u8], offset: usize) -> u32 {
    let byte = |index: usize| buffer.get(index).copied().unwrap_or(0) as u32;
    (byte(offset) << 24) | (byte(offset + 1) << 16) | (byte(offset + 2) << 8) | byte(offset + 3)
}

fn read_u32_le(buffer: &[u8], offset: usize) -> u32 {
    let byte = |index: usize| buffer.get(index).copied().unwrap_or(0) as u32;
    byte(offset) | (byte(offset + 1) << 8) | (byte(offset + 2) << 16) | (byte(offset + 3) << 24)
}

fn starts_with(buffer: &[u8], prefix: &[u8]) -> bool {
    buffer.len() >= prefix.len() && buffer[..prefix.len()] == *prefix
}

fn starts_with_ascii(buffer: &[u8], offset: usize, text: &str) -> bool {
    let bytes = text.as_bytes();
    if buffer.len() < offset + bytes.len() {
        return false;
    }
    buffer[offset..offset + bytes.len()] == *bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32, animated: bool) -> Vec<u8> {
        let mut buffer = PNG_SIGNATURE.to_vec();
        // IHDR chunk: length 13, "IHDR", 13 data bytes, CRC.
        buffer.extend_from_slice(&13u32.to_be_bytes());
        buffer.extend_from_slice(b"IHDR");
        buffer.extend_from_slice(&width.to_be_bytes());
        buffer.extend_from_slice(&height.to_be_bytes());
        buffer.extend_from_slice(&[8, 6, 0, 0, 0]);
        buffer.extend_from_slice(&[0, 0, 0, 0]); // CRC
        if animated {
            // acTL chunk: length 8, "acTL", 8 data bytes, CRC.
            buffer.extend_from_slice(&8u32.to_be_bytes());
            buffer.extend_from_slice(b"acTL");
            buffer.extend_from_slice(&1u32.to_be_bytes());
            buffer.extend_from_slice(&0u32.to_be_bytes());
            buffer.extend_from_slice(&[0, 0, 0, 0]); // CRC
        }
        // IDAT chunk.
        buffer.extend_from_slice(&0u32.to_be_bytes());
        buffer.extend_from_slice(b"IDAT");
        buffer
    }

    #[test]
    fn detects_jpeg() {
        assert_eq!(
            detect_supported_image_mime_type(&[0xFF, 0xD8, 0xFF, 0xE0]),
            Some("image/jpeg")
        );
        assert_eq!(
            detect_supported_image_mime_type(&[0xFF, 0xD8, 0xFF, 0xF7]),
            None
        );
    }

    #[test]
    fn detects_png_variants() {
        assert_eq!(
            detect_supported_image_mime_type(&png(1, 1, false)),
            Some("image/png")
        );
        assert_eq!(detect_supported_image_mime_type(&png(1, 1, true)), None);
    }

    #[test]
    fn detects_gif_webp_bmp() {
        assert_eq!(
            detect_supported_image_mime_type(b"GIF89a...."),
            Some("image/gif")
        );
        assert_eq!(
            detect_supported_image_mime_type(b"GIF87a...."),
            Some("image/gif")
        );

        let mut webp = b"RIFF".to_vec();
        webp.extend_from_slice(&4u32.to_le_bytes());
        webp.extend_from_slice(b"WEBP");
        assert_eq!(detect_supported_image_mime_type(&webp), Some("image/webp"));

        // BMP: 14-byte file header + BITMAPINFOHEADER (40 bytes).
        let mut bmp = Vec::new();
        bmp.extend_from_slice(b"BM");
        bmp.extend_from_slice(&70u32.to_le_bytes());
        bmp.extend_from_slice(&[0, 0, 0, 0]);
        bmp.extend_from_slice(&54u32.to_le_bytes());
        bmp.extend_from_slice(&40u32.to_le_bytes());
        bmp.extend_from_slice(&1u32.to_le_bytes()); // width
        bmp.extend_from_slice(&1u32.to_le_bytes()); // height
        bmp.extend_from_slice(&1u16.to_le_bytes());
        bmp.extend_from_slice(&24u16.to_le_bytes());
        assert_eq!(detect_supported_image_mime_type(&bmp), Some("image/bmp"));
    }

    #[test]
    fn rejects_unknown_bytes() {
        assert_eq!(detect_supported_image_mime_type(b"plain text"), None);
        assert_eq!(detect_supported_image_mime_type(&[]), None);
        assert_eq!(detect_supported_image_mime_type(&[0xFF, 0xD8]), None);
    }
}
