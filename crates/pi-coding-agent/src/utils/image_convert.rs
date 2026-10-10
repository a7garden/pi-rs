//! Port of `packages/coding-agent/src/utils/image-convert.ts` — PNG
//! normalization for terminal display.
//!
//! Upstream converts arbitrary image bytes to PNG through photon (which also
//! applies EXIF orientation via [`super::exif_orientation`]). The codec sits
//! behind the [`super::photon::PhotonEngine`] seam, so every function here
//! returns `None`/failure when no engine is registered, exactly like upstream
//! when the WASM module cannot be imported (crate docs, deviation D14).
//! `base64Encode`/`Buffer.from(..., "base64")` are hand-rolled
//! ([`base64_encode`]/[`base64_decode`]) since no base64 crate is available.

use super::exif_orientation::apply_exif_orientation;
use super::photon::load_photon;

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// `Buffer.from(bytes).toString("base64")`.
pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(BASE64_ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(BASE64_ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            BASE64_ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            BASE64_ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn base64_value(byte: u8) -> Option<u32> {
    match byte {
        b'A'..=b'Z' => Some((byte - b'A') as u32),
        b'a'..=b'z' => Some((byte - b'a' + 26) as u32),
        b'0'..=b'9' => Some((byte - b'0' + 52) as u32),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// `Buffer.from(text, "base64")` — stops at the first non-alphabet byte and
/// ignores trailing remainder bits, like node.
pub(crate) fn base64_decode(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for &byte in text.as_bytes() {
        if let Some(value) = base64_value(byte) {
            acc = (acc << 6) | value;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push((acc >> bits) as u8);
            }
        } else {
            break;
        }
    }
    out
}

/// `encodePng` — decode `bytes`, apply EXIF orientation, re-encode as PNG.
pub(crate) fn encode_png(
    engine: &dyn super::photon::PhotonEngine,
    bytes: &[u8],
) -> Option<Vec<u8>> {
    let image = engine.new_from_byteslice(bytes)?;
    let image = apply_exif_orientation(image, bytes);
    engine.get_bytes(&image)
}

/// Upstream `convertImageBytesToPng`.
pub async fn convert_image_bytes_to_png(bytes: &[u8]) -> Option<Vec<u8>> {
    let engine = load_photon().await?;
    encode_png(&*engine, bytes)
}

/// Convert image to PNG format for terminal display.
/// Kitty graphics protocol requires PNG format (f=100) — `convertToPng`.
pub async fn convert_to_png(base64_data: &str, mime_type: &str) -> Option<(String, String)> {
    // Already PNG, no conversion needed
    if mime_type == "image/png" {
        return Some((base64_data.to_string(), mime_type.to_string()));
    }

    let bytes = base64_decode(base64_data);
    let png_bytes = convert_image_bytes_to_png(&bytes).await?;
    Some((base64_encode(&png_bytes), "image/png".to_string()))
}

/// Encode a base64 payload through the PNG pipeline for pi-tui's Kitty image
/// rendering (`loadPngTranscoder`'s closure). Returns `None` if photon cannot
/// be loaded.
pub async fn png_transcoder_encode(base64_data: &str) -> Option<String> {
    let engine = load_photon().await?;
    encode_png(&*engine, &base64_decode(base64_data)).map(|png_bytes| base64_encode(&png_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_node_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(&[0, 255, 16]), "AP8Q");
    }

    #[test]
    fn base64_decodes_round_trip() {
        for input in [&b""[..], b"f", b"fo", b"foo", b"foob", &[0u8, 255, 10]] {
            assert_eq!(base64_decode(&base64_encode(input)), input);
        }
    }

    #[tokio::test]
    async fn convert_to_png_passes_png_through() {
        let data = "aGVsbG8=";
        assert_eq!(
            convert_to_png(data, "image/png").await,
            Some((data.to_string(), "image/png".to_string()))
        );
    }

    #[tokio::test]
    async fn convert_to_png_fails_without_a_photon_engine() {
        assert_eq!(convert_to_png("aGVsbG8=", "image/jpeg").await, None);
    }

    #[tokio::test]
    async fn png_transcoder_is_unavailable_without_a_photon_engine() {
        assert_eq!(png_transcoder_encode("aGVsbG8=").await, None);
    }
}
