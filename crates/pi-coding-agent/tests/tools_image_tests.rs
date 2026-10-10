//! Ported cases of `test/tools.test.ts` that require the photon seam
//! (read-tool image processing). They live in a separate integration binary
//! because the photon engine registry is process-global: registering the
//! fixture engine here cannot poison sibling in-module tests that assert
//! no-engine behavior (crate docs D36).

use std::io::Write as _;
use std::path::Path;
use std::sync::LazyLock;

use pi_agent::{AgentTool, AgentToolResult};
use serde_json::json;

use pi_coding_agent::core::tools::read::create_read_tool;
use pi_coding_agent::utils::photon::{PhotonEngine, PhotonImage};

fn first_image_block(result: &AgentToolResult) -> Option<&pi_ai::types::ImageContent> {
    result.content.iter().find_map(|block| match block {
        pi_ai::types::UserContent::Image(image) => Some(image),
        pi_ai::types::UserContent::Text(_) => None,
    })
}
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn png_chunk(tag: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut chunk = Vec::with_capacity(data.len() + 12);
    chunk.extend_from_slice(&(data.len() as u32).to_be_bytes());
    chunk.extend_from_slice(tag);
    chunk.extend_from_slice(data);
    chunk.extend_from_slice(&crc32(&[tag.as_slice(), data].concat()).to_be_bytes());
    chunk
}

/// Encode RGBA pixels as a minimal valid PNG (RGB, 8-bit).
fn encode_png_rgba(width: u32, height: u32, rgba: &[u8]) -> Option<Vec<u8>> {
    if rgba.len() != (width as usize) * (height as usize) * 4 {
        return None;
    }
    let mut raw = Vec::with_capacity((width * height * 3) as usize + height as usize);
    for y in 0..height as usize {
        raw.push(0); // filter: none
        for x in 0..width as usize {
            let base = (y * width as usize + x) * 4;
            raw.extend_from_slice(&rgba[base..base + 3]);
        }
    }
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&raw).ok()?;
    let idat = encoder.finish().ok()?;

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    png.extend_from_slice(&png_chunk(b"IHDR", &ihdr));
    png.extend_from_slice(&png_chunk(b"IDAT", &idat));
    png.extend_from_slice(&png_chunk(b"IEND", &[]));
    Some(png)
}

/// Decode a PNG with 8-bit RGB/RGBA (possibly filtered) into RGBA.
fn decode_png_rgba(bytes: &[u8]) -> Option<PhotonImage> {
    if bytes.len() < 8 || &bytes[..8] != [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return None;
    }
    let mut position = 8;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut idat = Vec::new();
    while position + 8 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[position..position + 4].try_into().ok()?) as usize;
        let tag = &bytes[position + 4..position + 8];
        let data = &bytes[position + 8..position + 8 + length];
        match tag {
            b"IHDR" => {
                width = u32::from_be_bytes(data[0..4].try_into().ok()?);
                height = u32::from_be_bytes(data[4..8].try_into().ok()?);
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        position += 12 + length;
    }
    let mut inflater = flate2::read::ZlibDecoder::new(&idat[..]);
    let mut raw = Vec::new();
    std::io::Read::read_to_end(&mut inflater, &mut raw).ok()?;
    let channels = 3usize; // our encoder emits RGB; upstream fixture PNGs are RGB/RGBA
    let stride = width as usize * channels;
    let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
    let mut previous = vec![0u8; stride];
    for y in 0..height as usize {
        let row_start = y * (stride + 1);
        if row_start >= raw.len() {
            return None;
        }
        let filter = raw[row_start];
        let mut row = raw[row_start + 1..(row_start + 1 + stride).min(raw.len())].to_vec();
        if row.len() < stride {
            return None;
        }
        for x in 0..stride {
            let left = if x >= channels { row[x - channels] } else { 0 };
            let up = previous[x];
            let up_left = if x >= channels {
                previous[x - channels]
            } else {
                0
            };
            row[x] = match filter {
                0 => row[x],
                1 => row[x].wrapping_add(left),
                2 => row[x].wrapping_add(up),
                3 => row[x].wrapping_add((((left as u16) + (up as u16)) / 2) as u8),
                4 => {
                    let prediction = left as i16 + up as i16 - up_left as i16;
                    let pa = (prediction as i16 - left as i16).unsigned_abs();
                    let pb = (prediction as i16 - up as i16).unsigned_abs();
                    let pc = (prediction as i16 - up_left as i16).unsigned_abs();
                    if pa <= pb && pa <= pc {
                        row[x].wrapping_add(left)
                    } else if pb <= pc {
                        row[x].wrapping_add(up)
                    } else {
                        row[x].wrapping_add(up_left)
                    }
                }
                _ => return None,
            };
        }
        for x in 0..width as usize {
            let base = x * 4;
            let source = x * channels;
            rgba[y * width as usize * 4 + base..y * width as usize * 4 + base + 3]
                .copy_from_slice(&row[source..source + 3]);
            rgba[y * width as usize * 4 + base + 3] = 255;
        }
        previous = row;
    }
    Some(PhotonImage::new(width, height, rgba))
}

/// Decode the 24-bit uncompressed BMP fixtures into RGBA.
fn decode_bmp_rgba(bytes: &[u8]) -> Option<PhotonImage> {
    if bytes.len() < 54 || &bytes[0..2] != b"BM" {
        return None;
    }
    let data_offset = u32::from_le_bytes(bytes[10..14].try_into().ok()?) as usize;
    let header_size = u32::from_le_bytes(bytes[14..18].try_into().ok()?) as usize;
    let width = i32::from_le_bytes(bytes[18..22].try_into().ok()?);
    let height = i32::from_le_bytes(bytes[22..26].try_into().ok()?);
    let planes = u16::from_le_bytes(bytes[26..28].try_into().ok()?);
    let bits = u16::from_le_bytes(bytes[28..30].try_into().ok()?);
    if planes != 1 || bits != 24 {
        return None;
    }
    let _ = header_size;
    let width = width.unsigned_abs() as usize;
    let height = height.unsigned_abs() as usize;
    let bottom_up = i32::from_le_bytes(bytes[22..26].try_into().ok()?) > 0;
    let row_size = (width * 3 + 3) / 4 * 4;
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        let source_y = if bottom_up { height - 1 - y } else { y };
        let row_start = data_offset + source_y * row_size;
        if row_start + width * 3 > bytes.len() {
            return None;
        }
        for x in 0..width {
            let base = x * 3;
            let b = bytes[row_start + base];
            let g = bytes[row_start + base + 1];
            let r = bytes[row_start + base + 2];
            let target = (y * width + x) * 4;
            rgba[target] = r;
            rgba[target + 1] = g;
            rgba[target + 2] = b;
            rgba[target + 3] = 255;
        }
    }
    Some(PhotonImage::new(width as u32, height as u32, rgba))
}

/// Minimal [`PhotonEngine`] standing in for the photon WASM module (D36):
/// decodes the fixtures' PNG/BMP and re-encodes as PNG.
struct TestPhotonEngine;

impl PhotonEngine for TestPhotonEngine {
    fn new_from_byteslice(&self, bytes: &[u8]) -> Option<PhotonImage> {
        decode_png_rgba(bytes).or_else(|| decode_bmp_rgba(bytes))
    }

    fn get_bytes(&self, image: &PhotonImage) -> Option<Vec<u8>> {
        encode_png_rgba(image.width, image.height, &image.rgba)
    }

    fn get_bytes_jpeg(&self, _image: &PhotonImage, _quality: u8) -> Option<Vec<u8>> {
        None
    }

    fn resize(&self, image: &PhotonImage, width: u32, height: u32) -> Option<PhotonImage> {
        if width == image.width && height == image.height {
            return Some(image.clone());
        }
        let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
        for y in 0..height as usize {
            let source_y = y * image.height as usize / height as usize;
            for x in 0..width as usize {
                let source_x = x * image.width as usize / width as usize;
                let source = (source_y * image.width as usize + source_x) * 4;
                let target = (y * width as usize + x) * 4;
                rgba[target..target + 4].copy_from_slice(&image.rgba[source..source + 4]);
            }
        }
        Some(PhotonImage::new(width, height, rgba))
    }
}

/// Register the fixture engine once per process (photon is a global seam).
fn ensure_test_photon_engine() {
    static REGISTER: LazyLock<()> = LazyLock::new(|| {
        pi_coding_agent::utils::photon::set_photon_engine(std::sync::Arc::new(TestPhotonEngine));
    });
    LazyLock::force(&REGISTER);
}

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::new();
    for chunk in data.chunks(3) {
        let buffer = (chunk[0] as u32) << 16
            | (chunk.get(1).copied().unwrap_or(0) as u32) << 8
            | chunk.get(2).copied().unwrap_or(0) as u32;
        output.push(TABLE[(buffer >> 18) as usize & 63] as char);
        output.push(TABLE[(buffer >> 12) as usize & 63] as char);
        output.push(if chunk.len() > 1 {
            TABLE[(buffer >> 6) as usize & 63] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            TABLE[buffer as usize & 63] as char
        } else {
            '='
        });
    }
    output
}

#[tokio::test]
async fn read_detects_image_mime_type_from_file_magic_not_extension() {
    ensure_test_photon_engine();
    let dir = test_dir("read-image-magic");
    let png_1x1_base64 =
		"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg==";
    // The upstream fixture PNG is written verbatim (no re-encoding).
    let png_bytes = base64_decode(&png_1x1_base64);
    let test_file = dir.join("image.txt");
    std::fs::write(&test_file, &png_bytes).unwrap();

    let result = execute_tool(
        &read_tool(),
        "test-call-img-1",
        json!({"path": test_file.to_string_lossy()}),
    )
    .await
    .unwrap();

    assert!(matches!(
        result.content.first(),
        Some(pi_ai::types::UserContent::Text(_))
    ));
    assert!(get_text_output(&result).contains("Read image file [image/png]"));

    let image_block = first_image_block(&result).expect("image block present");
    assert_eq!(image_block.mime_type, "image/png");
    assert!(!image_block.data.is_empty());
}

#[tokio::test]
async fn read_reads_bmp_files_from_disk_as_png_image_attachments() {
    ensure_test_photon_engine();
    let dir = test_dir("read-bmp");
    let test_file = dir.join("image.bmp");
    std::fs::write(&test_file, create_tiny_bmp_1x1_red_24bpp()).unwrap();

    let result = execute_tool(
        &read_tool(),
        "test-call-img-bmp",
        json!({"path": test_file.to_string_lossy()}),
    )
    .await
    .unwrap();

    assert!(get_text_output(&result).contains("Read image file [image/png]"));
    assert!(get_text_output(&result).contains("[Image converted from image/bmp to image/png.]"));

    let image_block = first_image_block(&result).expect("image block present");
    assert_eq!(image_block.mime_type, "image/png");
    let decoded = base64_decode(&image_block.data);
    assert_eq!(decoded[0], 0x89);
}

#[tokio::test]
async fn read_treats_files_with_image_extension_but_non_image_content_as_text() {
    let dir = test_dir("read-not-image");
    let test_file = dir.join("not-an-image.png");
    write_file(&test_file, "definitely not a png");

    let result = execute_tool(
        &read_tool(),
        "test-call-img-2",
        json!({"path": test_file.to_string_lossy()}),
    )
    .await
    .unwrap();
    let output = get_text_output(&result);

    assert!(output.contains("definitely not a png"));
    assert!(first_image_block(&result).is_none());
}

// ---------------------------------------------------------------------------
// write tool
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Support shared with the in-module suite (duplicated: separate binary)
// ---------------------------------------------------------------------------

fn get_text_output(result: &AgentToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|block| match block {
            pi_ai::types::UserContent::Text(text) => Some(text.text.clone()),
            pi_ai::types::UserContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn read_tool() -> AgentTool {
    create_read_tool(".", None)
}

async fn execute_tool(
    tool: &AgentTool,
    call_id: &str,
    args: serde_json::Value,
) -> Result<AgentToolResult, String> {
    let map = args.as_object().cloned().expect("args must be an object");
    let execute = std::sync::Arc::clone(&tool.execute);
    execute(call_id.to_string(), map, None, None).await
}

fn test_dir(label: &str) -> std::path::PathBuf {
    static COUNTER: LazyLock<std::sync::atomic::AtomicU64> =
        LazyLock::new(|| std::sync::atomic::AtomicU64::new(0));
    let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("pi-rs-img-{label}-{}-{id}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create test dir");
    dir
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent dir");
    }
    std::fs::write(path, content).expect("write test file");
}

fn create_tiny_bmp_1x1_red_24bpp() -> Vec<u8> {
    let mut buffer = vec![0u8; 58];
    buffer[0] = b'B';
    buffer[1] = b'M';
    let total_size = buffer.len() as u32;
    buffer[2..6].copy_from_slice(&total_size.to_le_bytes());
    buffer[10..14].copy_from_slice(&54u32.to_le_bytes());
    buffer[14..18].copy_from_slice(&40u32.to_le_bytes());
    buffer[18..22].copy_from_slice(&1i32.to_le_bytes());
    buffer[22..26].copy_from_slice(&1i32.to_le_bytes());
    buffer[26..28].copy_from_slice(&1u16.to_le_bytes());
    buffer[28..30].copy_from_slice(&24u16.to_le_bytes());
    buffer[30..34].copy_from_slice(&0u32.to_le_bytes());
    buffer[34..38].copy_from_slice(&4u32.to_le_bytes());
    buffer[56] = 0xff;
    buffer
}

fn base64_decode(input: &str) -> Vec<u8> {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    fn value_of(byte: u8) -> Option<u32> {
        TABLE
            .iter()
            .position(|&c| c == byte)
            .map(|index| index as u32)
    }
    let mut output = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for byte in input.bytes() {
        if byte == b'=' || byte.is_ascii_whitespace() {
            continue;
        }
        let value = value_of(byte).expect("valid base64");
        buffer = (buffer << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((buffer >> bits) as u8);
        }
    }
    output
}
