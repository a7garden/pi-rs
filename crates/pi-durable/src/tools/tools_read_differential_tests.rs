//! Port of `packages/durable/test/tools-read-differential.test.ts` — the `read` tool must return
//! exactly what a whole-file decode, split, slice, and `truncateHead` returns, while reading only
//! a bounded part of the file. Same seeds, offsets, limits, and byte pieces as upstream.
//!
//! D1: JSON cannot carry `NaN`/`Infinity` args, so the differential loop drives the fallible
//! body `execute_read`'s inner `read_text` with raw `f64` offsets/limits — the same function the
//! JSON path feeds — keeping `NaN`/`1e20` cases from upstream's arg matrix covered.

use std::sync::Arc;

use pi_chord::context::BACKGROUND_CONTEXT;
use serde_json::{json, Value};

use super::image::detect_supported_image_mime_type;
use crate::env::{get_or_throw, FileSystem};
use crate::harness::api::ToolExecutionResult;
use crate::harness::output::character_end;
use crate::tools::read::read_text;
use crate::truncate::{format_size, truncate_head, TruncationResult, DEFAULT_MAX_BYTES};

use super::tools_tests::node_env;

/// The `read` tool before bounded reads, kept as the reference: it decodes the whole file, splits
/// it into lines, and truncates the selection. The tool must give the same result while reading
/// only a bounded part of the file.
async fn reference_read(
    bytes: &[u8],
    path: &str,
    offset: Option<f64>,
    limit: Option<f64>,
) -> Result<Value, String> {
    if let Some(mime_type) = detect_supported_image_mime_type(bytes) {
        return Ok(json!({
            "content": [],
            "isError": true,
            "diagnostics": [{
                "severity": "error",
                "code": "unsupported_image",
                "message": format!("{path} is an image ({mime_type}); reading images is not supported"),
            }],
        }));
    }
    let text_content = String::from_utf8_lossy(bytes).into_owned();
    // JS `new TextDecoder()` strips a leading UTF-8 BOM by default; match it.
    let text_content = text_content
        .strip_prefix('\u{FEFF}')
        .map(str::to_string)
        .unwrap_or(text_content);
    let all_lines: Vec<&str> = text_content.split('\n').collect();
    let total_file_lines = all_lines.len() as f64;
    // JS: `offset ? Math.max(0, offset - 1) : 0` — `0` and `NaN` are falsy.
    let start_line = match offset {
        Some(offset) if offset != 0.0 && !offset.is_nan() => (offset - 1.0).max(0.0),
        _ => 0.0,
    };
    let start_line_display = start_line + 1.0;
    if start_line >= total_file_lines {
        return Err(format!(
            "Offset {} is beyond end of file ({} lines total)",
            offset
                .map(js_number)
                .unwrap_or_else(|| "undefined".to_string()),
            total_file_lines
        ));
    }
    let selected_content = match limit {
        Some(limit) => {
            let end_line = js_min(start_line + limit, total_file_lines);
            let slice = js_slice(&all_lines, start_line, end_line);
            slice
        }
        None => {
            let slice = js_slice(&all_lines, start_line, f64::INFINITY);
            slice
        }
    };
    let user_limited_lines =
        limit.map(|limit| js_min(start_line + limit, total_file_lines) - start_line);
    let TruncationResult {
        content: head_text, ..
    } = truncate_head(&selected_content, Default::default());
    let mut diagnostics: Vec<Value> = Vec::new();
    let mut output_text = head_text;
    let mut details: Option<Value> = None;
    let truncation = truncate_head(&selected_content, Default::default());
    if truncation.first_line_exceeds_limit {
        let line_bytes = all_lines[start_line as usize].as_bytes();
        let end = character_end(line_bytes, DEFAULT_MAX_BYTES);
        output_text = String::from_utf8_lossy(&line_bytes[..end]).into_owned();
        diagnostics.push(json!({
			"severity": "warn",
			"code": "truncated",
			"message": format!(
				"Line {} is {}, exceeds the {} limit; showing its first {}. Use bash: sed -n '{}p' {path} | tail -c +{}",
				js_number(start_line_display),
				format_size(line_bytes.len()),
				format_size(DEFAULT_MAX_BYTES),
				format_size(end),
				js_number(start_line_display),
				end + 1,
			),
		}));
        let mut value = serde_json::to_value(&truncation).expect("truncation serializes");
        if let Value::Object(map) = &mut value {
            map.shift_remove("content");
        }
        value["outputBytes"] = json!(end);
        value["outputLines"] = json!(1);
        details = Some(json!({ "truncation": value }));
    } else if truncation.truncated {
        let end_line_display = start_line_display + truncation.output_lines as f64 - 1.0;
        let next_offset = end_line_display + 1.0;
        let limit_text = if truncation.truncated_by == Some(crate::truncate::TruncationLimit::Lines)
        {
            String::new()
        } else {
            format!(" ({} limit)", format_size(DEFAULT_MAX_BYTES))
        };
        diagnostics.push(json!({
            "severity": "info",
            "code": "truncated",
            "message": format!(
                "Showing lines {}-{} of {}{}. Use offset={} to continue.",
                js_number(start_line_display),
                js_number(end_line_display),
                total_file_lines,
                limit_text,
                js_number(next_offset),
            ),
        }));
        let mut value = serde_json::to_value(&truncation).expect("truncation serializes");
        if let Value::Object(map) = &mut value {
            map.shift_remove("content");
        }
        details = Some(json!({ "truncation": value }));
    } else if let Some(user_limited) = user_limited_lines {
        if start_line + user_limited < total_file_lines {
            let remaining = total_file_lines - (start_line + user_limited);
            let next_offset = start_line + user_limited + 1.0;
            diagnostics.push(json!({
                "severity": "info",
                "message": format!(
                    "{} more lines in file. Use offset={} to continue.",
                    js_number(remaining),
                    js_number(next_offset),
                ),
            }));
        }
    }
    let mut value = json!({
        "content": if output_text.is_empty() { json!([]) } else { json!([{ "type": "text", "text": output_text }]) },
        "diagnostics": diagnostics,
    });
    if let Some(details) = details {
        value["details"] = details;
    }
    Ok(value)
}

/// JS `Math.min` (NaN-propagating).
fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

fn js_number(value: f64) -> String {
    format!("{value}")
}

/// `Array.prototype.slice(start, end)` on line strings, with JS index conversion
/// (`sliceIndex`) and negative-end-from-end semantics.
fn js_slice(lines: &[&str], start: f64, end: f64) -> String {
    let slice_index = |value: f64| if value.is_nan() { 0.0 } else { value.trunc() };
    let start = slice_index(start);
    let relative_end = slice_index(end);
    let count = lines.len() as f64;
    let from = start.clamp(0.0, count) as usize;
    let to = if relative_end < 0.0 {
        (count + relative_end).max(0.0)
    } else {
        relative_end
    }
    .clamp(0.0, count) as usize;
    let to = to.max(from);
    lines[from..to].join("\n")
}

fn random(seed: u32) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x6d2b79f5);
        let mut t = state;
        t = (t ^ (t >> 15)).wrapping_mul(1u32.wrapping_add(t));
        t ^= t.wrapping_add(t << 7);
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

const PIECES: &[&[u8]] = &[
    &[0x0a],
    &[0x0a],
    &[0x61],
    &[0x62, 0x63, 0x64, 0x65],
    &[0xef, 0xbb, 0xbf],
    &[0xc3, 0xa9],
    &[0xe2, 0x82, 0xac],
    &[0xf0, 0x9f, 0x98, 0x80],
    &[0xe2, 0x82],
    &[0xff],
    &[0x0d, 0x0a],
];

/// Mostly small files, some over the line limit, some over the byte limit, some with one huge line.
fn random_file(next: &mut impl FnMut() -> f64) -> Vec<u8> {
    let kind = next();
    if kind < 0.1 {
        let mut bytes = vec![0xefu8, 0xbb, 0xbf];
        bytes.extend("x\n".repeat(2100).into_bytes());
        return bytes;
    }
    if kind < 0.2 {
        let mut line = vec![0x61u8; DEFAULT_MAX_BYTES + 200 + (next() * 400.0) as usize];
        line[DEFAULT_MAX_BYTES - 1 + (next() * 3.0) as usize] = 0xc3;
        line.push(0x0a);
        line.push(0x62);
        return line;
    }
    if kind < 0.3 {
        return format!("{}\n", "0123456789".repeat(30))
            .repeat(200)
            .into_bytes();
    }
    let mut bytes: Vec<u8> = Vec::new();
    let pieces = (next() * 80.0) as usize;
    for _ in 0..pieces {
        bytes.extend_from_slice(PIECES[(next() * PIECES.len() as f64) as usize]);
    }
    bytes
}

const OFFSETS: [Option<f64>; 12] = [
    None,
    Some(0.0),
    Some(1.0),
    Some(2.0),
    Some(3.0),
    Some(2.5),
    Some(-4.0),
    Some(50.0),
    Some(2001.0),
    Some(3000.0),
    Some(1e20),
    Some(f64::NAN),
];
const LIMITS: [Option<f64>; 12] = [
    None,
    Some(0.0),
    Some(1.0),
    Some(2.0),
    Some(-3.0),
    Some(-1e20),
    Some(1.5),
    Some(7.0),
    Some(2000.0),
    Some(2500.0),
    Some(1e20),
    Some(f64::NAN),
];

async fn outcome_value(run: impl std::future::Future<Output = Result<Value, String>>) -> Value {
    match run.await {
        Ok(value) => value,
        Err(error) => json!({ "error": error }),
    }
}

/// Normalizes a `ToolExecutionResult` to the reference's JSON shape (key order mirrors it;
/// object equality is key-based anyway).
fn result_value(result: &ToolExecutionResult) -> Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "content".to_string(),
        result.content.clone().unwrap_or_else(|| json!([])),
    );
    if let Some(is_error) = result.is_error {
        map.insert("isError".to_string(), json!(is_error));
    }
    map.insert(
        "diagnostics".to_string(),
        json!(result.diagnostics.clone().unwrap_or_default()),
    );
    if let Some(details) = &result.details {
        map.insert("details".to_string(), details.clone());
    }
    Value::Object(map)
}

#[tokio::test(flavor = "multi_thread")]
async fn read_returns_exactly_what_reading_the_whole_file_returned() {
    let dir = std::env::temp_dir().join(format!(
        "pi-durable-read-{}-{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos(),
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let env = Arc::new(node_env(dir.to_string_lossy().into_owned()));
    for seed in 1u32..=400 {
        let mut next = random(seed);
        let file = random_file(&mut next);
        std::fs::write(dir.join("f.txt"), &file).expect("write fixture");
        for _ in 0..4 {
            let offset = OFFSETS[(next() * OFFSETS.len() as f64) as usize];
            let limit = LIMITS[(next() * LIMITS.len() as f64) as usize];
            let actual = outcome_value(async {
                let reader = get_or_throw(
                    env.open_binary_reader("f.txt", None, &BACKGROUND_CONTEXT)
                        .await,
                );
                let info = get_or_throw(reader.info(&BACKGROUND_CONTEXT).await);
                read_text(
                    reader.as_ref(),
                    &info,
                    "f.txt",
                    offset,
                    limit,
                    &BACKGROUND_CONTEXT,
                )
                .await
                .map(|result| result_value(&result))
            });
            let expected = outcome_value(reference_read(&file, "f.txt", offset, limit));
            let (actual, expected) = tokio::join!(actual, expected);
            assert_eq!(
                actual, expected,
                "seed {seed} offset {offset:?} limit {limit:?}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn read_detects_animated_pngs_whose_actl_chunk_lies_far_beyond_the_header() {
    let dir = std::env::temp_dir().join(format!(
        "pi-durable-read-{}-{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos(),
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let chunk = |chunk_type: &str, length: usize| -> Vec<u8> {
        let mut bytes = vec![
            ((length >> 24) & 0xff) as u8,
            ((length >> 16) & 0xff) as u8,
            ((length >> 8) & 0xff) as u8,
            (length & 0xff) as u8,
        ];
        bytes.extend_from_slice(chunk_type.as_bytes());
        bytes.extend(std::iter::repeat(0u8).take(length + 4));
        bytes
    };
    let mut png = vec![0x89u8, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    png.extend(chunk("IHDR", 13));
    png.extend(chunk("iCCP", 200_000));
    png.extend(chunk("acTL", 8));
    png.extend(chunk("IDAT", 10));
    std::fs::write(dir.join("a.png"), &png).expect("write fixture");
    let env = Arc::new(node_env(dir.to_string_lossy().into_owned()));
    let actual = outcome_value(async {
        let reader = get_or_throw(
            env.open_binary_reader("a.png", None, &BACKGROUND_CONTEXT)
                .await,
        );
        let info = get_or_throw(reader.info(&BACKGROUND_CONTEXT).await);
        read_text(
            reader.as_ref(),
            &info,
            "a.png",
            None,
            None,
            &BACKGROUND_CONTEXT,
        )
        .await
        .map(|result| result_value(&result))
    });
    let expected = outcome_value(reference_read(&png, "a.png", None, None));
    let (actual, expected) = tokio::join!(actual, expected);
    assert_eq!(actual, expected);
}
