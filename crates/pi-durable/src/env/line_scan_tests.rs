//! Port of `test/env-line-scan.test.ts`.
//!
//! Deviations:
//! - D1: negative and fractional line numbers are unrepresentable in the Rust `u64` signature,
//!   so the "rejects empty or invalid ranges" test covers the empty range only.

use crate::env::{LineScanner, StreamDecoder};

/// Upstream `random(seed)`: the exact mulberry32 PRNG, so the same corpus is exercised.
fn random(seed: u32) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x6d2b79f5);
        let mut t = state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        (t ^ (t >> 14)) as f64 / 4294967296.0
    }
}

// Newlines, ASCII, a byte-order mark, valid multi-byte sequences, and bytes that form invalid or truncated sequences.
const PIECES: &[&[u8]] = &[
    &[0x0a],
    &[0x0a],
    &[0x61],
    &[0x62, 0x63],
    &[0xef, 0xbb, 0xbf],
    &[0xc3, 0xa9],
    &[0xe2, 0x82, 0xac],
    &[0xf0, 0x9f, 0x98, 0x80],
    &[0xe2, 0x82],
    &[0xff],
    &[0x80],
    &[0xf0, 0x9f],
    &[0x0d, 0x0a],
];

fn random_file(next: &mut impl FnMut() -> f64) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::new();
    let pieces = (next() * 60.0) as usize;
    for _ in 0..pieces {
        bytes.extend_from_slice(PIECES[(next() * PIECES.len() as f64) as usize]);
    }
    bytes
}

fn utf8(text: &str) -> u64 {
    text.len() as u64
}

/// Upstream `new TextDecoder().decode(bytes)` (BOM stripping, replacement semantics): the
/// decoder is streaming, so the call must be ended with `decode(None)` like upstream.
fn decode_all(bytes: &[u8]) -> String {
    let mut decoder = crate::env::decode::TextDecoder::new();
    let text = decoder.decode(Some(bytes));
    text + &decoder.decode(None)
}

/// Upstream `new TextDecoder("utf-8", { ignoreBOM: from > 0 }).decode(...)` on a file slice.
/// JS `subarray` clamps out-of-range bounds; mirror that.
fn decode_range(file: &[u8], from: u64, to: u64) -> String {
    let from = (from as usize).min(file.len());
    let to = (to as usize).clamp(from, file.len());
    let mut decoder = if from > 0 {
        crate::env::range_decoder()
    } else {
        crate::env::decode::TextDecoder::new()
    };
    let text = decoder.decode(Some(&file[from..to]));
    text + &decoder.decode(None)
}

#[test]
fn stream_decoder_keeps_a_byte_order_mark_that_does_not_start_the_stream() {
    // Node's streaming TextDecoder with BOM handling dropped this U+FEFF, which follows an invalid sequence.
    let bytes: &[u8] = &[0xe2, 0x82, 0xef, 0xbb, 0xbf, 0x61];
    let mut decoder = StreamDecoder::new();
    let mut text = String::new();
    for byte in bytes {
        text.push_str(&decoder.decode(Some(&[*byte])));
    }
    text.push_str(&decoder.decode(None));
    assert_eq!(text, decode_all(bytes));
    assert_eq!(text, "\u{FFFD}\u{FEFF}a");
}

#[test]
fn stream_decoder_decodes_any_chunking_like_decoding_the_whole_stream() {
    for seed in 1u32..=5000 {
        let mut next = random(seed);
        let bytes = random_file(&mut next);
        let mut decoder = StreamDecoder::new();
        let mut text = String::new();
        let mut offset = 0usize;
        while offset < bytes.len() {
            let size = 1 + (next() * 7.0) as usize;
            text.push_str(&decoder.decode(Some(&bytes[offset..(offset + size).min(bytes.len())])));
            offset += size;
        }
        text.push_str(&decoder.decode(None));
        assert_eq!(text, decode_all(&bytes), "seed {seed}");
    }
}

#[test]
fn line_scanner_agrees_with_decoding_and_splitting_the_whole_file() {
    for seed in 1u32..=5000 {
        let mut next = random(seed);
        let file = random_file(&mut next);
        let whole = decode_all(&file);
        let lines: Vec<&str> = whole.split('\n').collect();
        let start_line = (next() * (lines.len() as f64 + 2.0)) as u64;
        let end_line = if next() < 0.3 {
            None
        } else {
            Some(start_line + 1 + (next() * (lines.len() as f64 + 1.0)) as u64)
        };
        let mut scanner = LineScanner::new(start_line, end_line).expect("valid range");
        let mut offset = 0usize;
        while offset < file.len() {
            let size = 1 + (next() * 7.0) as usize;
            scanner.push(&file[offset..(offset + size).min(file.len())]);
            offset += size;
        }
        let scan = scanner.finish();
        let message = format!("seed {seed}");
        assert_eq!(scan.newlines, (lines.len() - 1) as u64, "{message}");
        let selected = if end_line.is_none() {
            &lines[(start_line as usize).min(lines.len())..]
        } else {
            let start = (start_line as usize).min(lines.len());
            &lines[start..(end_line.unwrap() as usize).max(start).min(lines.len())]
        };
        let selected_text = selected.join("\n");
        assert_eq!(
            decode_range(&file, scan.start, scan.end),
            selected_text,
            "{message}"
        );
        assert_eq!(scan.selected_bytes, utf8(&selected_text), "{message}");
        if (start_line as usize) < lines.len() {
            assert_eq!(
                decode_range(&file, scan.start, scan.first_line_end),
                lines[start_line as usize],
                "{message}"
            );
            assert_eq!(
                scan.first_line_bytes,
                utf8(lines[start_line as usize]),
                "{message}"
            );
            let last_line = (end_line.map_or(lines.len(), |end| (end as usize).min(lines.len()))
                - 1)
            .max(start_line as usize);
            let last_line_start = (scan.last_line_start as usize).min(file.len());
            let last_end = if last_line + 1 < lines.len() {
                file[last_line_start..]
                    .iter()
                    .position(|byte| *byte == 0x0a)
                    .map(|position| last_line_start as u64 + position as u64)
                    .unwrap()
            } else {
                file.len() as u64
            };
            assert_eq!(
                decode_range(&file, scan.last_line_start, last_end),
                lines[last_line],
                "{message}"
            );
        } else {
            assert_eq!(scan.start, file.len() as u64, "{message}");
            assert_eq!(scan.end, file.len() as u64, "{message}");
            assert_eq!(scan.selected_bytes, 0, "{message}");
        }
    }
}

#[test]
fn line_scanner_rejects_empty_ranges() {
    // D1: negative and fractional starts cannot be expressed in the `u64` signature.
    assert!(LineScanner::new(2, Some(2)).is_err());
}
