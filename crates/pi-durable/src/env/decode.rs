//! Port of `packages/durable/src/env/decode.ts`.
//!
//! Decoding that matches `new TextDecoder().decode(bytes)` of the whole input when the input
//! arrives in pieces. WHATWG UTF-8 replacement semantics (invalid byte → one U+FFFD per maximal
//! invalid subpart) are reproduced with `std::str::from_utf8`: valid prefixes decode, incomplete
//! trailing sequences are held back for the next chunk, invalid bytes become U+FFFD immediately.
//!
//! Deviations:
//! - D1: `ignoreBOM: true` decoders are modeled by [`TextDecoder::with_ignore_bom`]; upstream
//!   reaches the same state by turning BOM handling off and dropping a leading mark itself.
//! - D2: `TextEncoder.encode(text).length` collapses to `str::len` (Rust strings are UTF-8);
//!   the JS UTF-16 length artifact has identical byte results.

use std::str;

/// A streaming UTF-8 decoder; `decode(None)` ends the stream. With BOM handling on (the
/// default), a leading byte-order mark is dropped from the stream's first output, like
/// `new TextDecoder()`. Callers that start at the beginning of a file and drop a leading mark
/// themselves use [`TextDecoder::with_ignore_bom`] (upstream `rangeDecoder`).
pub struct TextDecoder {
    ignore_bom: bool,
    /// Bytes of an incomplete trailing sequence, held back until more input arrives.
    pending: Vec<u8>,
    started: bool,
}

impl Default for TextDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TextDecoder {
    /// Upstream `new TextDecoder("utf-8")`: strips one leading mark from the stream.
    pub fn new() -> Self {
        Self {
            ignore_bom: false,
            pending: Vec::new(),
            started: false,
        }
    }

    /// Upstream `new TextDecoder("utf-8", { ignoreBOM: true })`: keeps any U+FEFF.
    pub fn with_ignore_bom() -> Self {
        Self {
            ignore_bom: true,
            pending: Vec::new(),
            started: false,
        }
    }

    /// Text for `bytes`, holding back an incomplete character; `None` ends the stream.
    pub fn decode(&mut self, bytes: Option<&[u8]>) -> String {
        let mut input: Vec<u8> = std::mem::take(&mut self.pending);
        match bytes {
            Some(bytes) => input.extend_from_slice(bytes),
            None => {}
        }
        let ended = bytes.is_none();
        let mut text = String::new();
        let mut position = 0;
        while position < input.len() {
            match str::from_utf8(&input[position..]) {
                Ok(valid) => {
                    text.push_str(valid);
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    // SAFETY-free variant: slice is valid UTF-8 per the error's own bound.
                    text.push_str(
                        str::from_utf8(&input[position..position + valid_up_to])
                            .unwrap_or_default(),
                    );
                    position += valid_up_to;
                    match error.error_len() {
                        // Incomplete sequence at the end: hold it back, or emit U+FFFD at the end of the stream.
                        None => {
                            if ended {
                                text.push('\u{FFFD}');
                            } else {
                                self.pending = input[position..].to_vec();
                            }
                            break;
                        }
                        // Invalid byte(s): WHATWG replaces the maximal invalid subpart with one U+FFFD.
                        Some(invalid_len) => {
                            text.push('\u{FFFD}');
                            position += invalid_len;
                        }
                    }
                }
            }
        }
        if !self.started && !text.is_empty() {
            self.started = true;
            if !self.ignore_bom && text.starts_with('\u{FEFF}') {
                return text[3..].to_string();
            }
        }
        text
    }
}

/// Whether decoding the whole input drops its first three bytes as a byte-order mark.
pub fn starts_with_bom(first_bytes: &[u8]) -> bool {
    first_bytes.first() == Some(&0xef)
        && first_bytes.get(1) == Some(&0xbb)
        && first_bytes.get(2) == Some(&0xbf)
}

/// Upstream `rangeDecoder`: a decoder that never strips a mark (callers drop it themselves).
pub fn range_decoder() -> TextDecoder {
    TextDecoder::with_ignore_bom()
}

/// Decodes one stream chunk by chunk exactly like decoding all of it at once, dropping a
/// leading mark (a leading U+FEFF is exactly a byte-order mark at the stream's start, since
/// U+FEFF encodes only as EF BB BF).
pub struct StreamDecoder {
    decoder: TextDecoder,
    started: bool,
}

impl Default for StreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self {
            decoder: range_decoder(),
            started: false,
        }
    }

    /// Text for `bytes`, holding back an incomplete character; without `bytes`, the end of the stream.
    pub fn decode(&mut self, bytes: Option<&[u8]>) -> String {
        let text = self.decoder.decode(bytes);
        if self.started || text.is_empty() {
            return text;
        }
        self.started = true;
        text.strip_prefix('\u{FEFF}')
            .map(str::to_string)
            .unwrap_or(text)
    }
}
