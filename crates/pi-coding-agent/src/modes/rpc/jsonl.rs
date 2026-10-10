//! Port of `packages/coding-agent/src/modes/rpc/jsonl.ts` — strict JSONL
//! framing for the RPC stdin/stdout protocol.
//!
//! Framing is LF-only. Payload strings may contain other Unicode separators
//! such as U+2028 and U+2029; clients must split records on `\n` only. The
//! upstream `StringDecoder`/stream listener pair becomes a push decoder fed
//! byte chunks (Rust stdin is read directly; D92).

/// Serialize a single strict JSONL record.
pub fn serialize_json_line<T: serde::Serialize + ?Sized>(value: &T) -> String {
    let payload = serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
    format!("{payload}\n")
}

/// LF-only JSONL decoder (`attachJsonlLineReader`'s buffer + `StringDecoder`).
/// LF never occurs inside a multi-byte UTF-8 sequence, so raw byte scanning
/// is boundary-safe; `finish` replaces truncated sequences lossily, matching
/// `StringDecoder.end()`.
#[derive(Default)]
pub struct JsonlLineReader {
    buffer: Vec<u8>,
}

impl JsonlLineReader {
    pub fn new() -> Self {
        Self::default()
    }

    /// `onData`: feed one chunk, emitting every complete line.
    pub fn feed(&mut self, chunk: &[u8], on_line: &mut dyn FnMut(String)) {
        self.buffer.extend_from_slice(chunk);
        while let Some(newline_index) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.buffer.drain(..=newline_index).collect();
            emit_line(&line[..newline_index], on_line);
        }
    }

    /// `onEnd`: emit a final line without trailing LF.
    pub fn finish(&mut self, on_line: &mut dyn FnMut(String)) {
        if self.buffer.is_empty() {
            return;
        }
        let remainder = std::mem::take(&mut self.buffer);
        emit_line(&remainder, on_line);
    }
}

fn emit_line(line: &[u8], on_line: &mut dyn FnMut(String)) {
    let mut line = String::from_utf8_lossy(line).into_owned();
    if line.ends_with('\r') {
        line.truncate(line.len() - 1);
    }
    on_line(line);
}

/// Blocking stream loop over a reader (`attachJsonlLineReader`'s production
/// wiring). Returns once the reader is exhausted; each complete line goes to
/// `on_line`.
pub fn read_jsonl_lines<R: std::io::Read>(mut reader: R, mut on_line: impl FnMut(String)) {
    let mut decoder = JsonlLineReader::new();
    let mut chunk = [0u8; 8192];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => {
                let mut forward = |line: String| on_line(line);
                decoder.feed(&chunk[..read], &mut forward);
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
    decoder.finish(&mut on_line);
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    /// `Readable.from` → chunks fed through the decoder.
    fn read_lines(chunks: Vec<&[u8]>) -> Vec<String> {
        let lines: parking_lot::Mutex<Vec<String>> = parking_lot::Mutex::new(Vec::new());
        let sink = &lines;
        let mut reader = JsonlLineReader::new();
        for chunk in chunks {
            reader.feed(chunk, &mut |line| sink.lock().push(line));
        }
        reader.finish(&mut |line| sink.lock().push(line));
        lines.into_inner()
    }

    #[test]
    fn serializes_strict_jsonl_records_without_escaping_unicode_separators() {
        let line = serialize_json_line(&json!({ "text": "a\u{2028}b\u{2029}c" }));

        assert!(line.contains("a\u{2028}b\u{2029}c"));
        assert!(line.ends_with('\n'));
        let parsed: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(parsed, json!({ "text": "a\u{2028}b\u{2029}c" }));
    }

    #[test]
    fn splits_on_lf_only_and_preserves_u2028_u2029_inside_payloads() {
        let payload = serialize_json_line(&json!({ "text": "a\u{2028}b\u{2029}c" }));
        let lines = read_lines(vec![payload.as_bytes()]);

        assert_eq!(lines.len(), 1);
        let parsed: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(parsed, json!({ "text": "a\u{2028}b\u{2029}c" }));
    }

    #[test]
    fn handles_crlf_delimited_input() {
        let lines = read_lines(vec![b"{\"a\":1}\r\n{\"b\":2}\r\n"]);

        assert_eq!(
            lines,
            vec!["{\"a\":1}".to_string(), "{\"b\":2}".to_string()]
        );
    }

    #[test]
    fn emits_a_final_line_without_trailing_lf() {
        let lines = read_lines(vec![b"{\"a\":1}"]);

        assert_eq!(lines, vec!["{\"a\":1}".to_string()]);
    }
}
