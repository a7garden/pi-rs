//! Port of `packages/coding-agent/src/core/tools/output-accumulator.ts` —
//! incrementally tracks streaming output with bounded memory.
//!
//! Appends decode chunks with a streaming UTF-8 decoder, keeps only a decoded
//! tail for display snapshots, and opens a temp file when the full output
//! needs to be preserved. The Node `TextDecoder({stream:true})` becomes a
//! hand-rolled incremental decoder that holds back incomplete trailing
//! sequences and replaces invalid bytes with U+FFFD, matching the non-fatal
//! default (D34).

use std::io::Write as _;

use pi_agent::BoxFuture;

use crate::core::tools::truncate::{
	truncate_tail, TruncationOptions, TruncationResult, TruncatedBy, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES,
};
use crate::utils::output_files::create_output_file_stream_sync;

/// Constructor options (`OutputAccumulatorOptions`).
#[derive(Debug, Clone, Default)]
pub struct OutputAccumulatorOptions {
	pub max_lines: Option<usize>,
	pub max_bytes: Option<usize>,
	pub temp_file_prefix: Option<String>,
}

/// A display snapshot (`OutputSnapshot`).
#[derive(Debug, Clone)]
pub struct OutputSnapshot {
	pub content: String,
	pub truncation: TruncationResult,
	pub full_output_path: Option<String>,
}

/// The complete output (`FullOutput`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullOutput {
	pub content: String,
	/// Whether `content` omits part of the output.
	pub truncated: bool,
}

/// Incremental counterpart of Node's `TextDecoder({ stream: true })`: holds
/// back an incomplete trailing sequence until more bytes arrive; invalid
/// sequences decode as U+FFFD (non-fatal default).
#[derive(Default)]
struct StreamingUtf8 {
	pending: Vec<u8>,
}

impl StreamingUtf8 {
	/// Length of the incomplete trailing sequence, if the buffer ends inside
	/// one.
	fn trailing_partial_len(bytes: &[u8]) -> usize {
		let len = bytes.len();
		let scan_start = len.saturating_sub(3);
		for start in scan_start..len {
			let byte = bytes[start];
			let sequence_len = if byte & 0x80 == 0 {
				continue; // ASCII: complete.
			} else if byte & 0xe0 == 0xc0 {
				2
			} else if byte & 0xf0 == 0xe0 {
				3
			} else if byte & 0xf8 == 0xf0 {
				4
			} else {
				continue; // Stray continuation byte: handled by lossy decode.
			};
			let available = len - start;
			if available < sequence_len {
				return available;
			}
		}
		0
	}

	fn decode(&mut self, chunk: &[u8]) -> String {
		self.pending.extend_from_slice(chunk);
		let complete = self.pending.len() - Self::trailing_partial_len(&self.pending);
		let text = String::from_utf8_lossy(&self.pending[..complete]).into_owned();
		self.pending.drain(..complete);
		text
	}

	/// Flush any held-back bytes; a trailing partial sequence decodes as
	/// U+FFFD (`decoder.decode()` with no argument).
	fn flush(&mut self) -> String {
		let text = String::from_utf8_lossy(&self.pending).into_owned();
		self.pending.clear();
		text
	}
}

/// Incremental streaming-output tracker (`OutputAccumulator`).
pub struct OutputAccumulator {
	max_lines: usize,
	max_bytes: usize,
	max_rolling_bytes: usize,
	temp_file_prefix: String,
	decoder: StreamingUtf8,

	raw_chunks: Vec<Vec<u8>>,
	tail_text: String,
	tail_bytes: usize,
	tail_starts_at_line_boundary: bool,
	total_raw_bytes: usize,
	total_decoded_bytes: usize,
	completed_lines: usize,
	total_lines: usize,
	current_line_bytes: usize,
	has_open_line: bool,
	finished: bool,

	temp_file_path: Option<String>,
	temp_file: Option<std::fs::File>,
}

impl Default for OutputAccumulator {
	fn default() -> Self {
		Self::new(OutputAccumulatorOptions::default())
	}
}

impl OutputAccumulator {
	pub fn new(options: OutputAccumulatorOptions) -> Self {
		let max_lines = options.max_lines.unwrap_or(DEFAULT_MAX_LINES);
		let max_bytes = options.max_bytes.unwrap_or(DEFAULT_MAX_BYTES);
		let max_rolling_bytes = (max_bytes * 2).max(1);
		OutputAccumulator {
			max_lines,
			max_bytes,
			max_rolling_bytes,
			temp_file_prefix: options.temp_file_prefix.unwrap_or_else(|| "pi-output".to_string()),
			decoder: StreamingUtf8::default(),
			raw_chunks: Vec::new(),
			tail_text: String::new(),
			tail_bytes: 0,
			tail_starts_at_line_boundary: true,
			total_raw_bytes: 0,
			total_decoded_bytes: 0,
			completed_lines: 0,
			total_lines: 0,
			current_line_bytes: 0,
			has_open_line: false,
			finished: false,
			temp_file_path: None,
			temp_file: None,
		}
	}

	/// Append a raw output chunk (`append`).
	///
	/// # Panics
	/// When called after [`finish`](Self::finish) (upstream throws).
	pub fn append(&mut self, data: &[u8]) {
		assert!(!self.finished, "Cannot append to a finished output accumulator");

		self.total_raw_bytes += data.len();
		let text = self.decoder.decode(data);
		self.append_decoded_text(&text);

		if self.temp_file.is_some() || self.should_use_temp_file() {
			self.ensure_temp_file();
			if let Some(file) = self.temp_file.as_mut() {
				let _ = file.write_all(data);
			}
		} else if !data.is_empty() {
			self.raw_chunks.push(data.to_vec());
		}
	}

	/// Flush the decoder (`finish`).
	pub fn finish(&mut self) {
		if self.finished {
			return;
		}
		self.finished = true;
		let text = self.decoder.flush();
		self.append_decoded_text(&text);
		if self.should_use_temp_file() {
			self.ensure_temp_file();
		}
	}

	/// Display snapshot (`snapshot`); `persist_if_truncated` opens the temp
	/// file when the output is truncated.
	pub fn snapshot(&mut self, persist_if_truncated: bool) -> OutputSnapshot {
		let tail_truncation = truncate_tail(
			&self.get_snapshot_text(),
			TruncationOptions {
				max_lines: Some(self.max_lines),
				max_bytes: Some(self.max_bytes),
			},
		);
		let truncated = self.total_lines > self.max_lines || self.total_decoded_bytes > self.max_bytes;
		let truncated_by = if truncated {
			Some(tail_truncation.truncated_by.unwrap_or(if self.total_decoded_bytes > self.max_bytes {
				TruncatedBy::Bytes
			} else {
				TruncatedBy::Lines
			}))
		} else {
			None
		};
		let mut truncation = tail_truncation;
		truncation.truncated = truncated;
		truncation.truncated_by = truncated_by;
		truncation.total_lines = self.total_lines;
		truncation.total_bytes = self.total_decoded_bytes;
		truncation.max_lines = self.max_lines;
		truncation.max_bytes = self.max_bytes;

		if persist_if_truncated && truncation.truncated {
			self.ensure_temp_file();
		}

		OutputSnapshot {
			content: truncation.content.clone(),
			truncation,
			full_output_path: self.temp_file_path.clone(),
		}
	}

	/// Flush and detach the temp file (`closeTempFile`).
	pub fn close_temp_file(&mut self) {
		if let Some(mut file) = self.temp_file.take() {
			let _ = file.flush();
		}
	}

	/// The complete output, for callers that can take more than the display
	/// snapshot. Call after [`finish`](Self::finish) and
	/// [`close_temp_file`](Self::close_temp_file). Output longer than
	/// `max_bytes` raw bytes keeps its first and last `max_bytes / 2` bytes
	/// around an omission marker (`readFullOutput`). Owns clones of the
	/// backing data so the future is `'static` (callers hold the accumulator
	/// behind a sync lock and must not carry a guard across the await).
	pub fn read_full_output(&self, max_bytes: usize) -> BoxFuture<'static, FullOutput> {
		let temp_file_path = self.temp_file_path.clone();
		let raw_chunks = self.raw_chunks.clone();
		Box::pin(async move {
			let Some(temp_file_path) = temp_file_path else {
				let mut bytes = Vec::new();
				for chunk in &raw_chunks {
					bytes.extend_from_slice(chunk);
				}
				return FullOutput {
					content: String::from_utf8_lossy(&bytes).into_owned(),
					truncated: false,
				};
			};
			let bytes = match tokio::fs::read(temp_file_path).await {
				Ok(bytes) => bytes,
				Err(_) => return FullOutput { content: String::new(), truncated: false },
			};
			let size = bytes.len();
			if size <= max_bytes {
				return FullOutput {
					content: String::from_utf8_lossy(&bytes).into_owned(),
					truncated: false,
				};
			}
			let head_bytes = max_bytes / 2;
			let tail_bytes = max_bytes - head_bytes;
			let head = &bytes[..head_bytes];
			let tail = &bytes[size - tail_bytes..];
			// Cut at character boundaries: the head decodes with a streaming
			// decoder (holding back an incomplete trailing sequence), and the
			// tail skips leading continuation bytes.
			let mut decoder = StreamingUtf8::default();
			let head_text = decoder.decode(head) + decoder.flush().as_str();
			let mut tail_start = 0;
			while tail_start < tail.len() && (tail[tail_start] & 0xc0) == 0x80 {
				tail_start += 1;
			}
			let tail_text = String::from_utf8_lossy(&tail[tail_start..]);
			let omitted = size - head_bytes - tail_bytes;
			FullOutput {
				content: format!("{head_text}\n\n[... {omitted} bytes omitted ...]\n\n{tail_text}"),
				truncated: true,
			}
		})
	}

	/// Bytes accumulated for the line currently being streamed
	/// (`getLastLineBytes`).
	pub fn get_last_line_bytes(&self) -> usize {
		self.current_line_bytes
	}

	fn append_decoded_text(&mut self, text: &str) {
		if text.is_empty() {
			return;
		}

		let bytes = text.len();
		self.total_decoded_bytes += bytes;
		self.tail_text.push_str(text);
		self.tail_bytes += bytes;
		if self.tail_bytes > self.max_rolling_bytes * 2 {
			self.trim_tail();
		}

		let mut newlines = 0;
		let mut last_newline = None;
		for (index, byte) in text.bytes().enumerate() {
			if byte == b'\n' {
				newlines += 1;
				last_newline = Some(index);
			}
		}
		if newlines == 0 {
			self.current_line_bytes += bytes;
			self.has_open_line = true;
		} else {
			self.completed_lines += newlines;
			let last = last_newline.unwrap_or(text.len() - 1);
			let tail = &text[last + 1..];
			self.current_line_bytes = tail.len();
			self.has_open_line = !tail.is_empty();
		}
		self.total_lines = self.completed_lines + if self.has_open_line { 1 } else { 0 };
	}

	fn trim_tail(&mut self) {
		let buffer = self.tail_text.as_bytes();
		if buffer.len() <= self.max_rolling_bytes {
			self.tail_bytes = buffer.len();
			return;
		}

		let mut start = buffer.len() - self.max_rolling_bytes;
		while start < buffer.len() && (buffer[start] & 0xc0) == 0x80 {
			start += 1;
		}

		self.tail_starts_at_line_boundary = if start == 0 {
			self.tail_starts_at_line_boundary
		} else {
			buffer[start - 1] == 0x0a
		};
		self.tail_text = String::from_utf8_lossy(&buffer[start..]).into_owned();
		self.tail_bytes = self.tail_text.len();
	}

	fn get_snapshot_text(&self) -> String {
		if self.tail_starts_at_line_boundary {
			return self.tail_text.clone();
		}

		match self.tail_text.find('\n') {
			Some(first_newline) => self.tail_text[first_newline + 1..].to_string(),
			None => self.tail_text.clone(),
		}
	}

	fn should_use_temp_file(&self) -> bool {
		self.total_raw_bytes > self.max_bytes
			|| self.total_decoded_bytes > self.max_bytes
			|| self.total_lines > self.max_lines
	}

	fn ensure_temp_file(&mut self) {
		if self.temp_file.is_some() {
			return;
		}
		// Node's fs.WriteStream buffers synchronously; the sync std file keeps
		// append ordering identical (D34).
		let (path, mut file) = create_output_file_stream_sync(&self.temp_file_prefix, ".log")
			.unwrap_or_else(|error| panic!("failed to create output temp file: {error}"));
		for chunk in &self.raw_chunks {
			let _ = file.write_all(chunk);
		}
		self.raw_chunks = Vec::new();
		self.temp_file_path = Some(path);
		self.temp_file = Some(file);
	}
}
