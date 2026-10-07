//! Port of `packages/durable/src/env/line-scan.ts`.
//!
//! Computes a [`LineScan`] from a file's bytes fed in order, so an environment can scan any size
//! in bounded memory. Decoded sizes use the same streaming decoder as decoding the whole file at
//! once ([`crate::env::decode`]).
//!
//! Deviations:
//! - D1: upstream validates `startLine`/`endLine` as non-negative safe integers (`RangeError`
//!   for negatives, fractions, NaN); the Rust signature `u64, Option<u64>` makes those inputs
//!   unrepresentable, so only `endLine > startLine` is checked. The `Invalid line range` message
//!   (and its `FileError` mapping in `node.rs`) is preserved verbatim.
//! - D2: `encoder.encode(text).length` collapses to `str::len` (see `decode` D2).

use crate::env::decode::{range_decoder, starts_with_bom, TextDecoder};

const NEWLINE: u8 = 0x0a;

/// UTF-8 byte length of decoded text.
fn decoded_bytes(text: &str) -> usize {
	text.len()
}

/// Error thrown by [`LineScanner::new`] for an empty or invalid range (upstream `RangeError`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeError;

impl std::fmt::Display for RangeError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str("Invalid line range")
	}
}

impl std::error::Error for RangeError {}

/// Computes a `LineScan` from a file's bytes fed in order.
pub struct LineScanner {
	start_line: u64,
	end_line: Option<u64>,
	position: u64,
	newlines: u64,
	line_start: u64,
	start: Option<u64>,
	end: Option<u64>,
	first_line_end: Option<u64>,
	last_line_start: Option<u64>,
	selected_bytes: u64,
	first_line_bytes: u64,
	selection: Option<TextDecoder>,
	first_line: Option<TextDecoder>,
	/// The first bytes, held until it is known whether they are a byte-order mark.
	head: Option<Vec<u8>>,
	bom: bool,
}

impl LineScanner {
	/// `startLine` and `endLine` must satisfy `endLine > startLine`; `endLine` absent: to the end.
	pub fn new(start_line: u64, end_line: Option<u64>) -> Result<Self, RangeError> {
		let end_line = end_line.unwrap_or(u64::MAX);
		if !(end_line > start_line) {
			return Err(RangeError);
		}
		let mut scanner = Self {
			start_line,
			end_line: if end_line == u64::MAX { None } else { Some(end_line) },
			position: 0,
			newlines: 0,
			line_start: 0,
			start: None,
			end: None,
			first_line_end: None,
			last_line_start: None,
			selected_bytes: 0,
			first_line_bytes: 0,
			selection: None,
			first_line: None,
			head: Some(Vec::new()),
			bom: false,
		};
		if start_line == 0 {
			scanner.begin(0);
		}
		Ok(scanner)
	}

	pub fn push(&mut self, chunk: &[u8]) {
		let mut chunk = chunk;
		if self.head.is_some() {
			let mut head = self.head.take().unwrap_or_default();
			let take = std::cmp::min(3 - head.len(), chunk.len());
			head.extend_from_slice(&chunk[..take]);
			if head.len() < 3 {
				self.head = Some(head);
				return;
			}
			self.release_head(head);
			chunk = &chunk[take..];
		}
		self.process(chunk);
	}

	fn release_head(&mut self, head: Vec<u8>) {
		self.bom = starts_with_bom(&head);
		self.process(&head);
	}

	fn process(&mut self, chunk: &[u8]) {
		let base = self.position;
		let mut from = 0;
		let mut index = 0;
		while let Some(offset) = find_newline(chunk, index) {
			// The newline ends line `self.newlines`. It belongs to the selection between selected lines only.
			self.feed(chunk, base, from, offset);
			let line = self.newlines;
			let position = base + offset as u64;
			if line == self.start_line {
				self.end_first_line(position);
			}
			if self.line_ends_selection(line) {
				self.end_selection(position);
			}
			self.feed(chunk, base, offset, offset + 1);
			from = offset + 1;
			self.newlines += 1;
			self.line_start = position + 1;
			if self.newlines == self.start_line {
				self.begin(self.line_start);
			}
			if self.selection_starts_last_line() {
				self.last_line_start = Some(self.line_start);
			}
			index = offset + 1;
		}
		self.feed(chunk, base, from, chunk.len());
		self.position += chunk.len() as u64;
	}

	/// Whether newline-terminated line `line` is the last line of the selection.
	fn line_ends_selection(&self, line: u64) -> bool {
		match self.end_line {
			None => false,
			Some(end_line) => line == end_line - 1,
		}
	}

	/// Whether the line about to start (line `self.newlines`) is the last selected line.
	fn selection_starts_last_line(&self) -> bool {
		self.line_ends_selection(self.newlines)
	}

	pub fn finish(mut self) -> LineScan {
		if let Some(head) = self.head.take() {
			self.release_head(head);
		}
		let size = self.position;
		let Some(start) = self.start else {
			return LineScan {
				newlines: self.newlines,
				start: size,
				end: size,
				first_line_end: size,
				last_line_start: size,
				selected_bytes: 0,
				first_line_bytes: 0,
			};
		};
		if self.first_line_end.is_none() {
			self.end_first_line(size);
		}
		if self.end.is_none() {
			self.end_selection(size);
		}
		LineScan {
			newlines: self.newlines,
			start,
			end: self.end.unwrap_or(size),
			first_line_end: self.first_line_end.unwrap_or(size),
			// A selection that reaches past the last line ends with the last line.
			last_line_start: self.last_line_start.unwrap_or(self.line_start),
			selected_bytes: self.selected_bytes,
			first_line_bytes: self.first_line_bytes,
		}
	}

	fn begin(&mut self, start: u64) {
		self.start = Some(start);
		if self.line_ends_selection(self.start_line) {
			self.last_line_start = Some(start);
		}
		self.selection = Some(range_decoder());
		self.first_line = Some(range_decoder());
	}

	fn end_first_line(&mut self, position: u64) {
		self.first_line_end = Some(position);
		if let Some(mut first_line) = self.first_line.take() {
			self.first_line_bytes += decoded_bytes(&first_line.decode(None)) as u64;
		}
	}

	fn end_selection(&mut self, position: u64) {
		self.end = Some(position);
		if let Some(mut selection) = self.selection.take() {
			self.selected_bytes += decoded_bytes(&selection.decode(None)) as u64;
		}
	}

	/// Feed `chunk[from, to)`, which starts at file offset `base`, to the decoders of the ranges still open.
	fn feed(&mut self, chunk: &[u8], base: u64, from: usize, to: usize) {
		let mut from = from;
		// Decoding the whole file drops a leading byte-order mark.
		if self.bom && (base + from as u64) < 3 {
			from = (3 - base).min(to as u64) as usize;
		}
		if to <= from {
			return;
		}
		let bytes = &chunk[from..to];
		if let Some(selection) = self.selection.as_mut() {
			self.selected_bytes += decoded_bytes(&selection.decode(Some(bytes))) as u64;
		}
		if let Some(first_line) = self.first_line.as_mut() {
			self.first_line_bytes += decoded_bytes(&first_line.decode(Some(bytes))) as u64;
		}
	}
}

fn find_newline(chunk: &[u8], from: usize) -> Option<usize> {
	chunk[from..].iter().position(|byte| *byte == NEWLINE).map(|offset| offset + from)
}

/// Where lines of a file are, as `BinaryReader.scanLines` found them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LineScan {
	/// Newline bytes in the whole file; it has `newlines + 1` lines.
	pub newlines: u64,
	/// Byte range of the selected lines: from the start of the first to the end of the last,
	/// without the newline that ends it. A selection past the last line is empty at the end of the file.
	pub start: u64,
	pub end: u64,
	/// Where the first selected line ends: its newline, or the end of the file.
	pub first_line_end: u64,
	/// Where the last selected line starts.
	pub last_line_start: u64,
	/// UTF-8 byte length of the decoded selection and of its first line.
	pub selected_bytes: u64,
	pub first_line_bytes: u64,
}
