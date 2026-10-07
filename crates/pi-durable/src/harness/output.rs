//! Port of `packages/durable/src/harness/output.ts`.
//!
//! Deviations: D1 — `Progress` uses a tokio task + notify machinery instead
//! of `setTimeout`; `stop()` returns the waiters the final commit must
//! settle as tokio oneshot senders (upstream `PromiseWithResolvers`).
//! D2 — `OutputBuffer::push` takes `&str` or bytes via an enum (`Chunk`)
//! since Rust has no union parameter; incomplete-character carry uses a
//! manual UTF-8 incremental decoder.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::Arc;

use pi_chord::context::Context;
use tokio::sync::{Mutex, Notify, oneshot};

use crate::env::ShellOutputSkip;
use crate::truncate::utf8_byte_length;

/// Retention limits of one tool's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputLimits {
	pub max_bytes: usize,
	pub max_lines: usize,
	pub retain: Retain,
}

/// Which end of the stream retention keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retain {
	Head,
	Tail,
}

/// Retained output and what the limits dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedOutput {
	pub text: String,
	pub dropped_bytes: usize,
	pub dropped_lines: usize,
}

/// An exact slice of the input within the limits, and what it left out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputSlice {
	pub text: String,
	pub bytes: usize,
	pub dropped_bytes: usize,
	pub dropped_lines: usize,
}

const NEWLINE: u8 = 0x0a;

fn is_invalid_output(c: char) -> bool {
	matches!(c, '\u{0}'..='\u{8}' | '\u{b}'..='\u{1f}' | '\u{fff9}'..='\u{fffb}')
}

/// Remove control characters that break display and transcripts; tabs and
/// newlines stay.
pub fn sanitize_output(text: &str) -> String {
	text.chars().filter(|c| !is_invalid_output(*c)).collect()
}

/// Bound `text` to whole lines within the limits: the first lines for `head`,
/// the last lines for `tail`. The result is an exact slice, trailing newline
/// included. A single line longer than `max_bytes` is cut at the byte limit
/// on a character boundary.
pub fn bound_output(text: &str, limits: OutputLimits) -> OutputSlice {
	let bytes = text.as_bytes();
	let (from, to) = match limits.retain {
		Retain::Head => head_range(bytes, limits),
		Retain::Tail => tail_range(bytes, limits),
	};
	let kept = &bytes[from..to];
	OutputSlice {
		text: if kept.len() == bytes.len() { text.to_string() } else { String::from_utf8_lossy(kept).into_owned() },
		bytes: kept.len(),
		dropped_bytes: bytes.len() - kept.len(),
		dropped_lines: line_count(bytes) - line_count(kept),
	}
}

fn head_range(bytes: &[u8], limits: OutputLimits) -> (usize, usize) {
	if limits.max_lines == 0 || limits.max_bytes == 0 {
		return (0, 0);
	}
	let mut end = bytes.len();
	let mut lines = 0usize;
	for (index, b) in bytes.iter().enumerate() {
		if *b == NEWLINE {
			lines += 1;
			if lines == limits.max_lines {
				end = index + 1;
				break;
			}
		}
	}
	if end > limits.max_bytes {
		let newline = bytes[..limits.max_bytes].iter().rposition(|b| *b == NEWLINE);
		end = match newline {
			Some(at) => at + 1,
			None => character_end(bytes, limits.max_bytes),
		};
	}
	(0, end)
}

fn tail_range(bytes: &[u8], limits: OutputLimits) -> (usize, usize) {
	if bytes.is_empty() || limits.max_lines == 0 || limits.max_bytes == 0 {
		return (bytes.len(), bytes.len());
	}
	// A trailing newline ends the last line rather than starting another.
	let len = bytes.len() as isize;
	let last = len - if bytes[len as usize - 1] == NEWLINE { 2 } else { 1 };
	let mut start = 0usize;
	let mut lines = 1usize;
	let mut index = if last < 0 { None } else { bytes[..=last as usize].iter().rposition(|b| *b == NEWLINE) };
	while let Some(at) = index {
		if lines == limits.max_lines {
			start = at + 1;
			break;
		}
		lines += 1;
		index = if at == 0 { None } else { bytes[..at].iter().rposition(|b| *b == NEWLINE) };
	}
	if bytes.len() - start > limits.max_bytes {
		let from = bytes.len() - limits.max_bytes;
		// The first line starting inside the byte window, or a cut of the last
		// line when it alone is too long.
		let newline = if from >= 1 { bytes[from - 1..].iter().position(|b| *b == NEWLINE).map(|at| at + from - 1) } else { None };
		start = match newline {
			Some(at) if at + 1 < bytes.len() => at + 1,
			_ => character_start(bytes, from),
		};
	}
	(start, bytes.len())
}

/// The last character boundary at or before `index`.
pub fn character_end(bytes: &[u8], index: usize) -> usize {
	let mut end = index;
	while end > 0 && (bytes[end] & 0xc0) == 0x80 {
		end -= 1;
	}
	end
}

/// The first character boundary at or after `index`.
fn character_start(bytes: &[u8], index: usize) -> usize {
	let mut start = index;
	while start < bytes.len() && (bytes[start] & 0xc0) == 0x80 {
		start += 1;
	}
	start
}

fn line_count(bytes: &[u8]) -> usize {
	if bytes.is_empty() {
		return 0;
	}
	let newlines = bytes.iter().filter(|b| **b == NEWLINE).count();
	newlines + if bytes[bytes.len() - 1] == NEWLINE { 0 } else { 1 }
}

fn count_newlines(text: &str) -> usize {
	text.bytes().filter(|b| *b == NEWLINE).count()
}

/// Lines of text with `newlines` newlines; a final unterminated line counts.
fn lines_of(newlines: usize, terminated: bool) -> usize {
	newlines + if terminated { 0 } else { 1 }
}

/// One accepted chunk of the stream.
#[derive(Debug, Clone)]
pub enum Chunk {
	Text(String),
	Bytes(Vec<u8>),
}

struct StoredChunk {
	text: String,
	bytes: usize,
	newlines: usize,
}

/// Bounded running output of one tool call. Accepting a chunk costs time
/// proportional to the chunk: head retention stops storing once the window is
/// full, and tail retention drops stored text the window no longer needs when
/// it snapshots. Counts of the whole stream are kept so the dropped totals
/// stay exact.
pub struct OutputBuffer {
	limits: OutputLimits,
	/// Carry of an incomplete trailing character from the last byte chunk.
	pending: Vec<u8>,
	chunks: VecDeque<StoredChunk>,
	stored_bytes: usize,
	stored_newlines: usize,
	full: bool,
	total_bytes: usize,
	total_newlines: usize,
	ends_with_newline: bool,
}

impl OutputBuffer {
	pub fn new(limits: OutputLimits) -> Self {
		Self {
			limits,
			pending: Vec::new(),
			chunks: VecDeque::new(),
			stored_bytes: 0,
			stored_newlines: 0,
			full: false,
			total_bytes: 0,
			total_newlines: 0,
			ends_with_newline: true,
		}
	}

	/// Bytes currently held; bounded by the limits plus one chunk.
	pub fn stored_bytes(&self) -> usize {
		self.stored_bytes
	}

	/// Accept a chunk; returns whether anything was accepted. `skipped` is
	/// output omitted right before the chunk, which must be more than the
	/// tail window by at least one byte or line (`ShellOutputSkip`); only
	/// tail retention accepts it.
	pub fn push(&mut self, chunk: Chunk, skipped: Option<ShellOutputSkip>) -> bool {
		let carry = std::mem::take(&mut self.pending);
		let pending_text = match (&chunk, &skipped) {
			(Chunk::Text(_), _) | (_, Some(_)) => String::from_utf8_lossy(&carry).into_owned(),
			_ => String::new(),
		};
		let text = match chunk {
			Chunk::Text(text) => text,
			Chunk::Bytes(bytes) => {
				// Keep an incomplete carry as bytes so a character split across
				// byte chunks can be decoded once it is complete.
				let mut all = if skipped.is_some() { Vec::new() } else { carry };
				all.extend_from_slice(&bytes);
				let split = split_complete(&all);
				self.pending = all[split..].to_vec();
				String::from_utf8_lossy(&all[..split]).into_owned()
			}
		};
		if let Some(skipped) = skipped {
			if self.limits.retain != Retain::Tail {
				panic!("Skipped output requires tail retention");
			}
			self.accept(&pending_text);
			self.skip(skipped);
			self.accept(&text);
			true
		} else {
			let mut accepted = self.accept(&pending_text);
			accepted |= self.accept(&text);
			accepted
		}
	}

	/// Count omitted output; nothing stored before it can be in the window
	/// once the text after it arrives.
	fn skip(&mut self, skipped: ShellOutputSkip) {
		if skipped.bytes == 0 {
			return;
		}
		self.total_bytes += skipped.bytes as usize;
		self.total_newlines += skipped.newlines as usize;
		self.ends_with_newline = skipped.ends_with_newline;
		self.chunks.clear();
		self.stored_bytes = 0;
		self.stored_newlines = 0;
	}

	/// Flush an incomplete trailing character as a replacement character;
	/// call when the stream ends.
	pub fn end(&mut self) {
		let pending = std::mem::take(&mut self.pending);
		let text = String::from_utf8_lossy(&pending).into_owned();
		self.accept(&text);
	}

	fn accept(&mut self, text: &str) -> bool {
		if text.is_empty() {
			return false;
		}
		let bytes = utf8_byte_length(text);
		let newlines = count_newlines(text);
		self.total_bytes += bytes;
		self.total_newlines += newlines;
		self.ends_with_newline = text.ends_with('\n');
		if self.full {
			return true;
		}
		self.chunks.push_back(StoredChunk { text: text.to_string(), bytes, newlines });
		self.stored_bytes += bytes;
		self.stored_newlines += newlines;
		if self.limits.retain == Retain::Head {
			// Nothing past a full window is ever needed.
			self.full = self.stored_bytes > self.limits.max_bytes || self.stored_newlines >= self.limits.max_lines;
			return true;
		}
		// Drop leading chunks while the rest still holds more than a window:
		// more than `max_bytes` bytes or `max_lines` newlines, plus one, so
		// the window's line start can still be found. Each chunk is dropped
		// once.
		while self.chunks.len() > 1 {
			let bytes_after = self.stored_bytes - self.chunks.front().unwrap().bytes;
			let newlines_after = self.stored_newlines - self.chunks.front().unwrap().newlines;
			if bytes_after <= self.limits.max_bytes + 1 && newlines_after <= self.limits.max_lines + 1 {
				break;
			}
			self.chunks.pop_front();
			self.stored_bytes = bytes_after;
			self.stored_newlines = newlines_after;
		}
		true
	}

	/// Retained, sanitized output and what the limits dropped from the whole
	/// stream.
	pub fn snapshot(&mut self) -> BoundedOutput {
		let stored: String = if self.chunks.len() == 1 {
			self.chunks.front().unwrap().text.clone()
		} else {
			self.chunks.iter().map(|c| c.text.as_str()).collect()
		};
		let kept = bound_output(&stored, self.limits);
		let stored_lines = lines_of(self.stored_newlines, stored.is_empty() || stored.ends_with('\n'));
		let kept_lines = stored_lines - kept.dropped_lines;
		// Tail windows never reach back before this one, but finding a later
		// window's first line needs what precedes it: keep the shortest
		// suffix longer than the window by a byte or a line, as `accept` does.
		if self.limits.retain == Retain::Tail || self.chunks.len() > 1 {
			let text = if self.limits.retain == Retain::Tail {
				tail_margin(&stored, self.limits)
			} else {
				stored.clone()
			};
			let bytes = if self.limits.retain == Retain::Tail { utf8_byte_length(&text) } else { self.stored_bytes };
			let newlines = count_newlines(&text);
			self.stored_newlines = if text.is_empty() { 0 } else { newlines };
			self.chunks.clear();
			if !text.is_empty() {
				self.chunks.push_back(StoredChunk { text, bytes, newlines });
			}
			self.stored_bytes = bytes;
		}
		BoundedOutput {
			text: sanitize_output(&kept.text),
			dropped_bytes: self.total_bytes - kept.bytes,
			dropped_lines: lines_of(self.total_newlines, self.ends_with_newline) - kept_lines,
		}
	}
}


/// Length of the longest prefix of `all` that contains only complete UTF-8
/// characters.
fn split_complete(all: &[u8]) -> usize {
	let mut split = all.len();
	while split > 0 && (all[split - 1] & 0xc0) == 0x80 {
		split -= 1;
	}
	if split > 0 && (all[split - 1] & 0x80) != 0 {
		// The last candidate starts a multibyte character; check completeness.
		let start = split - 1;
		let width = match all[start] {
			b if b & 0xe0 == 0xc0 => 2,
			b if b & 0xf0 == 0xe0 => 3,
			b if b & 0xf8 == 0xf0 => 4,
			_ => return split,
		};
		if all.len() - start < width {
			return start;
		}
		return all.len();
	}
	split
}

/// The shortest suffix of `text` with more than `max_bytes` bytes or more
/// than `max_lines` newlines, or all of it. The tail window of any text that
/// ends with this suffix, followed by anything, is the same as of `text`
/// followed by it.
fn tail_margin(text: &str, limits: OutputLimits) -> String {
	let bytes = text.as_bytes();
	let byte_start = if bytes.len() > limits.max_bytes {
		character_end(bytes, bytes.len() - limits.max_bytes - 1)
	} else {
		0
	};
	let mut line_start = 0usize;
	let mut newlines = 0usize;
	let mut index = bytes.iter().rposition(|b| *b == NEWLINE);
	while let Some(at) = index {
		newlines += 1;
		if newlines > limits.max_lines {
			line_start = at;
			break;
		}
		if at == 0 {
			break;
		}
		index = bytes[..at].iter().rposition(|b| *b == NEWLINE);
	}
	let start = byte_start.max(line_start);
	if start == 0 {
		text.to_string()
	} else {
		String::from_utf8_lossy(&bytes[start..]).into_owned()
	}
}

/// Each progress commit also buys a pause proportional to what it wrote.
pub const PROGRESS_BYTES_PER_SECOND: usize = 100 * 1024;

/// Adaptive progress commits (upstream `Progress`): the first change after an
/// idle period commits at once; each commit then delays the next by at least
/// `min_interval_ms` and by its written size at 100 KiB/s. At most one commit
/// is in flight; changes made meanwhile coalesce into the next one.
pub struct Progress {
	inner: Arc<ProgressInner>,
}

struct ProgressInner {
	write: Box<dyn Fn() -> std::pin::Pin<Box<dyn Future<Output = Result<usize, String>> + Send>> + Send + Sync>,
	on_error: Box<dyn Fn(String) + Send + Sync>,
	min_interval_ms: u64,
	state: Mutex<ProgressState>,
	schedule: Notify,
}

struct ProgressState {
	waiters: Vec<oneshot::Sender<Result<(), String>>>,
	next_at: u64,
	dirty: bool,
	stopped: bool,
	in_flight: bool,
}

impl Progress {
	pub fn new(
		write: impl Fn() -> std::pin::Pin<Box<dyn Future<Output = Result<usize, String>> + Send>> + Send + Sync + 'static,
		on_error: impl Fn(String) + Send + Sync + 'static,
		min_interval_ms: u64,
	) -> Self {
		Self {
			inner: Arc::new(ProgressInner {
				write: Box::new(write),
				on_error: Box::new(on_error),
				min_interval_ms,
				state: Mutex::new(ProgressState {
					waiters: Vec::new(),
					next_at: 0,
					dirty: false,
					stopped: false,
					in_flight: false,
				}),
				schedule: Notify::new(),
			}),
		}
	}

	/// Spawn the driver loop (upstream: the timer/closure machinery starts
	/// implicitly).
	pub fn run(&self) {
		let inner = self.inner.clone();
		tokio::spawn(async move {
			loop {
				inner.schedule.notified().await;
				let should_flush = loop {
					let wait = {
						let mut state = inner.state.lock().await;
						if state.stopped || state.in_flight || !state.dirty {
							break false;
						}
						let now = now_ms();
						if state.next_at > now {
							state.next_at - now
						} else {
							state.dirty = false;
							state.in_flight = true;
							break true;
						}
					};
					tokio::time::sleep(std::time::Duration::from_millis(wait)).await;
				};
				if !should_flush {
					continue;
				}
				let inner2 = inner.clone();
				tokio::spawn(async move {
					let waiters = {
						let mut state = inner2.state.lock().await;
						std::mem::take(&mut state.waiters)
					};
					let started = now_ms();
					let result = (inner2.write)().await;
					let mut state = inner2.state.lock().await;
					match result {
						Ok(bytes) => {
							state.next_at =
								started + std::cmp::max(inner2.min_interval_ms, (bytes as u64 * 1000) / PROGRESS_BYTES_PER_SECOND as u64);
							for waiter in waiters {
								let _ = waiter.send(Ok(()));
							}
						}
						Err(error) => {
							state.next_at = started + inner2.min_interval_ms;
							for waiter in waiters {
								let _ = waiter.send(Err(error.clone()));
							}
							(inner2.on_error)(error);
						}
					}
					state.in_flight = false;
					let dirty = state.dirty && !state.stopped;
					drop(state);
					if dirty {
						inner2.schedule.notify_one();
					}
				});
			}
		});
	}

	/// Schedule a commit.
	pub async fn mark(&self) {
		let mut state = self.inner.state.lock().await;
		state.dirty = true;
		drop(state);
		self.inner.schedule.notify_one();
	}

	/// Schedule a commit; the returned receiver settles with the commit that
	/// includes this change.
	pub async fn mark_and_wait(&self) -> oneshot::Receiver<Result<(), String>> {
		let (tx, rx) = oneshot::channel();
		{
			let mut state = self.inner.state.lock().await;
			state.waiters.push(tx);
			state.dirty = true;
		}
		self.inner.schedule.notify_one();
		rx
	}

	/// Stop committing and wait for the commit in flight; returns the waiters
	/// the final commit must settle.
	pub async fn stop(&self) -> Vec<oneshot::Sender<Result<(), String>>> {
		let mut state = self.inner.state.lock().await;
		state.stopped = true;
		let waiters = std::mem::take(&mut state.waiters);
		let in_flight = state.in_flight;
		drop(state);
		// Wait until no commit is in flight.
		loop {
			if !self.inner.state.lock().await.in_flight {
				break;
			}
			tokio::time::sleep(std::time::Duration::from_millis(1)).await;
		}
		let _ = in_flight;
		waiters
	}
}

fn now_ms() -> u64 {
	std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|d| d.as_millis() as u64)
		.unwrap_or(0)
}
