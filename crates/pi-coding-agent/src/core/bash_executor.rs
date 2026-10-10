//! Port of `packages/coding-agent/src/core/bash-executor.ts` — bash command
//! execution with streaming support and cancellation, shared by
//! `AgentSession.executeBash()` and direct callers.
//!
//! Replaces the `BashResult` stand-in in `core/extensions/types.rs` (crate
//! docs D29 lifecycle). Deviation (crate docs D35): the streamed temp file is
//! a `std::fs::File` created through
//! [`crate::utils::output_files::create_output_file_stream_sync`], because
//! the `BashOperations::exec` output callback is synchronous; chunk
//! sanitize/decode matches the upstream ordering exactly.

use std::io::Write;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::core::tools::bash::{BashExecOptions, BashOperations};
use crate::core::tools::truncate::{truncate_tail, TruncationOptions, DEFAULT_MAX_BYTES};
use crate::utils::abort::AbortSignal;
use crate::utils::ansi::strip_ansi;
use crate::utils::output_files::create_output_file_stream_sync;
use crate::utils::shell::sanitize_binary_output;

/// Options of [`execute_bash_with_operations`] (`BashExecutorOptions`).
#[derive(Default)]
pub struct BashExecutorOptions {
    /// Callback for streaming output chunks (already sanitized).
    pub on_chunk: Option<Arc<dyn Fn(&str) + Send + Sync>>,
    /// Signal for cancellation.
    pub signal: Option<AbortSignal>,
}

/// Result of executing a shell command (`BashResult`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashResult {
    /// Combined stdout + stderr output (sanitized, possibly truncated).
    pub output: String,
    /// Process exit code (`None` if killed/cancelled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Whether the command was cancelled via signal.
    pub cancelled: bool,
    /// Whether the output was truncated.
    pub truncated: bool,
    /// Path to the temp file with full output, when truncation hit the
    /// threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
}

/// Incremental counterpart of Node's `TextDecoder({ stream: true })`: holds
/// back an incomplete trailing sequence until more bytes arrive; invalid
/// sequences decode as U+FFFD (non-fatal default).
struct StreamingUtf8 {
    pending: Vec<u8>,
}

impl StreamingUtf8 {
    fn new() -> Self {
        StreamingUtf8 {
            pending: Vec::new(),
        }
    }

    fn trailing_partial_len(bytes: &[u8]) -> usize {
        let len = bytes.len();
        let scan_start = len.saturating_sub(3);
        for start in scan_start..len {
            let byte = bytes[start];
            let sequence_len = if byte & 0x80 == 0 {
                continue;
            } else if byte & 0xe0 == 0xc0 {
                2
            } else if byte & 0xf0 == 0xe0 {
                3
            } else if byte & 0xf8 == 0xf0 {
                4
            } else {
                continue;
            };
            if len - start < sequence_len {
                return len - start;
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

    /// Node's `decoder.decode()` with no argument: emit the held-back bytes,
    /// with incomplete sequences replaced (U+FFFD).
    fn flush(&mut self) -> String {
        let rest = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        rest
    }
}

/// Rolling chunk state shared between the output callback and the caller.
struct Accumulator {
    decoder: StreamingUtf8,
    /// Unfinished escape sequence at the end of the previous chunk, completed
    /// by the next chunk (`pendingAnsi`).
    pending_ansi: String,
    output_chunks: Vec<String>,
    output_bytes: usize,
    total_bytes: usize,
    temp_file_path: Option<String>,
    temp_file: Option<std::fs::File>,
}

impl Accumulator {
    /// The upstream `onData`: decode, split an unfinished trailing escape
    /// sequence off, sanitize, start/append the temp file past the threshold,
    /// keep a bounded rolling buffer, and stream the sanitized chunk to
    /// `on_chunk`.
    fn on_data(
        &mut self,
        data: &[u8],
        max_output_bytes: usize,
        on_chunk: Option<&Arc<dyn Fn(&str) + Send + Sync>>,
    ) {
        self.total_bytes += data.len();
        let decoded = self.decoder.decode(data);
        let (complete, pending) = crate::utils::ansi::split_incomplete_ansi_suffix(&format!(
            "{}{}",
            self.pending_ansi, decoded
        ));
        self.pending_ansi = pending;
        self.append_text(&complete, max_output_bytes, on_chunk);
    }

    /// The upstream `flushOutput`: emit the held-back escape sequence and any
    /// decoder remainder.
    fn flush_output(
        &mut self,
        max_output_bytes: usize,
        on_chunk: Option<&Arc<dyn Fn(&str) + Send + Sync>>,
    ) {
        let rest = format!("{}{}", self.pending_ansi, self.decoder.flush());
        self.pending_ansi.clear();
        self.append_text(&rest, max_output_bytes, on_chunk);
    }

    fn append_text(
        &mut self,
        raw_text: &str,
        max_output_bytes: usize,
        on_chunk: Option<&Arc<dyn Fn(&str) + Send + Sync>>,
    ) {
        // Sanitize: strip ANSI, replace binary garbage, normalize newlines.
        let text = sanitize_binary_output(&strip_ansi(raw_text)).replace('\r', "");
        if text.is_empty() {
            return;
        }

        // Start writing to the temp file once the threshold is exceeded.
        if self.total_bytes > DEFAULT_MAX_BYTES && self.temp_file_path.is_none() {
            match create_output_file_stream_sync("pi-bash", ".log") {
                Ok((path, mut file)) => {
                    for chunk in &self.output_chunks {
                        let _ = file.write_all(chunk.as_bytes());
                    }
                    self.temp_file_path = Some(path);
                    self.temp_file = Some(file);
                }
                Err(_) => {
                    // Upstream's WriteStream emits an async error event that
                    // nothing observes here; the port degrades to no temp
                    // file (full_output_path stays absent).
                }
            }
        }

        if let Some(file) = self.temp_file.as_mut() {
            let _ = file.write_all(text.as_bytes());
        }

        // Keep the rolling buffer bounded.
        self.output_chunks.push(text.clone());
        self.output_bytes += text.len();
        while self.output_bytes > max_output_bytes && self.output_chunks.len() > 1 {
            let removed = self.output_chunks.remove(0);
            self.output_bytes -= removed.len();
        }

        // Stream to the callback.
        if let Some(on_chunk) = on_chunk {
            on_chunk(&text);
        }
    }

    fn full_output(&self) -> String {
        self.output_chunks.join("")
    }

    fn end_temp_file(&mut self) {
        if let Some(file) = self.temp_file.as_mut() {
            let _ = file.flush();
        }
    }
}

/// Execute a bash command using custom [`BashOperations`]
/// (`executeBashWithOperations`). Used for remote execution (SSH, containers,
/// etc.).
pub async fn execute_bash_with_operations(
    command: &str,
    cwd: &str,
    operations: Arc<dyn BashOperations>,
    options: Option<BashExecutorOptions>,
) -> Result<BashResult, String> {
    let options = options.unwrap_or_default();
    let max_output_bytes = DEFAULT_MAX_BYTES * 2;

    let shared = Arc::new(Mutex::new(Accumulator {
        decoder: StreamingUtf8::new(),
        pending_ansi: String::new(),
        output_chunks: Vec::new(),
        output_bytes: 0,
        total_bytes: 0,
        temp_file_path: None,
        temp_file: None,
    }));
    let on_chunk = options.on_chunk.clone();

    let on_data: Arc<dyn Fn(Vec<u8>) + Send + Sync> = {
        let shared = Arc::clone(&shared);
        let on_chunk = on_chunk.clone();
        Arc::new(move |data: Vec<u8>| {
            if let Ok(mut accumulator) = shared.lock() {
                accumulator.on_data(&data, max_output_bytes, on_chunk.as_ref());
            }
        })
    };

    let exec_options = BashExecOptions {
        on_data,
        signal: options.signal.clone(),
        timeout: None,
        env: None,
    };

    let result = operations.exec(command, cwd, exec_options).await;

    // An aborted command still returns the output it produced so far.
    let cancelled = options
        .signal
        .as_ref()
        .is_some_and(|signal| signal.aborted());
    if result.is_err() && !cancelled {
        {
            let mut guard = shared.lock().expect("bash executor state lock poisoned");
            guard.end_temp_file();
        }
        return Err(result.err().expect("exec error"));
    }

    // Emit the held-back escape sequence and decoder remainder before the
    // final result (`flushOutput`).
    {
        let mut guard = shared.lock().expect("bash executor state lock poisoned");
        guard.flush_output(max_output_bytes, on_chunk.as_ref());
    }

    let accumulator = {
        let mut guard = shared.lock().expect("bash executor state lock poisoned");
        guard.end_temp_file();
        // Move the accumulated state out; the temp file handle drops here,
        // which closes it (upstream `stream.end()`).
        std::mem::replace(
            &mut *guard,
            Accumulator {
                decoder: StreamingUtf8::new(),
                pending_ansi: String::new(),
                output_chunks: Vec::new(),
                output_bytes: 0,
                total_bytes: 0,
                temp_file_path: None,
                temp_file: None,
            },
        )
    };

    let full_output = accumulator.full_output();
    let truncation_result = truncate_tail(&full_output, TruncationOptions::default());
    let mut accumulator = accumulator;
    if truncation_result.truncated && accumulator.temp_file_path.is_none() {
        // `ensureTempFile` at the truncation check (backfills chunks).
        match create_output_file_stream_sync("pi-bash", ".log") {
            Ok((path, mut file)) => {
                for chunk in &accumulator.output_chunks {
                    let _ = file.write_all(chunk.as_bytes());
                }
                accumulator.temp_file_path = Some(path);
            }
            Err(_) => {}
        }
    }
    Ok(BashResult {
        output: if truncation_result.truncated {
            truncation_result.content
        } else {
            full_output
        },
        exit_code: if cancelled {
            None
        } else {
            result.ok().and_then(|outcome| outcome.exit_code)
        },
        cancelled,
        truncated: truncation_result.truncated,
        full_output_path: accumulator.temp_file_path,
    })
}
