//! Port of `packages/durable/src/tools/read.ts` — bounded text reads with `slice` line selection.
//!
//! D1: tool args are JSON, so `offset`/`limit` travel as `Option<f64>`; JS number semantics
//! (truthiness, `sliceIndex` truncation, `Math.min`/`max` NaN propagation) are reproduced with
//! local helpers, so fractional, huge, and negative arguments select exactly like upstream.
//! D2: a thrown error becomes `Err(message)` (see `bash` module D1); the error text matches the
//! upstream throw, including JS `String(number)` formatting and `undefined` for a missing offset.
//! D3: `characterEnd` lives in `crate::harness::output` (upstream `harness/output.ts`).
//! D4: `truncate_head_of` is a pure function, so the port recomputes it where upstream
//! destructures one result; results are identical.

use std::sync::Arc;

use pi_chord::context::Context;
use serde_json::{json, Value};

use crate::env::{
    get_or_throw, range_decoder, starts_with_bom, BinaryReader, FileInfo, LineScan,
    ScanLinesOptions,
};
use crate::harness::api::{
    define_tool, ToolDiagnostic, ToolDiagnosticSeverity, ToolExecutionApi, ToolExecutionResult,
    ToolRegistration,
};
use crate::harness::output::character_end;
use crate::truncate::{
    format_size, truncate_head_of, utf8_byte_length, Totals, TruncationLimit, TruncationOptions,
    TruncationResult, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES,
};

use super::env::require_env;
use super::image::{detect_supported_image_mime_type_of, ByteSource};
use super::path_utils::resolve_read_tool_path;

pub fn read_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Path to the file to read (relative or absolute)" },
            "offset": { "type": "number", "description": "Line number to start reading from (1-indexed)" },
            "limit": { "type": "number", "description": "Maximum number of lines to read" },
        },
        "required": ["path"],
    })
}

/// Upstream `ReadToolInput` (`Static<typeof readSchema>`); erased to validated JSON (D1).
pub type ReadToolInput = Value;

/// Upstream `ReadToolDetails`: `{ truncation?: TruncationResult without content }`.
pub type ReadToolDetails = Value;

const READ_CHUNK: usize = 64 * 1024;

/// `Array.prototype.slice`'s conversion of an index: NaN is 0, other values truncate toward zero.
fn slice_index(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.trunc()
    }
}

fn is_safe_integer(value: f64) -> bool {
    value.is_finite() && value.fract() == 0.0 && value.abs() <= 9_007_199_254_740_991.0
}

/// JS `Math.min` (NaN-propagating).
fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// JS `Math.max` (NaN-propagating).
fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Formats a number like `String(number)`: integral values print without a decimal point.
fn js_number(value: f64) -> String {
    format!("{value}")
}

/// The decoded start of bytes `[start, end)` of the file, decoded as part of the whole file: all
/// of it, or enough for `truncate_head_of` (more than `DEFAULT_MAX_BYTES + 1` bytes, or
/// `DEFAULT_MAX_LINES` newlines).
async fn read_head(
    reader: &dyn BinaryReader,
    start: u64,
    end: u64,
    skip_bom: bool,
    context: &Context,
) -> String {
    let mut decoder = range_decoder();
    let mut text = String::new();
    let mut newlines = 0usize;
    let bom_bytes = u64::from(skip_bom && start == 0) * 3;
    let mut position = start + bom_bytes;
    while position < end {
        let bytes = get_or_throw(
            reader
                .read(position, READ_CHUNK.min((end - position) as usize), context)
                .await,
        );
        if bytes.is_empty() {
            break;
        }
        position += bytes.len() as u64;
        let decoded = decoder.decode(Some(&bytes));
        newlines += decoded.matches('\n').count();
        text.push_str(&decoded);
        if newlines >= DEFAULT_MAX_LINES || utf8_byte_length(&text) > DEFAULT_MAX_BYTES + 1 {
            return text;
        }
    }
    text + &decoder.decode(None)
}

/// Fallible body of one `read` call.
pub async fn execute(
    args: Value,
    api: Arc<dyn ToolExecutionApi>,
    context: Context,
) -> Result<ToolExecutionResult, String> {
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let offset = args.get("offset").and_then(Value::as_f64);
    let limit = args.get("limit").and_then(Value::as_f64);
    let env = require_env(api.as_ref())?;
    let absolute_path = resolve_read_tool_path(env.as_ref(), &path, &context)
        .await
        .map_err(|error| error.to_string())?;
    let reader = get_or_throw(env.open_binary_reader(&absolute_path, None, &context).await);
    let outcome = async {
        // A concurrent writer can change the file between the scan and the reads. Appending (a
        // growing log) leaves the scanned bytes as they were; a file that shrank or was
        // rewritten in place is read again once.
        for attempt in 0..=1u8 {
            let before = get_or_throw(reader.info(&context).await);
            let result =
                read_text(reader.as_ref(), &before, &path, offset, limit, &context).await?;
            let after = get_or_throw(reader.info(&context).await);
            if after.size > before.size
                || (after.size == before.size && after.mtime_ms == before.mtime_ms)
            {
                return Ok(result);
            }
            if attempt == 1 {
                return Err(format!("{path} changed while it was read"));
            }
        }
        unreachable!("retry loop always returns")
    };
    let result = outcome.await;
    reader.close(&context).await;
    result
}

/// The read result for the opened file. It equals decoding the whole file with `TextDecoder`,
/// splitting it on `\n`, and bounding the selected lines with `truncateHead`, while reading only
/// one scan's worth of the file plus the head.
pub(crate) async fn read_text(
    reader: &dyn BinaryReader,
    info: &FileInfo,
    path: &str,
    offset: Option<f64>,
    limit: Option<f64>,
    context: &Context,
) -> Result<ToolExecutionResult, String> {
    let source = ReaderByteSource {
        reader,
        context,
        size: info.size,
    };
    if let Some(mime_type) = detect_supported_image_mime_type_of(&source).await {
        // Image content is not supported yet.
        return Ok(ToolExecutionResult {
            content: Some(json!([])),
            is_error: Some(true),
            diagnostics: Some(vec![ToolDiagnostic {
                severity: ToolDiagnosticSeverity::Error,
                code: Some("unsupported_image".to_string()),
                message: format!(
                    "{path} is an image ({mime_type}); reading images is not supported"
                ),
            }]),
            ..Default::default()
        });
    }

    // JS: `offset ? Math.max(0, offset - 1) : 0` — `0` and `NaN` are falsy.
    let start_line = match offset {
        Some(offset) if offset != 0.0 && !offset.is_nan() => js_max(0.0, offset - 1.0),
        _ => 0.0,
    };
    let start_line_display = start_line + 1.0;
    // Lines are selected like `allLines.slice(startLine, endLine)`, which truncates fractional indices.
    let slice_start = slice_index(start_line);
    // One pass finds the line count and the selection; a selection past the last line ends with
    // it, as `slice` does, and an empty one (a zero or negative limit) is scanned as one line and
    // then ignored. A start beyond any file is scanned from 0 only to count lines; the offset
    // check below then fails as before.
    let scan_start = if is_safe_integer(slice_start) {
        slice_start
    } else {
        0.0
    };
    let requested_end =
        limit.map(|limit| js_max(scan_start + 1.0, slice_index(start_line + limit)));
    let scan_end = requested_end
        .filter(|end| is_safe_integer(*end))
        .map(|end| end as u64);
    let mut scan: LineScan = get_or_throw(
        reader
            .scan_lines(
                ScanLinesOptions {
                    start_line: scan_start as u64,
                    end_line: scan_end,
                },
                context,
            )
            .await,
    );
    let total_file_lines = scan.newlines + 1;
    if start_line >= total_file_lines as f64 {
        return Err(format!(
            "Offset {} is beyond end of file ({} lines total)",
            offset
                .map(js_number)
                .unwrap_or_else(|| "undefined".to_string()),
            total_file_lines
        ));
    }

    let mut user_limited_lines: Option<f64> = None;
    let mut selected_line_count = total_file_lines as f64 - slice_start;
    if let Some(limit) = limit {
        let end_line = js_min(start_line + limit, total_file_lines as f64);
        user_limited_lines = Some(end_line - start_line);
        // `slice` counts a negative end from the end of the lines, which only the line count
        // tells; scan again for it.
        let relative_end = slice_index(end_line);
        let slice_end = if relative_end < 0.0 {
            js_max(total_file_lines as f64 + relative_end, 0.0)
        } else {
            relative_end
        };
        selected_line_count = js_max(0.0, slice_end - slice_start);
        if selected_line_count > 0.0 && relative_end < 0.0 {
            scan = get_or_throw(
                reader
                    .scan_lines(
                        ScanLinesOptions {
                            start_line: scan_start as u64,
                            end_line: Some(slice_end as u64),
                        },
                        context,
                    )
                    .await,
            );
        }
    }
    let empty = selected_line_count == 0.0;
    // Counted like `truncateHead`: a trailing newline adds no line, and empty text has none.
    let ends_with_newline =
        !empty && scan.last_line_start == scan.end && scan.last_line_start > scan.start;
    let totals = Totals {
        lines: if empty || scan.selected_bytes == 0 {
            0
        } else {
            selected_line_count as usize - usize::from(ends_with_newline)
        },
        bytes: if empty {
            0
        } else {
            scan.selected_bytes as usize
        },
    };
    let first_bytes = get_or_throw(reader.read(0, 3, context).await);
    let head = if empty {
        String::new()
    } else {
        read_head(
            reader,
            scan.start,
            scan.end,
            starts_with_bom(&first_bytes),
            context,
        )
        .await
    };

    let TruncationResult {
        content: head_text, ..
    } = truncate_head_of(&head, totals, TruncationOptions::default());
    let truncation_value = {
        let mut value = serde_json::to_value(truncate_head_of(
            &head,
            totals,
            TruncationOptions::default(),
        ))
        .expect("truncation serializes");
        if let Value::Object(map) = &mut value {
            map.shift_remove("content");
        }
        value
    };
    let mut diagnostics: Vec<ToolDiagnostic> = Vec::new();
    let mut output_text = head_text;
    let mut details: Option<Value> = None;
    if truncate_head_of(&head, totals, TruncationOptions::default()).first_line_exceeds_limit {
        // Show the start of the line, cut at the byte limit on a character boundary. Like
        // `allLines[startLine]`, a fractional start line names no line.
        let integral = start_line.fract() == 0.0;
        let line_bytes: Vec<u8> = if integral {
            head.split('\n').next().unwrap_or("").as_bytes().to_vec()
        } else {
            Vec::new()
        };
        let line_size = if integral {
            scan.first_line_bytes as usize
        } else {
            0
        };
        let end = character_end(&line_bytes, DEFAULT_MAX_BYTES);
        output_text = String::from_utf8_lossy(&line_bytes[..end]).into_owned();
        diagnostics.push(ToolDiagnostic {
			severity: ToolDiagnosticSeverity::Warn,
			code: Some("truncated".to_string()),
			message: format!(
				"Line {} is {}, exceeds the {} limit; showing its first {}. Use bash: sed -n '{}p' {path} | tail -c +{}",
				js_number(start_line_display),
				format_size(line_size),
				format_size(DEFAULT_MAX_BYTES),
				format_size(end),
				js_number(start_line_display),
				end + 1,
			),
		});
        let mut truncation = truncation_value.clone();
        truncation["outputBytes"] = json!(end);
        truncation["outputLines"] = json!(1);
        details = Some(json!({ "truncation": truncation }));
    } else {
        let truncation = truncate_head_of(&head, totals, TruncationOptions::default());
        if truncation.truncated {
            let end_line_display = start_line_display + truncation.output_lines as f64 - 1.0;
            let next_offset = end_line_display + 1.0;
            let limit_text = if truncation.truncated_by == Some(TruncationLimit::Lines) {
                String::new()
            } else {
                format!(" ({} limit)", format_size(DEFAULT_MAX_BYTES))
            };
            diagnostics.push(ToolDiagnostic {
                severity: ToolDiagnosticSeverity::Info,
                code: Some("truncated".to_string()),
                message: format!(
                    "Showing lines {}-{} of {}{}. Use offset={} to continue.",
                    js_number(start_line_display),
                    js_number(end_line_display),
                    total_file_lines,
                    limit_text,
                    js_number(next_offset),
                ),
            });
            details = Some(json!({ "truncation": truncation_value }));
        } else if let Some(user_limited) = user_limited_lines {
            if start_line + user_limited < total_file_lines as f64 {
                let remaining = total_file_lines as f64 - (start_line + user_limited);
                let next_offset = start_line + user_limited + 1.0;
                diagnostics.push(ToolDiagnostic {
                    severity: ToolDiagnosticSeverity::Info,
                    code: None,
                    message: format!(
                        "{} more lines in file. Use offset={} to continue.",
                        js_number(remaining),
                        js_number(next_offset),
                    ),
                });
            }
        }
    }

    Ok(ToolExecutionResult {
        content: if output_text.is_empty() {
            None
        } else {
            Some(json!([{ "type": "text", "text": output_text }]))
        },
        details,
        diagnostics: if diagnostics.is_empty() {
            None
        } else {
            Some(diagnostics)
        },
        ..Default::default()
    })
}

/// Upstream `ByteSource` adapter over the opened reader (`info.size` of the opened file).
struct ReaderByteSource<'a> {
    reader: &'a dyn BinaryReader,
    context: &'a Context,
    size: u64,
}

#[async_trait::async_trait]
impl ByteSource for ReaderByteSource<'_> {
    fn size(&self) -> u64 {
        self.size
    }

    async fn read(&self, offset: u64, length: usize) -> Vec<u8> {
        get_or_throw(self.reader.read(offset, length, self.context).await)
    }
}

pub fn create_read_tool() -> ToolRegistration {
    define_tool(ToolRegistration {
		tool: pi_ai::Tool {
			name: "read".to_string(),
			description: format!(
				"Read the contents of a text file. Output is truncated to {} lines or {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
				DEFAULT_MAX_LINES,
				DEFAULT_MAX_BYTES / 1024
			),
			parameters: read_parameters(),
			constrained_sampling: None,
		},
		replay: None,
		execution_mode: None,
		prepare_arguments: None,
		output_limits: None,
		execute: super::wrap_execute(move |args, api, context| execute(args, api, context)),
	})
}
