//! Port of `upstream/pi/packages/coding-agent/src/core/tools/grep.ts` — the
//! `grep` tool (ripgrep content search).
//!
//! Deviations:
//! - Node readline/stdout streaming becomes tokio buffered lines over the
//!   piped `ChildStdout` (D38); `child.kill()` maps to the SIGTERM default
//!   (`Child::start_kill`). The upstream `settled` flag becomes a single
//!   async return/`Err` per execute; an abort watcher task kills the child
//!   and the close handler reports `Operation aborted`.
//! - `path.relative`/`path.basename` in `formatPath` become the hand-rolled
//!   [`crate::core::tools::find::lexical_relative`] /
//!   [`crate::core::tools::find::path_basename`] helpers (D38): posix
//!   (`/`) output on all platforms.
//! - `ensureTool("rg")` reuses the [`crate::utils::tools_manager`] seam
//!   unchanged.
//! - Renderers are deferred with the interactive mode (D32).

use std::collections::HashMap;
use std::sync::Arc;

use pi_agent::{AgentTool, AgentToolResult, BoxFuture};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use tokio::io::AsyncBufReadExt;

use crate::core::extensions::types::{ExtensionToolContext, ToolDefinition, ToolExecute};
use crate::core::tools::find::{lexical_relative, path_basename};
use crate::core::tools::path_utils::resolve_to_cwd;
use crate::core::tools::tool_definition_wrapper::wrap_tool_definition;
use crate::core::tools::truncate::{
    format_size, truncate_head, truncate_line, TruncationOptions, TruncationResult,
    DEFAULT_MAX_BYTES, GREP_MAX_LINE_LENGTH,
};
use crate::utils::abort::AbortSignal;
use crate::utils::child_process::spawn_process;
use crate::utils::tools_manager::ToolName;

/// `grepSchema` (`Type.Object`) — JSON Schema for the tool parameters.
pub fn grep_tool_input_schema() -> JsonValue {
    json!({
        "type": "object",
        "properties": {
            "pattern": { "type": "string", "description": "Search pattern (regex or literal string)" },
            "path": { "type": "string", "description": "Directory or file to search (default: current directory)" },
            "glob": { "type": "string", "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'" },
            "ignoreCase": { "type": "boolean", "description": "Case-insensitive search (default: false)" },
            "literal": { "type": "boolean", "description": "Treat pattern as literal string instead of regex (default: false)" },
            "context": { "type": "number", "description": "Number of lines to show before and after each match (default: 0)" },
            "limit": { "type": "number", "description": "Maximum number of matches to return (default: 100)" }
        },
        "required": ["pattern"]
    })
}

/// `grepToolSystemPromptContribution`.
pub const GREP_TOOL_SYSTEM_PROMPT_SNIPPET: &str =
    "Search file contents for patterns (respects .gitignore)";

/// `GrepToolInput` (`Static<typeof grepSchema>`).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GrepToolInput {
    pub pattern: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glob: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore_case: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub literal: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
}

const DEFAULT_LIMIT: usize = 100;

/// `GrepToolDetails`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GrepToolDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub match_limit_reached: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lines_truncated: Option<bool>,
}

/// Pluggable operations for the grep tool.
///
/// Override these to delegate search to remote systems (for example SSH).
pub trait GrepOperations: Send + Sync {
    /// Check if path is a directory. Throws if path does not exist.
    fn is_directory(&self, absolute_path: String) -> BoxFuture<'static, Result<bool, String>>;
    /// Read file contents for context lines.
    fn read_file(&self, absolute_path: String) -> BoxFuture<'static, Result<String, String>>;
}

/// `defaultGrepOperations`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalGrepOperations;

impl GrepOperations for LocalGrepOperations {
    fn is_directory(&self, absolute_path: String) -> BoxFuture<'static, Result<bool, String>> {
        Box::pin(async move {
            std::fs::metadata(&absolute_path)
                .map(|metadata| metadata.is_dir())
                .map_err(|e| e.to_string())
        })
    }
    fn read_file(&self, absolute_path: String) -> BoxFuture<'static, Result<String, String>> {
        Box::pin(async move {
            tokio::fs::read_to_string(&absolute_path)
                .await
                .map_err(|e| e.to_string())
        })
    }
}

/// `GrepToolOptions`.
#[derive(Clone, Default)]
pub struct GrepToolOptions {
    /// Custom operations for grep. Default: local filesystem plus ripgrep
    pub operations: Option<Arc<dyn GrepOperations>>,
}

/// One streamed ripgrep match (`matches` entries upstream).
struct RawMatch {
    file_path: String,
    line_number: usize,
    line_text: Option<String>,
}

/// Strip CR the way the context-line sanitizer does (`replace(/\r/g, "")`).
fn strip_cr(text: &str) -> String {
    text.replace('\r', "")
}

/// `createGrepToolDefinition`.
pub fn create_grep_tool_definition(cwd: &str, options: Option<GrepToolOptions>) -> ToolDefinition {
    let custom_ops = options.and_then(|options| options.operations);
    let cwd = cwd.to_string();
    let execute: ToolExecute = Arc::new(
        move |_tool_call_id: String,
              args: JsonValue,
              signal: Option<AbortSignal>,
              _on_update,
              ctx: Option<ExtensionToolContext>|
              -> BoxFuture<'static, Result<AgentToolResult, String>> {
            // Fn closure: clone captures for each call before the async block
            // consumes them.
            let cwd = cwd.clone();
            let custom_ops = custom_ops.clone();
            Box::pin(async move {
                if signal.as_ref().is_some_and(AbortSignal::aborted) {
                    return Err("Operation aborted".to_string());
                }
                let input: GrepToolInput = serde_json::from_value(args)
                    .map_err(|e| format!("Invalid arguments for grep tool: {e}"))?;
                let effective_cwd = ctx
                    .as_ref()
                    .map(|c| c.cwd.clone())
                    .filter(|c| !c.is_empty())
                    .unwrap_or_else(|| cwd.clone());
                let ops: Arc<dyn GrepOperations> = custom_ops
                    .clone()
                    .unwrap_or_else(|| Arc::new(LocalGrepOperations));

                let Some(rg_path) =
                    crate::core::tools::find::ensure_tool_detached(ToolName::Rg).await
                else {
                    return Err(
                        "ripgrep (rg) is not available and could not be downloaded".to_string()
                    );
                };

                let search_path =
                    resolve_to_cwd(input.path.as_deref().unwrap_or("."), &effective_cwd);
                let is_directory = match ops.is_directory(search_path.clone()).await {
                    Ok(is_directory) => is_directory,
                    Err(_) => return Err(format!("Path not found: {search_path}")),
                };

                let context_value = match input.context {
                    Some(context) if context > 0.0 => context as usize,
                    _ => 0,
                };
                let effective_limit = std::cmp::max(
                    1,
                    input
                        .limit
                        .map(|limit| limit as usize)
                        .unwrap_or(DEFAULT_LIMIT),
                );
                let format_path = |file_path: &str| -> String {
                    if is_directory {
                        let relative = lexical_relative(&search_path, file_path);
                        if !relative.is_empty() && !relative.starts_with("..") {
                            return relative.replace('\\', "/");
                        }
                    }
                    path_basename(file_path)
                };

                // `getFileLines` file cache lives in `file_cache`; formatting is
                // sequential, so a plain local map mirrors the upstream Map.
                let mut file_cache: HashMap<String, Vec<String>> = HashMap::new();

                let mut args: Vec<String> = vec![
                    "--json".to_string(),
                    "--line-number".to_string(),
                    "--color=never".to_string(),
                    "--hidden".to_string(),
                ];
                if input.ignore_case.unwrap_or(false) {
                    args.push("--ignore-case".to_string());
                }
                if input.literal.unwrap_or(false) {
                    args.push("--fixed-strings".to_string());
                }
                if let Some(glob) = &input.glob {
                    args.push("--glob".to_string());
                    args.push(glob.clone());
                }
                args.push("--".to_string());
                args.push(input.pattern.clone());
                args.push(search_path.clone());

                let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
                let mut child = spawn_process(&rg_path, &arg_refs)
                    .map_err(|e| format!("Failed to run ripgrep: {e}"))?;
                let stdout = child.stdout.take();
                let stderr_handle = child.stderr.take();

                // Kill reasons: abort watcher vs match-limit stop (upstream
                // `onAbort`/`stopChild(dueToLimit)`).
                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                enum KillReason {
                    Abort,
                    Limit,
                }
                let (kill_tx, kill_rx) = tokio::sync::mpsc::unbounded_channel::<KillReason>();
                if let Some(signal) = signal.clone() {
                    let kill_tx = kill_tx.clone();
                    tokio::spawn(async move {
                        signal.aborted_future().await;
                        let _ = kill_tx.send(KillReason::Abort);
                    });
                }

                // Accumulate stderr while stdout streams (upstream "data" handler).
                let stderr_task = tokio::spawn(async move {
                    let mut stderr = String::new();
                    if let Some(stderr_stream) = stderr_handle {
                        let mut lines = tokio::io::BufReader::new(stderr_stream).lines();
                        while let Ok(Some(line)) = lines.next_line().await {
                            stderr.push_str(&line);
                            stderr.push('\n');
                        }
                    }
                    stderr
                });

                // Collect matches during streaming, then format them after rg exits.
                let mut matches: Vec<RawMatch> = Vec::new();
                let mut match_count: usize = 0;
                let mut match_limit_reached = false;
                let mut killed_due_to_limit = false;
                {
                    let mut kill_rx = Some(kill_rx);
                    if let Some(stdout) = stdout {
                        let mut lines_stream = tokio::io::BufReader::new(stdout).lines();
                        loop {
                            tokio::select! {
                                line = lines_stream.next_line() => match line {
                                    Ok(Some(line)) => {
                                        if line.trim().is_empty() || match_count >= effective_limit {
                                            continue;
                                        }
                                        let Ok(event) = serde_json::from_str::<JsonValue>(&line) else {
                                            continue;
                                        };
                                        if event.get("type").and_then(JsonValue::as_str) == Some("match") {
                                            match_count += 1;
                                            let file_path = event
                                                .pointer("/data/path/text")
                                                .and_then(JsonValue::as_str)
                                                .map(str::to_string);
                                            let line_number =
                                                event.pointer("/data/line_number").and_then(JsonValue::as_u64);
                                            let line_text = event
                                                .pointer("/data/lines/text")
                                                .and_then(JsonValue::as_str)
                                                .map(str::to_string);
                                            if let (Some(file_path), Some(line_number)) = (file_path, line_number) {
                                                matches.push(RawMatch { file_path, line_number: line_number as usize, line_text });
                                            }
                                            if match_count >= effective_limit {
                                                match_limit_reached = true;
                                                killed_due_to_limit = true;
                                                let _ = child.start_kill();
                                                let _ = kill_tx.send(KillReason::Limit);
                                            }
                                        }
                                    }
                                    Ok(None) | Err(_) => break,
                                },
                                kill = async { kill_rx.as_mut().unwrap().recv().await }, if kill_rx.is_some() => {
                                    match kill {
                                        None => kill_rx = None,
                                        Some(KillReason::Abort) => {
                                            let _ = child.start_kill();
                                        }
                                        Some(KillReason::Limit) => {
                                            let _ = child.start_kill();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let status = child.wait().await;
                let stderr = stderr_task.await.unwrap_or_default();

                if signal.as_ref().is_some_and(AbortSignal::aborted) {
                    return Err("Operation aborted".to_string());
                }
                let code = status.ok().and_then(|status| status.code());
                if !killed_due_to_limit && code != Some(0) && code != Some(1) {
                    return Err(if stderr.trim().is_empty() {
                        format!(
                            "ripgrep exited with code {}",
                            code.map(|c| c.to_string())
                                .unwrap_or_else(|| "null".to_string())
                        )
                    } else {
                        stderr.trim().to_string()
                    });
                }
                if match_count == 0 {
                    return Ok(AgentToolResult::new(
                        vec![pi_ai::types::UserContent::from(
                            pi_ai::types::TextContent::new("No matches found"),
                        )],
                        JsonValue::Null,
                    ));
                }

                let ops = ops.clone();
                let mut output_lines: Vec<String> = Vec::new();
                let mut lines_truncated = false;
                // Format matches after streaming finishes so custom readFile() backends
                // can be async.
                for match_item in &matches {
                    if context_value == 0 {
                        if let Some(line_text) = &match_item.line_text {
                            let relative_path = format_path(&match_item.file_path);
                            let sanitized = strip_cr(
                                &line_text
                                    .replace("\r\n", "\n")
                                    .strip_suffix('\n')
                                    .unwrap_or(line_text),
                            );
                            let truncated = truncate_line(&sanitized, GREP_MAX_LINE_LENGTH);
                            if truncated.was_truncated {
                                lines_truncated = true;
                            }
                            output_lines.push(format!(
                                "{}:{}: {}",
                                relative_path, match_item.line_number, truncated.text
                            ));
                            continue;
                        }
                    }
                    // formatBlock
                    let relative_path = format_path(&match_item.file_path);
                    let lines: Vec<String> = {
                        let file_path = match_item.file_path.clone();
                        if let Some(cached) = file_cache.get(&file_path) {
                            cached.clone()
                        } else {
                            let loaded = match ops.read_file(file_path.clone()).await {
                                Ok(content) => content
                                    .replace("\r\n", "\n")
                                    .replace('\r', "\n")
                                    .split('\n')
                                    .map(str::to_string)
                                    .collect(),
                                Err(_) => Vec::new(),
                            };
                            file_cache.insert(file_path, loaded.clone());
                            loaded
                        }
                    };
                    if lines.is_empty() {
                        output_lines.push(format!(
                            "{relative_path}:{}: (unable to read file)",
                            match_item.line_number
                        ));
                        continue;
                    }
                    let start = if context_value > 0 {
                        match_item.line_number.saturating_sub(context_value).max(1)
                    } else {
                        match_item.line_number
                    };
                    let end = if context_value > 0 {
                        std::cmp::min(lines.len(), match_item.line_number + context_value)
                    } else {
                        match_item.line_number
                    };
                    for current in start..=end {
                        let line_text = lines.get(current - 1).cloned().unwrap_or_default();
                        let sanitized = strip_cr(&line_text);
                        let truncated = truncate_line(&sanitized, GREP_MAX_LINE_LENGTH);
                        if truncated.was_truncated {
                            lines_truncated = true;
                        }
                        if current == match_item.line_number {
                            output_lines
                                .push(format!("{relative_path}:{current}: {}", truncated.text));
                        } else {
                            output_lines
                                .push(format!("{relative_path}-{current}- {}", truncated.text));
                        }
                    }
                }

                let raw_output = output_lines.join("\n");
                // Apply byte truncation. There is no line limit here because the match
                // limit already capped rows.
                let truncation = truncate_head(
                    &raw_output,
                    TruncationOptions {
                        max_lines: Some(usize::MAX),
                        max_bytes: Some(DEFAULT_MAX_BYTES),
                    },
                );
                let mut output = truncation.content.clone();
                let mut details = GrepToolDetails::default();
                // Build actionable notices for truncation and match limits.
                let mut notices: Vec<String> = Vec::new();
                if match_limit_reached {
                    notices.push(format!(
						"{effective_limit} matches limit reached. Use limit={} for more, or refine pattern",
						effective_limit * 2
					));
                    details.match_limit_reached = Some(effective_limit);
                }
                if truncation.truncated {
                    notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
                    details.truncation = Some(truncation);
                }
                if lines_truncated {
                    notices.push(format!(
						"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
					));
                    details.lines_truncated = Some(true);
                }
                if !notices.is_empty() {
                    output += &format!("\n\n[{}]", notices.join(". "));
                }
                let details_json = if details.truncation.is_none()
                    && details.match_limit_reached.is_none()
                    && details.lines_truncated.is_none()
                {
                    JsonValue::Null
                } else {
                    serde_json::to_value(&details).unwrap_or(JsonValue::Null)
                };
                Ok(AgentToolResult::new(
                    vec![pi_ai::types::UserContent::from(
                        pi_ai::types::TextContent::new(output),
                    )],
                    details_json,
                ))
            })
        },
    );
    ToolDefinition {
		name: "grep".to_string(),
		label: "grep".to_string(),
		description: format!(
			"Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} matches or {}KB (whichever is hit first). Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars.",
			DEFAULT_MAX_BYTES / 1024
		),
		prompt_snippet: Some(GREP_TOOL_SYSTEM_PROMPT_SNIPPET.to_string()),
		prompt_guidelines: None,
		parameters: grep_tool_input_schema(),
		constrained_sampling: None,
		render_shell: None,
		prepare_arguments: None,
		output_schema: None,
		exposure: None,
		namespace: None,
		annotations: None,
		default_active: None,
		prepare_loadout: None,
		execution_mode: None,
		execute,
		render_call: None,
		render_result: None,
	}
}

/// `createGrepTool`.
pub fn create_grep_tool(cwd: &str, options: Option<GrepToolOptions>) -> AgentTool {
    wrap_tool_definition(create_grep_tool_definition(cwd, options), None)
}
