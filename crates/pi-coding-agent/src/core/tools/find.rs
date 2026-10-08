//! Port of `upstream/pi/packages/coding-agent/src/core/tools/find.ts` — the
//! `find` tool (fd glob search).
//!
//! Deviations:
//! - Node readline/stdout streaming becomes tokio buffered lines over the
//!   piped `ChildStdout` (D38); `child.kill()` maps to the SIGTERM default
//!   (`Child::start_kill`). The upstream `settled` flag becomes a single
//!   async return/`Err` per execute; an abort watcher task kills the child
//!   and the close handler reports `Operation aborted`.
//! - `path.relative`/`path.basename` become [`lexical_relative`] /
//!   [`path_basename`] hand-rolled helpers (D38): std::path produces
//!   platform strings, while the upstream output is posix (`/` on every
//!   platform), so separators are normalized manually.
//! - `process.platform === "win32"` full-path separator-class rewrite is
//!   kept behind `cfg!(windows)` for parity.
//! - Renderers are deferred with the interactive mode (D32).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use pi_agent::{AgentTool, AgentToolResult, BoxFuture};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use tokio::io::AsyncBufReadExt;

use crate::core::extensions::types::{ExtensionToolContext, ToolDefinition, ToolExecute};
use crate::core::tools::path_utils::{path_exists, resolve_to_cwd};
use crate::core::tools::tool_definition_wrapper::wrap_tool_definition;
use crate::core::tools::truncate::{
    format_size, truncate_head, TruncationOptions, TruncationResult, DEFAULT_MAX_BYTES,
};
use crate::utils::abort::AbortSignal;
use crate::utils::child_process::spawn_process;
use crate::utils::tools_manager::{ensure_tool, ToolName};

/// `findSchema` (`Type.Object`) — JSON Schema for the tool parameters.
pub fn find_tool_input_schema() -> JsonValue {
    json!({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"
            },
            "path": { "type": "string", "description": "Directory to search in (default: current directory)" },
            "limit": { "type": "number", "description": "Maximum number of results (default: 1000)" }
        },
        "required": ["pattern"]
    })
}

/// `findToolSystemPromptContribution`.
pub const FIND_TOOL_SYSTEM_PROMPT_SNIPPET: &str =
    "Find files by glob pattern (respects .gitignore)";

/// `FindToolInput` (`Static<typeof findSchema>`).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FindToolInput {
    pub pattern: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
}

const DEFAULT_LIMIT: usize = 1000;

/// `FindToolDetails`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FindToolDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_limit_reached: Option<usize>,
}

/// Options of [`FindOperations::glob`].
#[derive(Debug, Clone)]
pub struct FindGlobOptions {
    pub ignore: Vec<String>,
    pub limit: usize,
}

/// Pluggable operations for the find tool.
///
/// Override these to delegate file search to remote systems (for example SSH).
pub trait FindOperations: Send + Sync {
    /// Check if path exists.
    fn exists(&self, absolute_path: String) -> BoxFuture<'static, bool>;
    /// Find files matching glob pattern. Returns relative or absolute paths.
    fn glob(
        &self,
        pattern: String,
        cwd: String,
        options: FindGlobOptions,
    ) -> BoxFuture<'static, Result<Vec<String>, String>>;
}

/// `defaultFindOperations`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultFindOperations;

impl FindOperations for DefaultFindOperations {
    fn exists(&self, absolute_path: String) -> BoxFuture<'static, bool> {
        Box::pin(async move { path_exists(&absolute_path).await })
    }
    // This is a placeholder. Actual fd execution happens in execute() when no
    // custom glob is provided.
    fn glob(
        &self,
        _pattern: String,
        _cwd: String,
        _options: FindGlobOptions,
    ) -> BoxFuture<'static, Result<Vec<String>, String>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
}

/// `FindToolOptions`.
#[derive(Clone, Default)]
pub struct FindToolOptions {
    /// Custom operations for find. Default: local filesystem plus fd
    pub operations: Option<Arc<dyn FindOperations>>,
}

/// Normalize to posix separators, treating `\` as a separator too.
fn to_posix(path: &str) -> String {
    path.replace('\\', "/")
}

/// Lexical `path.relative(from, to)` over posix separators: common-prefix
/// strip, then one `..` per remaining `from` component.
pub fn lexical_relative(from: &str, to: &str) -> String {
    let from = to_posix(from);
    let to = to_posix(to);
    if from == to {
        return String::new();
    }
    let from_prefix = if from.ends_with('/') {
        from.clone()
    } else {
        format!("{from}/")
    };
    if to.starts_with(&from_prefix) {
        return to[from_prefix.len()..].to_string();
    }
    let from_parts: Vec<&str> = from
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    let to_parts: Vec<&str> = to
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    let mut common = 0;
    while common < from_parts.len()
        && common < to_parts.len()
        && from_parts[common] == to_parts[common]
    {
        common += 1;
    }
    let mut parts: Vec<&str> = vec![".."; from_parts.len() - common];
    parts.extend(&to_parts[common..]);
    parts.join("/")
}

/// `path.basename` over both separators.
pub fn path_basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

/// Relativize a find result against the search root and normalize it to posix
/// separators (`relativizeFindResultPath`).
pub fn relativize_find_result_path(result_path: &str, search_path: &str) -> String {
    let had_trailing_separator = result_path.ends_with(std::path::MAIN_SEPARATOR)
        || (cfg!(windows) && result_path.ends_with('/'));
    let relative_path = if std::path::Path::new(result_path).is_absolute() {
        lexical_relative(search_path, result_path)
    } else {
        result_path.to_string()
    };
    let posix_path = to_posix(&relative_path);
    if had_trailing_separator && !posix_path.ends_with('/') {
        format!("{posix_path}/")
    } else {
        posix_path
    }
}

/// `ensureTool` with default fetcher/status (upstream `await ensureTool("fd")`).
///
/// The seam's `Option<&dyn Fn>` parameters make the raw future `!Send` even
/// when `None`; both references are absent here and never dereferenced, so
/// detaching via a `Send` wrapper is sound.
pub(crate) fn ensure_tool_detached(tool: ToolName) -> BoxFuture<'static, Option<String>> {
    struct AssertSendFuture<F>(F);
    impl<F: Future> Future for AssertSendFuture<F> {
        type Output = F::Output;
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            unsafe { self.as_mut().map_unchecked_mut(|s| &mut s.0) }.poll(cx)
        }
    }
    // SAFETY: the wrapped future carries only the `None` fetcher/status
    // references passed below; they are never dereferenced.
    unsafe impl<F> Send for AssertSendFuture<F> {}
    Box::pin(AssertSendFuture(ensure_tool(tool, None, None)))
}

/// Walk up from `search_path` looking for a `.git` entry (upstream loop).
async fn find_git_root(search_path: &str) -> bool {
    let mut current = search_path.to_string();
    loop {
        if path_exists(&crate::utils::paths::join(&current, ".git")).await {
            return true;
        }
        let Some(parent) = std::path::Path::new(&current).parent() else {
            break;
        };
        let parent = parent.to_string_lossy().into_owned();
        if parent == current {
            break;
        }
        current = parent;
    }
    false
}

/// Build the result text/details shared by the custom-ops and fd paths.
fn build_find_output(
    relativized: Vec<String>,
    effective_limit: usize,
    limit_notice: String,
) -> (String, JsonValue) {
    let result_limit_reached = relativized.len() >= effective_limit;
    let raw_output = relativized.join("\n");
    // Apply byte truncation. There is no line limit here because the result
    // limit already capped rows.
    let truncation = truncate_head(
        &raw_output,
        TruncationOptions {
            max_lines: Some(usize::MAX),
            max_bytes: Some(DEFAULT_MAX_BYTES),
        },
    );
    let mut result_output = truncation.content.clone();
    let mut details = FindToolDetails {
        truncation: None,
        result_limit_reached: None,
    };
    let mut notices: Vec<String> = Vec::new();
    if result_limit_reached {
        notices.push(limit_notice);
        details.result_limit_reached = Some(effective_limit);
    }
    if truncation.truncated {
        notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
        details.truncation = Some(truncation);
    }
    if !notices.is_empty() {
        result_output += &format!("\n\n[{}]", notices.join(". "));
    }
    let details = if details.truncation.is_none() && details.result_limit_reached.is_none() {
        JsonValue::Null
    } else {
        serde_json::to_value(&details).unwrap_or(JsonValue::Null)
    };
    (result_output, details)
}

/// `createFindToolDefinition`.
pub fn create_find_tool_definition(cwd: &str, options: Option<FindToolOptions>) -> ToolDefinition {
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
                let input: FindToolInput = serde_json::from_value(args)
                    .map_err(|e| format!("Invalid arguments for find tool: {e}"))?;
                let effective_cwd = ctx
                    .as_ref()
                    .map(|c| c.cwd.clone())
                    .filter(|c| !c.is_empty())
                    .unwrap_or_else(|| cwd.clone());
                let search_path =
                    resolve_to_cwd(input.path.as_deref().unwrap_or("."), &effective_cwd);
                let effective_limit = input
                    .limit
                    .map(|limit| limit as usize)
                    .unwrap_or(DEFAULT_LIMIT);
                let ops: Arc<dyn FindOperations> = custom_ops
                    .clone()
                    .unwrap_or_else(|| Arc::new(DefaultFindOperations));

                // If custom operations provide glob(), use that instead of fd.
                if custom_ops.is_some() {
                    if !ops.exists(search_path.clone()).await {
                        return Err(format!("Path not found: {search_path}"));
                    }
                    if signal.as_ref().is_some_and(AbortSignal::aborted) {
                        return Err("Operation aborted".to_string());
                    }
                    let results = ops
                        .glob(
                            input.pattern.clone(),
                            search_path.clone(),
                            FindGlobOptions {
                                ignore: vec![
                                    "**/node_modules/**".to_string(),
                                    "**/.git/**".to_string(),
                                ],
                                limit: effective_limit,
                            },
                        )
                        .await?;
                    if signal.as_ref().is_some_and(AbortSignal::aborted) {
                        return Err("Operation aborted".to_string());
                    }
                    if results.is_empty() {
                        return Ok(AgentToolResult::new(
                            vec![pi_ai::types::UserContent::from(
                                pi_ai::types::TextContent::new("No files found matching pattern"),
                            )],
                            JsonValue::Null,
                        ));
                    }

                    // Relativize paths against the search root for stable output.
                    let relativized: Vec<String> = results
                        .iter()
                        .map(|p| relativize_find_result_path(p, &search_path))
                        .collect();
                    let (result_output, details) = build_find_output(
                        relativized,
                        effective_limit,
                        format!("{effective_limit} results limit reached"),
                    );
                    return Ok(AgentToolResult::new(
                        vec![pi_ai::types::UserContent::from(
                            pi_ai::types::TextContent::new(result_output),
                        )],
                        details,
                    ));
                }

                // Default implementation uses fd.
                let Some(fd_path) = ensure_tool_detached(ToolName::Fd).await else {
                    return Err("fd is not available and could not be downloaded".to_string());
                };
                if signal.as_ref().is_some_and(AbortSignal::aborted) {
                    return Err("Operation aborted".to_string());
                }

                let mut args: Vec<String> = vec![
                    "--glob".to_string(),
                    "--color=never".to_string(),
                    "--hidden".to_string(),
                ];

                // fd normally ignores .gitignore outside git repos, so keep --no-require-git
                // there. Inside repos, use fd's default git-aware behavior so parent
                // .gitignore rules stop at nested repo boundaries:
                // https://github.com/earendil-works/pi/issues/5960
                let inside_git_repo = find_git_root(&search_path).await;
                if !inside_git_repo {
                    args.push("--no-require-git".to_string());
                }
                args.push("--max-results".to_string());
                args.push(effective_limit.to_string());

                // fd --glob matches against the basename unless --full-path is set; in --full-path
                // mode it matches against the absolute candidate path, so a path-containing
                // pattern like 'src/**/*.spec.ts' needs a leading '**/' to match anything.
                let mut effective_pattern = input.pattern.clone();
                if input.pattern.contains('/') {
                    args.push("--full-path".to_string());
                    if !input.pattern.starts_with('/')
                        && !input.pattern.starts_with("**/")
                        && input.pattern != "**"
                    {
                        effective_pattern = format!("**/{effective_pattern}");
                    }
                    // fd matches full paths using native separators on Windows.
                    if cfg!(windows) {
                        effective_pattern = effective_pattern.replace('/', "[/\\\\]");
                    }
                }
                args.push("--".to_string());
                args.push(effective_pattern);
                args.push(search_path.clone());

                let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
                let mut child = spawn_process(&fd_path, &arg_refs)
                    .map_err(|e| format!("Failed to run fd: {e}"))?;
                let stdout = child.stdout.take();
                let stderr_handle = child.stderr.take();

                // Abort watcher (upstream addEventListener("abort") → stopChild).
                let (kill_tx, kill_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
                if let Some(signal) = signal.clone() {
                    let kill_tx = kill_tx.clone();
                    tokio::spawn(async move {
                        signal.aborted_future().await;
                        let _ = kill_tx.send(());
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

                let mut lines: Vec<String> = Vec::new();
                {
                    let mut kill_rx = Some(kill_rx);
                    if let Some(stdout) = stdout {
                        let mut lines_stream = tokio::io::BufReader::new(stdout).lines();
                        loop {
                            tokio::select! {
                                line = lines_stream.next_line() => match line {
                                    Ok(Some(line)) => lines.push(line),
                                    Ok(None) | Err(_) => break,
                                },
                                kill = async { kill_rx.as_mut().unwrap().recv().await }, if kill_rx.is_some() => {
                                    if kill.is_none() {
                                        kill_rx = None;
                                    } else {
                                        let _ = child.start_kill();
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
                let output = lines.join("\n");
                let code = status.ok().and_then(|status| status.code());
                if code != Some(0) {
                    let error_msg = if stderr.trim().is_empty() {
                        format!(
                            "fd exited with code {}",
                            code.map(|c| c.to_string())
                                .unwrap_or_else(|| "null".to_string())
                        )
                    } else {
                        stderr.trim().to_string()
                    };
                    if output.is_empty() {
                        return Err(error_msg);
                    }
                }
                if output.is_empty() {
                    return Ok(AgentToolResult::new(
                        vec![pi_ai::types::UserContent::from(
                            pi_ai::types::TextContent::new("No files found matching pattern"),
                        )],
                        JsonValue::Null,
                    ));
                }

                let mut relativized: Vec<String> = Vec::new();
                for raw_line in &lines {
                    let line = raw_line.strip_suffix('\r').unwrap_or(raw_line).trim();
                    if line.is_empty() {
                        continue;
                    }
                    relativized.push(relativize_find_result_path(line, &search_path));
                }

                let (result_output, details) = build_find_output(
                    relativized,
                    effective_limit,
                    format!(
						"{effective_limit} results limit reached. Use limit={} for more, or refine pattern",
						effective_limit * 2
					),
                );
                Ok(AgentToolResult::new(
                    vec![pi_ai::types::UserContent::from(
                        pi_ai::types::TextContent::new(result_output),
                    )],
                    details,
                ))
            })
        },
    );
    ToolDefinition {
		name: "find".to_string(),
		label: "find".to_string(),
		description: format!(
			"Search for files by glob pattern. Returns matching file paths relative to the search directory. Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} results or {}KB (whichever is hit first).",
			DEFAULT_MAX_BYTES / 1024
		),
		prompt_snippet: Some(FIND_TOOL_SYSTEM_PROMPT_SNIPPET.to_string()),
		prompt_guidelines: None,
		parameters: find_tool_input_schema(),
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

/// `createFindTool`.
pub fn create_find_tool(cwd: &str, options: Option<FindToolOptions>) -> AgentTool {
    wrap_tool_definition(create_find_tool_definition(cwd, options), None)
}
