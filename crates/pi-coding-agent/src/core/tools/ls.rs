//! Port of `upstream/pi/packages/coding-agent/src/core/tools/ls.ts` — the
//! `ls` tool (directory listing).
//!
//! Deviations:
//! - Entries sort case-insensitively via `to_lowercase()` byte comparison
//!   (D39): upstream uses `localeCompare`, which is locale-aware; a stable
//!   lowercase byte compare is the crate's stand-in.
//! - `nodePath.join` becomes [`crate::utils::paths::join`].
//! - The upstream abort-event listener rejecting mid-flight becomes
//!   `aborted()` checks at the same points (entry and between steps);
//!   operations are short so no watcher task is needed.
//! - Renderers are deferred with the interactive mode (D32).

use std::sync::Arc;

use pi_agent::{AgentTool, AgentToolResult, BoxFuture};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

use crate::core::extensions::types::{ExtensionToolContext, ToolDefinition, ToolExecute};
use crate::core::tools::path_utils::{path_exists, resolve_to_cwd};
use crate::core::tools::tool_definition_wrapper::wrap_tool_definition;
use crate::core::tools::truncate::{
    format_size, truncate_head, TruncationOptions, TruncationResult, DEFAULT_MAX_BYTES,
};
use crate::utils::abort::AbortSignal;

/// `lsSchema` (`Type.Object`) — JSON Schema for the tool parameters.
pub fn ls_tool_input_schema() -> JsonValue {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Directory to list (default: current directory)" },
            "limit": { "type": "number", "description": "Maximum number of entries to return (default: 500)" }
        }
    })
}

/// `lsToolSystemPromptContribution`.
pub const LS_TOOL_SYSTEM_PROMPT_SNIPPET: &str = "List directory contents";

/// `LsToolInput` (`Static<typeof lsSchema>`).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LsToolInput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
}

const DEFAULT_LIMIT: usize = 500;

/// `LsToolDetails`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LsToolDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_limit_reached: Option<usize>,
}

/// Stats result of [`LsOperations::stat`] (upstream `{ isDirectory() }`).
#[derive(Debug, Clone, Copy)]
pub struct LsStat {
    pub is_directory: bool,
}

/// Pluggable operations for the ls tool.
///
/// Override these to delegate directory listing to remote systems (for
/// example SSH).
pub trait LsOperations: Send + Sync {
    /// Check if path exists.
    fn exists(&self, absolute_path: String) -> BoxFuture<'static, bool>;
    /// Get file or directory stats. Throws if not found.
    fn stat(&self, absolute_path: String) -> BoxFuture<'static, Result<LsStat, String>>;
    /// Read directory entries.
    fn readdir(&self, absolute_path: String) -> BoxFuture<'static, Result<Vec<String>, String>>;
}

/// `defaultLsOperations`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalLsOperations;

impl LsOperations for LocalLsOperations {
    fn exists(&self, absolute_path: String) -> BoxFuture<'static, bool> {
        Box::pin(async move { path_exists(&absolute_path).await })
    }
    fn stat(&self, absolute_path: String) -> BoxFuture<'static, Result<LsStat, String>> {
        Box::pin(async move {
            std::fs::metadata(&absolute_path)
                .map(|metadata| LsStat {
                    is_directory: metadata.is_dir(),
                })
                .map_err(|e| e.to_string())
        })
    }
    fn readdir(&self, absolute_path: String) -> BoxFuture<'static, Result<Vec<String>, String>> {
        Box::pin(async move {
            let mut entries: Vec<String> = Vec::new();
            let mut reader = tokio::fs::read_dir(&absolute_path)
                .await
                .map_err(|e| e.to_string())?;
            while let Some(entry) = reader.next_entry().await.map_err(|e| e.to_string())? {
                entries.push(entry.file_name().to_string_lossy().into_owned());
            }
            Ok(entries)
        })
    }
}

/// `LsToolOptions`.
#[derive(Clone, Default)]
pub struct LsToolOptions {
    /// Custom operations for directory listing. Default: local filesystem
    pub operations: Option<Arc<dyn LsOperations>>,
}

/// `createLsToolDefinition`.
pub fn create_ls_tool_definition(cwd: &str, options: Option<LsToolOptions>) -> ToolDefinition {
    let ops: Arc<dyn LsOperations> = options
        .and_then(|options| options.operations)
        .unwrap_or_else(|| Arc::new(LocalLsOperations));
    let cwd = cwd.to_string();
    let execute: ToolExecute = Arc::new(
        move |_tool_call_id: String,
              args: JsonValue,
              signal: Option<AbortSignal>,
              _on_update,
              ctx: Option<ExtensionToolContext>|
              -> BoxFuture<'static, Result<AgentToolResult, String>> {
            let ops = Arc::clone(&ops);
            let cwd = cwd.clone();
            Box::pin(async move {
                if signal.as_ref().is_some_and(AbortSignal::aborted) {
                    return Err("Operation aborted".to_string());
                }
                let input: LsToolInput = serde_json::from_value(args)
                    .map_err(|e| format!("Invalid arguments for ls tool: {e}"))?;
                let effective_cwd = ctx
                    .as_ref()
                    .map(|c| c.cwd.clone())
                    .filter(|c| !c.is_empty())
                    .unwrap_or_else(|| cwd.clone());
                let dir_path = resolve_to_cwd(input.path.as_deref().unwrap_or("."), &effective_cwd);
                let effective_limit = input
                    .limit
                    .map(|limit| limit as usize)
                    .unwrap_or(DEFAULT_LIMIT);

                // Check if path exists.
                if !ops.exists(dir_path.clone()).await {
                    return Err(format!("Path not found: {dir_path}"));
                }
                if signal.as_ref().is_some_and(AbortSignal::aborted) {
                    return Err("Operation aborted".to_string());
                }

                // Check if path is a directory.
                let stat = ops
                    .stat(dir_path.clone())
                    .await
                    .map_err(|e| e.to_string())?;
                if !stat.is_directory {
                    return Err(format!("Not a directory: {dir_path}"));
                }
                if signal.as_ref().is_some_and(AbortSignal::aborted) {
                    return Err("Operation aborted".to_string());
                }

                // Read directory entries.
                let mut entries = ops
                    .readdir(dir_path.clone())
                    .await
                    .map_err(|e| format!("Cannot read directory: {e}"))?;

                // Sort alphabetically, case-insensitive (D39: lowercase byte
                // compare stands in for `localeCompare`).
                entries.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));

                // Format entries with directory indicators.
                let mut results: Vec<String> = Vec::new();
                let mut entry_limit_reached = false;
                for entry in &entries {
                    if results.len() >= effective_limit {
                        entry_limit_reached = true;
                        break;
                    }

                    let full_path = crate::utils::paths::join(&dir_path, entry);
                    let mut suffix = "";
                    match ops.stat(full_path).await {
                        // Skip entries we cannot stat.
                        Err(_) => continue,
                        Ok(entry_stat) => {
                            if entry_stat.is_directory {
                                suffix = "/";
                            }
                        }
                    }
                    results.push(format!("{entry}{suffix}"));
                }

                if signal.as_ref().is_some_and(AbortSignal::aborted) {
                    return Err("Operation aborted".to_string());
                }

                if results.is_empty() {
                    return Ok(AgentToolResult::new(
                        vec![pi_ai::types::UserContent::from(
                            pi_ai::types::TextContent::new("(empty directory)"),
                        )],
                        JsonValue::Null,
                    ));
                }

                let raw_output = results.join("\n");
                // Apply byte truncation. There is no separate line limit because entry
                // count is already capped.
                let truncation = truncate_head(
                    &raw_output,
                    TruncationOptions {
                        max_lines: Some(usize::MAX),
                        max_bytes: Some(DEFAULT_MAX_BYTES),
                    },
                );
                let mut output = truncation.content.clone();
                let mut details = LsToolDetails::default();
                // Build actionable notices for truncation and entry limits.
                let mut notices: Vec<String> = Vec::new();
                if entry_limit_reached {
                    notices.push(format!(
                        "{effective_limit} entries limit reached. Use limit={} for more",
                        effective_limit * 2
                    ));
                    details.entry_limit_reached = Some(effective_limit);
                }
                if truncation.truncated {
                    notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
                    details.truncation = Some(truncation);
                }
                if !notices.is_empty() {
                    output += &format!("\n\n[{}]", notices.join(". "));
                }

                let details_json =
                    if details.truncation.is_none() && details.entry_limit_reached.is_none() {
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
		name: "ls".to_string(),
		label: "ls".to_string(),
		description: format!(
			"List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to {DEFAULT_LIMIT} entries or {}KB (whichever is hit first).",
			DEFAULT_MAX_BYTES / 1024
		),
		prompt_snippet: Some(LS_TOOL_SYSTEM_PROMPT_SNIPPET.to_string()),
		prompt_guidelines: None,
		parameters: ls_tool_input_schema(),
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

/// `createLsTool`.
pub fn create_ls_tool(cwd: &str, options: Option<LsToolOptions>) -> AgentTool {
    wrap_tool_definition(create_ls_tool_definition(cwd, options), None)
}
