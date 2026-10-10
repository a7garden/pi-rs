//! Port of `packages/durable/src/tools` (upstream `src/tools/*.ts`).
//!
//! Tools take the environment from `api.env`, stream output through `ToolExecutionApi::output`,
//! and report remarks as diagnostics. Upstream tools throw inside `execute`; the erased seam's
//! `ToolExecute` returns a result, so each tool's fallible body returns
//! `Result<ToolExecutionResult, String>` and [`wrap_execute`] builds the error result upstream's
//! harness produces for a throw: `{ isError: true, diagnostics: [tool_error] }` (D1).
//!
//! Deviations:
//! - D1: throwing tools → fallible bodies + [`wrap_execute`]; the upstream `aborted` rethrow
//!   (which fails the task) collapses to the same error result — abort supervision stays with
//!   the harness slice.
//! - D2: schema-static input types (`BashToolInput`, `ReadToolInput`, `EditToolInput`,
//!   `WriteToolInput`) are erased to validated `serde_json::Value` (matches `types.rs` D1).
//! - D3: `unicode.rs` is a local helper, not an upstream file: it embeds UCD tables replacing
//!   JS `String#normalize` for `path-utils.ts`/`edit-diff.ts` (workspace has no unicode-
//!   normalization dependency).
//! - D4: `coding_tools()` collapses upstream `defineExtension({ ... })` into plain `Extension`
//!   construction (see `harness::api` D1).
//! - D5: `Number` semantics of read offset/limit (f64, JS truthiness/`Math.min`/`max`) are
//!   reproduced in `read.rs` (see its D1); JSON args cannot carry `NaN`/`Infinity`, so the
//!   read-differential test drives `read_text` with raw `f64` values instead.
//! - D6: upstream `tools/index.ts` re-exports collapse to the items below; helper modules
//!   (`edit_diff`, `image`, `path_utils`, `unicode`, `file_mutation_queue`, `env`) stay
//!   crate-internal because nothing outside `tools` consumes them.

mod bash;
mod edit;
pub(crate) mod edit_diff;
pub(crate) mod env;
mod file_mutation_queue;
pub(crate) mod image;
pub(crate) mod path_utils;
pub(crate) mod read;
mod unicode;
mod write;

#[cfg(test)]
mod tools_read_differential_tests;
#[cfg(test)]
mod tools_tests;

use std::future::Future;
use std::sync::Arc;

use pi_chord::context::Context;
use serde_json::Value;

use crate::harness::api::{
    ToolDiagnostic, ToolDiagnosticSeverity, ToolExecute, ToolExecutionApi, ToolExecutionResult,
};
use crate::harness::types::Extension;

// Mirrors upstream `tools/index.ts` re-exports (erased forms; see D2/D6).
pub use bash::{
    create_bash_tool, create_power_shell_tool, BashExecution, BashPrepare, BashToolInput,
    BashToolOptions, PowerShellToolInput, PowerShellToolOptions,
};
pub use edit::{create_edit_tool, EditToolDetails, EditToolInput};
pub use read::{create_read_tool, ReadToolDetails, ReadToolInput};
pub use write::{create_write_tool, WriteToolInput};

/// Maps a tool's fallible body onto the erased seam: an `Err(message)` becomes the error result
/// upstream's harness builds from a thrown error (module D1).
pub(crate) fn wrap_execute<F, Fut>(body: F) -> ToolExecute
where
    F: Fn(Value, Arc<dyn ToolExecutionApi>, Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolExecutionResult, String>> + Send + 'static,
{
    Arc::new(move |args, api, context| {
        let body = body(args, Arc::clone(&api), context);
        Box::pin(async move {
            match body.await {
                Ok(result) => result,
                Err(message) => ToolExecutionResult {
                    is_error: Some(true),
                    diagnostics: Some(vec![ToolDiagnostic {
                        severity: ToolDiagnosticSeverity::Error,
                        code: Some("tool_error".to_string()),
                        message,
                    }]),
                    ..Default::default()
                },
            }
        })
    })
}

/// Upstream `CodingTools`: `read`, `write`, `edit`, and `bash`; nothing installs it automatically.
/// `create_power_shell_tool()` adds `powershell`.
pub fn coding_tools() -> Extension {
    Extension {
        name: "coding-tools".to_string(),
        tools: vec![
            Arc::new(create_read_tool()),
            Arc::new(create_write_tool()),
            Arc::new(create_edit_tool()),
            Arc::new(create_bash_tool(None)),
        ],
        ..Default::default()
    }
}
