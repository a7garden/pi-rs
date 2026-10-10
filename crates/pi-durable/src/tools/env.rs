//! Port of `packages/durable/src/tools/env.ts` — the call's execution environment.
//!
//! D1: upstream `requireEnv` throws synchronously; Rust tools propagate
//! `Result`, so the throw becomes an `Err` carrying the same message.

use std::sync::Arc;

use crate::env::ExecutionEnv;
use crate::harness::api::ToolExecutionApi;

/// The call's execution environment; a tool without one fails with an ordinary error result.
pub fn require_env(api: &dyn ToolExecutionApi) -> Result<Arc<dyn ExecutionEnv>, String> {
    api.env()
        .ok_or_else(|| "No execution environment is configured".to_string())
}
