//! Port of `packages/coding-agent/src/core/extensions/wrapper.ts` — tool
//! wrappers for extension-registered tools. These wrappers only adapt tool
//! execution so extension tools receive the runner context; tool call and
//! tool result interception is handled by AgentSession via agent-core hooks.

use std::sync::Arc;

use pi_agent::AgentTool;

use super::runner::ExtensionRunner;
use super::types::RegisteredTool;
use crate::core::tools::tool_definition_wrapper::wrap_tool_definition;

/// Wrap a [`RegisteredTool`] into an [`AgentTool`] (`wrapRegisteredTool`).
/// Uses the runner's `create_tool_context()` for consistent context across
/// tools and event handlers.
pub fn wrap_registered_tool(
    registered_tool: RegisteredTool,
    runner: Arc<ExtensionRunner>,
) -> AgentTool {
    wrap_tool_definition(
        registered_tool.definition,
        Some(Arc::new(move |tool_call_id, signal| {
            runner
                .create_tool_context(tool_call_id, signal)
                .unwrap_or_else(|message| panic!("{message}"))
        })),
    )
}

/// Wrap all registered tools into `AgentTool`s (`wrapRegisteredTools`).
pub fn wrap_registered_tools(
    registered_tools: Vec<RegisteredTool>,
    runner: Arc<ExtensionRunner>,
) -> Vec<AgentTool> {
    registered_tools
        .into_iter()
        .map(|tool| wrap_registered_tool(tool, Arc::clone(&runner)))
        .collect()
}

#[cfg(test)]
mod wrapper_tests {
    use super::*;
    use crate::core::extensions::loader::create_extension_runtime;
    use crate::core::extensions::types::{RegisteredTool, ToolDefinition};
    use serde_json::json;

    #[tokio::test]
    async fn wrapped_tool_receives_runner_context() {
        let runtime = create_extension_runtime();
        let runner =
            Arc::new(super::super::runner::runner_tests::test_runner_for_wrapper_tests(runtime));
        let mut definition =
            ToolDefinition::new("echo", "Echo", "Echoes.", json!({"type": "object"}));
        definition.execute = Arc::new(move |_tool_call_id, _args, _signal, _on_update, ctx| {
            Box::pin(async move {
                assert!(
                    ctx.is_some(),
                    "extension tools receive the runner tool context"
                );
                Ok(pi_agent::AgentToolResult {
                    content: Vec::new(),
                    details: json!({}),
                    structured_content: None,
                    is_error: Some(false),
                    terminate: None,
                    usage: None,
                })
            })
        });
        let registered = RegisteredTool {
            definition,
            source_info: crate::core::source_info::create_synthetic_source_info(
                "<wrapper>",
                "local",
                None,
                None,
                None,
            ),
        };

        let agent_tool = wrap_registered_tool(registered, Arc::clone(&runner));
        assert_eq!(agent_tool.tool.name, "echo");
    }
}
