//! Port of `packages/coding-agent/src/core/tools/tool-definition-wrapper.ts`
//! — bridge erased [`ToolDefinition`]s into [`pi_agent::AgentTool`] for the
//! core runtime.

use std::sync::Arc;

use pi_agent::{AgentTool, AgentToolExecute};
use serde_json::Value as JsonValue;

use crate::core::extensions::types::{ExtensionToolContext, ToolDefinition, ToolExecute};
use crate::utils::abort::AbortSignal;

/// Creates the context for one tool call (`ToolContextFactory`).
pub type ToolContextFactory =
    Arc<dyn Fn(&str, Option<AbortSignal>) -> ExtensionToolContext + Send + Sync>;

/// Bridge the crate-local [`AbortSignal`] (D9) onto the pi-ai signal the
/// agent loop's [`pi_agent::AgentToolExecute`] expects. A task forwards the
/// abort; the forwarder finishes on its own afterwards.
pub(crate) fn bridge_signal_to_ai(
    signal: Option<AbortSignal>,
) -> Option<pi_ai::utils::abort::AbortSignal> {
    let signal = signal?;
    let forwarded = pi_ai::utils::abort::AbortSignal::new();
    let target = forwarded.clone();
    tokio::spawn(async move {
        signal.aborted_future().await;
        pi_ai::utils::abort::AbortSignal::abort(&target);
    });
    Some(forwarded)
}

/// Bridge a pi-ai signal back onto the crate-local type (reverse of
/// [`bridge_signal_to_ai`]).
fn bridge_signal_from_ai(signal: Option<pi_ai::utils::abort::AbortSignal>) -> Option<AbortSignal> {
    let signal = signal?;
    let controller = crate::utils::abort::AbortController::new();
    let forwarded = controller.signal().clone();
    tokio::spawn(async move {
        signal.aborted_future().await;
        controller.abort();
    });
    Some(forwarded)
}

/// Wrap a [`ToolDefinition`] into an [`AgentTool`] (`wrapToolDefinition`).
/// Without a context factory, the definition's execute receives `None` and
/// falls back to its constructor cwd (upstream passes `undefined`).
pub fn wrap_tool_definition(
    definition: ToolDefinition,
    ctx_factory: Option<ToolContextFactory>,
) -> AgentTool {
    let ToolDefinition {
        name,
        label,
        description,
        parameters,
        constrained_sampling,
        output_schema,
        prepare_arguments,
        execution_mode,
        execute,
        ..
    } = definition;
    let execute = Arc::clone(&execute);
    let agent_execute: AgentToolExecute = Arc::new(
        move |tool_call_id: String,
              params: serde_json::Map<String, JsonValue>,
              signal: Option<pi_ai::utils::abort::AbortSignal>,
              on_update| {
            let crate_signal = bridge_signal_from_ai(signal.clone());
            let ctx = ctx_factory
                .as_ref()
                .map(|factory| factory(&tool_call_id, crate_signal.clone()));
            let params: JsonValue = params.into();
            let execute = Arc::clone(&execute);
            Box::pin(
                async move { execute(tool_call_id, params, crate_signal, on_update, ctx).await },
            )
        },
    );
    AgentTool {
        tool: pi_ai::types::Tool {
            name,
            description,
            parameters,
            constrained_sampling,
        },
        label,
        prepare_arguments,
        output_schema,
        execute: agent_execute,
        replay: None,
        execution_mode,
    }
}

/// Wrap multiple definitions (`wrapToolDefinitions`).
pub fn wrap_tool_definitions(
    definitions: Vec<ToolDefinition>,
    ctx_factory: Option<ToolContextFactory>,
) -> Vec<AgentTool> {
    definitions
        .into_iter()
        .map(|definition| wrap_tool_definition(definition, ctx_factory.clone()))
        .collect()
}

/// Synthesize a minimal [`ToolDefinition`] from an [`AgentTool`]
/// (`createToolDefinitionFromAgentTool`). This keeps AgentSession's internal
/// registry definition-first even when a caller provides plain `AgentTool`
/// overrides without prompt metadata or renderers.
pub fn create_tool_definition_from_agent_tool(tool: &AgentTool) -> ToolDefinition {
    let AgentTool {
        tool: declaration,
        label,
        prepare_arguments,
        output_schema,
        execute,
        execution_mode,
        ..
    } = tool;
    let execute = Arc::clone(execute);
    let definition_execute: ToolExecute = Arc::new(
        move |tool_call_id: String,
              params: JsonValue,
              signal: Option<AbortSignal>,
              on_update,
              _ctx: Option<ExtensionToolContext>| {
            let signal = bridge_signal_to_ai(signal);
            let Ok(params) = serde_json::from_value::<serde_json::Map<String, JsonValue>>(params)
            else {
                return Box::pin(async { Err("tool arguments must be an object".to_string()) });
            };
            let execute = Arc::clone(&execute);
            Box::pin(async move { execute(tool_call_id, params, signal, on_update).await })
        },
    );
    ToolDefinition {
        name: declaration.name.clone(),
        label: label.clone(),
        description: declaration.description.clone(),
        prompt_snippet: None,
        prompt_guidelines: None,
        parameters: declaration.parameters.clone(),
        constrained_sampling: declaration.constrained_sampling.clone(),
        render_shell: None,
        prepare_arguments: prepare_arguments.clone(),
        output_schema: output_schema.clone(),
        exposure: None,
        namespace: None,
        annotations: None,
        default_active: None,
        prepare_loadout: None,
        execution_mode: *execution_mode,
        execute: definition_execute,
        render_call: None,
        render_result: None,
    }
}
