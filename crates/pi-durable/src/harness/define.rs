//! Port of `packages/durable/src/harness/define.ts`.
//!
//! Deviations: D1 — erased generics; `hook` takes the task's registered name
//! plus a handler table object (`Arc<dyn HookHandler>`).

use std::sync::Arc;

use crate::harness::types::{
    Extension, HookRegistration, PromptSection, SectionRender, SectionWrap, ToolRegistration,
    ToolWrap, Wrap,
};
use crate::tasks::HookHandler;

/// Identity function that types an extension (erased; returns the value).
pub fn define_extension(extension: Extension) -> Extension {
    extension
}

/// Identity function that types a tool: `args` from `parameters`, details
/// from what it reports (erased; returns the registration).
pub fn define_tool_of(tool: ToolRegistration) -> ToolRegistration {
    tool
}

/// A prompt section; tagged unless `tag` is false.
pub fn section(key: impl Into<String>, render: SectionRender, tag: Option<bool>) -> PromptSection {
    PromptSection {
        key: key.into(),
        render,
        tag,
    }
}

/// Hook handlers for tasks with `task`'s name.
pub fn hook(
    task: &crate::harness::types::AnyTask,
    handlers: Arc<dyn HookHandler>,
) -> HookRegistration {
    HookRegistration {
        task: task.definition.name.clone(),
        handlers,
    }
}

/// Wrap the tool named like `tool` wherever the wrapping extension is
/// selected.
pub fn wrap_tool(
    tool: &ToolRegistration,
    wrapper: impl Fn(Arc<ToolRegistration>) -> Arc<ToolRegistration> + Send + Sync + 'static,
) -> Wrap {
    Wrap::Tool(ToolWrap {
        tool: tool.tool.name.clone(),
        wrap: Arc::new(wrapper),
    })
}

/// Wrap the section `key` wherever the wrapping extension is selected.
pub fn wrap_section(
    key: &str,
    wrapper: impl Fn(PromptSection) -> PromptSection + Send + Sync + 'static,
) -> Wrap {
    Wrap::Section(SectionWrap {
        section: key.to_string(),
        wrap: Arc::new(wrapper),
    })
}
