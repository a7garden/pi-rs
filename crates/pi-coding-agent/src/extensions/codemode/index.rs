//! Port of `packages/coding-agent/src/extensions/codemode/index.ts` — the
//! `codemode` tool as an extension factory.
//!
//! The CLI loads it as a built-in extension; SDK users add
//! [`create_codemode_extension`] to their extension factories. The JS-entry
//! (module evaluation) half stays behind the D40
//! [`crate::core::extensions::loader`] seam; this module ports the native
//! factory surface: options reading and the tool registration.
//!
//! `codemode` is registered inactive (`default_active: false`). Activate it
//! with `--tools`, the `defaultTools` setting, or `setActiveTools()`; the
//! MCP extension activates it when MCP tools are only reachable from
//! scripts.

use std::sync::Arc;

use serde_json::Value;

use crate::core::extensions::types::{ExtensionApiHandle, ExtensionFactory};
use crate::core::settings_manager::CodemodeMode;
use crate::extensions::codemode::tool::{create_codemode_tool_definition, CodemodeToolOptions};

/// Options of [`create_codemode_extension`] (`CodemodeExtensionOptions`).
#[derive(Debug, Clone, Default)]
pub struct CodemodeExtensionOptions {
    /// Overrides the `codemode.mode` setting.
    pub mode: Option<CodemodeMode>,
    /// Overrides the `codemode.inlineBudget` setting.
    pub inline_budget: Option<f64>,
    /// Expose the model catalog and classifiers to scripts as `models`.
    /// Default: `true`.
    pub models: Option<bool>,
}

/// `readMode(pi)`: the `codemode.mode` setting, `only` or `on`.
fn read_mode(pi: &ExtensionApiHandle) -> CodemodeMode {
    let settings = pi.get_settings();
    match settings
        .get("codemode")
        .and_then(|codemode| codemode.get("mode"))
        .and_then(Value::as_str)
    {
        Some("only") => CodemodeMode::Only,
        _ => CodemodeMode::On,
    }
}

/// `readInlineBudget(pi)`: the `codemode.inlineBudget` setting when it is a
/// finite non-negative number.
fn read_inline_budget(pi: &ExtensionApiHandle) -> Option<f64> {
    pi.get_settings()
        .get("codemode")
        .and_then(|codemode| codemode.get("inlineBudget"))
        .and_then(Value::as_f64)
        // `typeof budget === "number" && Number.isFinite(budget) && budget >= 0`.
        .filter(|budget| budget.is_finite() && *budget >= 0.0)
}

/// `createCodemodeExtension(options)`: registers the `codemode` tool.
pub fn create_codemode_extension(options: CodemodeExtensionOptions) -> ExtensionFactory {
    Arc::new(move |pi: ExtensionApiHandle| {
        let tool_options = CodemodeToolOptions {
            get_tool_namespace: {
                let pi = pi.clone();
                Some(Arc::new(move |tool_name: &str| {
                    pi.get_all_tools()
                        .into_iter()
                        .find(|tool| tool.name == tool_name)
                        .and_then(|tool| tool.namespace)
                }))
            },
            get_tool_guidelines: {
                let pi = pi.clone();
                Some(Arc::new(move || {
                    pi.get_all_tools()
                        .into_iter()
                        .map(|tool| (tool.name, tool.prompt_guidelines.unwrap_or_default()))
                        .collect()
                }))
            },
            models: Some(options.models.unwrap_or(true)),
            append_entry: {
                let pi = pi.clone();
                Some(Arc::new(move |custom_type: &str, data: Option<Value>| {
                    pi.append_entry(custom_type, data)
                }))
            },
            get_mode: {
                let pi = pi.clone();
                let mode = options.mode.clone();
                Some(Arc::new(move || {
                    mode.clone().unwrap_or_else(|| read_mode(&pi))
                }))
            },
            get_inline_budget: {
                let pi = pi.clone();
                let budget = options.inline_budget;
                Some(Arc::new(move || budget.or_else(|| read_inline_budget(&pi))))
            },
            ..Default::default()
        };
        let mut definition = create_codemode_tool_definition(tool_options);
        definition.default_active = Some(false);
        pi.register_tool(definition);
        Box::pin(async { Ok(()) })
    })
}
