//! Port of `packages/coding-agent/src/extensions/tool-search/` — the
//! `tool_search` extension: BM25 discovery over deferred tool metadata.
//!
//! | Upstream | Port | Status |
//! |---|---|---|
//! | `tool-search/index.ts` | [`self`] ([`create_tool_search_extension`]) | ported (factory; see D40 note below) |
//! | `tool-search/tool.ts` | [`tool`] | ported (BM25 ranker, document builder, tool definition) |
//! | `test/tool-search.test.ts` | inline `#[cfg(test)]` modules | ported 1:1 (all 8 cases) |
//!
//! # Factory (index.ts)
//!
//! The CLI loads the extension as a built-in; SDK users add
//! [`create_tool_search_extension()`] to their extension factories. The
//! registration itself is fully native — `pi.registerTool` with
//! `defaultActive: false` — so unlike the other packages' JS entries nothing
//! defers to the D40 JS-module seam here beyond the loader's own extension
//! evaluation boundary.
//!
//! `tool_search` is registered inactive. Activate it with `--tools`, the
//! `defaultTools` setting, or `setActiveTools()`.
//!
//! # Deviations
//!
//! - **D95**/**D95a**/**D95b** — see [`tool`] module docs (schema
//!   representation, `limit` typing, ranker signatures).
//! - **D96** — upstream's `default export createToolSearchExtension()` (a
//!   pre-built factory value) becomes the [`create_tool_search_extension()`]
//!   constructor; `ExtensionFactory` in Rust is a shared `Arc` closure, so
//!   the module exports the constructor instead of a value.

pub mod tool;

use crate::core::extensions::types::{ExtensionApiHandle, ExtensionFactory};
use tool::{ApiToolSource, ToolSearchToolOptions};

/// Build the `tool_search` extension factory (`createToolSearchExtension`).
pub fn create_tool_search_extension() -> ExtensionFactory {
    std::sync::Arc::new(|pi: ExtensionApiHandle| {
        Box::pin(async move {
            let mut definition = tool::create_tool_search_tool_definition(ToolSearchToolOptions {
                tools: Some(std::sync::Arc::new(ApiToolSource::new(pi.clone()))),
            });
            definition.default_active = Some(false);
            pi.register_tool(definition);
            Ok(())
        })
    })
}
