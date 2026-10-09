//! Port of `packages/coding-agent/src/extensions/` — the built-in extension
//! packages shipped with the coding agent.
//!
//! | Upstream | Port | Status |
//! |---|---|---|
//! | `extensions/mcp/` | [`mcp`] | see [`mcp`] docs (D80–D89) |
//! | `extensions/tool-search/` | [`tool_search`] | see [`tool_search`] docs (D95–D96) |
//! | `extensions/llama/` | [`llama`] | see [`llama`] docs (D97–D99) |
//! | `extensions/codemode/` | [`codemode`] | see [`codemode`] docs (D90–D94) |
//!
//! JS-entry files (each package's `index.ts` extension entry) stay behind the
//! D40 [`core::extensions::loader`] seam: their Rust-observable surface
//! (config parsing, tool/command/resource factories) is ported as native
//! factories; the JS module evaluation itself defers with the pi-runtime JS
//! host.

pub mod codemode;
pub mod llama;
pub mod mcp;
pub mod tool_search;
