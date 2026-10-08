//! Port of `packages/coding-agent/src/core/extensions/` — the extension
//! system. Only the type foundation is ported so far: [`types`] mirrors
//! upstream `extensions/types.ts` (contexts, tool definitions, events, the
//! `pi.*` API surface, loader/runtime shapes). The runner/loader
//! (`loader.ts`, `runner.ts`, `api.ts`, …) lands in a later pass on top of
//! these types; `core/tools/*` ports against [`types::ToolDefinition`].
//!
//! Extensions themselves run through the embedded JS runtime (pi-runtime),
//! so the Rust types here are the host-side model: serde data types for
//! everything crossing the bridge, erased `Arc<dyn Fn>` callbacks for the
//! behavior the host supplies (see crate docs D27–D31).

pub mod types;
