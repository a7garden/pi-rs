//! Port of `packages/coding-agent/src/core/extensions/` — the extension
//! system: [`types`] mirrors upstream `extensions/types.ts` (contexts, tool
//! definitions, events, the `pi.*` API surface, loader/runtime shapes),
//! [`loader`] is the loader stand-in per its D43 contract (runtime
//! construction, cache registry, factory pipeline; the jiti module seam is
//! D40), and [`runner`] ports `runner.ts` (lifecycle dispatch, shortcut
//! conflicts, contexts). `wrapper.ts` / `index.ts` / `virtual-modules.ts`
//! ports are [`wrapper`] / [`index`] / [`virtual_modules`].
//!
//! Extensions themselves run through the embedded JS runtime (pi-runtime),
//! so the Rust types here are the host-side model: serde data types for
//! everything crossing the bridge, erased `Arc<dyn Fn>` callbacks for the
//! behavior the host supplies (see crate docs D27–D31 and D40+).

pub mod index;
pub mod loader;
pub mod runner;
pub mod types;
pub mod virtual_modules;
pub mod wrapper;
