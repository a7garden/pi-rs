//! Port of upstream `experimental/durable/` — the durable pi client cluster:
//! harness assembly, sessions directory, pi prompt extension, the foreground
//! subagent extension, and the headless `openDurable` runtime.
//!
//! Slice map:
//! - [`harness_setup`] — `durable/harness-setup.ts`
//! - [`prompt`] — `durable/prompt.ts`
//! - [`runtime`] — `durable/runtime.ts` (hosts the view/controller machinery
//!   shared with `vacation/runtime.ts`, parameterized by registry + root
//!   agent injection)
//! - [`sessions`] — `durable/sessions.ts` (hosts the shared `selectSession`
//!   body behind the session-space name)
//! - [`subagent`] — `durable/subagent.ts`
//!
//! `durable/tui.ts` and `durable/main.ts` defer with interactive mode (TUI
//! exclusion, D152 policy); `open_durable` is the TUI-free entry surface.

pub mod harness_setup;
pub mod prompt;
pub mod runtime;
pub mod sessions;
pub mod subagent;
