//! Port of upstream `experimental/vacation/` — the vacation-planner demo
//! cluster: a tool-only registry (no coding tools, no pi prompt), a
//! background research task that reports back, and the headless
//! `openDurable` runtime.
//!
//! Slice map:
//! - [`harness_setup`] — `vacation/harness-setup.ts` (the shared
//!   harness-assembly surface re-exports from `durable`; only
//!   `createVacationRegistry` is local)
//! - [`vacation`] — `vacation.ts` (`search`/`research` tools, the
//!   `vacation.research` task, the `Search`/`Vacation` extensions)
//! - [`runtime`] — `vacation/runtime.ts` (shared machinery from
//!   `durable::runtime`, parameterized per D174)
//! - [`sessions`] — `vacation/sessions.ts` (shared `selectSession` body per
//!   D170)
//!
//! `vacation/tui.ts` and `vacation/main.ts` defer with interactive mode
//! (TUI exclusion, D152 policy).

pub mod harness_setup;
pub mod runtime;
pub mod sessions;
pub mod vacation;
