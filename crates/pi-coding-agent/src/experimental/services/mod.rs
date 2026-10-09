//! Port of `experimental/services/` — the chord-composed service layer for the
//! experimental client/server split: contracts, providers, and the assembly
//! set (connection, server, worker).
//!
//! Upstream files (S1 slice):
//! - `services/{sessions,models,plugins,presentation-ui,slash-commands,
//!   transcript,agent-controller}.ts` → the contract modules.
//! - `services/{agent-controller,models,slash-commands,transcript}-provider.ts`
//!   → the provider modules.
//! - `services/{connection,server,worker}.ts` → the assembly modules.
//! - `experimental/plugin.ts` → [`crate::experimental::plugin`].
//!
//! TUI-coupled portions are deferred with interactive mode: the
//! `PresentationUI` implementations (alternate-screen renderer integration and
//! selection dialogs) do not exist here — only the contract and the local
//! service token port.
//!
//! Slice deviations:
//! - **D144 — structural services → typed tokens.** Upstream services are
//!   structural objects behind `defineService<T>`; Rust tokens carry a marker
//!   (remote-only services), a concrete implementation type (local services),
//!   or a trait-object handle (`PresentationUI`). Remote facets expose
//!   upstream object methods as JSON members through
//!   `RemoteServiceObject::builder`; state members use chord's untyped JSON
//!   replicated states with typed DTOs at the seam.
//! - **D145 — busy detection.** pi-durable Rust errors arrive stringified, so
//!   the typed `ConversationBusy → { code: "busy" }` mapping checks the stable
//!   durable display text (`… is busy`).
//! - **D146 — timestamps.** `new Date().toISOString()` is formatted by a local
//!   RFC 3339 writer (no chrono dependency).
//! - **D147 — provider sharing.** `FacetHost::provider_arc` (pi-chord,
//!   additive default method) shares the assembled provider as an `Arc` for
//!   per-scope endpoints.
//! - **D148 — requirement cloning.** `LocalRequirement` is not `Clone`; the
//!   reload command lends its requirement out of a mutex cell for the
//!   synchronous handle acquisition.

pub mod agent_controller;
pub mod agent_controller_provider;
pub mod connection;
pub mod models;
pub mod models_provider;
pub mod plugins;
pub mod presentation_ui;
pub mod server;
pub mod sessions;
pub mod slash_commands;
pub mod slash_commands_provider;
pub mod transcript;
pub mod transcript_provider;
pub mod worker;

// The upstream `README.md` continuation table is documentation, not code; the
// server/client entrypoints (`pi-test.sh` flows) defer with the CLI slices.
