//! Port of upstream `experimental/` — the experimental client/server service
//! split.
//!
//! Wave slices:
//! - **S1 `services/`** ([`services`]) — contracts, providers, connection,
//!   server, worker (D138, D144–D148).
//! - **S2 process foundation** — [`process`], [`coordinator`],
//!   [`coordinator_entry`], [`session_catalog`] (D138–D141).
//! - **S3 `server-cluster`** — [`server`], [`session_worker`],
//!   [`session_worker_manager`], [`plugins`], [`radius_auth`],
//!   [`radius_relay`] (D149–D152).
//! - **S4a `app-clusters`** — [`durable`], [`vacation`] (D170–D176), plus
//!   the D151 session-worker factory release in [`session_worker`].
//! - **S5 `headless-client`** — [`client_runtime`], [`client`],
//!   [`commands`], [`cli`] (D177–D182): the non-interactive client runtime,
//!   the development-only `pi experimental server|client` dispatch, and the
//!   dev entry seam.
//!
//! Deferred with interactive mode (TUI-coupled): `client-tui.ts`,
//! `client-tui-chat.ts`, `durable/tui.ts` (+`main.ts`), `vacation/tui.ts`
//! (+`main.ts`), `source-resolver.ts` (not needed in Rust), and the
//! presentation-side `createPresentationFacetLoaders` of `plugins/bundled.ts`
//! (D152). The `client-tui` dispatch branch of `commands.ts` defers with
//! them (D178).

pub mod cli;
pub mod client;
pub mod client_runtime;
pub mod commands;
pub mod coordinator;
pub mod coordinator_entry;
pub mod durable;
pub mod lockfile;
pub mod plugin;
pub mod plugins;
pub mod process;
pub mod radius_auth;
pub mod radius_relay;

pub mod services;
pub mod server;
pub mod session_catalog;
pub mod session_worker;
pub mod session_worker_manager;
pub mod vacation;



/// Serializes tests that touch process environment variables (upstream tests
/// run in one process per file; Rust lib tests share one process). Also read
/// by sibling modules' tests.
#[cfg(test)]
pub static TEST_GUARD: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
