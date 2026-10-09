//! Port of `experimental/vacation/harness-setup.ts`.
//!
//! Deviations:
//! - **D175 — the shared harness-assembly surface re-exports from
//!   `durable::harness_setup`.** Upstream `configureHarnessHttp`,
//!   `createHarnessSettings`, `InitialModel`, and `findInitialAgentModel`
//!   are byte-identical between the two clusters; the port implements them
//!   once (D173 notes apply verbatim). Only `createVacationRegistry` is
//!   local.

pub use crate::experimental::durable::harness_setup::{
	configure_harness_http, create_harness_settings, create_model_runtime, create_settings_manager,
	find_initial_agent_model, CliModel, InitialModel, SharedSettingsManager,
};

use pi_durable::harness::registry::{create_registry, RegistryImpl};
use pi_durable::harness::types::Registry;

use super::vacation::{search_extension, vacation_extension};

/// A registry with the vacation planner and its research subagent's search.
/// No coding tools, no pi prompt (upstream `createVacationRegistry`).
pub fn create_vacation_registry() -> RegistryImpl {
	let registry = create_registry();
	registry.install(vacation_extension());
	registry.install(search_extension());
	registry
}
