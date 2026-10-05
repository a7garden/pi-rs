//! Port of `packages/ai/src/providers/` (the provider factories assigned to
//! this wave: anthropic, openai, meta, typesafe, plus the faux test provider
//! and their generated model catalogs).
//!
//! Upstream `all.ts` wires every built-in provider into the global provider
//! registry; only the five in-scope providers exist here, exposed through
//! [`builtin_provider_factories`] in registration order matching upstream
//! `all.ts` (anthropic, meta, openai, typesafe). Faux is not part of the
//! upstream registry — it is constructed explicitly by tests — so it is not
//! in the hook.
//!
//! Deviations continue from the crate docs: D15/D16 (anthropic.rs), D17
//! (factory `expect`s), D21 (faux registry omitted).

pub mod anthropic;
pub mod anthropic_models;
pub mod faux;
pub mod meta;
pub mod meta_models;
pub mod openai;
pub mod openai_models;
pub mod typesafe;
pub mod typesafe_models;

use std::sync::Arc;

use crate::models::Provider;

/// A built-in provider factory: `(provider id, constructor)` mirroring the
/// upstream `all.ts` registration list for the in-scope providers.
pub fn builtin_provider_factories() -> Vec<(&'static str, fn() -> Arc<dyn Provider>)> {
	vec![
		("anthropic", anthropic::anthropic_provider),
		("meta", meta::meta_provider),
		("openai", openai::openai_provider),
		("typesafe", typesafe::typesafe_provider),
	]
}
