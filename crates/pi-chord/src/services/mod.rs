//! Wiring for `packages/chord/src/services/*` (upstream has no `services/index.ts`;
//! members are imported directly).

pub mod consumer;
pub mod errors;
pub mod handle;
pub mod instances;
pub mod loopback;
pub mod provider;
pub mod state;
pub mod state_codec;
pub mod state_internals;
pub mod wire;
