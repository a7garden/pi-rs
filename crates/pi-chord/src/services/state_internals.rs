//! Port of `packages/chord/src/services/state-internals.ts`.
//!
//! Upstream registers internals in a `WeakMap` keyed by the state object. Rust
//! states expose internals through [`StateInternals`] trait objects handed to the
//! provider classification directly, so the registry is unnecessary; the trait
//! and its two accessors are the ported surface.

use crate::context::Context;
use crate::delta::Op;
use crate::json::JsonValue;
use crate::types::Unsubscribe;

/// Listener receiving published operation batches with their sequence numbers.
pub type SourceListener = Arc<dyn Fn(&[Op], u64, &Context) + Send + Sync>;

use std::sync::Arc;

/// Internals through which a provider observes one replicated state.
pub trait StateInternals: Send + Sync {
    /// Atomically capture the immutable value and its matching publication sequence.
    fn snapshot(&self) -> StateSnapshot;
    /// Subscribe to every published revision batch in order.
    fn subscribe(&self, listener: SourceListener) -> Unsubscribe;
}

#[derive(Clone)]
pub struct StateSnapshot {
    pub value: JsonValue,
    pub sequence: u64,
}
