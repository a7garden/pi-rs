//! Port of `packages/agent/src/stream-fn.ts` — the default `StreamFn`
//! registry used by [`crate::agent::Agent`] and the low-level loops when
//! callers omit one.
//!
//! Deviation (crate docs D2): upstream's unset lookup throws; Rust panics
//! with the identical message at the same call point.

use parking_lot::Mutex;
use std::sync::LazyLock;

use crate::types::StreamFn;

static DEFAULT_STREAM_FN: LazyLock<Mutex<Option<StreamFn>>> = LazyLock::new(|| Mutex::new(None));

/// Configure the fallback used by Agent and low-level loops when callers
/// omit streamFn.
///
/// Hosts that provide a default model runtime can install its stream
/// function here without making pi-agent depend on a provider catalog or
/// compatibility layer.
pub fn set_default_stream_fn(stream_fn: Option<StreamFn>) {
	*DEFAULT_STREAM_FN.lock() = stream_fn;
}

/// Upstream `getDefaultStreamFn`.
///
/// # Panics
/// With the upstream error message when no default is configured.
pub fn get_default_stream_fn() -> StreamFn {
	DEFAULT_STREAM_FN.lock().clone().unwrap_or_else(|| {
		panic!(
			"No default stream function configured. Pass streamFn explicitly or call setDefaultStreamFn()."
		)
	})
}
