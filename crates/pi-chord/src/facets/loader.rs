//! Port of `packages/chord/src/facets/loader.ts`.

use crate::services::errors::AnyError;
use crate::types::LoadedFacets;

/// Dispose every loaded facet generation and collect the failures.
///
/// Upstream awaits `Promise.allSettled` over the disposals; the Rust port runs
/// them sequentially (the combinators pass the array pre-reversed, so reverse
/// generation order is preserved) and returns the rejected reasons.
pub async fn dispose_loaded_facets(loaded: &mut [LoadedFacets]) -> Vec<AnyError> {
    let mut errors = Vec::new();
    for entry in loaded.iter_mut() {
        if let Err(error) = entry.dispose().await {
            errors.push(error);
        }
    }
    errors
}
