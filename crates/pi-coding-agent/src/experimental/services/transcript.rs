//! Port of `experimental/services/transcript.ts` — the root conversation's
//! durable view service.
//!
//! Deviations (D144):
//! - `ReplicatedState<ConversationView>` becomes chord's untyped
//!   [`pi_chord::AttachedReplicatedState`] over the serialized
//!   `pi_durable` conversation view JSON (typed at the seam).

use std::sync::{Arc, LazyLock};

use pi_chord::api::define_service;
use pi_chord::services::state::AttachedReplicatedState;

/// `defineService<Transcript>("pi.transcript")`; provided remotely by the
/// Session worker's transcript facet (D144 marker token).
pub struct TranscriptToken;

pub static TRANSCRIPT: LazyLock<pi_chord::Service<TranscriptService>> =
    LazyLock::new(|| define_service("pi.transcript", false).expect("service id is valid"));

/// Upstream `Transcript` service value: the attached durable view state.
#[derive(Clone)]
pub struct TranscriptService {
    pub state: Arc<AttachedReplicatedState>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_token_pins_the_upstream_id() {
        assert_eq!(TRANSCRIPT.id(), "pi.transcript");
    }
}
