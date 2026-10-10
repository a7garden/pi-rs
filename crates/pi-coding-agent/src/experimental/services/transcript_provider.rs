//! Port of `experimental/services/transcript-provider.ts` — serve the
//! conversation's durable view state. The facet owns and disposes the attached
//! state.
//!
//! Deviations (D144):
//! - `conversation.viewState(context)` maps to
//!   [`pi_durable::harness::Conversation::view_state`], which returns chord's
//!   untyped [`pi_chord::AttachedReplicatedState`].

use std::sync::Arc;

use pi_chord::context::Context;
use pi_chord::services::provider::RemoteServiceObject;
use pi_chord::services::state::AttachedReplicatedState;
use pi_chord::types::Facet;
use pi_durable::harness::harness::Conversation;

/// `createTranscriptServiceFacet(conversation, context)`.
pub async fn create_transcript_service_facet(
    conversation: &Arc<Conversation>,
    context: &Context,
) -> Result<Facet, String> {
    let state: Arc<AttachedReplicatedState> = conversation.view_state(context).await?;
    Ok(pi_chord::api::define_facet(pi_chord::types::Facet::new(
        "@pi/transcript",
        move |env| {
            env.own({
                let state = state.clone();
                Arc::new(move || {
                    state.dispose();
                    Box::pin(async { Ok(()) })
                })
            })?;
            env.provide_remote(
                super::transcript::TRANSCRIPT.id(),
                pi_chord::types::ServiceMode::Singleton,
                RemoteServiceObject::builder()
                    .state("state", state.internals())
                    .build(),
            )?;
            Ok(())
        },
    )))
}
