//! Port of `packages/durable/src/harness/provider.ts`.
//!
//! Deviations: D1 — `uuidv7()` via `pi_ai::utils::uuid::uuidv7_now`.
//! D2 — doc access goes through the erased Session `Tx` doc API
//! (`doc_conversation_mut`); reconcile with `crate::session` on landing.

use std::sync::Arc;

use pi_chord::context::Context;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::documents::{DocDefinition, DocToken};
use crate::harness::types::ConversationId;
use crate::session::Tx;
use crate::tasks::TaskRuntime;
use crate::types::{DocumentFork, DocumentHistory};

/// Stable provider-facing identity of one conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderState {
    pub session_id: String,
}

fn provider_definition() -> DocDefinition {
    DocDefinition::conversation(
        "pi.provider",
        1,
        DocumentHistory::Latest,
        DocumentFork::Initial,
        || {
            json!({
                "sessionId": pi_ai::utils::uuid::uuidv7_now().expect("uuidv7 generation failed"),
            })
            .as_object()
            .cloned()
            .unwrap_or_default()
        },
    )
    .with_checkpoint_when(|_, _, _| true)
}

/// Built-in provider state; every fork starts with a fresh identity instead
/// of copying its parent.
static PROVIDER_DOC: std::sync::LazyLock<DocToken> =
    std::sync::LazyLock::new(|| DocToken::new(provider_definition()));

/// The conversation provider-identity document.
pub fn provider_doc() -> DocToken {
    PROVIDER_DOC.clone()
}

/// Return the persisted identity without writing in the normal path. A
/// legacy conversation without `pi.provider` gets one migration commit whose
/// `tx.doc()` runs `initial()` before the provider request starts.
pub async fn ensure_provider_session_id(
    runtime: &dyn TaskRuntime,
    context: &Context,
) -> Result<String, String> {
    let existing = runtime
        .snapshot(&provider_doc(), runtime.conversation_id(), context)
        .await?;
    if let Some(existing) = existing {
        return existing
            .get("sessionId")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "pi.provider has no sessionId".to_string());
    }
    let created: Arc<parking_lot::Mutex<Option<String>>> = Arc::default();
    let conversation_id = runtime.conversation_id();
    let created_in = created.clone();
    runtime
        .commit(
            crate::tasks::commit_change(move |tx, _current| {
                let created_in = created_in.clone();
                let draft = tx
                    .doc(&provider_doc(), Some(conversation_id.0), None, None)
                    .map_err(|e| e.to_string())?;
                let value = draft.value().unwrap_or_default();
                *created_in.lock() = value
                    .get("sessionId")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                Ok(None)
            }),
            context,
        )
        .await?;
    let got = created.lock().clone();
    got.ok_or_else(|| {
        format!(
            "Conversation {} has no provider session ID",
            conversation_id
        )
    })
}
