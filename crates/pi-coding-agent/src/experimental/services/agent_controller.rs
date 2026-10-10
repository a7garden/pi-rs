//! Port of `experimental/services/agent-controller.ts` — the
//! presentation-safe command facade contract over the worker-owned root
//! conversation, plus its remote service token.
//!
//! Deviations (D144):
//! - Upstream `defineService<AgentController>("pi.agent-controller")` types the
//!   token with a structural object; Rust types the token with a marker and
//!   exposes the implementation shape through [`AgentControllerOps`] (see
//!   `agent_controller_provider`).
//! - DTOs are typed serde structs/enums with upstream wire names (camelCase).

use std::future::Future;
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use pi_chord::api::define_service;
use pi_chord::context::Context;

/// Upstream `AgentPromptImage`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentPromptImage {
    pub data: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
}

/// Upstream `AgentPromptRequest`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentPromptRequest {
    pub message: String,
    /// `null` upstream when absent; an empty vector serializes as `[]`, so
    /// absent images use `None` (D144).
    pub images: Option<Vec<AgentPromptImage>>,
}

/// Upstream `AgentOperationError`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentOperationError {
    pub code: String,
    pub message: String,
}

/// Upstream `AgentOperationResponse` (untagged union of two object shapes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentOperationResponse {
    pub accepted: bool,
    /// `operationId` identifies the durable submission of a prompt, or the task
    /// of a compaction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<AgentOperationError>,
}

impl AgentOperationResponse {
    pub fn accepted(operation_id: String) -> Self {
        AgentOperationResponse {
            accepted: true,
            operation_id: Some(operation_id),
            error: None,
        }
    }

    pub fn rejected(error: AgentOperationError) -> Self {
        AgentOperationResponse {
            accepted: false,
            operation_id: None,
            error: Some(error),
        }
    }
}

/// Upstream `AgentQueueResponse` (untagged union of two object shapes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentQueueResponse {
    pub accepted: bool,
    /// `entryId` identifies the durable submission; `cancelQueued()` withdraws
    /// it while it is still queued.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<AgentOperationError>,
}

impl AgentQueueResponse {
    pub fn accepted(entry_id: String) -> Self {
        AgentQueueResponse {
            accepted: true,
            entry_id: Some(entry_id),
            error: None,
        }
    }

    pub fn rejected(error: AgentOperationError) -> Self {
        AgentQueueResponse {
            accepted: false,
            entry_id: None,
            error: Some(error),
        }
    }
}

/// Upstream `AgentPromptResult` — the settled outcome of a prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum AgentPromptResult {
    /// `text` present; `reason` is `null` upstream.
    Done {
        text: String,
    },
    Unanswered {
        reason: String,
    },
}

/// Upstream `AgentCompactionRequest`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCompactionRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
}

/// The upstream `AgentController` service shape; methods are async and
/// context-passing. Implemented by [`crate::experimental::services::agent_controller_provider::AgentControllerService`].
pub trait AgentControllerOps: Send + Sync {
    fn prompt(
        &self,
        request: AgentPromptRequest,
        context: Context,
    ) -> BoxFutureAgent<'_, AgentOperationResponse>;
    fn steer(
        &self,
        request: AgentPromptRequest,
        context: Context,
    ) -> BoxFutureAgent<'_, AgentQueueResponse>;
    fn follow_up(
        &self,
        request: AgentPromptRequest,
        context: Context,
    ) -> BoxFutureAgent<'_, AgentQueueResponse>;
    fn cancel_queued(
        &self,
        entry_id: String,
        context: Context,
    ) -> BoxFutureAgent<'_, CancelQueuedOutcome>;
    fn abort(&self, context: Context) -> BoxFutureAgent<'_, Result<(), String>>;
    fn compact(
        &self,
        request: AgentCompactionRequest,
        context: Context,
    ) -> BoxFutureAgent<'_, AgentOperationResponse>;
    fn wait_for_prompt(
        &self,
        operation_id: String,
        context: Context,
    ) -> BoxFutureAgent<'_, Result<AgentPromptResult, String>>;
}

/// Boxed future alias used by [`AgentControllerOps`] (D144: Rust traits carry
/// explicit future types instead of `async_trait` on a hot facade).
pub type BoxFutureAgent<'a, T> = std::pin::Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Upstream `cancelQueued` outcome union.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelQueuedOutcome {
    Cancelled,
    AlreadyConsumed,
    NotFound,
}

/// `defineService<AgentController>("pi.agent-controller")`. The token is typed
/// with a marker because the service is only provided/consumed remotely (D144).
pub struct AgentControllerToken;

pub static AGENT_CONTROLLER: LazyLock<pi_chord::Service<AgentControllerToken>> =
    LazyLock::new(|| define_service("pi.agent-controller", false).expect("service id is valid"));

/// Convenience accessor matching upstream's exported constant name.
pub fn agent_controller_service() -> &'static pi_chord::Service<AgentControllerToken> {
    &AGENT_CONTROLLER
}

/// Parse an upstream stringified submission id (`^[1-9][0-9]*$`, safe integer;
/// Rust submission ids are `u64`, which is always safe, D144).
pub fn parse_submission_id(value: &str) -> Option<u64> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    if value.starts_with('0') {
        return None;
    }
    value.parse::<u64>().ok()
}

/// Remote facade members of the `pi.agent-controller` service (upstream object
/// method names).
pub mod members {
    pub const PROMPT: &str = "prompt";
    pub const STEER: &str = "steer";
    pub const FOLLOW_UP: &str = "followUp";
    pub const CANCEL_QUEUED: &str = "cancelQueued";
    pub const ABORT: &str = "abort";
    pub const COMPACT: &str = "compact";
    pub const WAIT_FOR_PROMPT: &str = "waitForPrompt";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_response_round_trips() {
        let response = AgentOperationResponse::accepted("7".into());
        let value = serde_json::to_value(&response).unwrap();
        assert_eq!(value["accepted"], serde_json::Value::Bool(true));
        assert_eq!(value["operationId"], "7");
        let rejected = AgentOperationResponse::rejected(AgentOperationError {
            code: "busy".into(),
            message: "busy".into(),
        });
        let value = serde_json::to_value(&rejected).unwrap();
        assert_eq!(value["error"]["code"], "busy");
        assert!(value.get("operationId").is_none());
    }

    #[test]
    fn prompt_result_uses_status_tag() {
        let value = serde_json::to_value(AgentPromptResult::Done { text: "hi".into() }).unwrap();
        assert_eq!(value["status"], "done");
        let value = serde_json::to_value(AgentPromptResult::Unanswered {
            reason: "aborted".into(),
        })
        .unwrap();
        assert_eq!(value["status"], "unanswered");
    }

    #[test]
    fn parse_submission_id_rejects_non_positive() {
        assert_eq!(parse_submission_id("12"), Some(12));
        assert_eq!(parse_submission_id("0"), None);
        assert_eq!(parse_submission_id("01"), None);
        assert_eq!(parse_submission_id("-1"), None);
        assert_eq!(parse_submission_id("x"), None);
        assert_eq!(parse_submission_id(""), None);
    }
}
