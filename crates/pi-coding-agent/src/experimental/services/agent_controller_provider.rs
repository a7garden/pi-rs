//! Port of `experimental/services/agent-controller-provider.ts` — builds the
//! presentation-safe `AgentController` facade over one durable conversation.
//!
//! Deviations (D144):
//! - The upstream object literal becomes [`AgentControllerService`], typed
//!   over `pi_durable`'s `Harness`/`Conversation`.
//! - `String(submission.id)` maps to the `u64` newtype's `Display` form.
//! - `harness.commit((tx) => tx.entry(answer))` maps to
//!   `Conversation::commit` with a transaction closure reading the answer
//!   entry.
//! - Errors: `ConversationBusy` → `busy`; everything else →
//!   `operation_failed` with the Rust error's `Display` text.

use std::sync::Arc;

use pi_chord::context::Context;
use pi_durable::harness::harness::{Conversation, Harness};
use pi_durable::harness::types::{
    InputSubmissionDraft, SubmissionAbort, SubmissionDraft, WhenBusy,
};
use pi_durable::harness::types::{Submission as _, UserInput};
use pi_durable::types::{DurableId as _, SubmissionId};

use super::agent_controller::{
    parse_submission_id, AgentCompactionRequest, AgentControllerOps, AgentOperationError,
    AgentOperationResponse, AgentPromptRequest, AgentPromptResult, AgentQueueResponse,
    CancelQueuedOutcome,
};
use crate::experimental::services::agent_controller::BoxFutureAgent;

/// `createAgentController(harness, conversation)` — the concrete facade.
pub struct AgentControllerService {
    pub harness: Arc<Harness>,
    pub conversation: Arc<Conversation>,
}

impl AgentControllerService {
    pub fn new(harness: Arc<Harness>, conversation: Arc<Conversation>) -> AgentControllerService {
        AgentControllerService {
            harness,
            conversation,
        }
    }

    async fn queue(
        &self,
        when_busy: WhenBusy,
        request: AgentPromptRequest,
        context: Context,
    ) -> AgentQueueResponse {
        match self
            .conversation
            .submit(
                SubmissionDraft::Input(InputSubmissionDraft {
                    request_id: None,
                    content: to_input(request),
                    when_busy: Some(when_busy),
                }),
                &context,
            )
            .await
        {
            Ok(submission) => AgentQueueResponse::accepted(submission.id.to_string()),
            Err(error) => AgentQueueResponse::rejected(to_agent_error(&error)),
        }
    }
}

impl AgentControllerOps for AgentControllerService {
    fn prompt(
        &self,
        request: AgentPromptRequest,
        context: Context,
    ) -> BoxFutureAgent<'_, AgentOperationResponse> {
        Box::pin(async move {
            match self
                .conversation
                .submit(
                    SubmissionDraft::Input(InputSubmissionDraft {
                        request_id: None,
                        content: to_input(request),
                        when_busy: Some(WhenBusy::Reject),
                    }),
                    &context,
                )
                .await
            {
                Ok(submission) => AgentOperationResponse::accepted(submission.id.to_string()),
                Err(error) => AgentOperationResponse::rejected(to_agent_error(&error)),
            }
        })
    }

    fn steer(
        &self,
        request: AgentPromptRequest,
        context: Context,
    ) -> BoxFutureAgent<'_, AgentQueueResponse> {
        Box::pin(self.queue(WhenBusy::Steer, request, context))
    }

    fn follow_up(
        &self,
        request: AgentPromptRequest,
        context: Context,
    ) -> BoxFutureAgent<'_, AgentQueueResponse> {
        Box::pin(self.queue(WhenBusy::FollowUp, request, context))
    }

    fn cancel_queued(
        &self,
        entry_id: String,
        context: Context,
    ) -> BoxFutureAgent<'_, CancelQueuedOutcome> {
        Box::pin(async move {
            let Some(id) = parse_submission_id(&entry_id) else {
                return CancelQueuedOutcome::NotFound;
            };
            let parsed = SubmissionId::from_number(id);
            match self
                .harness
                .abort_submission(parsed, &context, Some(self.conversation.id))
                .await
            {
                Ok(result) => match result {
                    Some(SubmissionAbort::Aborted) => CancelQueuedOutcome::Cancelled,
                    Some(SubmissionAbort::AlreadyPlaced) | Some(SubmissionAbort::Settled) => {
                        CancelQueuedOutcome::AlreadyConsumed
                    }
                    None => CancelQueuedOutcome::NotFound,
                },
                Err(_) => CancelQueuedOutcome::AlreadyConsumed,
            }
        })
    }

    fn abort(&self, context: Context) -> BoxFutureAgent<'_, Result<(), String>> {
        Box::pin(async move { self.conversation.abort(&context, None).await.map(|_| ()) })
    }

    fn compact(
        &self,
        request: AgentCompactionRequest,
        context: Context,
    ) -> BoxFutureAgent<'_, AgentOperationResponse> {
        Box::pin(async move {
            match self
                .conversation
                .compact(request.custom_instructions, &context)
                .await
            {
                Ok(task_id) => AgentOperationResponse::accepted(task_id.to_string()),
                Err(error) => AgentOperationResponse::rejected(to_agent_error(&error)),
            }
        })
    }

    fn wait_for_prompt(
        &self,
        operation_id: String,
        context: Context,
    ) -> BoxFutureAgent<'_, Result<AgentPromptResult, String>> {
        Box::pin(async move {
            let submission = match parse_submission_id(&operation_id) {
                Some(id) => {
                    self.harness
                        .submission(SubmissionId::from_number(id), &context)
                        .await?
                }
                None => None,
            };
            let Some(submission) = submission else {
                return Err(format!("Unknown prompt: {operation_id}"));
            };
            let settled = submission.wait(&context).await?;
            if settled.status == pi_durable::types::SubmissionStatus::Unanswered {
                return Ok(AgentPromptResult::Unanswered {
                    reason: settled.reason.unwrap_or_default(),
                });
            }
            // Upstream: commit a transaction reading the answer entry, then take
            // the first model message's concatenated text parts.
            let answer_entry = settled.answer;
            let text = match answer_entry {
                Some(answer) => {
                    self.conversation
                        .commit(
                            move |tx| {
                                Ok(tx
                                    .entry(answer)
                                    .map_err(|entry_error| entry_error.to_string())?
                                    .and_then(|entry| entry.model)
                                    .and_then(|messages| messages.first().cloned())
                                    .map(message_text)
                                    .unwrap_or_default())
                            },
                            &context,
                        )
                        .await?
                }
                None => String::new(),
            };
            Ok(AgentPromptResult::Done { text })
        })
    }
}

/// Upstream `message.content.flatMap(text).join("")` for the first model
/// message of the answer entry.
fn message_text(message: pi_ai::types::Message) -> String {
    match message {
        pi_ai::types::Message::Assistant(assistant) => assistant
            .content
            .into_iter()
            .map(|block| match block {
                pi_ai::types::AssistantContent::Text(text_block) => text_block.text,
                _ => String::new(),
            })
            .collect(),
        _ => String::new(),
    }
}

/// Upstream `toInput(request)`.
fn to_input(request: AgentPromptRequest) -> UserInput {
    let images = request.images.unwrap_or_default();
    if images.is_empty() {
        return UserInput::Text(request.message);
    }
    // Upstream builds `[{ type: "text", text }, ...images]` content blocks.
    let mut blocks = Vec::with_capacity(images.len() + 1);
    blocks.push(pi_ai::types::UserContent::Text(
        pi_ai::types::TextContent::new(request.message),
    ));
    for image in images {
        blocks.push(pi_ai::types::UserContent::Image(
            pi_ai::types::ImageContent {
                data: image.data,
                mime_type: image.mime_type,
            },
        ));
    }
    UserInput::Blocks(blocks)
}

/// Upstream `toAgentError(error)`.
///
/// D145: pi-durable Rust errors arrive stringified (`Result<_, String>`), so
/// the typed `ConversationBusy` mapping becomes a stable-message check against
/// the durable display text (`Conversation {id} is busy`).
fn to_agent_error(error: &str) -> AgentOperationError {
    if error.ends_with(" is busy") || error.contains(" is busy\n") {
        return AgentOperationError {
            code: "busy".into(),
            message: error.to_string(),
        };
    }
    AgentOperationError {
        code: "operation_failed".into(),
        message: error.to_string(),
    }
}

/// Build the remote singleton implementation for the `pi.agent-controller`
/// service: each upstream object method becomes a JSON-encoded member.
pub fn agent_controller_object(
    service: Arc<AgentControllerService>,
) -> Arc<pi_chord::services::provider::RemoteServiceObject> {
    use super::agent_controller::members;
    use pi_chord::services::errors::AnyError as ChordAnyError;
    use pi_chord::services::provider::RemoteServiceObject;

    let decode = |args: &[pi_chord::json::JsonValue],
                  member: &str|
     -> Result<pi_chord::json::JsonValue, ChordAnyError> {
        args.first().cloned().ok_or_else(|| {
            pi_chord::services::errors::type_error(format!("{member} requires a request"))
        })
    };

    let prompt = Arc::clone(&service);
    let steer = Arc::clone(&service);
    let follow_up = Arc::clone(&service);
    let cancel = Arc::clone(&service);
    let abort = Arc::clone(&service);
    let compact = Arc::clone(&service);
    let wait = Arc::clone(&service);

    RemoteServiceObject::builder()
        .method(members::PROMPT, move |args, context| {
            let service = Arc::clone(&prompt);
            Box::pin(async move {
                let request: AgentPromptRequest =
                    serde_json::from_value(decode(&args, members::PROMPT)?).map_err(
                        |serde_error| {
                            pi_chord::services::errors::type_error(serde_error.to_string())
                        },
                    )?;
                Ok(Some(
                    serde_json::to_value(service.prompt(request, context).await)
                        .expect("AgentOperationResponse serializes"),
                ))
            })
        })
        .method(members::STEER, move |args, context| {
            let service = Arc::clone(&steer);
            Box::pin(async move {
                let request: AgentPromptRequest =
                    serde_json::from_value(decode(&args, members::STEER)?).map_err(
                        |serde_error| {
                            pi_chord::services::errors::type_error(serde_error.to_string())
                        },
                    )?;
                Ok(Some(
                    serde_json::to_value(service.steer(request, context).await)
                        .expect("AgentQueueResponse serializes"),
                ))
            })
        })
        .method(members::FOLLOW_UP, move |args, context| {
            let service = Arc::clone(&follow_up);
            Box::pin(async move {
                let request: AgentPromptRequest =
                    serde_json::from_value(decode(&args, members::FOLLOW_UP)?).map_err(
                        |serde_error| {
                            pi_chord::services::errors::type_error(serde_error.to_string())
                        },
                    )?;
                Ok(Some(
                    serde_json::to_value(service.follow_up(request, context).await)
                        .expect("AgentQueueResponse serializes"),
                ))
            })
        })
        .method(members::CANCEL_QUEUED, move |args, context| {
            let service = Arc::clone(&cancel);
            Box::pin(async move {
                let entry_id = args
                    .first()
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| {
                        pi_chord::services::errors::type_error("cancelQueued requires an entry id")
                    })?
                    .to_string();
                let outcome = service.cancel_queued(entry_id, context).await;
                Ok(Some(
                    serde_json::to_value(&outcome).expect("CancelQueuedOutcome serializes"),
                ))
            })
        })
        .method(members::ABORT, move |_args, context| {
            let service = Arc::clone(&abort);
            Box::pin(async move {
                service.abort(context).await?;
                Ok(None)
            })
        })
        .method(members::COMPACT, move |args, context| {
            let service = Arc::clone(&compact);
            Box::pin(async move {
                let request: AgentCompactionRequest = serde_json::from_value(
                    args.first()
                        .cloned()
                        .unwrap_or(pi_chord::json::JsonValue::Null),
                )
                .map_err(|serde_error| {
                    pi_chord::services::errors::type_error(serde_error.to_string())
                })?;
                Ok(Some(
                    serde_json::to_value(service.compact(request, context).await)
                        .expect("AgentOperationResponse serializes"),
                ))
            })
        })
        .method(members::WAIT_FOR_PROMPT, move |args, context| {
            let service = Arc::clone(&wait);
            Box::pin(async move {
                let operation_id = args
                    .first()
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| {
                        pi_chord::services::errors::type_error(
                            "waitForPrompt requires an operation id",
                        )
                    })?
                    .to_string();
                let result = service.wait_for_prompt(operation_id, context).await?;
                Ok(Some(
                    serde_json::to_value(&result).expect("AgentPromptResult serializes"),
                ))
            })
        })
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_input_passes_plain_text_through() {
        let input = to_input(AgentPromptRequest {
            message: "hi".into(),
            images: None,
        });
        match input {
            UserInput::Text(text) => assert_eq!(text, "hi"),
            _ => panic!("expected text input"),
        }
    }

    #[test]
    fn to_input_prepends_text_block_for_images() {
        let input = to_input(AgentPromptRequest {
            message: "look".into(),
            images: Some(vec![super::super::agent_controller::AgentPromptImage {
                data: "Zm9v".into(),
                mime_type: "image/png".into(),
            }]),
        });
        match input {
            UserInput::Blocks(blocks) => assert_eq!(blocks.len(), 2),
            _ => panic!("expected blocks input"),
        }
    }

    #[test]
    fn to_agent_error_codes_failures() {
        let error = to_agent_error("boom");
        assert_eq!(error.code, "operation_failed");
        assert_eq!(error.message, "boom");
    }
}
