//! Port of `packages/coding-agent/src/core/messages.ts` — custom message
//! types and transformers for the coding agent. Extends the base
//! [`pi_agent::AgentMessage`] union with coding-agent specific message types,
//! and provides a transformer to convert them to LLM-compatible messages.
//!
//! Upstream attaches these types via TS declaration merging on
//! `CustomAgentMessages`; the Rust port represents them as typed structs that
//! (de)serialize to the same JSON objects and travel inside
//! [`AgentMessage::Custom`] (opaque JSON) — see crate docs, deviation D4.

use pi_agent::AgentMessage;
use pi_ai::types::{Message, UserContent, UserMessage};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;

use crate::utils::time::parse_timestamp_ms;

pub const COMPACTION_SUMMARY_PREFIX: &str =
	"The conversation history before this point was compacted into the following summary:\n\n<summary>\n";

pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";

pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";

pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

/// Literal `"bashExecution"` role tag of [`BashExecutionMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BashExecutionRole {
    #[serde(rename = "bashExecution")]
    BashExecution,
}

/// Literal `"custom"` role tag of [`CustomMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CustomRole {
    #[serde(rename = "custom")]
    Custom,
}

/// Literal `"branchSummary"` role tag of [`BranchSummaryMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchSummaryRole {
    #[serde(rename = "branchSummary")]
    BranchSummary,
}

/// Literal `"compactionSummary"` role tag of [`CompactionSummaryMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionSummaryRole {
    #[serde(rename = "compactionSummary")]
    CompactionSummary,
}

/// `string | (TextContent|ImageContent)[]` missing-field default: empty block
/// list (matches upstream `content ?? []` handling).
pub(crate) fn empty_user_content() -> pi_ai::types::UserContentField {
    pi_ai::types::UserContentField::Blocks(Vec::new())
}

/// Message type for bash executions via the `!` command (`BashExecutionMessage`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashExecutionMessage {
    pub role: BashExecutionRole,
    pub command: String,
    pub output: String,
    /// Absent mirrors upstream `exitCode: undefined`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    pub cancelled: bool,
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
    pub timestamp: i64,
    /// If true, this message is excluded from LLM context (`!` prefix).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_from_context: Option<bool>,
}

/// Message type for extension-injected messages via sendMessage()
/// (`CustomMessage`). Content reuses the pi-ai `string | (text|image)[]`
/// union; `null` content reads as empty blocks (pi-ai deviation D13).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessage {
    pub role: CustomRole,
    #[serde(rename = "customType")]
    pub custom_type: String,
    #[serde(default = "empty_user_content")]
    pub content: pi_ai::types::UserContentField,
    pub display: bool,
    /// Absent mirrors upstream `details?: undefined`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    pub timestamp: i64,
}

/// Summary of an abandoned branch carried into the new one
/// (`BranchSummaryMessage`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryMessage {
    pub role: BranchSummaryRole,
    pub summary: String,
    #[serde(default)]
    pub from_id: Option<String>,
    pub timestamp: i64,
}

/// Summary of compacted history injected at a compaction boundary
/// (`CompactionSummaryMessage`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSummaryMessage {
    pub role: CompactionSummaryRole,
    pub summary: String,
    #[serde(rename = "tokensBefore")]
    pub tokens_before: u64,
    pub timestamp: i64,
}

/// Convert a BashExecutionMessage to user message text for LLM context
/// (`bashExecutionToText`).
pub fn bash_execution_to_text(msg: &BashExecutionMessage) -> String {
    let mut text = format!("Ran `{}`\n", msg.command);
    if !msg.output.is_empty() {
        text += &format!("```\n{}\n```", msg.output);
    } else {
        text += "(no output)";
    }
    if msg.cancelled {
        text += "\n\n(command cancelled)";
    } else if msg.exit_code.is_some_and(|code| code != 0) {
        text += &format!(
            "\n\nCommand exited with code {}",
            msg.exit_code.unwrap_or(0)
        );
    }
    if msg.truncated {
        if let Some(full_output_path) = &msg.full_output_path {
            text += &format!("\n\n[Output truncated. Full output: {full_output_path}]");
        }
    }
    text
}

/// `new Date(timestamp).getTime()` with invalid input mapping to `0`
/// (upstream would produce `NaN`, which JSON serializes as `null`; pi always
/// passes valid ISO strings — crate docs D2).
fn timestamp_ms(timestamp: &str) -> i64 {
    parse_timestamp_ms(timestamp).unwrap_or(0)
}

pub fn create_branch_summary_message(
    summary: &str,
    from_id: &str,
    timestamp: &str,
) -> BranchSummaryMessage {
    BranchSummaryMessage {
        role: BranchSummaryRole::BranchSummary,
        summary: summary.to_string(),
        from_id: Some(from_id.to_string()),
        timestamp: timestamp_ms(timestamp),
    }
}

pub fn create_compaction_summary_message(
    summary: &str,
    tokens_before: u64,
    timestamp: &str,
) -> CompactionSummaryMessage {
    CompactionSummaryMessage {
        role: CompactionSummaryRole::CompactionSummary,
        summary: summary.to_string(),
        tokens_before,
        timestamp: timestamp_ms(timestamp),
    }
}

/// Convert CustomMessageEntry to AgentMessage format (`createCustomMessage`).
pub fn create_custom_message(
    custom_type: &str,
    content: pi_ai::types::UserContentField,
    display: bool,
    details: Option<Value>,
    timestamp: &str,
) -> CustomMessage {
    CustomMessage {
        role: CustomRole::Custom,
        custom_type: custom_type.to_string(),
        content,
        display,
        details,
        timestamp: timestamp_ms(timestamp),
    }
}

/// Serialize a typed custom message into the [`AgentMessage::Custom`] JSON
/// representation. Infallible for these value-only payloads.
pub fn custom_message_to_agent<T: Serialize>(message: &T) -> AgentMessage {
    match serde_json::to_value(message) {
        Ok(value) => AgentMessage::Custom(value),
        // These payloads contain no maps with non-string keys; serialization
        // cannot fail.
        Err(error) => unreachable!("custom message serialization failed: {error}"),
    }
}

/// Extract the typed payload of a custom agent message role.
pub fn custom_message_from_agent<T: DeserializeOwned>(message: &AgentMessage) -> Option<T> {
    match message {
        AgentMessage::Custom(value) => serde_json::from_value(value.clone()).ok(),
        _ => None,
    }
}

/// Extract the raw JSON payload of a custom agent message (upstream custom
/// role objects).
pub fn custom_value_from_agent(message: &AgentMessage) -> Option<&Value> {
    match message {
        AgentMessage::Custom(value) => Some(value),
        _ => None,
    }
}

/// Transform AgentMessages (including custom types) to LLM-compatible
/// Messages (`convertToLlm`). Used by the agent's transformToLlm option,
/// compaction summary generation, and extensions.
pub fn convert_to_llm(messages: &[AgentMessage]) -> Vec<Message> {
    messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Llm(llm) => Some(llm.clone()),
            AgentMessage::Custom(value) => {
                let role = value
                    .get("role")
                    .and_then(|role| role.as_str())
                    .unwrap_or("custom");
                match role {
                    "bashExecution" => {
                        let bash: BashExecutionMessage =
                            serde_json::from_value(value.clone()).ok()?;
                        // Skip messages excluded from context (! prefix)
                        if bash.exclude_from_context.unwrap_or(false) {
                            return None;
                        }
                        Some(Message::User(UserMessage {
                            content: pi_ai::types::UserContentField::Blocks(vec![
                                UserContent::Text(pi_ai::types::TextContent::new(
                                    bash_execution_to_text(&bash),
                                )),
                            ]),
                            timestamp: bash.timestamp,
                        }))
                    }
                    "custom" => {
                        let custom: CustomMessage = serde_json::from_value(value.clone()).ok()?;
                        let content = match &custom.content {
                            pi_ai::types::UserContentField::Text(text) => {
                                pi_ai::types::UserContentField::Blocks(vec![UserContent::Text(
                                    pi_ai::types::TextContent::new(text.clone()),
                                )])
                            }
                            blocks @ pi_ai::types::UserContentField::Blocks(_) => blocks.clone(),
                        };
                        Some(Message::User(UserMessage {
                            content,
                            timestamp: custom.timestamp,
                        }))
                    }
                    "branchSummary" => {
                        let summary: BranchSummaryMessage =
                            serde_json::from_value(value.clone()).ok()?;
                        Some(Message::User(UserMessage {
                            content: pi_ai::types::UserContentField::Blocks(vec![
                                UserContent::Text(pi_ai::types::TextContent::new(format!(
                                    "{BRANCH_SUMMARY_PREFIX}{}{BRANCH_SUMMARY_SUFFIX}",
                                    summary.summary
                                ))),
                            ]),
                            timestamp: summary.timestamp,
                        }))
                    }
                    "compactionSummary" => {
                        let summary: CompactionSummaryMessage =
                            serde_json::from_value(value.clone()).ok()?;
                        Some(Message::User(UserMessage {
                            content: pi_ai::types::UserContentField::Blocks(vec![
                                UserContent::Text(pi_ai::types::TextContent::new(format!(
                                    "{COMPACTION_SUMMARY_PREFIX}{}{COMPACTION_SUMMARY_SUFFIX}",
                                    summary.summary
                                ))),
                            ]),
                            timestamp: summary.timestamp,
                        }))
                    }
                    _ => None,
                }
            }
        })
        .collect()
}
