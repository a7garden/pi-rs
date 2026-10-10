//! Port of `packages/coding-agent/src/modes/rpc/rpc-types.ts` — RPC protocol
//! types for headless operation.
//!
//! Commands arrive as JSON lines on stdin; responses and events leave as
//! JSON lines on stdout.
//!
//! Deviation **D93** — the upstream discriminated unions deserialize through
//! an envelope ([`RpcCommandEnvelope`]) + per-command bodies (upstream reads
//! fields off the parsed object directly), and [`RpcResponse`] is a struct
//! with optional members — wire-identical to the union (`preserve_order`
//! keeps upstream key order). Aggregates the Rust port keeps as typed
//! structs ([`RpcSessionState`]) stay `Serialize` with camelCase renames.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::core::agent_session::{PromptDisposition, QueuedInputDisposition};
use crate::core::session_manager::SessionTreeNode;
use crate::core::source_info::SourceInfo;

// ============================================================================
// RPC Commands (stdin)
// ============================================================================

/// Envelope of every RPC command: optional correlation `id` plus a `type`
/// discriminator and the command body (`RpcCommand` union + `RpcCommandType`).
#[derive(Debug, Clone, Deserialize)]
pub struct RpcCommandEnvelope {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(flatten)]
    pub body: JsonValue,
}

/// `RpcCommandType` — every valid command discriminator.
pub const RPC_COMMAND_TYPES: &[&str] = &[
    "prompt",
    "steer",
    "follow_up",
    "abort",
    "clear_queue",
    "new_session",
    "get_state",
    "set_model",
    "cycle_model",
    "get_available_models",
    "set_thinking_level",
    "cycle_thinking_level",
    "get_available_thinking_levels",
    "set_steering_mode",
    "set_follow_up_mode",
    "compact",
    "set_auto_compaction",
    "set_auto_retry",
    "abort_retry",
    "bash",
    "abort_bash",
    "get_session_stats",
    "export_html",
    "switch_session",
    "fork",
    "clone",
    "get_fork_messages",
    "get_entries",
    "get_tree",
    "get_last_assistant_text",
    "set_session_name",
    "get_messages",
    "get_commands",
];

pub type RpcCommandType = String;

// -- command bodies ----------------------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptCommand {
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub images: Option<Vec<pi_ai::types::ImageContent>>,
    #[serde(default)]
    pub streaming_behavior: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageCommand {
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub images: Option<Vec<pi_ai::types::ImageContent>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionCommand {
    #[serde(default)]
    pub parent_session: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetModelCommand {
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub model_id: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetThinkingLevelCommand {
    #[serde(default)]
    pub level: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueueModeCommand {
    #[serde(default)]
    pub mode: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactCommand {
    #[serde(default)]
    pub custom_instructions: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetEnabledCommand {
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashCommand {
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub exclude_from_context: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportHtmlCommand {
    #[serde(default)]
    pub output_path: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SwitchSessionCommand {
    #[serde(default)]
    pub session_path: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ForkCommand {
    #[serde(default)]
    pub entry_id: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GetEntriesCommand {
    #[serde(default)]
    pub since: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetSessionNameCommand {
    #[serde(default)]
    pub name: String,
}

// ============================================================================
// RPC Slash Command (for get_commands response)
// ============================================================================

/// A command available for invocation via prompt (`RpcSlashCommand`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcSlashCommand {
    /// Command name (without leading slash).
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// What kind of command this is.
    pub source: RpcCommandSource,
    /// Source metadata for the owning resource.
    pub source_info: SourceInfo,
}

/// `RpcSlashCommand["source"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RpcCommandSource {
    #[serde(rename = "extension")]
    Extension,
    #[serde(rename = "prompt")]
    Prompt,
    #[serde(rename = "skill")]
    Skill,
}

// ============================================================================
// RPC State
// ============================================================================

/// Session state snapshot returned by `get_state` (`RpcSessionState`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcSessionState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<pi_ai::types::Model>,
    pub thinking_level: pi_agent::types::AgentThinkingLevel,
    pub is_streaming: bool,
    pub is_compacting: bool,
    pub steering_mode: pi_agent::types::QueueMode,
    pub follow_up_mode: pi_agent::types::QueueMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_file: Option<String>,
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_name: Option<String>,
    pub auto_compaction_enabled: bool,
    pub message_count: usize,
    pub pending_message_count: usize,
}

// ============================================================================
// RPC Responses (stdout)
// ============================================================================

/// Response line for a command (`RpcResponse` union): success carries
/// optional `data`, failure carries `error`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse {
    /// Correlation id echoed from the command (`undefined` keys are dropped
    /// upstream by `JSON.stringify`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub response_type: String,
    pub command: String,
    pub success: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl RpcResponse {
    /// `success(id, command)` / `success(id, command, data)`.
    pub fn success(id: Option<String>, command: impl Into<String>) -> Self {
        RpcResponse {
            id,
            response_type: "response".to_string(),
            command: command.into(),
            success: true,
            data: None,
            error: None,
        }
    }

    /// `success(id, command, data)`.
    pub fn success_with(id: Option<String>, command: impl Into<String>, data: JsonValue) -> Self {
        RpcResponse {
            data: Some(data),
            ..RpcResponse::success(id, command)
        }
    }

    /// `error(id, command, message)`.
    pub fn error(
        id: Option<String>,
        command: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        RpcResponse {
            id,
            response_type: "response".to_string(),
            command: command.into(),
            success: false,
            data: None,
            error: Some(message.into()),
        }
    }

    /// Extract `data` or the `error` message (`RpcClient.getData`).
    pub fn into_data(self) -> Result<JsonValue, String> {
        if !self.success {
            return Err(self.error.unwrap_or_default());
        }
        Ok(self.data.unwrap_or(JsonValue::Null))
    }
}

/// Disposition wire value (`PromptDisposition`).
pub fn prompt_disposition_wire(disposition: PromptDisposition) -> &'static str {
    match disposition {
        PromptDisposition::Handled => "handled",
        PromptDisposition::Queued => "queued",
        PromptDisposition::Started => "started",
    }
}

/// Disposition wire value (`QueuedInputDisposition`).
pub fn queued_disposition_wire(disposition: QueuedInputDisposition) -> &'static str {
    match disposition {
        QueuedInputDisposition::Handled => "handled",
        QueuedInputDisposition::Queued => "queued",
    }
}

// ============================================================================
// Extension UI Events (stdin/stdout)
// ============================================================================

/// Build an `extension_ui_request` line object (`RpcExtensionUIRequest`).
/// `method_fields` carries the per-method members spread after `id`
/// upstream.
pub fn extension_ui_request(id: &str, method_fields: JsonValue) -> JsonValue {
    let mut object = serde_json::Map::new();
    object.insert(
        "type".to_string(),
        JsonValue::String("extension_ui_request".to_string()),
    );
    object.insert("id".to_string(), JsonValue::String(id.to_string()));
    if let JsonValue::Object(fields) = method_fields {
        object.extend(fields);
    }
    JsonValue::Object(object)
}

/// Response to an extension UI request (`RpcExtensionUIResponse` union).
/// Upstream narrows by key presence; the Rust form keeps optional members
/// and matches the same way.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename = "extension_ui_response", tag = "type")]
pub struct RpcExtensionUIResponse {
    pub id: String,
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default)]
    pub confirmed: Option<bool>,
    #[serde(default)]
    pub cancelled: Option<bool>,
}

impl RpcExtensionUIResponse {
    /// `"cancelled" in response && response.cancelled`.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled == Some(true)
    }
}

/// `SessionTreeNode` wire shape (D93 projection — the Rust struct is not
/// `Serialize`).
pub fn session_tree_node_to_json(node: &SessionTreeNode) -> JsonValue {
    let mut object = serde_json::Map::new();
    object.insert(
        "entry".to_string(),
        serde_json::to_value(&node.entry).unwrap_or(JsonValue::Null),
    );
    object.insert(
        "children".to_string(),
        JsonValue::Array(
            node.children
                .iter()
                .map(session_tree_node_to_json)
                .collect(),
        ),
    );
    if let Some(label) = &node.label {
        object.insert("label".to_string(), JsonValue::String(label.clone()));
    }
    if let Some(label_timestamp) = &node.label_timestamp {
        object.insert(
            "labelTimestamp".to_string(),
            JsonValue::String(label_timestamp.clone()),
        );
    }
    JsonValue::Object(object)
}
