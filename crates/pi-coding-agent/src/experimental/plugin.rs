//! Port of `experimental/plugin.ts` — the barrel re-exporting the
//! presentation-facing service contracts.

pub use crate::experimental::services::agent_controller::{
	AgentCompactionRequest, AgentControllerToken, AgentOperationError, AgentOperationResponse,
	AgentPromptImage, AgentPromptRequest, AgentPromptResult, AgentQueueResponse,
};
pub use crate::experimental::services::presentation_ui::{
	PresentationSelectItem, PresentationUi, PresentationUiHandle, PRESENTATION_UI,
};
pub use crate::experimental::services::slash_commands::{
	SlashCommandCompletion, SlashCommandContribution, SlashCommandRunResult, SLASH_COMMANDS,
};
