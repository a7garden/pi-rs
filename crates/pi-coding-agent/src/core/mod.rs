//! Port of `packages/coding-agent/src/core/` — the ported surface so far is
//! [`messages`], [`session_manager`], [`defaults`], [`settings_manager`],
//! [`usage_totals`], [`compaction`] and the standalone [`agent_session`]
//! surface; the remaining core modules (tools, extensions, export-html,
//! modes, model-*/system-prompt, the AgentSession class) stay unported (see
//! crate docs).
pub mod agent_session;
pub mod compaction;
pub mod defaults;
pub mod messages;
pub mod session_manager;
pub mod settings_manager;
pub mod usage_totals;
