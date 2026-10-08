//! Port of `packages/coding-agent/src/core/` — the ported surface so far is
//! [`messages`], [`session_manager`], [`defaults`] and [`settings_manager`];
//! the remaining ~50 core modules (tools, compaction, extensions,
//! export-html, modes) stay unported (see crate docs).
pub mod defaults;
pub mod messages;
pub mod session_manager;
pub mod settings_manager;
