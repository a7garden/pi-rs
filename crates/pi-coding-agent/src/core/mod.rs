//! Port of `packages/coding-agent/src/core/` — the ported surface so far is
//! [`messages`], [`session_manager`], [`defaults`], [`settings_manager`],
//! [`usage_totals`], [`compaction`], [`diagnostics`], [`source_info`],
//! [`skills`], the [`extensions`] type foundation, the standalone
//! [`agent_session`]
//! surface, and the model layer ([`model_config`], [`model_resolver`],
//! [`provider_composer`], [`resolve_config_value`]); the remaining core
//! modules (export-html, modes, system-prompt, the AgentSession class) stay
//! unported (see crate docs).
pub mod agent_session;
pub mod bash_executor;
pub mod compaction;
pub mod defaults;
pub mod diagnostics;
pub mod event_bus;
pub mod extensions;
pub mod footer_data_provider;
pub mod messages;
pub mod model_config;
pub mod model_resolver;
pub mod package_manager;
pub mod pi_manifest;
pub mod prompt_templates;
pub mod provider_composer;
pub mod resolve_config_value;
pub mod resource_loader;
pub mod session_manager;
pub mod settings_manager;
pub mod skills;
pub mod source_info;
pub mod timings;
pub mod tools;
pub mod usage_totals;
