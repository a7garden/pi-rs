//! Port of `packages/coding-agent/src/core/` — the ported surface so far is
//! [`messages`], [`session_manager`], [`defaults`], [`settings_manager`],
//! [`usage_totals`], [`compaction`], [`diagnostics`], [`source_info`],
//! [`skills`], the [`extensions`] type foundation, the standalone
//! [`agent_session`]
//! surface, and the model layer ([`model_config`], [`model_resolver`],
//! [`provider_composer`], [`resolve_config_value`], [`virtual_models`]);
//! system-prompt and
//! slash-commands are ported ([`system_prompt`], [`slash_commands`]); the
//! remaining core
//! modules (export-html, modes, the AgentSession class) stay
//! unported (see crate docs).
pub mod agent_session;
pub mod agent_session_runtime;
pub mod agent_session_services;
pub mod auth_guidance;
pub mod bash_executor;
pub mod cache_warmer;
pub mod compaction;
pub mod defaults;
pub mod diagnostics;
pub mod event_bus;
pub mod export_html;
pub mod extensions;
pub mod footer_data_provider;
pub mod messages;
pub mod model_config;
pub mod model_registry;
pub mod model_resolver;
pub mod model_runtime;
pub mod models_store_wrapper;
pub mod nested_tool_calls;
pub mod output_guard;
pub mod package_manager;
pub mod pi_manifest;
pub mod prompt_templates;
pub mod provider_attribution;
pub mod provider_composer;
pub mod remote_catalog_provider;
pub mod resolve_config_value;
pub mod resource_loader;
pub mod runtime_credentials;
pub mod sdk;
pub mod session_cwd;
pub mod session_export;
pub mod session_manager;
pub mod settings_manager;
pub mod skills;
pub mod slash_commands;
pub mod source_info;
pub mod system_prompt;
pub mod telemetry;
pub mod timings;
pub mod tools;
pub mod usage_totals;
pub mod virtual_models;

pub mod auth_storage;
pub mod bug_report;
pub mod bug_report_upload;
pub mod cache_stats;
pub mod crash_log;
pub mod exec;
pub mod experimental;
pub mod http_dispatcher;
pub mod keybindings_migration;
pub mod mcp_servers;
#[cfg(test)]
#[path = "model_registry_tests.rs"]
mod model_registry_tests;
pub mod project_trust;
pub mod radius;
pub mod settings_diagnostics;
pub mod trust_manager;
