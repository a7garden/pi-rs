//! Port of `packages/ai/src/providers/` — every upstream provider factory,
//! their generated model catalogs, and `all.rs` (the upstream `all.ts`
//! built-in registry: [`builtin_provider_factories`], [`builtin_models`]).
//!
//! Faux is not part of the upstream registry — it is constructed explicitly
//! by tests — so it is not in the hook.
//!
//! Deviations continue from the crate docs: per-file D15+ ledgers (factory
//! `expect`s, oauth wiring notes, api-key fallbacks).

pub mod anthropic;
pub mod anthropic_models;
pub mod faux;
pub mod meta;
pub mod meta_models;
pub mod openai;
pub mod openai_models;
pub mod typesafe;
pub mod typesafe_models;
// Wave B modules (empty until their owning wave fills them):
pub mod all;
pub mod amazon_bedrock;
pub mod amazon_bedrock_models;
pub mod ant_ling;
pub mod ant_ling_models;
pub mod azure;
pub mod azure_models;
pub mod baseten;
pub mod baseten_models;
pub mod cerebras;
pub mod cerebras_models;
pub mod cloudflare_ai_gateway;
pub mod cloudflare_ai_gateway_models;
pub mod cloudflare_workers_ai;
pub mod cloudflare_workers_ai_models;
pub mod cloudflare_auth;
pub mod cloudflare_stream;
pub mod deepseek;
pub mod deepseek_models;
pub mod fireworks;
pub mod fireworks_models;
pub mod github_copilot;
pub mod github_copilot_models;
pub mod google;
pub mod google_models;
pub mod google_vertex;
pub mod google_vertex_models;
pub mod groq;
pub mod groq_models;
pub mod huggingface;
pub mod huggingface_models;
pub mod images;
pub mod kimi_coding;
pub mod kimi_coding_models;
pub mod minimax;
pub mod minimax_models;
pub mod minimax_cn;
pub mod minimax_cn_models;
pub mod mistral;
pub mod mistral_models;
pub mod moonshotai;
pub mod moonshotai_models;
pub mod moonshotai_cn;
pub mod moonshotai_cn_models;
pub mod nvidia;
pub mod nvidia_models;
pub mod opencode;
pub mod opencode_go;
pub mod opencode_go_models;
pub mod opencode_headers;
pub mod opencode_models;
pub mod openai_codex;
pub mod openai_codex_models;
pub mod openrouter;
pub mod openrouter_models;
pub mod qwen_token_plan;
pub mod qwen_token_plan_cn;
pub mod qwen_token_plan_cn_models;
pub mod qwen_token_plan_individual;
pub mod qwen_token_plan_individual_models;
pub mod qwen_token_plan_models;
pub mod radius;
pub mod radius_config;
pub mod radius_models;
pub mod together;
pub mod together_models;
pub mod vercel_ai_gateway;
pub mod vercel_ai_gateway_models;
pub mod xai;
pub mod xai_models;
pub mod xiaomi;
pub mod xiaomi_models;
pub mod xiaomi_token_plan_ams;
pub mod xiaomi_token_plan_ams_models;
pub mod xiaomi_token_plan_cn;
pub mod xiaomi_token_plan_cn_models;
pub mod xiaomi_token_plan_sgp;
pub mod xiaomi_token_plan_sgp_models;
pub mod zai;
pub mod zai_coding_cn;
pub mod zai_coding_cn_models;
pub mod zai_models;

pub use all::{builtin_models, builtin_provider_factories, builtin_providers};
