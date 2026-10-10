//! API implementation modules (upstream `src/api/*.ts`).

pub mod anthropic_messages;
pub mod classifier_shared;
pub mod constrained_sampling;
pub mod lazy;
pub mod openai_completions;
pub mod openai_decisions;
pub mod openai_prompt_cache;
pub mod openai_responses;
pub mod openai_responses_shared;
pub mod simple_options;
pub mod system_one_shared;
pub mod transform_messages;
pub mod typesafe_system_one;
// Wave B modules (empty until their owning wave fills them):
pub mod azure_openai_completions;
pub mod azure_openai_config;
pub mod azure_openai_responses;
pub mod bedrock_converse_stream;
pub mod cloudflare;
pub mod cloudflare_ai_binding;
pub mod cloudflare_workers_ai_system_one;
pub mod github_copilot_headers;
pub mod google_generative_ai;
pub mod google_shared;
pub mod google_vertex;
pub mod llama_cpp_classify;
pub mod mistral_conversations;
pub mod openai_codex_responses;
pub mod openrouter_images;
pub mod pi_messages;
