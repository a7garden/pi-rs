//! Port of `packages/ai/src/utils/provider-env.ts`.
//!
//! The Bun sandbox `/proc/self/environ` fallback is Node-specific and does not
//! apply to Rust; provider values resolve from scoped overrides and then the
//! process environment (crate docs, deviation D8).

use crate::types::ProviderEnv;

/// Resolve a provider env value from scoped overrides, then the process
/// environment (`getProviderEnvValue`). Empty override values do not match,
/// matching upstream's `env?.[name] ||` truthiness chain.
pub fn get_provider_env_value(name: &str, env: Option<&ProviderEnv>) -> Option<String> {
    if let Some(env) = env {
        if let Some(value) = env.get(name) {
            if !value.is_empty() {
                return Some(value.clone());
            }
        }
    }
    match std::env::var(name) {
        Ok(value) if !value.is_empty() => Some(value),
        _ => None,
    }
}
