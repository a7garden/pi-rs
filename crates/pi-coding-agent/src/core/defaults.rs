//! Port of `packages/coding-agent/src/core/defaults.ts` — the default
//! thinking level and the full option list.

use pi_agent::types::AgentThinkingLevel;

/// Upstream `DEFAULT_THINKING_LEVEL`.
pub const DEFAULT_THINKING_LEVEL: AgentThinkingLevel = AgentThinkingLevel::Medium;

/// Upstream `THINKING_LEVEL_OPTIONS`.
pub const THINKING_LEVEL_OPTIONS: &[AgentThinkingLevel] = &[
    AgentThinkingLevel::Off,
    AgentThinkingLevel::Minimal,
    AgentThinkingLevel::Low,
    AgentThinkingLevel::Medium,
    AgentThinkingLevel::High,
    AgentThinkingLevel::Xhigh,
    AgentThinkingLevel::Max,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_medium() {
        assert_eq!(DEFAULT_THINKING_LEVEL, AgentThinkingLevel::Medium);
    }

    #[test]
    fn options_match_upstream_order() {
        let rendered: Vec<String> = THINKING_LEVEL_OPTIONS
            .iter()
            .map(|level| {
                serde_json::to_value(level)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(
            rendered,
            ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
        );
    }
}
