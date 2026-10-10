//! Port of `packages/ai/src/api/simple-options.ts` — shared option-building
//! infrastructure used by the API adapters (`streamSimple` base options,
//! sampling parameter resolution, thinking budget math).
//!
//! Deviations:
//! - D20: upstream `model.contextWindow <= 0` guards an absent/zero window;
//!   Rust carries `contextWindow` as `u64`, so the guard is `== 0`. All
//!   token arithmetic saturates instead of wrapping.
//! - D21: upstream merges custom thinking budgets with JS object spread, so
//!   a custom budget explicitly set to `undefined` would shadow a default
//!   with `undefined`; the typed Rust `ThinkingBudgets` (`Option<u64>`
//!   fields) only overrides keys that carry a value, and a level budget is
//!   always present after the merge (upstream asserts it with `!`).

use std::cmp::{max, min};

use crate::models::clamp_thinking_level;
use crate::types::{
    Model, ModelThinkingLevel, SamplingParams, SimpleStreamOptions, StreamOptions, ThinkingBudgets,
    ThinkingLevel, TranscriptContext,
};
use crate::utils::estimate::estimate_context_tokens;

const CONTEXT_SAFETY_TOKENS: u64 = 4096;
const MIN_MAX_TOKENS: u64 = 1;

pub fn clamp_max_tokens_to_context(
    model: &Model,
    context: &TranscriptContext,
    max_tokens: u64,
) -> u64 {
    if model.context_window == 0 {
        return max(MIN_MAX_TOKENS, max_tokens);
    }
    let available = model
        .context_window
        .saturating_sub(estimate_context_tokens(context).tokens)
        .saturating_sub(CONTEXT_SAFETY_TOKENS);
    min(max_tokens, max(MIN_MAX_TOKENS, available))
}

pub fn resolve_sampling_params(
    model: &Model,
    thinking_level: ModelThinkingLevel,
    request_params: Option<&SamplingParams>,
) -> Option<SamplingParams> {
    let effective_thinking_level = clamp_thinking_level(model, thinking_level);
    let thinking_level_params = model
        .sampling_params_by_thinking_level
        .as_ref()
        .and_then(|params| params.get(&effective_thinking_level));
    if model.sampling_params.is_none()
        && thinking_level_params.is_none()
        && request_params.is_none()
    {
        return None;
    }
    // `{ ...model.samplingParams, ...thinkingLevelParams, ...requestParams }`:
    // later sources override earlier keys.
    let mut merged = SamplingParams::new();
    if let Some(params) = &model.sampling_params {
        merged.extend(params.clone());
    }
    if let Some(params) = thinking_level_params {
        merged.extend(params.clone());
    }
    if let Some(params) = request_params {
        merged.extend(params.clone());
    }
    Some(merged)
}

pub fn build_base_options(
    model: &Model,
    context: &TranscriptContext,
    options: &SimpleStreamOptions,
    api_key: Option<&str>,
) -> StreamOptions {
    let thinking_level: ModelThinkingLevel = match options.reasoning {
        Some(level) => level.into(),
        None => ModelThinkingLevel::Off,
    };
    let sampling_params =
        resolve_sampling_params(model, thinking_level, options.base.sampling_params.as_ref());
    let base = &options.base.base;
    StreamOptions {
        base: crate::types::ProviderRequestOptions {
            signal: base.signal.clone(),
            telemetry_context: base.telemetry_context.clone(),
            // `apiKey || options?.apiKey`: an empty explicit key falls through.
            api_key: match api_key.filter(|key| !key.is_empty()) {
                Some(key) => Some(key.to_string()),
                None => base.api_key.clone(),
            },
            fetch: base.fetch.clone(),
            env: base.env.clone(),
            on_payload: base.on_payload.clone(),
            on_response: base.on_response.clone(),
            headers: base.headers.clone(),
            timeout_ms: base.timeout_ms,
            max_retries: base.max_retries,
            max_retry_delay_ms: base.max_retry_delay_ms,
        },
        on_provider_stream_event: options.base.on_provider_stream_event.clone(),
        anthropic: Default::default(),
        openai_completions: Default::default(),
        openai_responses: Default::default(),
        temperature: options.base.temperature,
        sampling_params,
        max_tokens: Some(clamp_max_tokens_to_context(
            model,
            context,
            options.base.max_tokens.unwrap_or(model.max_tokens),
        )),
        transport: options.base.transport,
        cache_retention: options.base.cache_retention,
        session_id: options.base.session_id.clone(),
        websocket_connect_timeout_ms: options.base.websocket_connect_timeout_ms,
        metadata: options.base.metadata.clone(),
    }
}

/// Tokens always left for the answer when a thinking budget shares the response
/// ceiling.
pub const MIN_ANSWER_TOKENS: u64 = 1024;

pub const DEFAULT_THINKING_BUDGETS: ThinkingBudgets = ThinkingBudgets {
    minimal: Some(1024),
    low: Some(2048),
    medium: Some(8192),
    high: Some(16384),
};

fn clamp_reasoning_level(effort: ThinkingLevel) -> ThinkingLevel {
    match effort {
        ThinkingLevel::Xhigh | ThinkingLevel::Max => ThinkingLevel::High,
        level => level,
    }
}

pub fn clamp_reasoning(effort: Option<ThinkingLevel>) -> Option<ThinkingLevel> {
    effort.map(clamp_reasoning_level)
}

/// `{ ...DEFAULT_THINKING_BUDGETS, ...customBudgets }` with deviation D21.
fn merged_budgets(custom_budgets: Option<&ThinkingBudgets>) -> ThinkingBudgets {
    let mut budgets = DEFAULT_THINKING_BUDGETS;
    if let Some(custom) = custom_budgets {
        if custom.minimal.is_some() {
            budgets.minimal = custom.minimal;
        }
        if custom.low.is_some() {
            budgets.low = custom.low;
        }
        if custom.medium.is_some() {
            budgets.medium = custom.medium;
        }
        if custom.high.is_some() {
            budgets.high = custom.high;
        }
    }
    budgets
}

pub fn thinking_budget_for_level(
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> u64 {
    let budgets = merged_budgets(custom_budgets);
    let level = clamp_reasoning_level(reasoning_level);
    let budget = match level {
        ThinkingLevel::Minimal => budgets.minimal,
        ThinkingLevel::Low => budgets.low,
        ThinkingLevel::Medium => budgets.medium,
        ThinkingLevel::High => budgets.high,
        // clamp_reasoning_level never yields these (deviation D21: upstream's
        // non-null assertion is always satisfied after the merge).
        ThinkingLevel::Xhigh | ThinkingLevel::Max => None,
    };
    budget.unwrap_or(0)
}

/// Cap a thinking budget so at least MIN_ANSWER_TOKENS remain under a shared
/// response ceiling.
pub fn clamp_thinking_budget_to_answer_room(thinking_budget: u64, ceiling: u64) -> u64 {
    min(thinking_budget, ceiling.saturating_sub(MIN_ANSWER_TOKENS))
}

/// Upstream `{ maxTokens, thinkingBudget }` return of
/// [`adjust_max_tokens_for_thinking`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdjustedThinkingTokens {
    pub max_tokens: u64,
    pub thinking_budget: u64,
}

pub fn adjust_max_tokens_for_thinking(
    // Undefined means no explicit caller cap. Use the model cap and fit thinking inside it.
    base_max_tokens: Option<u64>,
    model_max_tokens: u64,
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> AdjustedThinkingTokens {
    let mut thinking_budget = thinking_budget_for_level(reasoning_level, custom_budgets);
    let max_tokens = match base_max_tokens {
        None => model_max_tokens,
        Some(base) => min(base.saturating_add(thinking_budget), model_max_tokens),
    };

    if max_tokens <= thinking_budget {
        thinking_budget = clamp_thinking_budget_to_answer_room(thinking_budget, max_tokens);
    }

    AdjustedThinkingTokens {
        max_tokens,
        thinking_budget,
    }
}

#[cfg(test)]
mod tests {
    // Port of the parts of test/sampling-options.test.ts that belong to this
    // module. The upstream suite asserts the merged sampling parameters as they
    // appear in captured provider request bodies (`stream`/`streamSimple` over
    // the OpenAI-compatible and Anthropic adapters); the capture, API-parameter
    // mapping (`reasoningEffort`/`reasoningSummary` → request fields), and
    // non-OpenAI-API ignorance assertions exercise the adapter slices and
    // travel with those modules. The sampling-merge matrix they observe is
    // decided here, so it is ported against `resolve_sampling_params` /
    // `build_base_options` directly.
    use super::*;
    use crate::types::{
        BaseModelFields, Message, ModelCost, ModelCostRates, ModelInput, ProviderRequestOptions,
        UserContentField, UserMessage,
    };

    fn make_model(
        sampling_params: Option<SamplingParams>,
        overrides: impl FnOnce(&mut Model),
    ) -> Model {
        let mut model = Model {
            base: BaseModelFields {
                id: "custom-model".to_string(),
                name: "Custom Model".to_string(),
                api: "openai-completions".to_string(),
                provider: "custom-provider".to_string(),
                base_url: "http://127.0.0.1:9/v1".to_string(),
                input: vec![ModelInput::Text],
                input_limits: None,
                cost: ModelCost {
                    rates: ModelCostRates {
                        input: 0.0,
                        output: 0.0,
                        cache_read: 0.0,
                        cache_write: 0.0,
                    },
                    tiers: None,
                },
                headers: None,
            },
            model_type: None,
            reasoning: false,
            thinking_level_map: None,
            prompt_cache: None,
            context_window: 128_000,
            max_tokens: 16_384,
            sampling_params,
            sampling_params_by_thinking_level: None,
            compat: None,
        };
        overrides(&mut model);
        model
    }

    fn params(entries: &[(&str, serde_json::Value)]) -> SamplingParams {
        entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect()
    }

    fn empty_context() -> TranscriptContext {
        crate::utils::transcript::normalize_context(&crate::types::Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage {
                content: UserContentField::Text("Hello".to_string()),
                timestamp: 0,
            })],
            tools: None,
        })
    }

    fn model_level_map(
        model: &mut Model,
        low: Option<Option<String>>,
        medium: Option<Option<String>>,
    ) {
        let mut map = crate::types::ThinkingLevelMap::new();
        if let Some(low) = low {
            map.insert(ModelThinkingLevel::Low, low);
        }
        if let Some(medium) = medium {
            map.insert(ModelThinkingLevel::Medium, medium);
        }
        model.thinking_level_map = Some(map);
    }

    #[test]
    fn merges_request_sampling_params() {
        let model = make_model(None, |_| {});
        let merged = resolve_sampling_params(
            &model,
            ModelThinkingLevel::Off,
            Some(&params(&[
                ("top_p", 0.95.into()),
                ("top_k", 0.into()),
                ("min_p", 0.into()),
            ])),
        )
        .expect("request params set");
        assert_eq!(merged.get("top_p"), Some(&serde_json::json!(0.95)));
        assert_eq!(merged.get("top_k"), Some(&serde_json::json!(0)));
        assert_eq!(merged.get("min_p"), Some(&serde_json::json!(0)));
    }

    #[test]
    fn omits_sampling_params_when_neither_options_nor_model_set_them() {
        let model = make_model(None, |_| {});
        assert_eq!(
            resolve_sampling_params(&model, ModelThinkingLevel::Off, None),
            None
        );
    }

    #[test]
    fn applies_model_level_sampling_params_with_request_keys_taking_precedence() {
        // Ported for `openai-completions`; the `openai-responses` and
        // `azure-openai-responses` upstream variants assert the same merge with
        // different adapters — the resolution is api-independent.
        let model = make_model(
            Some(params(&[("top_p", 0.95.into()), ("min_p", 0.05.into())])),
            |_| {},
        );
        let merged = resolve_sampling_params(
            &model,
            ModelThinkingLevel::Off,
            Some(&params(&[("top_p", 0.5.into())])),
        )
        .expect("params set");
        assert_eq!(merged.get("top_p"), Some(&serde_json::json!(0.5)));
        assert_eq!(merged.get("min_p"), Some(&serde_json::json!(0.05)));
    }

    #[test]
    fn passes_request_sampling_params_through_stream_simple() {
        let model = make_model(None, |_| {});
        let options = SimpleStreamOptions {
            base: StreamOptions {
                sampling_params: Some(params(&[("top_p", 0.5.into())])),
                ..Default::default()
            },
            ..Default::default()
        };
        let built = build_base_options(&model, &empty_context(), &options, None);
        assert_eq!(
            built.sampling_params.as_ref().and_then(|p| p.get("top_p")),
            Some(&serde_json::json!(0.5))
        );
    }

    #[test]
    fn applies_sampling_params_for_the_effective_thinking_level_over_model_defaults() {
        let mut model = make_model(
            Some(params(&[
                ("temperature", 1.0.into()),
                ("top_p", 0.95.into()),
            ])),
            |model| {
                model.reasoning = true;
            },
        );
        model.sampling_params_by_thinking_level = Some(
            [(
                ModelThinkingLevel::High,
                params(&[("temperature", 0.8.into()), ("top_k", 64.into())]),
            )]
            .into_iter()
            .collect(),
        );
        model_level_map(&mut model, Some(None), Some(None));

        // `reasoning: "low"` is clamped to `high` (low/medium are unsupported).
        let built = build_base_options(
            &model,
            &empty_context(),
            &SimpleStreamOptions {
                base: StreamOptions::default(),
                reasoning: Some(ThinkingLevel::Low),
                ..Default::default()
            },
            None,
        );
        let merged = built
            .sampling_params
            .as_ref()
            .expect("thinking-level params");
        assert_eq!(merged.get("temperature"), Some(&serde_json::json!(0.8)));
        assert_eq!(merged.get("top_p"), Some(&serde_json::json!(0.95)));
        assert_eq!(merged.get("top_k"), Some(&serde_json::json!(64)));
    }

    #[test]
    fn applies_off_sampling_params_when_reasoning_is_disabled() {
        let model = make_model(None, |model| {
            model.sampling_params_by_thinking_level = Some(
                [(
                    ModelThinkingLevel::Off,
                    params(&[("temperature", 0.7.into())]),
                )]
                .into_iter()
                .collect(),
            );
        });

        let built = build_base_options(
            &model,
            &empty_context(),
            &SimpleStreamOptions::default(),
            None,
        );
        let merged = built.sampling_params.as_ref().expect("off params");
        assert_eq!(merged.get("temperature"), Some(&serde_json::json!(0.7)));
    }

    #[test]
    fn merges_stream_option_keys_over_thinking_level_keys() {
        let mut model = make_model(None, |model| {
            model.reasoning = true;
        });
        model.sampling_params_by_thinking_level = Some(
            [(
                ModelThinkingLevel::Low,
                params(&[("temperature", 0.6.into()), ("top_p", 0.95.into())]),
            )]
            .into_iter()
            .collect(),
        );

        let built = build_base_options(
            &model,
            &empty_context(),
            &SimpleStreamOptions {
                base: StreamOptions {
                    sampling_params: Some(params(&[("top_p", 0.5.into())])),
                    ..Default::default()
                },
                reasoning: Some(ThinkingLevel::Low),
                ..Default::default()
            },
            None,
        );
        let merged = built.sampling_params.as_ref().expect("params");
        assert_eq!(merged.get("temperature"), Some(&serde_json::json!(0.6)));
        assert_eq!(merged.get("top_p"), Some(&serde_json::json!(0.5)));
    }

    #[test]
    fn applies_thinking_level_params_between_model_and_request_params() {
        // Same merge as the upstream `openai-completions` /
        // `openai-responses` / `azure-openai-responses` parameterized case; the
        // `reasoningEffort` → `reasoning.effort` body mapping is adapter-owned.
        let mut model = make_model(
            Some(params(&[
                ("temperature", 1.0.into()),
                ("top_p", 0.95.into()),
            ])),
            |model| {
                model.reasoning = true;
            },
        );
        model.sampling_params_by_thinking_level = Some(
            [(
                ModelThinkingLevel::Low,
                params(&[("temperature", 0.6.into()), ("top_k", 64.into())]),
            )]
            .into_iter()
            .collect(),
        );

        let merged = resolve_sampling_params(
            &model,
            ModelThinkingLevel::Low,
            Some(&params(&[("top_p", 0.5.into())])),
        )
        .expect("params");
        assert_eq!(merged.get("temperature"), Some(&serde_json::json!(0.6)));
        assert_eq!(merged.get("top_p"), Some(&serde_json::json!(0.5)));
        assert_eq!(merged.get("top_k"), Some(&serde_json::json!(64)));
    }

    #[test]
    fn uses_medium_sampling_params_for_summary_only_requests() {
        // The adapter maps `reasoningSummary` to the medium effort; the
        // sampling selection for that level lives here.
        let model = make_model(None, |model| {
            model.reasoning = true;
            model.sampling_params_by_thinking_level = Some(
                [
                    (
                        ModelThinkingLevel::Off,
                        params(&[("temperature", 0.7.into())]),
                    ),
                    (
                        ModelThinkingLevel::Medium,
                        params(&[("temperature", 0.8.into())]),
                    ),
                ]
                .into_iter()
                .collect(),
            );
        });

        let merged =
            resolve_sampling_params(&model, ModelThinkingLevel::Medium, None).expect("params");
        assert_eq!(merged.get("temperature"), Some(&serde_json::json!(0.8)));
    }

    #[test]
    fn build_base_options_carries_named_fields_alongside_sampling_params() {
        // Upstream asserts the adapter emits sampling params after the named
        // `temperature` field; at this layer the named field and the merged
        // params must simply coexist on the built options.
        let model = make_model(None, |_| {});
        let options = SimpleStreamOptions {
            base: StreamOptions {
                temperature: Some(0.0),
                sampling_params: Some(params(&[("temperature", 1.0.into())])),
                base: ProviderRequestOptions {
                    api_key: Some("fake-key".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let built = build_base_options(&model, &empty_context(), &options, None);
        assert_eq!(built.temperature, Some(0.0));
        assert_eq!(
            built
                .sampling_params
                .as_ref()
                .and_then(|p| p.get("temperature")),
            Some(&serde_json::json!(1.0))
        );
        assert_eq!(built.api_key.as_deref(), Some("fake-key"));
    }

    #[test]
    fn clamps_max_tokens_to_context() {
        let model = make_model(None, |_| {});
        let context = empty_context();
        // maxTokens below the available window stays untouched.
        assert_eq!(
            clamp_max_tokens_to_context(&model, &context, 16_384),
            16_384
        );
        // maxTokens above the window is clamped to the remaining room
        // (contextWindow - estimate - 4096 safety).
        let clamped = clamp_max_tokens_to_context(&model, &context, 200_000);
        assert!(clamped < 200_000);
        // Zero-context models keep the caller cap.
        let mut no_window = make_model(None, |_| {});
        no_window.context_window = 0;
        assert_eq!(
            clamp_max_tokens_to_context(&no_window, &context, 8_000),
            8_000
        );
        assert_eq!(clamp_max_tokens_to_context(&no_window, &context, 0), 1);
    }

    #[test]
    fn thinking_budget_helpers() {
        assert_eq!(clamp_reasoning(None), None);
        assert_eq!(
            clamp_reasoning(Some(ThinkingLevel::Xhigh)),
            Some(ThinkingLevel::High)
        );
        assert_eq!(
            clamp_reasoning(Some(ThinkingLevel::Max)),
            Some(ThinkingLevel::High)
        );
        assert_eq!(
            clamp_reasoning(Some(ThinkingLevel::Medium)),
            Some(ThinkingLevel::Medium)
        );

        assert_eq!(thinking_budget_for_level(ThinkingLevel::Low, None), 2048);
        let custom = ThinkingBudgets {
            low: Some(4096),
            ..Default::default()
        };
        assert_eq!(
            thinking_budget_for_level(ThinkingLevel::Low, Some(&custom)),
            4096
        );
        // Other levels keep their defaults (D21: only provided keys override).
        assert_eq!(
            thinking_budget_for_level(ThinkingLevel::High, Some(&custom)),
            16384
        );

        assert_eq!(clamp_thinking_budget_to_answer_room(2048, 4096), 2048);
        assert_eq!(clamp_thinking_budget_to_answer_room(8192, 4096), 3072);
        assert_eq!(clamp_thinking_budget_to_answer_room(1024, 512), 0);
    }

    #[test]
    fn adjust_max_tokens_for_thinking_fits_budget() {
        // No explicit caller cap: the model cap is used as-is and the budget fits.
        let adjusted = adjust_max_tokens_for_thinking(None, 16_384, ThinkingLevel::Low, None);
        assert_eq!(
            adjusted,
            AdjustedThinkingTokens {
                max_tokens: 16_384,
                thinking_budget: 2048
            }
        );

        // Explicit cap: budget is added on top before clamping to the model cap.
        // maxTokens (16384) <= budget (16384), so the budget then shrinks to
        // leave MIN_ANSWER_TOKENS (upstream `maxTokens <= thinkingBudget` branch).
        let adjusted =
            adjust_max_tokens_for_thinking(Some(10_000), 16_384, ThinkingLevel::High, None);
        assert_eq!(adjusted.max_tokens, 16_384);
        assert_eq!(adjusted.thinking_budget, 15_360);

        // Tight ceiling: the budget shrinks to leave MIN_ANSWER_TOKENS.
        let adjusted = adjust_max_tokens_for_thinking(Some(0), 2_048, ThinkingLevel::Medium, None);
        assert_eq!(adjusted.max_tokens, 2_048);
        assert_eq!(
            adjusted.thinking_budget,
            clamp_thinking_budget_to_answer_room(8192, 2_048)
        );
    }
}
