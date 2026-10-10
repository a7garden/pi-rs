//! Port of `packages/coding-agent/src/core/cache-stats.ts` — prompt-cache
//! miss accounting over the [`SessionEntry`] stream.
//!
//! Deviations:
//! - **D118 — entry-id keying, minimal pricing trait, timestamp handling.**
//!   Upstream keys `collectCacheMisses` by the `AssistantMessage` object
//!   reference; Rust messages carry no stable id, so the map is keyed by the
//!   owning [`SessionMessageEntry`] base id — the stable handle consumers
//!   hold while rebuilding the chat from entries (resume, post-compaction
//!   rebuild). Upstream's `ModelPriceSource`
//!   (`getModel(provider, modelId).cost.cacheRead`) becomes the minimal
//!   [`ModelPriceSource::cache_read_price`] trait ($/million tokens),
//!   implemented for [`ModelRuntime`]. Usage-entry timestamps that fail to
//!   parse are treated as epoch 0 where upstream's `Date.parse` NaN
//!   propagates into `idleMs`.

use std::collections::HashMap;

use pi_agent::AgentMessage;
use pi_ai::models::Models;
use pi_ai::types::{AssistantMessage, Message};

use crate::core::model_runtime::ModelRuntime;
use crate::core::session_manager::SessionEntry;
use crate::utils::time::parse_timestamp_ms;

/// Prompt-cache TTL: idle gaps longer than this are worth mentioning as the
/// likely cause of a miss. Anthropic's default cache TTL is 5 minutes.
pub const CACHE_TTL_MS: u64 = 5 * 60 * 1000;

/// Per-turn misses at or below this are cache breakpoint granularity noise.
const NOISE_FLOOR_TOKENS: i64 = 1024;

/// A counted cache miss on a single assistant message (`CacheMiss`).
#[derive(Debug, Clone, PartialEq)]
pub struct CacheMiss {
    /// Prompt tokens that were in the previous turn's prompt but not read from cache.
    pub missed_tokens: u64,
    /// Extra dollars paid vs. a full cache hit; 0 when pricing is unknown.
    pub missed_cost: f64,
    /// Milliseconds since the previous request (which last refreshed the cache).
    pub idle_ms: i64,
    /// True when the model changed relative to the previous request.
    pub model_changed: bool,
}

/// Cumulative cache waste across a session (`CacheWasteTotals`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CacheWasteTotals {
    pub missed_tokens: u64,
    pub missed_cost: f64,
    /// Number of counted misses (turns above the noise floor).
    pub miss_count: usize,
}

/// Minimal pricing lookup, satisfied by `ModelRuntime` (`ModelPriceSource`).
/// Cost is $/million tokens.
pub trait ModelPriceSource {
    fn cache_read_price(&self, provider: &str, model_id: &str) -> Option<f64>;
}

impl ModelPriceSource for ModelRuntime {
    fn cache_read_price(&self, provider: &str, model_id: &str) -> Option<f64> {
        Models::get_model(self, provider, model_id).map(|model| model.base.cost.rates.cache_read)
    }
}

/// The last request seen by the scan; everything in its prompt should be
/// cached (`PreviousRequest`).
#[derive(Debug, Clone)]
struct PreviousRequest {
    prompt_tokens: u64,
    model_key: String,
    timestamp: i64,
    /// Sticky: some earlier request in this scan segment reported cache
    /// activity. Distinguishes a total miss on a cache-read-only provider
    /// (OpenAI-style, writes unreported) from a provider that never reports
    /// caching at all.
    reported_cache: bool,
}

/// Compute the cache miss for one assistant message relative to the previous
/// request (`detectMiss`). Returns `None` when nothing is counted: first
/// turn, after a reset, no cache activity ever reported (provider without
/// cache support), or miss below the noise floor.
fn detect_miss(
    prev: Option<&PreviousRequest>,
    message: &AssistantMessage,
    models: &dyn ModelPriceSource,
) -> Option<CacheMiss> {
    let prev = prev?;
    let usage = &message.usage;
    let prompt_tokens = usage.input + usage.cache_read + usage.cache_write;
    // A zero-cache turn only counts when cache activity was reported before:
    // on cache-read-only providers that is a total miss, while on providers
    // that never report caching it means nothing.
    if prompt_tokens == 0 || (usage.cache_read + usage.cache_write == 0 && !prev.reported_cache) {
        return None;
    }

    let missed_tokens = prev.prompt_tokens.min(prompt_tokens) as i64 - usage.cache_read as i64;
    if missed_tokens <= NOISE_FLOOR_TOKENS {
        return None;
    }

    // Extra cost = missed tokens billed at the actual paid rate (input/cacheWrite,
    // incl. write premium) instead of the cache-read rate. Missed tokens can only
    // land in the input or cacheWrite buckets, so the paid rate comes straight
    // from this message's own cost breakdown.
    let paid_tokens = usage.input + usage.cache_write;
    let paid_per_token = if paid_tokens > 0 {
        (usage.cost.input + usage.cost.cache_write) / paid_tokens as f64
    } else {
        0.0
    };
    let read_per_token = if usage.cache_read > 0 {
        usage.cost.cache_read / usage.cache_read as f64
    } else {
        models
            .cache_read_price(&message.provider, &message.model)
            .unwrap_or(0.0)
            / 1_000_000.0
    };

    Some(CacheMiss {
        missed_tokens: missed_tokens as u64,
        missed_cost: missed_tokens as f64 * (paid_per_token - read_per_token).max(0.0),
        idle_ms: (message.timestamp - prev.timestamp).max(0),
        model_changed: format!("{}/{}", message.provider, message.model) != prev.model_key,
    })
}

/// `asPreviousRequest`.
fn as_previous_request(
    message: &AssistantMessage,
    reported_cache: bool,
) -> Option<PreviousRequest> {
    let usage = &message.usage;
    let prompt_tokens = usage.input + usage.cache_read + usage.cache_write;
    if prompt_tokens == 0 {
        return None;
    }
    Some(PreviousRequest {
        prompt_tokens,
        model_key: format!("{}/{}", message.provider, message.model),
        timestamp: message.timestamp,
        reported_cache: reported_cache || usage.cache_read + usage.cache_write > 0,
    })
}

/// `scan` — state accumulated over one pass of the entry stream.
struct Scan {
    prev: Option<PreviousRequest>,
    totals: CacheWasteTotals,
    misses: HashMap<String, CacheMiss>,
}

fn scan(entries: &[SessionEntry], models: &dyn ModelPriceSource) -> Scan {
    let mut prev: Option<PreviousRequest> = None;
    let mut totals = CacheWasteTotals {
        missed_tokens: 0,
        missed_cost: 0.0,
        miss_count: 0,
    };
    let mut misses: HashMap<String, CacheMiss> = HashMap::new();

    for entry in entries {
        match entry {
            SessionEntry::Compaction(_) | SessionEntry::BranchSummary(_) => {
                // The context legitimately changed; the next turn's prompt is new content,
                // not re-billed content. Model switches are NOT exempt: they re-bill the
                // full prompt and should be counted.
                prev = None;
            }
            SessionEntry::Usage(usage) if usage.kind == "cache_warm" => {
                let prompt_tokens =
                    usage.usage.input + usage.usage.cache_read + usage.usage.cache_write;
                if prompt_tokens > 0 {
                    prev = Some(PreviousRequest {
                        prompt_tokens,
                        model_key: format!("{}/{}", usage.provider, usage.model),
                        timestamp: parse_timestamp_ms(&usage.base.timestamp).unwrap_or(0), // D118
                        reported_cache: true,
                    });
                }
            }
            SessionEntry::Message(message_entry) => {
                let AgentMessage::Llm(Message::Assistant(message)) = &message_entry.message else {
                    continue;
                };
                if let Some(miss) = detect_miss(prev.as_ref(), message, models) {
                    totals.missed_tokens += miss.missed_tokens;
                    totals.missed_cost += miss.missed_cost;
                    totals.miss_count += 1;
                    misses.insert(message_entry.base.id.clone(), miss); // D118
                }
                prev = as_previous_request(
                    message,
                    prev.as_ref().is_some_and(|prev| prev.reported_cache),
                )
                .or(prev);
            }
            _ => {}
        }
    }
    Scan {
        prev,
        totals,
        misses,
    }
}

/// Cumulative cache waste across a session: prompt tokens that should have been
/// cache reads (they were in the previous turn's prompt) but were re-billed
/// (`computeCacheWaste`).
pub fn compute_cache_waste(
    entries: &[SessionEntry],
    models: &dyn ModelPriceSource,
) -> CacheWasteTotals {
    scan(entries, models).totals
}

/// All counted cache misses across a session, keyed by the session entry id
/// of the assistant message that paid for them (D118). Used to re-derive
/// transcript notices when rebuilding the chat from entries (resume,
/// post-compaction rebuild).
pub fn collect_cache_misses(
    entries: &[SessionEntry],
    models: &dyn ModelPriceSource,
) -> HashMap<String, CacheMiss> {
    scan(entries, models).misses
}

/// Detect a cache miss on a just-completed assistant message
/// (`detectCacheMiss`). `entries` must not yet contain `message`
/// (message_end fires before persistence).
pub fn detect_cache_miss(
    entries: &[SessionEntry],
    message: &AssistantMessage,
    models: &dyn ModelPriceSource,
) -> Option<CacheMiss> {
    detect_miss(scan(entries, models).prev.as_ref(), message, models)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::session_manager::{
        CompactionEntry, SessionEntryBase, SessionMessageEntry, UsageEntry,
    };
    use crate::utils::time::to_iso_timestamp;
    use pi_ai::types::{StopReason, Usage, UsageCost};

    /// The upstream test model source: $0.3/M cache read as the fallback price.
    struct TestPrices;

    impl ModelPriceSource for TestPrices {
        fn cache_read_price(&self, _provider: &str, _model_id: &str) -> Option<f64> {
            Some(0.3)
        }
    }

    fn usage(input: u64, cache_read: u64, cache_write: u64, cost: UsageCost) -> Usage {
        Usage {
            input,
            output: 10,
            cache_read,
            cache_write,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 0,
            cost,
        }
    }

    fn cost(input: f64, cache_read: f64, cache_write: f64) -> UsageCost {
        UsageCost {
            input,
            output: 0.0,
            cache_read,
            cache_write,
            total: 0.0,
        }
    }

    fn assistant(
        input: u64,
        cache_read: u64,
        cache_write: u64,
        cost: UsageCost,
        model: &str,
        timestamp: i64,
    ) -> AssistantMessage {
        AssistantMessage {
            content: Vec::new(),
            api: "anthropic-messages".to_string(),
            provider: "test".to_string(),
            model: model.to_string(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            thinking_level: None,
            diagnostics: None,
            usage: usage(input, cache_read, cache_write, cost),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp,
            duration_ms: None,
        }
    }

    fn entry(message: AssistantMessage) -> SessionEntry {
        SessionEntry::Message(SessionMessageEntry {
            base: SessionEntryBase {
                id: "x".to_string(),
                parent_id: None,
                timestamp: String::new(),
            },
            message: AgentMessage::Llm(Message::Assistant(message)),
        })
    }

    fn usage_entry(kind: &str, timestamp_ms: i64) -> SessionEntry {
        SessionEntry::Usage(UsageEntry {
            base: SessionEntryBase {
                id: format!("usage-{kind}"),
                parent_id: None,
                timestamp: to_iso_timestamp(timestamp_ms),
            },
            kind: kind.to_string(),
            provider: "test".to_string(),
            model: "test-model".to_string(),
            usage: usage(0, 100_000, 0, UsageCost::default()),
            note: None,
        })
    }

    fn compaction_entry() -> SessionEntry {
        SessionEntry::Compaction(CompactionEntry {
            base: SessionEntryBase {
                id: "c".to_string(),
                parent_id: None,
                timestamp: String::new(),
            },
            summary: String::new(),
            first_kept_entry_id: String::new(),
            tokens_before: 0,
            details: None,
            usage: None,
            from_hook: None,
            system_message: None,
            first_kept_entry_index: None,
        })
    }

    // Turn 1: fresh 100k cache write at $3.75/M.
    fn turn1() -> AssistantMessage {
        assistant(0, 0, 100_000, cost(0.0, 0.0, 0.375), "test-model", 0)
    }
    // Turn 2: healthy, everything read back at $0.30/M.
    fn turn2() -> AssistantMessage {
        assistant(
            0,
            100_000,
            5_000,
            cost(0.0, 0.03, 0.019),
            "test-model",
            60_000,
        )
    }

    fn models() -> TestPrices {
        TestPrices
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    // Upstream "accumulates missed tokens and cost across turns".
    #[test]
    fn accumulates_missed_tokens_and_cost_across_turns() {
        // Turn 3: full miss, previous 105k prompt re-billed at $3.75/M write.
        let turn3 = assistant(0, 0, 110_000, cost(0.0, 0.0, 0.4125), "test-model", 120_000);
        let totals =
            compute_cache_waste(&[entry(turn1()), entry(turn2()), entry(turn3)], &models());
        assert_eq!(totals.missed_tokens, 105_000);
        // 105k at ($3.75 - $0.30)/M.
        assert_close(totals.missed_cost, 0.36225);
        assert_eq!(totals.miss_count, 1);
    }

    // Upstream "counts nothing for healthy sessions".
    #[test]
    fn counts_nothing_for_healthy_sessions() {
        let totals = compute_cache_waste(&[entry(turn1()), entry(turn2())], &models());
        assert_eq!(totals.missed_tokens, 0);
        assert_eq!(totals.missed_cost, 0.0);
        assert_eq!(totals.miss_count, 0);
    }

    // Upstream "skips the turn after a compaction reset".
    #[test]
    fn skips_the_turn_after_a_compaction_reset() {
        let after_reset = assistant(0, 0, 20_000, cost(0.0, 0.0, 0.075), "test-model", 0);
        let totals = compute_cache_waste(
            &[entry(turn1()), compaction_entry(), entry(after_reset)],
            &models(),
        );
        assert_eq!(totals.missed_tokens, 0);
    }

    // Upstream "counts misses caused by model switches".
    #[test]
    fn counts_misses_caused_by_model_switches() {
        let other_model = assistant(0, 0, 100_000, cost(0.0, 0.0, 0.375), "other-model", 0);
        let totals = compute_cache_waste(&[entry(turn1()), entry(other_model)], &models());
        assert_eq!(totals.missed_tokens, 100_000);
        assert_eq!(totals.miss_count, 1);
    }

    // Upstream "skips providers that report no cache activity".
    #[test]
    fn skips_providers_that_report_no_cache_activity() {
        let a = assistant(100_000, 0, 0, UsageCost::default(), "test-model", 0);
        let b = assistant(110_000, 0, 0, UsageCost::default(), "test-model", 60_000);
        let totals = compute_cache_waste(&[entry(a), entry(b)], &models());
        assert_eq!(totals.missed_tokens, 0);
    }

    // Upstream "maps counted misses to their assistant messages by reference"
    // (D118: keyed by session entry id).
    #[test]
    fn maps_counted_misses_to_their_session_entries() {
        let miss_turn = assistant(0, 0, 110_000, cost(0.0, 0.0, 0.4125), "test-model", 120_000);
        let misses = collect_cache_misses(
            &[entry(turn1()), entry(turn2()), entry(miss_turn)],
            &models(),
        );
        assert_eq!(misses.len(), 1);
        assert_eq!(misses.get("x").unwrap().missed_tokens, 105_000);
    }

    // Upstream "detects a miss on a just-completed message with idle time".
    #[test]
    fn detects_a_miss_on_a_just_completed_message_with_idle_time() {
        let miss_message = assistant(0, 0, 110_000, cost(0.0, 0.0, 0.4125), "test-model", 600_000);
        let miss =
            detect_cache_miss(&[entry(turn1()), entry(turn2())], &miss_message, &models()).unwrap();
        assert_eq!(miss.missed_tokens, 105_000);
        assert_close(miss.missed_cost, 0.36225);
        // 600s - 60s since the previous request.
        assert_eq!(miss.idle_ms, 540_000);
        assert!(!miss.model_changed);
    }

    // Upstream "flags model switches on detected misses".
    #[test]
    fn flags_model_switches_on_detected_misses() {
        let other_model = assistant(
            0,
            0,
            110_000,
            cost(0.0, 0.0, 0.4125),
            "other-model",
            120_000,
        );
        let miss =
            detect_cache_miss(&[entry(turn1()), entry(turn2())], &other_model, &models()).unwrap();
        assert_eq!(miss.missed_tokens, 105_000);
        assert!(miss.model_changed);
    }

    // Upstream "uses only cache-warm usage entries as cache refreshes".
    #[test]
    fn uses_only_cache_warm_usage_entries_as_cache_refreshes() {
        let miss_message = assistant(0, 0, 110_000, cost(0.0, 0.0, 0.4125), "test-model", 600_000);

        let after_cache_warm = detect_cache_miss(
            &[entry(turn1()), usage_entry("cache_warm", 500_000)],
            &miss_message,
            &models(),
        );
        let after_other_usage = detect_cache_miss(
            &[entry(turn1()), usage_entry("custom_operation", 500_000)],
            &miss_message,
            &models(),
        );

        assert_eq!(after_cache_warm.unwrap().idle_ms, 100_000);
        assert_eq!(after_other_usage.unwrap().idle_ms, 600_000);
    }

    // Upstream "returns undefined for healthy turns".
    #[test]
    fn returns_none_for_healthy_turns() {
        let healthy = assistant(
            0,
            105_000,
            2_000,
            cost(0.0, 0.0315, 0.0075),
            "test-model",
            120_000,
        );
        assert!(
            detect_cache_miss(&[entry(turn1()), entry(turn2())], &healthy, &models()).is_none()
        );
    }

    // Upstream "returns undefined for the first turn of a session".
    #[test]
    fn returns_none_for_the_first_turn_of_a_session() {
        assert!(detect_cache_miss(&[], &turn1(), &models()).is_none());
    }

    #[test]
    fn cache_ttl_constant_pins_the_anthropic_default() {
        assert_eq!(CACHE_TTL_MS, 300_000);
    }
}
