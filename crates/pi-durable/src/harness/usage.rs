//! Port of `packages/durable/src/harness/usage.ts`.
//!
//! Deviations: D1 — `Usage` values are stored as their serde JSON
//! representation in the erased document (`JsonRepresentation<Usage>` →
//! `serde_json::Value` with `Usage`'s serde shape). D2 — `Object.hasOwn`
//! checks become plain map lookups; `Object.defineProperty` (the
//! `__proto__` guard) has no Rust counterpart.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::documents::{DocDefinition, DocToken};
use crate::harness::types::ConversationId;
use crate::session::Tx;
use crate::types::{DocumentFork, DocumentHistory};
use pi_ai::Usage;

/// Ledger of one conversation's own spend: its entries, and compaction
/// summarization attempts, which have none.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageState {
    /// Assistant entries and summarization attempts, keyed `provider/modelId`.
    #[serde(default)]
    pub models: Map<String, Value>,
    /// Tool results, keyed by tool name; their usage has no model identity.
    #[serde(default)]
    pub tools: Map<String, Value>,
}

/// Bucket of [`UsageState`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageBucket {
    Models,
    Tools,
}

impl UsageBucket {
    fn key(self) -> &'static str {
        match self {
            UsageBucket::Models => "models",
            UsageBucket::Tools => "tools",
        }
    }
}

fn usage_definition() -> DocDefinition {
    DocDefinition::conversation(
        "pi.usage",
        1,
        DocumentHistory::Latest,
        DocumentFork::Initial,
        || {
            serde_json::json!({ "models": {}, "tools": {} })
                .as_object()
                .cloned()
                .unwrap_or_default()
        },
    )
    .with_checkpoint_when(|_, _, _| true)
}

static USAGE_DOC: std::sync::LazyLock<DocToken> =
    std::sync::LazyLock::new(|| DocToken::new(usage_definition()));

/// Built-in usage ledger document of a conversation.
pub fn usage_doc() -> DocToken {
    USAGE_DOC.clone()
}

/// Add `usage` to one bucket of the conversation's `pi.usage`, in the commit
/// that records the response.
pub fn record_usage(
    tx: &Tx<'_>,
    conversation_id: ConversationId,
    bucket: UsageBucket,
    key: &str,
    usage: &Usage,
) -> Result<(), String> {
    let draft = tx
        .doc(&usage_doc(), Some(conversation_id.0), None, None)
        .map_err(|e| e.to_string())?;
    let mut value = Value::Object(draft.value().unwrap_or_default());
    if !value.is_object() {
        value = Value::Object(Map::new());
    }
    {
        let map = value
            .as_object_mut()
            .expect("object")
            .entry(bucket.key().to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if !map.is_object() {
            *map = Value::Object(Map::new());
        }
        let bucket_map = map.as_object_mut().expect("object");
        // Own keys only: a tool may be called `toString`.
        match bucket_map.get_mut(key) {
            // Providers may leave optional counters `undefined`; drafts take
            // strict JSON.
            None => {
                bucket_map.insert(
                    key.to_string(),
                    serde_json::to_value(usage).expect("Usage serializes"),
                );
            }
            Some(total) => {
                let mut total_usage: Usage = serde_json::from_value(total.clone())
                    .map_err(|error| format!("Stored usage is not a Usage: {error}"))?;
                add_usage(&mut total_usage, usage);
                *total = serde_json::to_value(&total_usage).expect("Usage serializes");
            }
        }
    }
    use pi_chord::delta::Seg;
    let bucket_value = value
        .get(bucket.key())
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    draft
        .set(&[Seg::key(bucket.key())], bucket_value)
        .map_err(|error| error.to_string())?;
    Ok(())
}

/// Add every counter of `usage` to `total`; optional counters are added once
/// either side reports them.
pub fn add_usage(total: &mut Usage, usage: &Usage) {
    total.input += usage.input;
    total.output += usage.output;
    total.cache_read += usage.cache_read;
    total.cache_write += usage.cache_write;
    total.total_tokens += usage.total_tokens;
    if let Some(value) = usage.cache_write_1h {
        total.cache_write_1h = Some(total.cache_write_1h.unwrap_or(0) + value);
    }
    if let Some(value) = usage.reasoning {
        total.reasoning = Some(total.reasoning.unwrap_or(0) + value);
    }
    total.cost.input += usage.cost.input;
    total.cost.output += usage.cost.output;
    total.cost.cache_read += usage.cost.cache_read;
    total.cost.cache_write += usage.cost.cache_write;
    total.cost.total += usage.cost.total;
}

fn state_bucket_entries(state: &UsageState, bucket: UsageBucket) -> Vec<(String, Value)> {
    match bucket {
        UsageBucket::Models => state
            .models
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        UsageBucket::Tools => state
            .tools
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    }
}

/// Add every bucket of `state` into `sum`.
pub fn add_usage_state(sum: &mut UsageState, state: &UsageState) {
    for bucket in [UsageBucket::Models, UsageBucket::Tools] {
        for (key, usage) in state_bucket_entries(state, bucket) {
            let existing = match bucket {
                UsageBucket::Models => sum.models.get(&key).cloned(),
                UsageBucket::Tools => sum.tools.get(&key).cloned(),
            };
            match existing {
                Some(total) if serde_json::from_value::<Usage>(total.clone()).is_err() => {}
                Some(total) => {
                    if let (Ok(mut total_usage), Ok(usage)) = (
                        serde_json::from_value::<Usage>(total.clone()),
                        serde_json::from_value::<Usage>(usage.clone()),
                    ) {
                        add_usage(&mut total_usage, &usage);
                        let merged = serde_json::to_value(&total_usage).expect("Usage serializes");
                        match bucket {
                            UsageBucket::Models => {
                                sum.models.insert(key.clone(), merged);
                            }
                            UsageBucket::Tools => {
                                sum.tools.insert(key.clone(), merged);
                            }
                        }
                    }
                }
                None => match bucket {
                    UsageBucket::Models => {
                        sum.models.insert(key.clone(), usage.clone());
                    }
                    UsageBucket::Tools => {
                        sum.tools.insert(key.clone(), usage.clone());
                    }
                },
            }
        }
    }
}
