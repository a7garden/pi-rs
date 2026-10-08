//! Port of `packages/coding-agent/src/core/usage-totals.ts` — running usage
//! totals and per-model cost breakdowns over session entries. Ported alongside
//! [`super::compaction`], which consumes [`combine_usage`] for split-turn
//! summary accounting; `agent-session.ts` (deferred, crate docs) consumes the
//! rest.

use indexmap::IndexMap;
use pi_ai::types::Usage;

use super::session_manager::SessionEntry;

/// Accumulated usage and cost across a session (`UsageTotals`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UsageTotals {
	pub input: u64,
	pub output: u64,
	pub cache_read: u64,
	pub cache_write: u64,
	pub cost: f64,
}

pub fn create_usage_totals() -> UsageTotals {
	UsageTotals::default()
}

pub fn add_usage_to_totals(totals: &mut UsageTotals, usage: &Usage) {
	totals.input += usage.input;
	totals.output += usage.output;
	totals.cache_read += usage.cache_read;
	totals.cache_write += usage.cache_write;
	totals.cost += usage.cost.total;
}

/// Sum of two usages, keeping the optional token splits when either side
/// reports them (`combineUsage`).
pub fn combine_usage(first: &Usage, second: &Usage) -> Usage {
	Usage {
		input: first.input + second.input,
		output: first.output + second.output,
		cache_read: first.cache_read + second.cache_read,
		cache_write: first.cache_write + second.cache_write,
		cache_write_1h: if first.cache_write_1h.is_some() || second.cache_write_1h.is_some() {
			Some(first.cache_write_1h.unwrap_or(0) + second.cache_write_1h.unwrap_or(0))
		} else {
			None
		},
		reasoning: if first.reasoning.is_some() || second.reasoning.is_some() {
			Some(first.reasoning.unwrap_or(0) + second.reasoning.unwrap_or(0))
		} else {
			None
		},
		total_tokens: first.total_tokens + second.total_tokens,
		cost: pi_ai::types::UsageCost {
			input: first.cost.input + second.cost.input,
			output: first.cost.output + second.cost.output,
			cache_read: first.cost.cache_read + second.cost.cache_read,
			cache_write: first.cost.cache_write + second.cost.cache_write,
			total: first.cost.total + second.cost.total,
		},
	}
}

/// One row of [`get_usage_cost_breakdown`] (`UsageCostBreakdownEntry`).
#[derive(Debug, Clone, PartialEq)]
pub struct UsageCostBreakdownEntry {
	pub key: String,
	pub cost: f64,
	pub tokens: u64,
}

/// Group model-attributed usage by model and all other usage into a separate
/// bucket (`getUsageCostBreakdown`). Rows keep first-seen order for equal
/// costs (upstream `Array.from(map)` + stable sort).
pub fn get_usage_cost_breakdown(entries: &[SessionEntry]) -> Vec<UsageCostBreakdownEntry> {
	let mut totals_by_key: IndexMap<String, UsageTotals> = IndexMap::new();

	for entry in entries {
		let mut key: Option<String> = None;
		let mut usage: Option<&Usage> = None;
		match entry {
			SessionEntry::Message(message) => match message.message.as_llm() {
				Some(pi_ai::types::Message::Assistant(assistant)) => {
					key = Some(format!(
						"{}/{}",
						assistant.provider,
						assistant.response_model.as_deref().unwrap_or(&assistant.model)
					));
					usage = Some(&assistant.usage);
				}
				Some(pi_ai::types::Message::ToolResult(result)) => {
					if let Some(result_usage) = &result.usage {
						key = Some("Tools/summaries".to_string());
						usage = Some(result_usage);
					}
				}
				_ => {}
			},
			SessionEntry::Usage(usage_entry) => {
				key = Some(format!("{}/{}", usage_entry.provider, usage_entry.model));
				usage = Some(&usage_entry.usage);
			}
			SessionEntry::BranchSummary(summary) if summary.usage.is_some() => {
				key = Some("Tools/summaries".to_string());
				usage = summary.usage.as_ref();
			}
			SessionEntry::Compaction(compaction) if compaction.usage.is_some() => {
				key = Some("Tools/summaries".to_string());
				usage = compaction.usage.as_ref();
			}
			_ => {}
		}
		let (Some(key), Some(usage)) = (key, usage) else { continue };

		let totals = totals_by_key.entry(key).or_default();
		add_usage_to_totals(totals, usage);
	}

	let mut rows: Vec<UsageCostBreakdownEntry> = totals_by_key
		.into_iter()
		.map(|(key, totals)| UsageCostBreakdownEntry {
			tokens: totals.input + totals.output + totals.cache_read + totals.cache_write,
			key,
			cost: totals.cost,
		})
		.filter(|entry| entry.cost > 0.0 || entry.tokens > 0)
		.collect();
	rows.sort_by(|a, b| b.cost.partial_cmp(&a.cost).unwrap_or(std::cmp::Ordering::Equal));
	rows
}

#[cfg(test)]
mod usage_totals_tests {
	//! Port of the standalone `getUsageCostBreakdown` case from upstream
	//! `test/agent-session-stats.test.ts` ("groups tool and summary usage
	//! separately from model-attributed usage"); the remaining cases of that
	//! file exercise the deferred `AgentSession` class.

	use super::*;
	use crate::core::session_manager::SessionManager;
	use pi_agent::AgentMessage;
	use pi_ai::types::{AssistantMessage, StopReason, TextContent, UsageCost, UserContent, UserContentField};
	use pi_ai::types::AssistantContent;

	fn usage(total_tokens: u64, total_cost: f64) -> Usage {
		Usage {
			input: total_tokens,
			output: 0,
			cache_read: 0,
			cache_write: 0,
			cache_write_1h: None,
			reasoning: None,
			total_tokens,
			cost: UsageCost { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0, total: total_cost },
		}
	}

	fn assistant_message(text: &str, total_tokens: u64, timestamp: i64, total_cost: f64) -> AgentMessage {
		AgentMessage::assistant(AssistantMessage {
			content: vec![AssistantContent::Text(TextContent::new(text))],
			api: "anthropic-messages".to_string(),
			provider: "anthropic".to_string(),
			model: "claude-sonnet-4-5".to_string(),
			response_model: None,
			response_id: None,
			provider_thinking_level: None,
			thinking_level: None,
			diagnostics: None,
			usage: usage(total_tokens, total_cost),
			stop_reason: StopReason::Stop,
			deferred: None,
			error_message: None,
			raw_stop_reason: None,
			end_turn: None,
			timestamp,
		})
	}

	fn user_message(text: &str, timestamp: i64) -> AgentMessage {
		AgentMessage::user(pi_ai::types::UserMessage {
			content: UserContentField::Text(text.to_string()),
			timestamp,
		})
	}

	fn tool_result_message(total_tokens: u64, total_cost: f64) -> AgentMessage {
		AgentMessage::tool_result(pi_ai::types::ToolResultMessage {
			tool_call_id: "call-1".to_string(),
			tool_name: "read".to_string(),
			content: vec![UserContent::Text(TextContent::new("ok"))],
			details: None,
			usage: Some(usage(total_tokens, total_cost)),
			nested_calls: None,
			is_error: false,
			timestamp: 3,
		})
	}

	#[test]
	fn groups_tool_and_summary_usage_separately_from_model_attributed_usage() {
		let mut session = SessionManager::in_memory(None, None, None).expect("session");
		let root_id = session.append_message(user_message("hello", 1)).expect("append");
		session.append_message(assistant_message("response", 100, 2, 0.5)).expect("append");
		session.append_message(tool_result_message(100, 1.0)).expect("append");
		session
			.append_compaction("summary", Some(&root_id), 100, None, Some(false), Some(usage(100, 2.0)))
			.expect("append");
		session
			.branch_with_summary(None, "branch summary", None, Some(false), Some(usage(100, 3.0)))
			.expect("append");

		assert_eq!(
			get_usage_cost_breakdown(&session.get_entries()),
			vec![
				UsageCostBreakdownEntry { key: "Tools/summaries".to_string(), cost: 6.0, tokens: 300 },
				UsageCostBreakdownEntry {
					key: "anthropic/claude-sonnet-4-5".to_string(),
					cost: 0.5,
					tokens: 100,
				},
			]
		);
	}

	#[test]
	fn combine_usage_sums_optional_splits_when_either_side_reports_them() {
		let mut first = usage(10, 0.0);
		first.cache_write_1h = Some(4);
		let second = usage(5, 0.0);

		let combined = combine_usage(&first, &second);
		assert_eq!(combined.input, 15);
		assert_eq!(combined.cache_write_1h, Some(4));

		let plain = combine_usage(&second, &second);
		assert_eq!(plain.cache_write_1h, None);
		assert_eq!(plain.reasoning, None);
	}
}
