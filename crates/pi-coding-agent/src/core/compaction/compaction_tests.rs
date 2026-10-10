//! Port of upstream `test/compaction.test.ts` (26 offline cases; the two
//! `ANTHROPIC_OAUTH_TOKEN`-gated LLM cases stay unported — upstream skips
//! them without credentials too), `test/branch-summarization.test.ts` (4
//! cases), and the AgentSession-free cases of
//! `test/compaction-summary-reasoning.test.ts` (13 of 14 cases: the two
//! "Anthropic refusal fallback" cases collapse into one because the
//! `refusalFallbacks` option does not exist on the Rust
//! `SimpleStreamOptions`). Upstream mocks `completeSimple`; the port
//! observes the same calls through the [`pi_agent::StreamFn`] capture seam
//! (crate docs D23).

use std::sync::{Arc, Mutex};

use super::*;
use indexmap::IndexMap;
use pi_agent::AgentMessage;
use pi_ai::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Message, SimpleStreamOptions,
    StopReason, StreamDoneReason, SystemContent, SystemMessage, TextContent, ToolCall,
    TranscriptContext, Usage, UsageCost, UserContent, UserContentField,
};
use pi_ai::utils::abort::AbortSignal;
use pi_ai::utils::event_stream::create_assistant_message_event_stream;
use serde_json::{json, Value};

use crate::core::compaction::branch_summarization::{
    generate_branch_summary, GenerateBranchSummaryOptions,
};
use crate::core::compaction::utils::create_file_ops;
use crate::core::messages::{custom_message_from_agent, CompactionSummaryMessage};
use crate::core::session_manager::{
    build_session_context, migrate_session_entries, parse_session_entries, CompactionEntry,
    CustomMessageEntry, Leaf, ModelChangeEntry, SessionEntryBase, SessionMessageEntry,
    SessionModelRef, ThinkingLevelChangeEntry,
};
use crate::utils::time::{now_ms, to_iso_timestamp};

// ============================================================================
// Test fixtures (upstream compaction.test.ts helpers)
// ============================================================================

fn load_large_session_entries() -> Vec<SessionEntry> {
    const CONTENT: &str = include_str!("test_fixtures/large-session.jsonl");
    let mut entries = parse_session_entries(CONTENT);
    migrate_session_entries(&mut entries); // Add id/parentId for v1 fixtures
    entries
        .into_iter()
        .filter_map(|entry| entry.as_entry().cloned())
        .collect()
}

fn create_mock_usage(input: u64, output: u64, cache_read: u64, cache_write: u64) -> Usage {
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: input + output + cache_read + cache_write,
        cost: UsageCost::default(),
    }
}

fn create_user_message(text: &str) -> AgentMessage {
    AgentMessage::user(pi_ai::types::UserMessage {
        content: UserContentField::Text(text.to_string()),
        timestamp: now_ms(),
    })
}

fn create_assistant_message(text: &str, usage: Option<Usage>) -> AssistantMessage {
    pi_ai::types::AssistantMessage {
        content: vec![AssistantContent::Text(TextContent::new(text))],
        usage: usage.unwrap_or_else(|| create_mock_usage(100, 50, 0, 0)),
        stop_reason: StopReason::Stop,
        timestamp: now_ms(),
        api: "anthropic-messages".to_string(),
        provider: "anthropic".to_string(),
        model: "claude-sonnet-4-5".to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        duration_ms: None,
    }
}

/// Sequential entry builder with predictable ids (`test-id-N`); each test
/// constructs a fresh one, mirroring upstream's `beforeEach` counter reset.
struct EntryFactory {
    counter: usize,
    last_id: Option<String>,
}

impl EntryFactory {
    fn new() -> Self {
        EntryFactory {
            counter: 0,
            last_id: None,
        }
    }

    fn next_base(&mut self) -> SessionEntryBase {
        let id = format!("test-id-{}", self.counter);
        self.counter += 1;
        SessionEntryBase {
            id,
            parent_id: self.last_id.clone(),
            timestamp: to_iso_timestamp(now_ms()),
        }
    }

    fn finish(&mut self, base: &SessionEntryBase) {
        self.last_id = Some(base.id.clone());
    }

    fn create_message_entry(&mut self, message: AgentMessage) -> SessionMessageEntry {
        let base = self.next_base();
        self.finish(&base);
        SessionMessageEntry { base, message }
    }

    fn create_compaction_entry(
        &mut self,
        summary: &str,
        first_kept_entry_id: &str,
    ) -> CompactionEntry {
        let base = self.next_base();
        self.finish(&base);
        CompactionEntry {
            base,
            summary: summary.to_string(),
            first_kept_entry_id: first_kept_entry_id.to_string(),
            tokens_before: 10000,
            details: None,
            usage: None,
            from_hook: None,
            system_message: None,
            first_kept_entry_index: None,
        }
    }

    fn create_model_change_entry(&mut self, provider: &str, model_id: &str) -> ModelChangeEntry {
        let base = self.next_base();
        self.finish(&base);
        ModelChangeEntry {
            base,
            provider: provider.to_string(),
            model_id: model_id.to_string(),
        }
    }

    fn create_thinking_level_entry(&mut self, thinking_level: &str) -> ThinkingLevelChangeEntry {
        let base = self.next_base();
        self.finish(&base);
        ThinkingLevelChangeEntry {
            base,
            thinking_level: thinking_level.to_string(),
        }
    }

    fn create_custom_message_entry(&mut self, content: &str) -> CustomMessageEntry {
        let base = self.next_base();
        self.finish(&base);
        CustomMessageEntry {
            base,
            custom_type: "test".to_string(),
            content: UserContentField::Text(content.to_string()),
            details: None,
            display: true,
        }
    }
}

fn extract_text(messages: &[AgentMessage]) -> String {
    messages
        .iter()
        .map(|message| match message {
            AgentMessage::Llm(Message::User(user)) => match &user.content {
                UserContentField::Text(text) => text.clone(),
                UserContentField::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|block| match block {
                        UserContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            },
            AgentMessage::Llm(Message::Assistant(assistant)) => assistant
                .content
                .iter()
                .filter_map(|block| match block {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
            AgentMessage::Llm(Message::ToolResult(result)) => result
                .content
                .iter()
                .filter_map(|block| match block {
                    UserContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
            AgentMessage::Llm(Message::System(_)) => String::new(),
            AgentMessage::Custom(value) => {
                let role = value
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("custom");
                match role {
                    "branchSummary" | "compactionSummary" => value
                        .get("summary")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    "bashExecution" => {
                        format!(
                            "{}\n{}",
                            value.get("command").and_then(Value::as_str).unwrap_or(""),
                            value.get("output").and_then(Value::as_str).unwrap_or("")
                        )
                    }
                    "custom" | "toolResult" => match value.get("content") {
                        Some(Value::String(text)) => text.clone(),
                        Some(Value::Array(blocks)) => blocks
                            .iter()
                            .filter_map(|block| {
                                block.get("text").and_then(Value::as_str).filter(|_| {
                                    block.get("type").and_then(Value::as_str) == Some("text")
                                })
                            })
                            .collect::<Vec<_>>()
                            .join(" "),
                        _ => String::new(),
                    },
                    _ => String::new(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ============================================================================
// Unit tests — upstream "Token calculation"
// ============================================================================

#[test]
fn should_calculate_total_context_tokens_from_usage() {
    let usage = create_mock_usage(1000, 500, 200, 100);
    assert_eq!(calculate_context_tokens(&usage), 1800);
}

#[test]
fn should_handle_zero_values() {
    let usage = create_mock_usage(0, 0, 0, 0);
    assert_eq!(calculate_context_tokens(&usage), 0);
}

// ============================================================================
// Upstream "getLastAssistantUsage"
// ============================================================================

#[test]
fn get_last_assistant_usage_finds_the_last_non_aborted_assistant_message_usage() {
    let mut factory = EntryFactory::new();
    let entries = vec![
        SessionEntry::Message(factory.create_message_entry(create_user_message("Hello"))),
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("Hi", Some(create_mock_usage(100, 50, 0, 0))),
        ))),
        SessionEntry::Message(factory.create_message_entry(create_user_message("How are you?"))),
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("Good", Some(create_mock_usage(200, 100, 0, 0))),
        ))),
    ];

    let usage = get_last_assistant_usage(&entries);
    assert!(usage.is_some());
    assert_eq!(usage.unwrap().input, 200);
}

#[test]
fn get_last_assistant_usage_skips_aborted_messages() {
    let mut factory = EntryFactory::new();
    let mut aborted = create_assistant_message("Aborted", Some(create_mock_usage(300, 150, 0, 0)));
    aborted.stop_reason = StopReason::Aborted;

    let entries = vec![
        SessionEntry::Message(factory.create_message_entry(create_user_message("Hello"))),
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("Hi", Some(create_mock_usage(100, 50, 0, 0))),
        ))),
        SessionEntry::Message(factory.create_message_entry(create_user_message("How are you?"))),
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(aborted))),
    ];

    let usage = get_last_assistant_usage(&entries);
    assert!(usage.is_some());
    assert_eq!(usage.unwrap().input, 100);
}

#[test]
fn get_last_assistant_usage_skips_all_zero_assistant_usage() {
    let mut factory = EntryFactory::new();
    let entries = vec![
        SessionEntry::Message(factory.create_message_entry(create_user_message("Hello"))),
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("Hi", Some(create_mock_usage(100, 50, 0, 0))),
        ))),
        SessionEntry::Message(factory.create_message_entry(create_user_message("continue"))),
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("Partial", Some(create_mock_usage(0, 0, 0, 0))),
        ))),
    ];

    let usage = get_last_assistant_usage(&entries);
    assert!(usage.is_some());
    assert_eq!(usage.unwrap().input, 100);
}

#[test]
fn get_last_assistant_usage_returns_none_if_no_assistant_messages() {
    let mut factory = EntryFactory::new();
    let entries = vec![SessionEntry::Message(
        factory.create_message_entry(create_user_message("Hello")),
    )];
    assert!(get_last_assistant_usage(&entries).is_none());
}

// ============================================================================
// Upstream "estimateContextTokens"
// ============================================================================

#[test]
fn estimate_context_tokens_uses_the_last_non_zero_assistant_usage_as_the_context_anchor() {
    let messages = vec![
        create_user_message("Hello"),
        AgentMessage::assistant(create_assistant_message(
            "Hi",
            Some(create_mock_usage(100, 50, 0, 0)),
        )),
        create_user_message("continue"),
        AgentMessage::assistant(create_assistant_message(
            "Partial thinking",
            Some(create_mock_usage(0, 0, 0, 0)),
        )),
    ];

    let estimate = estimate_context_tokens(&messages);

    assert_eq!(estimate.usage_tokens, 150);
    assert_eq!(estimate.last_usage_index, Some(1));
    assert!(estimate.trailing_tokens > 0);
    assert_eq!(estimate.tokens, 150 + estimate.trailing_tokens);
}

// ============================================================================
// Upstream "shouldCompact"
// ============================================================================

#[test]
fn should_compact_returns_true_when_context_exceeds_threshold() {
    let settings = CompactionSettings {
        enabled: true,
        reserve_tokens: 10000,
        keep_recent_tokens: 20000,
    };

    assert!(should_compact(95000, 100000, &settings));
    assert!(!should_compact(89000, 100000, &settings));
}

#[test]
fn should_compact_returns_false_when_disabled() {
    let settings = CompactionSettings {
        enabled: false,
        reserve_tokens: 10000,
        keep_recent_tokens: 20000,
    };

    assert!(!should_compact(95000, 100000, &settings));
}

// ============================================================================
// Upstream "findCutPoint"
// ============================================================================

#[test]
fn find_cut_point_should_find_cut_point_based_on_actual_token_differences() {
    let mut factory = EntryFactory::new();
    let mut entries: Vec<SessionEntry> = Vec::new();
    for i in 0..10u64 {
        entries.push(SessionEntry::Message(
            factory.create_message_entry(create_user_message(&format!("User {i}"))),
        ));
        entries.push(SessionEntry::Message(factory.create_message_entry(
            AgentMessage::assistant(create_assistant_message(
                &format!("Assistant {i}"),
                Some(create_mock_usage(0, 100, (i + 1) * 1000, 0)),
            )),
        )));
    }

    // 20 entries, last assistant has 10000 tokens
    // keepRecentTokens = 2500: keep entries where diff < 2500
    let result = find_cut_point(&entries, 0, entries.len(), 2500);

    // Should cut at a valid cut point (user or assistant message)
    let SessionEntry::Message(cut_entry) = &entries[result.first_kept_entry_index] else {
        panic!("expected message entry");
    };
    let role = cut_entry.message.role();
    assert!(role == "user" || role == "assistant");
}

#[test]
fn find_cut_point_should_return_start_index_if_no_valid_cut_points_in_range() {
    let mut factory = EntryFactory::new();
    let entries = vec![SessionEntry::Message(factory.create_message_entry(
        AgentMessage::assistant(create_assistant_message("a", None)),
    ))];
    let result = find_cut_point(&entries, 0, entries.len(), 1000);
    assert_eq!(result.first_kept_entry_index, 0);
}

#[test]
fn find_cut_point_should_keep_everything_if_all_messages_fit_within_budget() {
    let mut factory = EntryFactory::new();
    let entries = vec![
        SessionEntry::Message(factory.create_message_entry(create_user_message("1"))),
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("a", Some(create_mock_usage(0, 50, 500, 0))),
        ))),
        SessionEntry::Message(factory.create_message_entry(create_user_message("2"))),
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("b", Some(create_mock_usage(0, 50, 1000, 0))),
        ))),
    ];

    let result = find_cut_point(&entries, 0, entries.len(), 50000);
    assert_eq!(result.first_kept_entry_index, 0);
}

#[test]
fn find_cut_point_should_indicate_split_turn_when_cutting_at_assistant_message() {
    let mut factory = EntryFactory::new();
    // Create a scenario where we cut at an assistant message mid-turn
    let entries = vec![
        SessionEntry::Message(factory.create_message_entry(create_user_message("Turn 1"))),
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("A1", Some(create_mock_usage(0, 100, 1000, 0))),
        ))),
        SessionEntry::Message(factory.create_message_entry(create_user_message("Turn 2"))), // index 2
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("A2-1", Some(create_mock_usage(0, 100, 5000, 0))),
        ))), // index 3
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("A2-2", Some(create_mock_usage(0, 100, 8000, 0))),
        ))), // index 4
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("A2-3", Some(create_mock_usage(0, 100, 10000, 0))),
        ))), // index 5
    ];

    // With keepRecentTokens = 3000, should cut somewhere in Turn 2
    let result = find_cut_point(&entries, 0, entries.len(), 3000);

    // If cut at assistant message (not user), should indicate split turn
    let SessionEntry::Message(cut_entry) = &entries[result.first_kept_entry_index] else {
        panic!("expected message entry");
    };
    if cut_entry.message.role() == "assistant" {
        assert!(result.is_split_turn);
        assert_eq!(result.turn_start_index, Some(2)); // Turn 2 starts at index 2
    }
}

#[test]
fn find_cut_point_should_budget_context_visible_custom_message_entries() {
    let mut factory = EntryFactory::new();
    let entries = vec![
        SessionEntry::Message(factory.create_message_entry(create_user_message("hi"))),
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("hello", None),
        ))),
        SessionEntry::CustomMessage(factory.create_custom_message_entry("x".repeat(4000).as_str())),
        SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
            create_assistant_message("ok", None),
        ))),
    ];

    let tiny_budget = find_cut_point(&entries, 0, entries.len(), 1);
    assert_eq!(tiny_budget.first_kept_entry_index, 3);
    assert!(tiny_budget.is_split_turn);
    assert_eq!(tiny_budget.turn_start_index, Some(2));

    let custom_fits_budget = find_cut_point(&entries, 0, entries.len(), 2);
    assert_eq!(custom_fits_budget.first_kept_entry_index, 2);
    assert!(!custom_fits_budget.is_split_turn);
    assert_eq!(custom_fits_budget.turn_start_index, None);
}

// Regression test for #9740.
#[test]
fn find_cut_point_should_fall_back_to_the_latest_valid_cut_point_before_oversized_trailing_tool_results(
) {
    let mut factory = EntryFactory::new();
    let old_user = factory.create_message_entry(create_user_message("old history"));
    let old_assistant = factory.create_message_entry(AgentMessage::assistant(
        create_assistant_message("old answer", None),
    ));
    let current_user = factory.create_message_entry(create_user_message("read the large file"));
    let mut tool_call_assistant = create_assistant_message("", None);
    tool_call_assistant.content = vec![AssistantContent::ToolCall(ToolCall {
        id: "call-1".to_string(),
        name: "read".to_string(),
        arguments: serde_json::from_str(r#"{ "path": "big.txt" }"#).unwrap(),
        thought_signature: None,
        namespace: None,
    })];
    tool_call_assistant.stop_reason = StopReason::ToolUse;
    let tool_call = factory.create_message_entry(AgentMessage::assistant(tool_call_assistant));
    let tool_result =
        factory.create_message_entry(AgentMessage::tool_result(pi_ai::types::ToolResultMessage {
            tool_call_id: "call-1".to_string(),
            tool_name: "read".to_string(),
            content: vec![UserContent::Text(TextContent::new("x".repeat(8000)))],
            details: None,
            usage: None,
            nested_calls: None,
            is_error: false,
            timestamp: now_ms(),
            duration_ms: None,
        }));
    let old_user_message = old_user.message.clone();
    let old_assistant_message = old_assistant.message.clone();
    let current_user_message = current_user.message.clone();
    let tool_call_id = tool_call.base.id.clone();
    let entries = vec![
        SessionEntry::Message(old_user),
        SessionEntry::Message(old_assistant),
        SessionEntry::Message(current_user),
        SessionEntry::Message(tool_call),
        SessionEntry::Message(tool_result),
    ];

    let result = find_cut_point(&entries, 0, entries.len(), 1000);
    assert_eq!(
        result,
        CutPointResult {
            first_kept_entry_index: 3,
            turn_start_index: Some(2),
            is_split_turn: true
        }
    );

    let preparation = prepare_compaction(
        &entries,
        &CompactionSettings {
            keep_recent_tokens: 1000,
            ..DEFAULT_COMPACTION_SETTINGS
        },
    );
    let preparation = preparation.expect("preparation");
    assert_eq!(preparation.first_kept_entry_id, tool_call_id);
    assert_eq!(
        preparation.messages_to_summarize,
        vec![old_user_message, old_assistant_message]
    );
    assert_eq!(preparation.turn_prefix_messages, vec![current_user_message]);
}

// ============================================================================
// Upstream "buildSessionContext"
// ============================================================================

#[test]
fn build_context_should_load_all_messages_when_no_compaction() {
    let mut factory = EntryFactory::new();
    let entries =
        vec![
            SessionEntry::Message(factory.create_message_entry(create_user_message("1"))),
            SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
                create_assistant_message("a", None),
            ))),
            SessionEntry::Message(factory.create_message_entry(create_user_message("2"))),
            SessionEntry::Message(factory.create_message_entry(AgentMessage::assistant(
                create_assistant_message("b", None),
            ))),
        ];

    let loaded = build_session_context(&entries, Leaf::Undefined, None);
    assert_eq!(loaded.messages.len(), 4);
    assert_eq!(loaded.thinking_level, "off");
    assert_eq!(
        loaded.model,
        Some(SessionModelRef {
            provider: "anthropic".to_string(),
            model_id: "claude-sonnet-4-5".to_string()
        })
    );
}

#[test]
fn build_context_should_handle_single_compaction() {
    let mut factory = EntryFactory::new();
    let u1 = factory.create_message_entry(create_user_message("1"));
    let a1 =
        factory.create_message_entry(AgentMessage::assistant(create_assistant_message("a", None)));
    let u2 = factory.create_message_entry(create_user_message("2"));
    let a2 =
        factory.create_message_entry(AgentMessage::assistant(create_assistant_message("b", None)));
    let u2_id = u2.base.id.clone();
    let compaction = factory.create_compaction_entry("Summary of 1,a,2,b", &u2_id); // keep from u2 onwards
    let u3 = factory.create_message_entry(create_user_message("3"));
    let a3 =
        factory.create_message_entry(AgentMessage::assistant(create_assistant_message("c", None)));

    let entries = vec![
        SessionEntry::Message(u1),
        SessionEntry::Message(a1),
        SessionEntry::Message(u2),
        SessionEntry::Message(a2),
        SessionEntry::Compaction(compaction),
        SessionEntry::Message(u3),
        SessionEntry::Message(a3),
    ];

    let loaded = build_session_context(&entries, Leaf::Undefined, None);
    // summary + kept (u2, a2) + after (u3, a3) = 5
    assert_eq!(loaded.messages.len(), 5);
    assert_eq!(loaded.messages[0].role(), "compactionSummary");
    let summary: CompactionSummaryMessage =
        custom_message_from_agent(&loaded.messages[0]).expect("compaction summary message");
    assert!(summary.summary.contains("Summary of 1,a,2,b"));
}

#[test]
fn build_context_should_handle_multiple_compactions_only_latest_matters() {
    let mut factory = EntryFactory::new();
    // First batch
    let u1 = factory.create_message_entry(create_user_message("1"));
    let a1 =
        factory.create_message_entry(AgentMessage::assistant(create_assistant_message("a", None)));
    let u1_id = u1.base.id.clone();
    let compact1 = factory.create_compaction_entry("First summary", &u1_id);
    // Second batch
    let u2 = factory.create_message_entry(create_user_message("2"));
    let b =
        factory.create_message_entry(AgentMessage::assistant(create_assistant_message("b", None)));
    let u3 = factory.create_message_entry(create_user_message("3"));
    let c =
        factory.create_message_entry(AgentMessage::assistant(create_assistant_message("c", None)));
    let u3_id = u3.base.id.clone();
    let compact2 = factory.create_compaction_entry("Second summary", &u3_id); // keep from u3 onwards
                                                                              // After second compaction
    let u4 = factory.create_message_entry(create_user_message("4"));
    let d =
        factory.create_message_entry(AgentMessage::assistant(create_assistant_message("d", None)));

    let entries = vec![
        SessionEntry::Message(u1),
        SessionEntry::Message(a1),
        SessionEntry::Compaction(compact1),
        SessionEntry::Message(u2),
        SessionEntry::Message(b),
        SessionEntry::Message(u3),
        SessionEntry::Message(c),
        SessionEntry::Compaction(compact2),
        SessionEntry::Message(u4),
        SessionEntry::Message(d),
    ];

    let loaded = build_session_context(&entries, Leaf::Undefined, None);
    // summary + kept from u3 (u3, c) + after (u4, d) = 5
    assert_eq!(loaded.messages.len(), 5);
    let summary: CompactionSummaryMessage =
        custom_message_from_agent(&loaded.messages[0]).expect("compaction summary message");
    assert!(summary.summary.contains("Second summary"));
}

#[test]
fn build_context_should_keep_all_messages_when_first_kept_entry_id_is_first_entry() {
    let mut factory = EntryFactory::new();
    let u1 = factory.create_message_entry(create_user_message("1"));
    let a1 =
        factory.create_message_entry(AgentMessage::assistant(create_assistant_message("a", None)));
    let u1_id = u1.base.id.clone();
    let compact1 = factory.create_compaction_entry("First summary", &u1_id); // keep from first entry
    let u2 = factory.create_message_entry(create_user_message("2"));
    let b =
        factory.create_message_entry(AgentMessage::assistant(create_assistant_message("b", None)));

    let entries = vec![
        SessionEntry::Message(u1),
        SessionEntry::Message(a1),
        SessionEntry::Compaction(compact1),
        SessionEntry::Message(u2),
        SessionEntry::Message(b),
    ];

    let loaded = build_session_context(&entries, Leaf::Undefined, None);
    // summary + all messages (u1, a1, u2, b) = 5
    assert_eq!(loaded.messages.len(), 5);
}

#[test]
fn build_context_should_track_model_and_thinking_level_changes() {
    let mut factory = EntryFactory::new();
    let entries = vec![
        SessionEntry::Message(factory.create_message_entry(create_user_message("1"))),
        SessionEntry::ModelChange(factory.create_model_change_entry("openai", "gpt-4")),
        SessionEntry::Message(
            factory
                .create_message_entry(AgentMessage::assistant(create_assistant_message("a", None))),
        ),
        SessionEntry::ThinkingLevelChange(factory.create_thinking_level_entry("high")),
    ];

    let loaded = build_session_context(&entries, Leaf::Undefined, None);
    // model_change is later overwritten by assistant message's model info
    assert_eq!(
        loaded.model,
        Some(SessionModelRef {
            provider: "anthropic".to_string(),
            model_id: "claude-sonnet-4-5".to_string()
        })
    );
    assert_eq!(loaded.thinking_level, "high");
}

// ============================================================================
// Upstream "prepareCompaction"
// ============================================================================

#[test]
fn prepare_compaction_does_not_treat_system_messages_as_conversation_history() {
    let mut factory = EntryFactory::new();
    let mut sections: IndexMap<String, Option<String>> = IndexMap::new();
    sections.insert("preamble".to_string(), Some("current prompt".to_string()));
    let system = factory.create_message_entry(AgentMessage::system(SystemMessage {
        content: SystemContent::Text(String::new()),
        sections: Some(sections),
        tools_added: None,
        tools_removed: None,
        timestamp: now_ms(),
    }));
    let user = factory.create_message_entry(create_user_message("one long turn"));
    let assistant = factory.create_message_entry(AgentMessage::assistant(
        create_assistant_message("assistant suffix", None),
    ));
    let user_message = user.message.clone();
    let assistant_id = assistant.base.id.clone();
    let entries = vec![
        SessionEntry::Message(system),
        SessionEntry::Message(user),
        SessionEntry::Message(assistant),
    ];
    let preparation = prepare_compaction(
        &entries,
        &CompactionSettings {
            keep_recent_tokens: 1,
            ..DEFAULT_COMPACTION_SETTINGS
        },
    );

    let preparation = preparation.expect("preparation");
    assert_eq!(preparation.first_kept_entry_id, assistant_id);
    assert!(preparation.is_split_turn);
    assert!(preparation.messages_to_summarize.is_empty());
    assert_eq!(preparation.turn_prefix_messages, vec![user_message]);
}

// ============================================================================
// Upstream "prepareCompaction with previous compaction"
// ============================================================================

#[test]
fn prepare_compaction_should_skip_repeated_compactions_when_kept_messages_still_fit() {
    let mut factory = EntryFactory::new();
    let u1 = factory.create_message_entry(create_user_message(
        "user msg 1 (summarized by compaction1)",
    ));
    let a1 = factory.create_message_entry(AgentMessage::assistant(create_assistant_message(
        "assistant msg 1",
        None,
    )));
    let u2 = factory.create_message_entry(create_user_message("user msg 2 - kept by compaction1"));
    let a2 = factory.create_message_entry(AgentMessage::assistant(create_assistant_message(
        "assistant msg 2",
        None,
    )));
    let u3 = factory.create_message_entry(create_user_message("user msg 3 - kept by compaction1"));
    let a3 = factory.create_message_entry(AgentMessage::assistant(create_assistant_message(
        "assistant msg 3",
        Some(create_mock_usage(5000, 1000, 0, 0)),
    )));
    let u2_id = u2.base.id.clone();
    let compaction1 = factory.create_compaction_entry("First summary", &u2_id);
    let u4 =
        factory.create_message_entry(create_user_message("user msg 4 (new after compaction1)"));
    let a4 = factory.create_message_entry(AgentMessage::assistant(create_assistant_message(
        "assistant msg 4",
        Some(create_mock_usage(8000, 2000, 0, 0)),
    )));

    let path_entries = vec![
        SessionEntry::Message(u1),
        SessionEntry::Message(a1),
        SessionEntry::Message(u2),
        SessionEntry::Message(a2),
        SessionEntry::Message(u3),
        SessionEntry::Message(a3),
        SessionEntry::Compaction(compaction1),
        SessionEntry::Message(u4),
        SessionEntry::Message(a4),
    ];
    let preparation = prepare_compaction(&path_entries, &DEFAULT_COMPACTION_SETTINGS);

    assert!(preparation.is_none());
}

#[test]
fn prepare_compaction_should_re_summarize_previously_kept_messages_when_the_recent_window_moves_past_them(
) {
    let mut factory = EntryFactory::new();
    let u1 = factory.create_message_entry(create_user_message(
        "user msg 1 (summarized by compaction1)".repeat(4).as_str(),
    ));
    let a1 = factory.create_message_entry(AgentMessage::assistant(create_assistant_message(
        "assistant msg 1".repeat(4).as_str(),
        None,
    )));
    let u2 = factory.create_message_entry(create_user_message(
        "user msg 2 - kept by compaction1 ".repeat(12).as_str(),
    ));
    let a2 = factory.create_message_entry(AgentMessage::assistant(create_assistant_message(
        "assistant msg 2 ".repeat(12).as_str(),
        None,
    )));
    let u3 = factory.create_message_entry(create_user_message(
        "user msg 3 - kept by compaction1 ".repeat(12).as_str(),
    ));
    let a3 = factory.create_message_entry(AgentMessage::assistant(create_assistant_message(
        "assistant msg 3 ".repeat(12).as_str(),
        Some(create_mock_usage(5000, 1000, 0, 0)),
    )));
    let u2_id = u2.base.id.clone();
    let compaction1 = factory.create_compaction_entry("First summary", &u2_id);
    let u4 = factory.create_message_entry(create_user_message(
        "user msg 4 (new after compaction1) ".repeat(12).as_str(),
    ));
    let a4 = factory.create_message_entry(AgentMessage::assistant(create_assistant_message(
        "assistant msg 4 ".repeat(12).as_str(),
        Some(create_mock_usage(8000, 2000, 0, 0)),
    )));

    let settings = CompactionSettings {
        keep_recent_tokens: 100,
        ..DEFAULT_COMPACTION_SETTINGS
    };
    let preparation = prepare_compaction(
        &[
            SessionEntry::Message(u1),
            SessionEntry::Message(a1),
            SessionEntry::Message(u2),
            SessionEntry::Message(a2),
            SessionEntry::Message(u3),
            SessionEntry::Message(a3),
            SessionEntry::Compaction(compaction1),
            SessionEntry::Message(u4),
            SessionEntry::Message(a4),
        ],
        &settings,
    );

    let preparation = preparation.expect("preparation");
    let summarized_text = extract_text(&preparation.messages_to_summarize);
    assert!(summarized_text.contains("user msg 2 - kept by compaction1"));
    assert!(summarized_text.contains("user msg 3 - kept by compaction1"));
    assert!(!summarized_text.contains("First summary"));
    assert_eq!(
        preparation.previous_summary.as_deref(),
        Some("First summary")
    );
}

// ============================================================================
// Integration tests with real session data — upstream "Large session fixture"
// ============================================================================

#[test]
fn large_session_should_parse_the_large_session() {
    let entries = load_large_session_entries();
    assert!(entries.len() > 100);

    let message_count = entries
        .iter()
        .filter(|entry| matches!(entry, SessionEntry::Message(_)))
        .count();
    assert!(message_count > 100);
}

#[test]
fn large_session_should_find_cut_point_in_large_session() {
    let entries = load_large_session_entries();
    let result = find_cut_point(
        &entries,
        0,
        entries.len(),
        DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens,
    );

    // Cut point should be at a message entry (user or assistant)
    let SessionEntry::Message(cut_entry) = &entries[result.first_kept_entry_index] else {
        panic!("expected message entry");
    };
    let role = cut_entry.message.role();
    assert!(role == "user" || role == "assistant");
}

#[test]
fn large_session_should_load_session_correctly() {
    let entries = load_large_session_entries();
    let loaded = build_session_context(&entries, Leaf::Undefined, None);

    assert!(loaded.messages.len() > 100);
    assert!(loaded.model.is_some());
}

// ============================================================================
// Upstream test/branch-summarization.test.ts
// ============================================================================

fn branch_test_model(max_tokens: u64) -> pi_ai::types::Model {
    serde_json::from_value(json!({
        "id": "test-model",
        "name": "Test Model",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "baseUrl": "https://api.anthropic.com",
        "reasoning": false,
        "input": ["text"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": 200000,
        "maxTokens": max_tokens,
    }))
    .expect("model")
}

fn branch_entries() -> Vec<SessionEntry> {
    vec![SessionEntry::Message(SessionMessageEntry {
        base: SessionEntryBase {
            id: "branch-user".to_string(),
            parent_id: None,
            timestamp: to_iso_timestamp(1),
        },
        message: AgentMessage::user(pi_ai::types::UserMessage {
            content: UserContentField::Text("Abandoned request".to_string()),
            timestamp: 1,
        }),
    })]
}

fn branch_response(
    model: &pi_ai::types::Model,
    content: Vec<AssistantContent>,
) -> AssistantMessage {
    let mut message = pi_ai::faux_assistant_message(
        pi_ai::FauxContent::Blocks(
            content
                .into_iter()
                .map(|block| match block {
                    AssistantContent::Text(text) => pi_ai::FauxContentBlock::Text(text),
                    AssistantContent::Thinking(thinking) => {
                        pi_ai::FauxContentBlock::Thinking(thinking)
                    }
                    AssistantContent::ToolCall(call) => pi_ai::FauxContentBlock::ToolCall(
                        pi_ai::faux_tool_call(call.name, call.arguments, Some(call.id.as_str())),
                    ),
                })
                .collect(),
        ),
        pi_ai::FauxMessageOptions::default(),
    );
    message.api = model.api().to_string();
    message.provider = model.provider().to_string();
    message.model = model.id().to_string();
    message
}

/// StreamFn mock capturing the options of every call and resolving with a
/// single `done` event (upstream `queueMicrotask` pushes).
fn capturing_stream_fn(
    response: AssistantMessage,
) -> (pi_agent::StreamFn, Arc<Mutex<Vec<SimpleStreamOptions>>>) {
    let captured: Arc<Mutex<Vec<SimpleStreamOptions>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    let stream_fn: pi_agent::StreamFn = Arc::new(move |_model, _context, options| {
        sink.lock()
            .expect("captured lock")
            .push(options.clone().unwrap_or_default());
        let stream = create_assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: StreamDoneReason::Stop,
            message: response.clone(),
        });
        Box::pin(async move { Ok(stream) })
    });
    (stream_fn, captured)
}

fn branch_options<'a>(
    model: pi_ai::types::Model,
    stream_fn: pi_agent::StreamFn,
) -> GenerateBranchSummaryOptions<'a> {
    GenerateBranchSummaryOptions {
        model,
        api_key: None,
        headers: None,
        env: None,
        signal: AbortSignal::new(),
        custom_instructions: None,
        replace_instructions: false,
        reserve_tokens: None,
        stream_fn: Some(stream_fn),
        retry: None,
        callbacks: None,
    }
}

#[tokio::test]
async fn branch_summary_does_not_override_tool_choice() {
    let model = branch_test_model(8192);
    let response = branch_response(
        &model,
        vec![AssistantContent::Text(TextContent::new("summary"))],
    );
    let (stream_fn, captured) = capturing_stream_fn(response);

    let result = generate_branch_summary(&branch_entries(), branch_options(model, stream_fn)).await;

    assert!(result.error.is_none());
    let options = captured.lock().expect("captured lock");
    assert_eq!(options.len(), 1);
    assert_eq!(options[0].base.max_tokens, Some(4096));
    assert!(options[0].tool_choice.is_none());
}

#[tokio::test]
async fn branch_summary_clamps_the_output_cap_to_the_model_limit() {
    let model = branch_test_model(1024);
    let response = branch_response(
        &model,
        vec![AssistantContent::Text(TextContent::new("summary"))],
    );
    let (stream_fn, captured) = capturing_stream_fn(response);

    let result = generate_branch_summary(&branch_entries(), branch_options(model, stream_fn)).await;

    assert!(result.error.is_none());
    let options = captured.lock().expect("captured lock");
    assert_eq!(options.len(), 1);
    assert_eq!(options[0].base.max_tokens, Some(1024));
}

#[tokio::test]
async fn branch_summary_rejects_tool_calls() {
    let model = branch_test_model(8192);
    let response = branch_response(
        &model,
        vec![AssistantContent::ToolCall(ToolCall {
            id: "tool-call-1".to_string(),
            name: "read".to_string(),
            arguments: serde_json::from_str(r#"{ "path": "README.md" }"#).unwrap(),
            thought_signature: None,
            namespace: None,
        })],
    );
    let (stream_fn, _captured) = capturing_stream_fn(response);

    let result = generate_branch_summary(&branch_entries(), branch_options(model, stream_fn)).await;

    assert_eq!(
        result.error.as_deref(),
        Some("Branch summarization attempted to call a tool")
    );
}

#[tokio::test]
async fn branch_summary_rejects_length_limited_summaries() {
    let model = branch_test_model(8192);
    let mut response = branch_response(
        &model,
        vec![AssistantContent::Text(TextContent::new("partial"))],
    );
    response.stop_reason = StopReason::Length;
    let stream_fn: pi_agent::StreamFn = Arc::new(move |_model, _context, _options| {
        let stream = create_assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: StreamDoneReason::Length,
            message: response.clone(),
        });
        Box::pin(async move { Ok(stream) })
    });

    let result = generate_branch_summary(&branch_entries(), branch_options(model, stream_fn)).await;

    assert_eq!(
		result.error.as_deref(),
		Some("Branch summarization failed: generation hit the token cap and the summary is incomplete")
	);
}

// ============================================================================
// Upstream test/compaction-summary-reasoning.test.ts
// ============================================================================

fn mock_summary_response() -> AssistantMessage {
    pi_ai::types::AssistantMessage {
        content: vec![AssistantContent::Text(TextContent::new(
            "## Goal\nTest summary",
        ))],
        api: "anthropic-messages".to_string(),
        provider: "anthropic".to_string(),
        model: "claude-sonnet-4-5".to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: Usage {
            input: 10,
            output: 10,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 20,
            cost: UsageCost::default(),
        },
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
        duration_ms: None,
    }
}

fn summary_model(reasoning: bool, max_tokens: u64, compat: Option<Value>) -> pi_ai::types::Model {
    let mut value = json!({
        "id": if reasoning { "reasoning-model" } else { "non-reasoning-model" },
        "name": if reasoning { "Reasoning Model" } else { "Non-reasoning Model" },
        "api": "anthropic-messages",
        "provider": "anthropic",
        "baseUrl": "https://api.anthropic.com",
        "reasoning": reasoning,
        "input": ["text"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": 200000,
        "maxTokens": max_tokens,
    });
    if let Some(compat) = compat {
        value["compat"] = compat;
    }
    serde_json::from_value(value).expect("model")
}

fn summarize_messages() -> Vec<AgentMessage> {
    vec![create_user_message("Summarize this.")]
}

fn summary_response_stream_fn(
    response: AssistantMessage,
) -> (
    pi_agent::StreamFn,
    Arc<Mutex<Vec<SimpleStreamOptions>>>,
    Arc<Mutex<Vec<TranscriptContext>>>,
) {
    let captured: Arc<Mutex<Vec<SimpleStreamOptions>>> = Arc::new(Mutex::new(Vec::new()));
    let contexts: Arc<Mutex<Vec<TranscriptContext>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    let context_sink = contexts.clone();
    let stream_fn: pi_agent::StreamFn = Arc::new(move |_model, context, options| {
        sink.lock()
            .expect("captured lock")
            .push(options.clone().unwrap_or_default());
        context_sink
            .lock()
            .expect("context lock")
            .push(context.clone());
        let stream = create_assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: StreamDoneReason::Stop,
            message: response.clone(),
        });
        Box::pin(async move { Ok(stream) })
    });
    (stream_fn, captured, contexts)
}

#[tokio::test]
async fn generate_summary_uses_the_provided_thinking_level_for_reasoning_capable_models() {
    let (stream_fn, captured, _contexts) = summary_response_stream_fn(mock_summary_response());

    let result = generate_summary_with_usage(
        &summarize_messages(),
        &summary_model(true, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(AgentThinkingLevel::Medium),
        Some(stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    assert_eq!(result.text, "## Goal\nTest summary");
    assert_eq!(result.usage, mock_summary_response().usage);

    let options = captured.lock().expect("captured lock");
    assert_eq!(options.len(), 1);
    assert_eq!(
        options[0].reasoning,
        Some(pi_ai::types::ThinkingLevel::Medium)
    );
    assert_eq!(options[0].base.base.api_key.as_deref(), Some("test-key"));
}

#[tokio::test]
async fn generate_summary_preserves_the_string_result() {
    let (stream_fn, _captured, _contexts) = summary_response_stream_fn(mock_summary_response());

    let result = generate_summary(
        &summarize_messages(),
        &summary_model(false, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        None,
        Some(stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    assert_eq!(result, "## Goal\nTest summary");
}

#[tokio::test]
async fn generate_summary_uses_fresh_routing_sessions_without_prompt_caching() {
    let (stream_fn_a, captured_a, _c) = summary_response_stream_fn(mock_summary_response());
    let (stream_fn_b, captured_b, _c2) = summary_response_stream_fn(mock_summary_response());

    let _ = generate_summary(
        &summarize_messages(),
        &summary_model(false, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        None,
        Some(stream_fn_a),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");
    let _ = generate_summary(
        &summarize_messages(),
        &summary_model(false, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        None,
        Some(stream_fn_b),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    let options_a = captured_a.lock().expect("captured lock");
    let options_b = captured_b.lock().expect("captured lock");
    assert_eq!(options_a.len(), 1);
    assert_eq!(options_b.len(), 1);
    assert_eq!(
        options_a[0].base.cache_retention,
        Some(pi_ai::types::CacheRetention::None)
    );
    assert_eq!(
        options_b[0].base.cache_retention,
        Some(pi_ai::types::CacheRetention::None)
    );
    let session_a = options_a[0].base.session_id.clone().expect("session id");
    let session_b = options_b[0].base.session_id.clone().expect("session id");
    assert_ne!(session_a, session_b);
}

#[tokio::test]
async fn complete_summarization_honors_caller_supplied_routing_session_and_tool_choice_without_prompt_caching(
) {
    let (stream_fn, captured, _contexts) = summary_response_stream_fn(mock_summary_response());

    let mut options = SimpleStreamOptions::default();
    options.base.session_id = Some("current-routing-session".to_string());
    options.base.cache_retention = Some(pi_ai::types::CacheRetention::Long);
    options.tool_choice = Some(pi_ai::types::ToolChoice::Auto);

    let context = pi_ai::utils::transcript::normalize_context(&Context {
        system_prompt: Some("Summarize".to_string()),
        messages: Vec::new(),
        tools: None,
    });

    complete_summarization(
        &summary_model(false, 8192, None),
        context,
        options,
        Some(stream_fn),
        None,
        None,
    )
    .await
    .expect("completion");

    let captured = captured.lock().expect("captured lock");
    assert_eq!(captured.len(), 1);
    assert_eq!(
        captured[0].base.session_id.as_deref(),
        Some("current-routing-session")
    );
    assert_eq!(
        captured[0].base.cache_retention,
        Some(pi_ai::types::CacheRetention::None)
    );
    assert_eq!(
        captured[0].tool_choice,
        Some(pi_ai::types::ToolChoice::Auto)
    );
}

#[tokio::test]
async fn compact_preserves_the_previous_summary_without_an_empty_history_request_for_a_split_turn()
{
    let (stream_fn, _captured, contexts) = summary_response_stream_fn(mock_summary_response());

    let preparation = CompactionPreparation {
        first_kept_entry_id: "entry-keep".to_string(),
        messages_to_summarize: Vec::new(),
        turn_prefix_messages: summarize_messages(),
        is_split_turn: true,
        tokens_before: 100,
        previous_summary: Some("previous checkpoint".to_string()),
        file_ops: create_file_ops(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens: 2000,
            keep_recent_tokens: 20,
        },
    };

    let result = compact(
        &preparation,
        &summary_model(false, 8192, None),
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("compaction");

    assert!(result.summary.contains("previous checkpoint"));
    let contexts = contexts.lock().expect("context lock");
    assert_eq!(contexts.len(), 1);
    let prompt = serde_json::to_string(contexts[0].messages()).expect("json");
    // Regression test for #9652: clear boundaries and continuation wording
    // avoid the reasoning-extraction false positive.
    assert!(prompt.contains("# Conversation\\n[User]: Summarize this."));
    assert!(prompt.contains(
        "# Instructions\\nThe messages above are earlier context from an ongoing conversation."
    ));
}

#[tokio::test]
async fn generate_summary_rejects_tool_calls_from_conversation_summaries() {
    let mut tool_call_response = mock_summary_response();
    tool_call_response.content = vec![AssistantContent::ToolCall(ToolCall {
        id: "tool-call-1".to_string(),
        name: "read".to_string(),
        arguments: serde_json::from_str(r#"{ "path": "README.md" }"#).unwrap(),
        thought_signature: None,
        namespace: None,
    })];
    tool_call_response.stop_reason = StopReason::ToolUse;
    let (stream_fn, _captured, _contexts) = summary_response_stream_fn(tool_call_response);

    let error = generate_summary_with_usage(
        &summarize_messages(),
        &summary_model(false, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        None,
        Some(stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect_err("tool calls rejected");

    assert_eq!(error, "Summarization attempted to call a tool");
}

#[tokio::test]
async fn compact_rejects_tool_calls_from_split_turn_summaries() {
    let mut tool_call_response = mock_summary_response();
    tool_call_response.content = vec![AssistantContent::ToolCall(ToolCall {
        id: "tool-call-1".to_string(),
        name: "read".to_string(),
        arguments: serde_json::from_str(r#"{ "path": "README.md" }"#).unwrap(),
        thought_signature: None,
        namespace: None,
    })];
    tool_call_response.stop_reason = StopReason::ToolUse;
    let (stream_fn, _captured, _contexts) = summary_response_stream_fn(tool_call_response);
    let preparation = CompactionPreparation {
        first_kept_entry_id: "entry-keep".to_string(),
        messages_to_summarize: Vec::new(),
        turn_prefix_messages: summarize_messages(),
        is_split_turn: true,
        tokens_before: 100,
        previous_summary: None,
        file_ops: create_file_ops(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens: 2000,
            keep_recent_tokens: 20,
        },
    };

    let error = compact(
        &preparation,
        &summary_model(false, 8192, None),
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect_err("tool calls rejected");

    assert_eq!(error, "Turn prefix summarization attempted to call a tool");
}

#[tokio::test]
async fn generate_summary_rejects_a_length_limited_history_summary() {
    let mut length_response = mock_summary_response();
    length_response.stop_reason = StopReason::Length;
    length_response.content = vec![AssistantContent::Text(TextContent::new("partial"))];
    let (stream_fn, _captured, _contexts) = summary_response_stream_fn(length_response);

    let error = generate_summary_with_usage(
        &summarize_messages(),
        &summary_model(false, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        None,
        Some(stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect_err("length limited summaries rejected");

    assert!(error.contains("generation hit the token cap"));
}

#[tokio::test]
async fn compact_rejects_a_length_limited_split_turn_summary() {
    let mut length_response = mock_summary_response();
    length_response.stop_reason = StopReason::Length;
    length_response.content = vec![AssistantContent::Text(TextContent::new("partial"))];
    let (stream_fn, _captured, _contexts) = summary_response_stream_fn(length_response);
    let preparation = CompactionPreparation {
        first_kept_entry_id: "entry-keep".to_string(),
        messages_to_summarize: Vec::new(),
        turn_prefix_messages: summarize_messages(),
        is_split_turn: true,
        tokens_before: 100,
        previous_summary: None,
        file_ops: create_file_ops(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens: 2000,
            keep_recent_tokens: 20,
        },
    };

    let error = compact(
        &preparation,
        &summary_model(false, 8192, None),
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect_err("length limited summaries rejected");

    assert!(error.contains("generation hit the token cap"));
}

#[tokio::test]
async fn generate_summary_does_not_set_reasoning_when_thinking_is_off() {
    let (stream_fn, captured, _contexts) = summary_response_stream_fn(mock_summary_response());

    let _ = generate_summary(
        &summarize_messages(),
        &summary_model(true, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(AgentThinkingLevel::Off),
        Some(stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    let options = captured.lock().expect("captured lock");
    assert_eq!(options.len(), 1);
    assert_eq!(options[0].base.base.api_key.as_deref(), Some("test-key"));
    assert!(options[0].reasoning.is_none());
}

#[tokio::test]
async fn generate_summary_does_not_set_reasoning_for_non_reasoning_models() {
    let (stream_fn, captured, _contexts) = summary_response_stream_fn(mock_summary_response());

    let _ = generate_summary(
        &summarize_messages(),
        &summary_model(false, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(AgentThinkingLevel::Medium),
        Some(stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    let options = captured.lock().expect("captured lock");
    assert_eq!(options.len(), 1);
    assert_eq!(options[0].base.base.api_key.as_deref(), Some("test-key"));
    assert!(options[0].reasoning.is_none());
}

#[tokio::test]
async fn generate_summary_leaves_anthropic_refusal_fallback_handling_to_pi_ai_model_metadata() {
    // Upstream passes `compat.allowedFallbackModels` and asserts the request
    // options carry no `refusalFallbacks`; that option does not exist in the
    // Rust `SimpleStreamOptions`, so the port asserts the call goes through
    // with model metadata intact and no reasoning override.
    let compat = json!({
        "allowedFallbackModels": [
            {
                "provider": "anthropic",
                "model": "claude-opus-4-8",
                "cost": { "input": 5, "output": 25, "cacheRead": 0.5, "cacheWrite": 6.25 },
            },
        ],
    });
    let (stream_fn, captured, _contexts) = summary_response_stream_fn(mock_summary_response());

    let _ = generate_summary(
        &summarize_messages(),
        &summary_model(true, 8192, Some(compat)),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        None,
        Some(stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    let options = captured.lock().expect("captured lock");
    assert_eq!(options.len(), 1);
    assert!(options[0].reasoning.is_none());
}

#[tokio::test]
async fn generate_summary_clamps_compaction_summary_max_tokens_to_the_model_output_cap() {
    let (stream_fn, captured, _contexts) = summary_response_stream_fn(mock_summary_response());
    let messages = summarize_messages();
    let preparation = CompactionPreparation {
        first_kept_entry_id: "entry-keep".to_string(),
        messages_to_summarize: messages.clone(),
        turn_prefix_messages: messages,
        is_split_turn: true,
        tokens_before: 600000,
        previous_summary: None,
        file_ops: create_file_ops(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens: 500000,
            keep_recent_tokens: 20000,
        },
    };

    let result = compact(
        &preparation,
        &summary_model(false, 128000, None),
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("compaction");

    let mut expected_usage = mock_summary_response().usage;
    expected_usage.input = 20;
    expected_usage.output = 20;
    expected_usage.total_tokens = 40;
    expected_usage.cost = UsageCost::default();
    assert_eq!(result.usage, Some(expected_usage));

    let captured = captured.lock().expect("captured lock");
    let max_tokens: Vec<Option<u64>> = captured
        .iter()
        .map(|options| options.base.max_tokens)
        .collect();
    assert_eq!(max_tokens, vec![Some(128000), Some(128000)]);
}
