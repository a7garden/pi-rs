use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_ai::models::MutableModels;
use pi_ai::providers::faux::{
	faux_assistant_message, faux_provider, faux_tool_call, FauxContentBlock, FauxMessageOptions,
	FauxModelDefinition, FauxResponseStep, RegisterFauxProviderOptions,
};
use pi_ai::types::{
	AssistantContent, AssistantMessage, Message, ModelThinkingLevel, StopReason, SystemContent,
	SystemMessage, TextContent, ThinkingContent, ToolResultMessage, UserContent, UserContentField,
	UserMessage,
};
use serde_json::{json, Value};

use super::support::{context, root};
use crate::harness::compaction::{select_cut, serialize_conversation};
use crate::harness::harness::{open_harness, Conversation, Harness};
use crate::harness::live::{live_doc, LiveState};
use crate::harness::registry::{create_registry, RegistryImpl};
use crate::harness::types::{
	AgentChange, BeforeCompactDecision, BeforeCompactPayload, CompactionHooks, CompactionPolicyOverrides,
	CompactionReason, ConversationStreamOptions, Extension, HookRegistration, HarnessOptions, HarnessSettings,
	InputSubmissionDraft, ModelRef, Registry, RetryPolicyOverrides, SetOrClear, Submission, SubmissionDraft, SubmissionStatus, WriteSubmissionDraft,
};
use crate::storage::memory::MemoryStorage;
use crate::types::{EntryDraft, EntryHead, EntryId, EntryRecord, Storage, TaskOutcome, TaskOutcomeStatus};

const WINDOW: u64 = 100_000;
const OVERFLOW: &str = "prompt is too long: 250000 tokens > 200000 maximum";

#[derive(Clone)]
struct CapturedRequest {
	messages: Vec<Message>,
	options: Option<pi_ai::types::SimpleStreamOptions>,
	model: String,
}

enum Reply {
	Message(AssistantMessage),
	Gate {
		message: AssistantMessage,
		release: tokio::sync::oneshot::Receiver<()>,
		reached: Option<tokio::sync::oneshot::Sender<()>>,
	},
	Action {
		message: AssistantMessage,
		before_return: Box<dyn FnOnce() + Send>,
	},
}

#[derive(Default)]
struct Script {
	agent: Mutex<VecDeque<Reply>>,
	summaries: Mutex<VecDeque<Reply>>,
	agent_requests: Mutex<Vec<CapturedRequest>>,
	summary_requests: Mutex<Vec<CapturedRequest>>,
	reports: Mutex<Vec<String>>,
}

impl Script {
	fn push_agent(&self, message: AssistantMessage) {
		self.agent.lock().push_back(Reply::Message(message));
	}

	fn push_summary(&self, message: AssistantMessage) {
		self.summaries.lock().push_back(Reply::Message(message));
	}
	fn push_summary_action(&self, message: AssistantMessage, before_return: impl FnOnce() + Send + 'static) {
		self.summaries.lock().push_back(Reply::Action { message, before_return: Box::new(before_return) });
	}

	fn gate_agent(
		&self,
		message: AssistantMessage,
	) -> (tokio::sync::oneshot::Sender<()>, tokio::sync::oneshot::Receiver<()>) {
		let (release_tx, release_rx) = tokio::sync::oneshot::channel();
		let (reached_tx, reached_rx) = tokio::sync::oneshot::channel();
		self.agent.lock().push_back(Reply::Gate {
			message,
			release: release_rx,
			reached: Some(reached_tx),
		});
		(release_tx, reached_rx)
	}

	fn gate_summary(
		&self,
		message: AssistantMessage,
	) -> (tokio::sync::oneshot::Sender<()>, tokio::sync::oneshot::Receiver<()>) {
		let (release_tx, release_rx) = tokio::sync::oneshot::channel();
		let (reached_tx, reached_rx) = tokio::sync::oneshot::channel();
		self.summaries.lock().push_back(Reply::Gate {
			message,
			release: release_rx,
			reached: Some(reached_tx),
		});
		(release_tx, reached_rx)
	}

	fn requests(&self, summary: bool) -> Vec<CapturedRequest> {
		if summary { self.summary_requests.lock().clone() } else { self.agent_requests.lock().clone() }
	}
}

struct Chat {
	storage: Arc<dyn Storage>,
	registry: Arc<RegistryImpl>,
	harness: Arc<Harness>,
	root: Arc<Conversation>,
	faux: Arc<pi_ai::providers::faux::FauxProviderHandle>,
	script: Arc<Script>,
}

fn reply_factory(script: Arc<Script>) -> pi_ai::providers::faux::FauxResponseFactory {
	Arc::new(move |transcript, options, _state, model| {
		let messages = transcript.messages().to_vec();
		let is_summary = messages.first().is_some_and(|message| match message {
			Message::System(system) => matches!(&system.content, SystemContent::Text(text) if text.starts_with("You are a context summarization assistant")),
			_ => false,
		});
		let request = CapturedRequest {
			messages,
			options: options.cloned(),
			model: model.id().to_string(),
		};
		let queue = if is_summary { &script.summaries } else { &script.agent };
		let requests = if is_summary { &script.summary_requests } else { &script.agent_requests };
		let signal = options.and_then(|options| options.signal.clone());
		requests.lock().push(request);
		let reply = queue.lock().pop_front();
		Box::pin(async move {
			match reply {
				Some(Reply::Message(message)) => Ok(message),
				Some(Reply::Gate { message, release, reached }) => {
					if let Some(reached) = reached { let _ = reached.send(()); }
					if let Some(signal) = signal {
						let _ = pi_ai::utils::abort::race_with_abort_signal(release, signal).await;
					} else {
						let _ = release.await;
					}
					Ok(message)
				}
				Some(Reply::Action { message, before_return }) => {
					before_return();
					Ok(message)
				}
				None => Ok(faux_assistant_message("unscripted", FauxMessageOptions::default())),
			}
		})
	})
}

fn text(label: &str, tokens: usize) -> String {
	format!("{label} {}", "x".repeat(tokens.saturating_mul(4).saturating_sub(label.len() + 1)))
}

fn answer(text: impl Into<String>) -> AssistantMessage {
	faux_assistant_message(text.into(), FauxMessageOptions::default())
}

fn failure(message: impl Into<String>) -> AssistantMessage {
	faux_assistant_message(
		"",
		FauxMessageOptions {
			stop_reason: Some(StopReason::Error),
			error_message: Some(message.into()),
			..Default::default()
		},
	)
}

fn policy(enabled: bool, reserve_tokens: u64, keep_recent_tokens: u64, background_tokens: u64) -> CompactionPolicyOverrides {
	CompactionPolicyOverrides {
		enabled: Some(enabled),
		reserve_tokens: Some(reserve_tokens),
		keep_recent_tokens: Some(keep_recent_tokens),
		background_tokens: Some(background_tokens),
	}
}

async fn open_chat(window: u64, compaction: CompactionPolicyOverrides) -> Chat {
	open_chat_with_retry(window, compaction, 1, None).await
}

async fn open_chat_with_retry(
	window: u64,
	compaction: CompactionPolicyOverrides,
	retry_delay_ms: u64,
	now: Option<Arc<dyn Fn() -> i64 + Send + Sync>>,
) -> Chat {
	let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
	open_chat_on(storage, window, compaction, retry_delay_ms, now, None).await
}

async fn open_chat_on(
	storage: Arc<dyn Storage>,
	window: u64,
	compaction: CompactionPolicyOverrides,
	retry_delay_ms: u64,
	now: Option<Arc<dyn Fn() -> i64 + Send + Sync>>,
	settings_provider: Option<crate::harness::types::HarnessSettingsProvider>,
) -> Chat {
	let registry = Arc::new(create_registry());
	let faux = Arc::new(faux_provider(RegisterFauxProviderOptions {
		models: vec![
			FauxModelDefinition { id: "faux-1".into(), context_window: Some(window), max_tokens: Some(900), ..Default::default() },
			FauxModelDefinition { id: "faux-2".into(), context_window: Some(window), max_tokens: Some(900), ..Default::default() },
		],
		..Default::default()
	}));
	let models = pi_ai::create_models(None);
	models.set_provider(faux.provider.clone());
	let script = Arc::new(Script::default());
	faux.set_responses((0..512).map(|_| FauxResponseStep::Factory(reply_factory(script.clone()))).collect());
	let report_script = script.clone();
	let on_report: crate::harness::types::ReportHook = Arc::new(move |error| report_script.reports.lock().push(error));
	let harness = open_harness(
		storage.clone(),
		HarnessOptions {
			models,
			registry: registry.clone(),
			now,
			settings_provider,
			settings: Some(HarnessSettings {
				stream: Some(ConversationStreamOptions {
					cache_retention: Some(pi_ai::types::CacheRetention::None),
					..Default::default()
				}),
				retry: Some(RetryPolicyOverrides {
					enabled: Some(true),
					max_retries: Some(2),
					base_delay_ms: Some(retry_delay_ms),
					..Default::default()
				}),
				compaction: Some(compaction),
				..Default::default()
			}),
			on_report: Some(on_report),
			..Default::default()
		},
		&context(),
	)
	.await
	.expect("open harness");
	let root = Arc::new(root(&harness).await);
	root.configure(
		AgentChange {
			model: Some(SetOrClear::Set(ModelRef { provider: "faux".into(), model_id: "faux-1".into() })),
			..Default::default()
		},
		&context(),
	)
	.await
	.expect("select faux model");
	harness.resume();
	Chat { storage, registry, harness, root, faux, script }
}

/// Fresh sqlite file storage for reopen tests: closing a harness closes the
/// storage backend it opened with (upstream semantics), so these tests need a
/// backend that persists across harness lifetimes via its file.
async fn open_reopen_storage(path: &std::path::Path) -> Arc<dyn Storage> {
	Arc::new(crate::storage::sqlite::SqliteStorage::open_path(
		path,
		&crate::storage::sqlite::NodeSqliteStorageOptions::default(),
	).await.expect("open sqlite storage"))
}

async fn turn(chat: &Chat, input: impl Into<String>, output: impl Into<String>) {
	chat.script.push_agent(answer(output));
	let submission = chat.root.submit(
		SubmissionDraft::Input(InputSubmissionDraft {
			request_id: None,
			content: UserContentField::Text(input.into()),
			when_busy: None,
		}),
		&context(),
	).await.expect("submit input");
	assert_eq!(submission.wait(&context()).await.expect("settle input").status, SubmissionStatus::Done);
}

async fn history(chat: &Chat) {
	for index in 1..=3 {
		turn(chat, text(&format!("u{index}"), 100), text(&format!("a{index}"), 100)).await;
	}
}

async fn outcome(chat: &Chat, id: crate::types::TaskId) -> TaskOutcome {
	chat.harness.wait_for_task(id, &context()).await.expect("wait for task").record.state.outcome.expect("terminal outcome")
}

async fn live(chat: &Chat) -> LiveState {
	let value = chat.harness.session.snapshot(&live_doc(), Some(chat.root.id.0), None, &context())
		.await.expect("read live document").unwrap_or_default();
	serde_json::from_value(Value::Object((*value).clone())).expect("decode live state")
}

async fn usage(chat: &Chat) -> Value {
	let value = chat.harness.session.snapshot(&crate::harness::usage::usage_doc(), Some(chat.root.id.0), None, &context())
		.await.expect("read usage").unwrap_or_default();
	Value::Object((*value).clone())
}

/// Upstream `allEntries()`: raw active entries, oldest first (the storage
/// scan returns newest first).
async fn raw_entries(conversation: &Conversation) -> Vec<crate::types::EntryRecord> {
	let mut entries = conversation
		.entries(crate::harness::types::ConversationEntryQuery::default(), 1000, None, &context())
		.await
		.expect("read raw entries")
		.items;
	entries.reverse();
	entries
}

async fn kinds(conversation: &Conversation) -> Vec<String> {
	raw_entries(conversation).await.into_iter().map(|entry| entry.kind).collect()
}

fn message_text(message: &Message) -> String {
	match message {
		Message::User(user) => match &user.content {
			UserContentField::Text(text) => text.clone(),
			UserContentField::Blocks(blocks) => blocks.iter().filter_map(|block| match block {
				UserContent::Text(text) => Some(text.text.clone()),
				_ => None,
			}).collect::<Vec<_>>().join("\n"),
		},
		Message::Assistant(assistant) => assistant.content.iter().filter_map(|block| match block {
			AssistantContent::Text(text) => Some(text.text.clone()),
			_ => None,
		}).collect::<Vec<_>>().join("\n"),
		Message::System(system) => match &system.content {
			SystemContent::Text(text) => text.clone(),
			SystemContent::Blocks(_) => String::new(),
		},
		Message::ToolResult(result) => result.content.iter().filter_map(|block| match block {
			UserContent::Text(text) => Some(text.text.clone()),
			_ => None,
		}).collect::<Vec<_>>().join("\n"),
	}
}

fn input(text: impl Into<String>) -> SubmissionDraft {
	SubmissionDraft::Input(InputSubmissionDraft {
		request_id: None,
		content: UserContentField::Text(text.into()),
		when_busy: None,
	})
}

async fn manual_result(chat: &Chat, instructions: Option<&str>) -> TaskOutcome {
	outcome(chat, chat.root.compact(instructions.map(str::to_string), &context()).await.expect("start compaction")).await
}

fn compaction_submission(outcome: &TaskOutcome) -> crate::types::SubmissionId {
	crate::types::SubmissionId(outcome.result.as_ref().expect("completed result")["submissionId"].as_u64().expect("submission id"))
}

async fn submission(chat: &Chat, id: crate::types::SubmissionId) -> crate::harness::submissions::SubmissionHandle {
	chat.harness.submission(id, &context()).await.expect("read submission").expect("submission exists")
}

fn selection_entry(id: u64, kind: &str, message: Message) -> EntryRecord {
	EntryRecord {
		id: EntryId(id),
		conversation_id: crate::types::ConversationId(1),
		kind: kind.to_string(),
		model: Some(vec![message]),
		data: None,
		head: None,
		edits: None,
		by_task_id: None,
	}
}

fn user_message(value: impl Into<String>) -> Message {
	Message::User(UserMessage { content: UserContentField::Text(value.into()), timestamp: 0 })
}

fn assistant_message(value: impl Into<String>, stop_reason: StopReason) -> Message {
	let mut message = answer(value);
	message.stop_reason = stop_reason;
	Message::Assistant(message)
}
fn assistant_tool_call(call_id: &str) -> Message {
	Message::Assistant(faux_assistant_message(
		FauxContentBlock::from(faux_tool_call("read", Default::default(), Some(call_id))),
		FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
	))
}

fn tool_result_message(call_id: &str, text: impl Into<String>) -> Message {
	Message::ToolResult(ToolResultMessage {
		tool_call_id: call_id.into(), tool_name: "read".into(),
		content: vec![UserContent::Text(TextContent::new(text))],
		details: None, usage: None, nested_calls: None, is_error: false, timestamp: 0,
	})
}

fn view(entries: Vec<EntryRecord>, head: Option<EntryRecord>) -> crate::session::ContextView {
	let all = head.into_iter().chain(entries).collect::<Vec<_>>();
	let contributions = all.iter().map(|entry| entry.model.clone().unwrap_or_default()).collect::<Vec<_>>();
	let messages = crate::harness::context::order_tool_results(contributions.iter().flatten().cloned().collect());
	crate::session::ContextView { head: all.first().cloned().filter(|head| head.kind == "pi.compaction"), entries: all, contributions, messages }
}

#[test]
fn compaction_range_selection_matches_the_long_tool_result_spec_example() {
	let entries = vec![
		selection_entry(1, "pi.user", user_message(text("1", 10))),
		selection_entry(2, "pi.assistant", assistant_tool_call("c1")),
		selection_entry(3, "pi.tool-result", tool_result_message("c1", text("3", 3000))),
		selection_entry(4, "pi.assistant", assistant_message(text("4", 10), StopReason::Stop)),
		selection_entry(5, "pi.user", user_message(text("5", 10))),
		selection_entry(6, "pi.assistant", assistant_message(text("6", 10), StopReason::Stop)),
	];
	assert_eq!(select_cut(&view(entries, None), 2000), Some(3));
}

#[test]
fn compaction_range_selection_preserves_message_and_tool_boundaries() {
	let entries = vec![
		selection_entry(1, "pi.user", user_message(text("u1", 100))),
		selection_entry(2, "pi.assistant", assistant_message(text("a1", 100), StopReason::Stop)),
		selection_entry(3, "pi.user", user_message(text("u2", 100))),
		selection_entry(4, "pi.assistant", assistant_message(text("a2", 100), StopReason::Stop)),
	];
	assert_eq!(select_cut(&view(entries, None), 150), Some(2));

	let mut run = vec![selection_entry(10, "pi.user", user_message("do it"))];
	for index in 0..5 {
		let call_id = format!("c{index}");
		run.push(selection_entry(11 + index * 2, "pi.assistant", assistant_tool_call(&call_id)));
		run.push(selection_entry(12 + index * 2, "pi.tool-result", tool_result_message(&call_id, text("result", 100))));
	}
	assert_eq!(select_cut(&view(run, None), 150), Some(9));

	let huge = vec![
		selection_entry(30, "pi.user", user_message("u")),
		selection_entry(31, "pi.assistant", assistant_tool_call("huge-call")),
		selection_entry(32, "pi.tool-result", tool_result_message("huge-call", text("large", 5000))),
	];
	assert_eq!(select_cut(&view(huge, None), 100), Some(1));
}

#[test]
fn compaction_range_selection_skips_non_candidates_and_omitted_entries() {
	let records = vec![
		selection_entry(1, "pi.user", user_message(text("u1", 100))),
		selection_entry(2, "pi.assistant", assistant_message(text("a1", 100), StopReason::Stop)),
		selection_entry(3, "pi.system", Message::System(SystemMessage {
			content: SystemContent::Text(text("system", 100)), sections: None, tools_added: None, tools_removed: None, timestamp: 0,
		})),
		selection_entry(4, "pi.assistant", assistant_message("failed", StopReason::Error)),
		selection_entry(5, "pi.assistant", assistant_message("stopped", StopReason::Aborted)),
		selection_entry(6, "pi.assistant", assistant_message(text("a2", 100), StopReason::Stop)),
	];
	let mut filtered = view(records.clone(), None);
	filtered.contributions[3].clear();
	filtered.contributions[4].clear();
	filtered.messages = filtered.contributions.iter().flatten().cloned().collect();
	assert_eq!(select_cut(&filtered, 150), Some(5));

	let omitted_at = view(vec![
		selection_entry(10, "pi.user", user_message(text("u1", 100))),
		selection_entry(11, "pi.assistant", assistant_message(text("a1", 100), StopReason::Stop)),
		selection_entry(12, "pi.user", user_message(text("u2", 100))),
		selection_entry(13, "pi.assistant", assistant_message(text("a2", 100), StopReason::Stop)),
	], None);
	let mut omitted = omitted_at.clone();
	omitted.contributions[2].clear();
	omitted.messages = omitted.contributions.iter().flatten().cloned().collect();
	assert_eq!(select_cut(&omitted_at, 150), Some(2));
	assert_eq!(select_cut(&omitted, 150), Some(1));

	let pending_result = vec![
		selection_entry(20, "pi.user", user_message(text("u1", 100))),
		selection_entry(21, "pi.assistant", assistant_tool_call("pending")),
		selection_entry(22, "pi.user", user_message(text("steer", 100))),
		selection_entry(23, "pi.tool-result", tool_result_message("pending", text("result", 100))),
		selection_entry(24, "pi.assistant", assistant_message(text("a2", 100), StopReason::Stop)),
	];
	assert_eq!(select_cut(&view(pending_result, None), 250), Some(4));
}

#[test]
fn compaction_range_selection_requires_old_content_and_summarizes_old_marker_first() {
	let small = vec![
		selection_entry(1, "pi.user", user_message("hi")),
		selection_entry(2, "pi.assistant", assistant_message("hello", StopReason::Stop)),
	];
	assert_eq!(select_cut(&view(small, None), 150), None);
	let marker = selection_entry(10, "pi.compaction", user_message("EARLIER"));
	let only_recent = view(vec![selection_entry(11, "pi.user", user_message(text("u", 200)))], Some(marker.clone()));
	assert_eq!(select_cut(&only_recent, 150), None);

	let selected = view(
		vec![
			selection_entry(11, "pi.user", user_message(text("u1", 100))),
			selection_entry(12, "pi.assistant", assistant_message(text("a1", 100), StopReason::Stop)),
			selection_entry(13, "pi.user", user_message(text("u2", 100))),
			selection_entry(14, "pi.assistant", assistant_message(text("a2", 100), StopReason::Stop)),
		],
		Some(marker),
	);
	assert_eq!(select_cut(&selected, 150), Some(3));
	let source = crate::harness::compaction::summarized_messages(&selected, 3);
	assert!(serialize_conversation(&source).starts_with("[User]: EARLIER"));
}

#[test]
fn compaction_serialization_omits_system_and_truncates_tool_results() {
	let messages = vec![
		Message::System(SystemMessage {
			content: SystemContent::Text("hidden".into()), sections: None, tools_added: None, tools_removed: None, timestamp: 0,
		}),
		user_message("hello"),
		Message::Assistant(AssistantMessage {
			content: vec![AssistantContent::Thinking(ThinkingContent::new("hmm")), AssistantContent::Text(TextContent::new("sure"))],
			..faux_assistant_message("", FauxMessageOptions::default())
		}),
		Message::Assistant(faux_assistant_message(
			FauxContentBlock::from(faux_tool_call("read", json!({"path":"a.ts"}).as_object().cloned().unwrap_or_default(), Some("call"))),
			FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
		)),
		Message::ToolResult(ToolResultMessage {
			tool_call_id: "call".into(), tool_name: "read".into(),
			content: vec![UserContent::Text(TextContent::new("y".repeat(2500)))], details: None, usage: None,
			nested_calls: None, is_error: false, timestamp: 0,
		}),
	];
	let serialized = serialize_conversation(&messages);
	assert!(!serialized.contains("hidden"));
	assert!(serialized.contains("[User]: hello"));
	assert!(serialized.contains("[Assistant thinking]: hmm"));
	assert!(serialized.contains("[Assistant]: sure"));
	assert!(serialized.contains("[Assistant tool calls]: read(path=\"a.ts\")"));
	assert!(serialized.contains(&format!("[Tool result]: {}\n\n[... 500 more characters truncated]", "y".repeat(2000))));
}

#[tokio::test]
async fn manual_compaction_records_summary_without_rewriting_raw_history() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let before = raw_entries(&chat.root).await;
	let usage_before = usage(&chat).await["models"]["faux/faux-1"]["input"].as_u64().unwrap_or(0);
	chat.script.push_summary(answer("SUMMARY"));
	let compact = chat.root.compact(Some("focus on files".into()), &context()).await.expect("start manual compaction");
	let task = outcome(&chat, compact).await;
	assert_eq!(task.status, TaskOutcomeStatus::Completed);
	let submission = submission(&chat, compaction_submission(&task)).await;
	let placed = submission.wait(&context()).await.expect("place summary");
	assert_eq!(placed.status, SubmissionStatus::Done);
	let after = raw_entries(&chat.root).await;
	assert_eq!(&after[..before.len()], before.as_slice());
	let marker = after.last().expect("summary marker");
	let kept = before.iter().find(|entry| entry.model.as_ref().is_some_and(|messages| messages.first().is_some_and(|message| message_text(message).starts_with("u3")))).unwrap();
	assert_eq!(marker.kind, "pi.compaction");
	assert_eq!(marker.head, Some(kept.id));
	assert_eq!(marker.data.as_ref().unwrap()["reason"], "manual");
	assert_eq!(placed.entry, Some(marker.id));
	assert_eq!(message_text(&marker.model.as_ref().unwrap()[0]), "The conversation history before this point was compacted into the following summary:\n\n<summary>\nSUMMARY\n</summary>");
	let view = chat.root.context(&context()).await.expect("read compacted context");
	assert_eq!(view.messages.iter().take(3).map(message_text).collect::<Vec<_>>(), vec![message_text(&marker.model.as_ref().unwrap()[0]), text("u3", 100), text("a3", 100)]);
	let request = chat.script.requests(true).remove(0);
	assert_eq!(request.messages.len(), 2);
	let prompt = message_text(&request.messages[1]);
	assert!(prompt.starts_with("<conversation>\n[User]: u1 "));
	assert!(prompt.contains("[Assistant]: a2 "));
	assert!(!prompt.contains("u3 "));
	assert!(prompt.ends_with("\n\nAdditional focus: focus on files"));
	let opts = request.options.expect("summary stream options");
	assert_eq!(opts.base.cache_retention, Some(pi_ai::types::CacheRetention::None));
	assert_eq!(opts.base.max_tokens, Some(800));
	let provider = chat.harness.session.snapshot(&crate::harness::provider::provider_doc(), Some(chat.root.id.0), None, &context())
		.await.unwrap().expect("provider state created");
	assert_eq!(opts.base.session_id.as_deref(), provider["sessionId"].as_str());
	assert!(opts.deferred.is_none());
	let usage_after = usage(&chat).await["models"]["faux/faux-1"]["input"].as_u64().unwrap_or(0);
	assert!(usage_after > usage_before);
	turn(&chat, "next", "done").await;
	let next = chat.script.requests(false).last().unwrap().messages.clone();
	assert_eq!(next.iter().map(|message| match message {
		Message::User(_) => "user",
		Message::Assistant(_) => "assistant",
		Message::System(_) => "system",
		Message::ToolResult(_) => "tool-result",
	}).collect::<Vec<_>>(), ["user", "user", "assistant", "user", "system"]);
	chat.harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn manual_compaction_is_not_busy_and_queues_at_the_next_final_boundary() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (release, reached) = chat.script.gate_summary(answer("SUMMARY"));
	let compact = chat.root.compact(None, &context()).await.expect("start compaction");
	reached.await.expect("summary reached provider");
	let (answer_release, answer_reached) = chat.script.gate_agent(answer("a4"));
	let input = chat.root.submit(input("u4"), &context()).await.expect("submit during summarize");
	answer_reached.await.expect("generation started without waiting for summarize");
	assert_eq!(input.status(&context()).await.expect("input status").status, SubmissionStatus::Placed);
	let first_request = chat.script.requests(false).last().unwrap().messages[0].clone();
	assert_eq!(message_text(&first_request), text("u1", 100));
	release.send(()).expect("release summary");
	let task = outcome(&chat, compact).await;
	let queued = submission(&chat, compaction_submission(&task)).await;
	assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	answer_release.send(()).expect("release generation");
	assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(queued.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(kinds(&chat.root).await.iter().rev().take(3).cloned().collect::<Vec<_>>(), ["pi.compaction", "pi.assistant", "pi.user"]);
	chat.harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn queued_manual_compaction_places_after_tool_results_and_continuation_uses_summary() {
	use crate::harness::define::define_tool_of;
	use crate::harness::types::{ToolExecutionResult, ToolRegistration};
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	let tool = define_tool_of(ToolRegistration {
		tool: pi_ai::types::Tool { name: "wait".into(), description: "wait".into(), parameters: json!({"type":"object","properties":{}}), constrained_sampling: None },
		replay: None, execution_mode: None, prepare_arguments: None, output_limits: None,
		execute: Arc::new(|_, _, _| Box::pin(async { ToolExecutionResult::default() })),
	});
	chat.registry.install(Extension::new("wait-tool").with_tools(vec![Arc::new(tool)]));
	history(&chat).await;
	let (release, reached) = chat.script.gate_agent(faux_assistant_message(
		FauxContentBlock::from(faux_tool_call("wait", Default::default(), Some("wait-call"))),
		FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
	));
	chat.script.push_agent(answer("after tools"));
	let input = chat.root.submit(input("use a tool"), &context()).await.expect("submit tool round");
	reached.await.expect("tool-call request started");
	chat.script.push_summary(answer("SUMMARY"));
	let compact = chat.root.compact(None, &context()).await.expect("start compaction");
	let task = outcome(&chat, compact).await;
	let queued = submission(&chat, compaction_submission(&task)).await;
	assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	release.send(()).expect("release tool call");
	assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(queued.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	let request = chat.script.requests(false).last().unwrap().messages.clone();
	assert!(message_text(&request[0]).contains("<summary>\nSUMMARY\n</summary>"));
	let kinds = kinds(&chat.root).await;
	let tail = &kinds[kinds.len() - 4..];
	assert_eq!(tail, ["pi.tool-result", "pi.compaction", "pi.system", "pi.assistant"]);
	chat.harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn failed_generation_leaves_follow_up_queued_behind_compaction() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (release, reached) = chat.script.gate_agent(failure("bad request"));
	let failed = chat.root.submit(input("fails"), &context()).await.expect("submit failing input");
	reached.await.expect("failed request started");
	let follow_up = chat.root.submit(input("follow-up"), &context()).await.expect("queue follow-up");
	release.send(()).expect("release failure");
	assert_eq!(failed.wait(&context()).await.unwrap().status, SubmissionStatus::Unanswered);
	assert_eq!(follow_up.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	chat.script.push_summary(answer("SUMMARY"));
	chat.script.push_agent(answer("followed"));
	let compact = chat.root.compact(None, &context()).await.expect("start compaction");
	let compact_outcome = outcome(&chat, compact).await;
	assert_eq!(outcome(&chat, compact).await.status, TaskOutcomeStatus::Completed);
	assert_eq!(submission(&chat, compaction_submission(&compact_outcome)).await.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(follow_up.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	let messages = chat.script.requests(false).last().unwrap().messages.clone();
	assert!(message_text(&messages[0]).contains("SUMMARY"));
	assert!(messages.iter().any(|message| message_text(message) == "follow-up"));
	chat.harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn reset_makes_an_in_flight_summary_stale_but_keeps_its_usage() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let before = usage(&chat).await["models"]["faux/faux-1"]["input"].as_u64().unwrap_or(0);
	let (release, reached) = chat.script.gate_summary(answer("SUMMARY"));
	let compact = chat.root.compact(None, &context()).await.expect("start compaction");
	reached.await.expect("summary started");
	chat.root.reset(None, &context()).await.expect("reset while summarizing");
	release.send(()).expect("release summary");
	let task = outcome(&chat, compact).await;
	let submission = submission(&chat, compaction_submission(&task)).await;
	let settled = submission.wait(&context()).await.expect("stale submission settles");
	assert_eq!(settled.status, SubmissionStatus::Unanswered);
	assert_eq!(settled.reason.as_deref(), Some("stale"));
	assert!(usage(&chat).await["models"]["faux/faux-1"]["input"].as_u64().unwrap_or(0) > before);
	assert!(!kinds(&chat.root).await.iter().any(|kind| kind == "pi.compaction"));
	assert_eq!(kinds(&chat.root).await.last().map(String::as_str), Some("pi.reset"));
	chat.harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn aborting_manual_compaction_removes_status_without_writing_a_marker() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (release, reached) = chat.script.gate_summary(answer("SUMMARY"));
	let task = chat.root.compact(None, &context()).await.expect("start compaction");
	reached.await.expect("summarizer started");
	assert_eq!(live(&chat).await.compactions.as_ref().unwrap().len(), 1);
	chat.harness.abort_task(task, &context()).await.expect("abort compaction");
	assert_eq!(outcome(&chat, task).await.status, TaskOutcomeStatus::Aborted);
	assert!(live(&chat).await.compactions.is_none());
	assert!(!kinds(&chat.root).await.iter().any(|kind| kind == "pi.compaction"));
	drop(release);
	chat.harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn manual_compaction_is_ordinary_idle_work_and_no_cut_completes_without_a_request() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	turn(&chat, "hi", "hello").await;
	let empty = manual_result(&chat, None).await;
	assert_eq!(empty.status, TaskOutcomeStatus::Completed);
	assert_eq!(empty.result, Some(json!({})));
	assert!(chat.script.requests(true).is_empty());
	assert!(live(&chat).await.compactions.is_none());
	history(&chat).await;
	let (release, reached) = chat.script.gate_summary(answer("SUMMARY"));
	let task = chat.root.compact(None, &context()).await.unwrap();
	reached.await.unwrap();
	let idle_context = context();
	let idle = chat.root.wait_for_idle(&idle_context);
	tokio::pin!(idle);
	assert!(tokio::time::timeout(std::time::Duration::from_millis(15), &mut idle).await.is_err());
	release.send(()).unwrap();
	idle.await.unwrap();
	assert_eq!(chat.harness.get_task(task, &context()).await.unwrap().unwrap().state.status, crate::types::TaskStateStatus::Terminal);
	chat.harness.close(&context()).await.unwrap();
}

static NEXT_COMPACTION_HOOK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn install_compaction_hook(
	registry: &RegistryImpl,
	callback: impl Fn(BeforeCompactPayload) -> Result<Option<BeforeCompactDecision>, String> + Send + Sync + 'static,
) {
	let callback = Arc::new(callback);
	let before = Arc::new(move |payload, _api, _context| {
		let callback = callback.clone();
		Box::pin(async move { callback(payload) })
			as crate::tasks::BoxFuture<Result<Option<BeforeCompactDecision>, String>>
	});
	registry.install(Extension::new(format!(
		"compaction-hooks-{}",
		NEXT_COMPACTION_HOOK.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
	)).with_hooks(vec![HookRegistration {
		task: "pi.compaction".into(),
		handlers: Arc::new(CompactionHooks { before_compact: Some(before) }),
	}]));
}

#[tokio::test]
async fn compaction_hooks_receive_selected_prefix_and_can_supply_or_decline_summary() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	let observed = Arc::new(Mutex::new(None::<BeforeCompactPayload>));
	let observed_hook = observed.clone();
	install_compaction_hook(&chat.registry, |_| Err("hook broke".into()));
	install_compaction_hook(&chat.registry, move |payload| {
		*observed_hook.lock() = Some(payload);
		Ok(Some(BeforeCompactDecision::Summary("FROM HOOK".into())))
	});
	history(&chat).await;
	install_compaction_hook(&chat.registry, |_| Ok(Some(BeforeCompactDecision::Decline)));
	let usage_before_hook_summary = usage(&chat).await;
	let result = manual_result(&chat, Some("why")).await;
	assert_eq!(result.status, TaskOutcomeStatus::Completed);
	assert!(chat.script.requests(true).is_empty());
	assert!(message_text(&chat.root.context(&context()).await.unwrap().messages[0]).contains("FROM HOOK"));
	let payload = observed.lock().clone().expect("hook payload");
	assert_eq!(payload.reason, CompactionReason::Manual);
	assert_eq!(payload.instructions.as_deref(), Some("why"));
	assert_eq!(payload.messages.iter().filter(|message| !matches!(message, Message::System(_))).map(message_text).collect::<Vec<_>>(), vec![text("u1", 100), text("a1", 100), text("u2", 100), text("a2", 100)]);
	assert_eq!(chat.script.reports.lock().as_slice(), &[String::from("hook broke")]);
	assert_eq!(usage(&chat).await, usage_before_hook_summary);
	chat.harness.close(&context()).await.unwrap();

	let declined = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	install_compaction_hook(&declined.registry, |_| Ok(Some(BeforeCompactDecision::Decline)));
	history(&declined).await;
	let result = manual_result(&declined, None).await;
	assert_eq!(result.status, TaskOutcomeStatus::Completed);
	assert_eq!(result.result, Some(json!({})));
	assert!(declined.script.requests(true).is_empty());
	declined.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn compaction_without_a_model_fails_and_clears_its_live_status() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	chat.root.configure(AgentChange { model: Some(SetOrClear::Clear), ..Default::default() }, &context()).await.unwrap();
	let result = manual_result(&chat, None).await;
	assert_eq!(result.status, TaskOutcomeStatus::Failed);
	assert_eq!(result.error.as_ref().and_then(|error| error.detail.as_ref()).unwrap()["reason"], "no_model");
	assert!(live(&chat).await.compactions.is_none());
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn summarization_retries_pin_request_and_count_every_attempt_once() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	chat.root.configure(AgentChange { thinking_level: Some(SetOrClear::Set(ModelThinkingLevel::High)), ..Default::default() }, &context()).await.unwrap();
	let before = usage(&chat).await["models"]["faux/faux-1"]["input"].as_u64().unwrap_or(0);
	let (release, reached) = chat.script.gate_summary(failure("overloaded"));
	chat.script.push_summary(answer("SUMMARY"));
	let id = chat.root.compact(None, &context()).await.unwrap();
	reached.await.unwrap();
	chat.root.configure(AgentChange { thinking_level: Some(SetOrClear::Set(ModelThinkingLevel::Low)), ..Default::default() }, &context()).await.unwrap();
	release.send(()).unwrap();
	let result = outcome(&chat, id).await;
	let requests = chat.script.requests(true);
	assert_eq!(requests.len(), 2);
	assert_eq!(requests[0].model, "faux-1");
	assert_eq!(requests[1].model, "faux-1");
	assert_eq!(requests[0].options.as_ref().unwrap().reasoning, Some(pi_ai::types::ThinkingLevel::High));
	assert_eq!(requests[1].options.as_ref().unwrap().reasoning, Some(pi_ai::types::ThinkingLevel::High));
	assert_eq!(message_text(&requests[0].messages[1]), message_text(&requests[1].messages[1]));
	assert!(usage(&chat).await["models"]["faux/faux-1"]["input"].as_u64().unwrap_or(0) > before);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn summarization_rejects_retries_exhaustion_length_tool_and_empty_outputs() {
	for (name, response, expected, expected_attempts) in [
		("retry exhaustion", failure("overloaded"), "Summarization failed: overloaded", 3usize),
		("nonretryable", failure("bad request"), "Summarization failed: bad request", 1usize),
		("length", faux_assistant_message("partial", FauxMessageOptions { stop_reason: Some(StopReason::Length), ..Default::default() }), "Summarization hit the token limit; the summary is incomplete", 1),
		("tool call", faux_assistant_message(FauxContentBlock::from(faux_tool_call("read", Default::default(), Some("summary-call"))), FauxMessageOptions::default()), "Summarization attempted to call a tool", 1),
		("empty", answer("  "), "Summarization produced no text", 1),
	] {
		let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
		history(&chat).await;
		for _ in 0..expected_attempts { chat.script.push_summary(response.clone()); }
		let result = manual_result(&chat, None).await;
		assert_eq!(result.status, TaskOutcomeStatus::Failed, "{name}");
		assert_eq!(result.error.as_ref().unwrap().message, expected, "{name}");
		assert_eq!(result.error.as_ref().unwrap().detail.as_ref().unwrap()["reason"], "model_error", "{name}");
		assert_eq!(chat.script.requests(true).len(), expected_attempts, "{name}");
		assert!(live(&chat).await.compactions.is_none(), "{name}");
		assert!(!kinds(&chat.root).await.iter().any(|kind| kind == "pi.compaction"), "{name}");
		chat.harness.close(&context()).await.unwrap();
	}
}

#[tokio::test]
async fn summarization_caps_output_and_never_offers_tools() {
	let chat = open_chat(WINDOW, policy(false, 2000, 150, 0)).await;
	let tool = crate::harness::define::define_tool_of(crate::harness::types::ToolRegistration {
		tool: pi_ai::types::Tool { name: "read".into(), description: "read".into(), parameters: json!({"type":"object","properties":{}}), constrained_sampling: None },
		replay: None, execution_mode: None, prepare_arguments: None, output_limits: None,
		execute: Arc::new(|_, _, _| Box::pin(async { crate::harness::api::ToolExecutionResult::default() })),
	});
	chat.registry.install(Extension::new("read-tool").with_tools(vec![Arc::new(tool)]));
	history(&chat).await;
	chat.script.push_summary(answer("SUMMARY"));
	let result = manual_result(&chat, None).await;
	assert_eq!(result.status, TaskOutcomeStatus::Completed);
	let request = chat.script.requests(true).remove(0);
	assert_eq!(request.options.unwrap().base.max_tokens, Some(900));
	assert_eq!(request.messages.len(), 2);
	assert!(request.messages.iter().all(|message| !matches!(message, Message::System(system) if system.tools_added.is_some())));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn threshold_compaction_runs_in_background_or_blocks_before_generation() {
	let current_policy = Arc::new(Mutex::new(policy(false, 500, 150, 1000)));
	let provider_policy = current_policy.clone();
	let settings_provider = Arc::new(move || HarnessSettings {
		compaction: Some(provider_policy.lock().clone()),
		..Default::default()
	});
	let background = open_chat_on(
		Arc::new(MemoryStorage::new()),
		2000,
		policy(false, 500, 150, 1000),
		1,
		None,
		Some(settings_provider),
	).await;
	history(&background).await;
	*current_policy.lock() = policy(true, 500, 150, 1000);
	let (release, reached) = background.script.gate_summary(answer("SUMMARY"));
	turn(&background, text("u4", 100), text("a4", 100)).await;
	reached.await.expect("background summary started");
	let tasks = background.harness.inspect(&context()).await.unwrap().tasks;
	let task = tasks.iter().find(|task| task.record.kind == "pi.compaction").unwrap();
	assert!(task.record.background);
	assert!(task.record.owner.is_none());
	assert_eq!(live(&background).await.compactions.as_ref().unwrap()[0].blocking, false);
	background.root.wait_for_idle(&context()).await.unwrap();
	background.root.abort(&context(), None).await.unwrap();
	assert_eq!(background.harness.get_task(task.record.id, &context()).await.unwrap().unwrap().state.status, crate::types::TaskStateStatus::Running);
	release.send(()).unwrap();
	assert_eq!(outcome(&background, task.record.id).await.status, TaskOutcomeStatus::Completed);
	assert!(message_text(&background.root.context(&context()).await.unwrap().messages[0]).contains("SUMMARY"));
	background.harness.close(&context()).await.unwrap();

	let blocking = open_chat(1000, policy(true, 300, 150, 0)).await;
	history(&blocking).await;
	let (release, reached) = blocking.script.gate_summary(answer("SUMMARY"));
	blocking.script.push_agent(answer("a4"));
	let input = blocking.root.submit(input(text("u4", 200)), &context()).await.unwrap();
	reached.await.unwrap();
	let task = blocking.harness.inspect(&context()).await.unwrap().tasks.into_iter().find(|task| task.record.kind == "pi.compaction").unwrap();
	assert_eq!(task.record.owner, Some(live(&blocking).await.run.as_ref().unwrap().task_id));
	let generation_id = live(&blocking).await.run.as_ref().unwrap().task_id;
	let waiting = blocking.harness.get_task(generation_id, &context()).await.unwrap().unwrap();
	assert_eq!(waiting.state.status, crate::types::TaskStateStatus::Waiting);
	assert_eq!(waiting.state.on.as_deref(), Some(&[task.record.id][..]));
	assert_eq!(waiting.state.checkpoint.as_ref().unwrap()["phase"], "prepare");
	assert_eq!(waiting.state.checkpoint.as_ref().unwrap()["compacted"], task.record.id.0);
	assert_eq!(live(&blocking).await.compactions.as_ref().unwrap()[0].attempt, 1);
	assert_eq!(live(&blocking).await.compactions.as_ref().unwrap()[0].blocking, true);
	assert!(kinds(&blocking.root).await.last().unwrap() == "pi.user");
	release.send(()).unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(outcome(&blocking, task.record.id).await.status, TaskOutcomeStatus::Completed);
	assert!(message_text(&blocking.script.requests(false).last().unwrap().messages[0]).contains("SUMMARY"));
	blocking.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn background_threshold_respects_disable_zero_budget_no_cut_and_task_abort() {
	for setting in [
		policy(false, 500, 150, 1000),
		policy(true, 500, 150, 0),
		policy(true, 500, 100_000, 1000),
	] {
		let chat = open_chat(2000, setting).await;
		history(&chat).await;
		turn(&chat, text("u4", 100), text("a4", 100)).await;
		assert!(chat.harness.inspect(&context()).await.unwrap().tasks.iter().all(|task| task.record.kind != "pi.compaction"));
		assert!(chat.script.requests(true).is_empty());
		chat.harness.close(&context()).await.unwrap();
	}
	let current_policy = Arc::new(Mutex::new(policy(false, 500, 150, 1000)));
	let provider_policy = current_policy.clone();
	let settings_provider = Arc::new(move || HarnessSettings {
		compaction: Some(provider_policy.lock().clone()),
		..Default::default()
	});
	let chat = open_chat_on(
		Arc::new(MemoryStorage::new()),
		2000,
		policy(false, 500, 150, 1000),
		1,
		None,
		Some(settings_provider),
	).await;
	history(&chat).await;
	*current_policy.lock() = policy(true, 500, 150, 1000);
	let (release, reached) = chat.script.gate_summary(answer("SUMMARY"));
	turn(&chat, text("u4", 100), text("a4", 100)).await;
	reached.await.unwrap();
	let task = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter().find(|task| task.record.kind == "pi.compaction").unwrap();
	chat.harness.abort_task(task.record.id, &context()).await.unwrap();
	assert_eq!(outcome(&chat, task.record.id).await.status, TaskOutcomeStatus::Aborted);
	assert!(live(&chat).await.compactions.is_none());
	drop(release);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn background_compaction_abort_task_and_background_conversation_abort_both_stop_it() {
	for conversation_abort in [false, true] {
		let current_policy = Arc::new(Mutex::new(policy(false, 500, 150, 1000)));
		let provider_policy = current_policy.clone();
		let settings_provider = Arc::new(move || HarnessSettings {
			retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(1), ..Default::default() }),
			compaction: Some(provider_policy.lock().clone()),
			..Default::default()
		});
		let chat = open_chat_on(
			Arc::new(MemoryStorage::new()),
			2000,
			policy(false, 500, 150, 1000),
			1,
			None,
			Some(settings_provider),
		).await;
		history(&chat).await;
		*current_policy.lock() = policy(true, 500, 150, 1000);
		let (release, reached) = chat.script.gate_summary(answer("SUMMARY"));
		turn(&chat, text("u4", 100), text("a4", 100)).await;
		reached.await.unwrap();
		let task = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
			.find(|task| task.record.kind == "pi.compaction").unwrap();
		if conversation_abort {
			chat.root.abort(&context(), Some(crate::session::ConversationAbortOptions { background: Some(true) }))
				.await.unwrap();
		} else {
			chat.harness.abort_task(task.record.id, &context()).await.unwrap();
		}
		assert_eq!(outcome(&chat, task.record.id).await.status, TaskOutcomeStatus::Aborted);
		assert!(live(&chat).await.compactions.is_none());
		drop(release);
		chat.harness.close(&context()).await.unwrap();
	}
}

#[tokio::test]
async fn blocking_compaction_keeps_its_limit_and_direct_abort_still_sends_the_request() {
	let chat = open_chat(1000, policy(true, 300, 700, 0)).await;
	history(&chat).await;
	chat.script.push_summary(answer("SUMMARY"));
	chat.script.push_agent(answer("a4"));
	let input_handle = chat.root.submit(input(text("u4", 400)), &context()).await.unwrap();
	assert_eq!(input_handle.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(chat.script.requests(true).len(), 1);
	assert_eq!(kinds(&chat.root).await.iter().filter(|kind| kind.as_str() == "pi.compaction").count(), 1);
	chat.harness.close(&context()).await.unwrap();

	let chat = open_chat(1000, policy(true, 300, 150, 0)).await;
	history(&chat).await;
	let (release, reached) = chat.script.gate_summary(answer("SUMMARY"));
	chat.script.push_agent(answer("a4"));
	let input_handle = chat.root.submit(input(text("u4", 200)), &context()).await.unwrap();
	reached.await.unwrap();
	let child = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
		.find(|task| task.record.kind == "pi.compaction").unwrap();
	chat.harness.abort_task(child.record.id, &context()).await.unwrap();
	assert_eq!(input_handle.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert!(!kinds(&chat.root).await.iter().any(|kind| kind == "pi.compaction"));
	drop(release);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn blocking_overflow_after_compaction_fails_instead_of_starting_another_cut() {
	let chat = open_chat(1000, policy(true, 300, 150, 0)).await;
	history(&chat).await;
	chat.script.push_summary(answer("SUMMARY"));
	chat.script.push_agent(failure(OVERFLOW));
	let input = chat.root.submit(input(text("u4", 200)), &context()).await.unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().reason.as_deref(), Some("model_error"));
	assert_eq!(chat.script.requests(true).len(), 1);
	assert_eq!(kinds(&chat.root).await.iter().filter(|kind| kind.as_str() == "pi.compaction").count(), 1);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn blocking_threshold_failure_or_decline_still_sends_generation_request() {
	for decline in [false, true] {
		let chat = open_chat(1000, policy(true, 300, 150, 0)).await;
		history(&chat).await;
		if decline {
			install_compaction_hook(&chat.registry, |_| Ok(Some(BeforeCompactDecision::Decline)));
		} else {
			chat.script.push_summary(failure("bad request"));
		}
		turn(&chat, text("u4", 200), "a4").await;
		assert!(!kinds(&chat.root).await.iter().any(|kind| kind == "pi.compaction"));
		assert_eq!(message_text(&chat.script.requests(false).last().unwrap().messages[0]), text("u1", 100));
		chat.harness.close(&context()).await.unwrap();
	}
}

#[tokio::test]
async fn overflow_compaction_retries_in_same_generation_without_the_error_entry() {
	let chat = open_chat(WINDOW, policy(true, 1000, 150, 0)).await;
	history(&chat).await;
	chat.script.push_summary(answer("SUMMARY"));
	chat.script.push_agent(failure(OVERFLOW));
	chat.script.push_agent(answer("fits"));
	let input = chat.root.submit(input(text("u4", 100)), &context()).await.unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	let kinds = kinds(&chat.root).await;
	assert!(kinds.iter().any(|kind| kind == "pi.compaction"));
	let request = chat.script.requests(false).last().unwrap().messages.clone();
	assert!(message_text(&request[0]).contains("SUMMARY"));
	assert!(request.iter().all(|message| !matches!(message, Message::Assistant(answer) if answer.stop_reason == StopReason::Error)));
	let marker = chat.root.context(&context()).await.unwrap().entries.into_iter().find(|entry| entry.kind == "pi.compaction").unwrap();
	assert_eq!(marker.data.unwrap()["reason"], "overflow");
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn overflow_compaction_fails_after_second_overflow_and_is_skipped_when_disabled() {
	let enabled = open_chat(WINDOW, policy(true, 1000, 150, 0)).await;
	history(&enabled).await;
	enabled.script.push_summary(answer("SUMMARY"));
	enabled.script.push_agent(failure(OVERFLOW));
	enabled.script.push_agent(failure(OVERFLOW));
	let enabled_input = enabled.root.submit(input(text("u4", 100)), &context()).await.unwrap();
	let failed = enabled_input.wait(&context()).await.unwrap();
	assert_eq!(failed.status, SubmissionStatus::Unanswered);
	assert_eq!(failed.reason.as_deref(), Some("model_error"));
	assert_eq!(failed.detail.as_ref().and_then(|detail| detail.as_str()), Some(OVERFLOW));
	assert_eq!(kinds(&enabled.root).await.iter().filter(|kind| *kind == "pi.compaction").count(), 1);
	enabled.harness.close(&context()).await.unwrap();

	let disabled = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&disabled).await;
	for _ in 0..4 { disabled.script.push_agent(failure(format!("overloaded: {OVERFLOW}"))); }
	let input_handle = disabled.root.submit(input(text("u4", 100)), &context()).await.unwrap();
	assert_eq!(input_handle.wait(&context()).await.unwrap().status, SubmissionStatus::Unanswered);
	assert_eq!(disabled.script.requests(false).len(), 4);
	assert!(disabled.script.requests(true).is_empty());
	disabled.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn second_compaction_summarizes_the_prior_summary_and_summary_is_pinned_to_model() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	chat.script.push_summary(answer("FIRST"));
	assert_eq!(manual_result(&chat, None).await.status, TaskOutcomeStatus::Completed);
	turn(&chat, text("u4", 100), text("a4", 100)).await;
	turn(&chat, text("u5", 100), text("a5", 100)).await;
	chat.script.push_summary(answer("SECOND"));
	assert_eq!(manual_result(&chat, None).await.status, TaskOutcomeStatus::Completed);
	let prompt = message_text(&chat.script.requests(true)[1].messages[1]);
	assert!(prompt.contains("FIRST"));
	assert!(message_text(&chat.root.context(&context()).await.unwrap().messages[0]).contains("SECOND"));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn busy_compaction_waits_for_boundary_and_queued_summary_survives_abort() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (answer_release, answer_reached) = chat.script.gate_agent(answer("late answer"));
	let input = chat.root.submit(input("busy"), &context()).await.unwrap();
	answer_reached.await.unwrap();
	chat.script.push_summary(answer("SUMMARY"));
	let compact = chat.root.compact(None, &context()).await.unwrap();
	let task = outcome(&chat, compact).await;
	let queued = submission(&chat, compaction_submission(&task)).await;
	assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	chat.root.abort(&context(), None).await.unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().reason.as_deref(), Some("aborted"));
	assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	queued.abort(&context()).await.unwrap();
	turn(&chat, "next", "ok").await;
	assert_eq!(queued.status(&context()).await.unwrap().reason.as_deref(), Some("aborted"));
	assert!(!kinds(&chat.root).await.iter().any(|kind| kind == "pi.compaction"));
	drop(answer_release);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn fork_compaction_writes_only_to_fork_and_fork_reset_makes_work_stale() {
	use crate::harness::types::ConversationCreateOptions;
	use crate::types::ConversationOwnership;
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let entries = chat.root.context(&context()).await.unwrap().entries;
	let fork = Arc::new(chat.root.fork(entries.last().unwrap().id, ConversationCreateOptions { ownership: ConversationOwnership::Ownerless, agent: None, init: None }, &context()).await.unwrap());
	chat.script.push_summary(answer("SUMMARY"));
	let compact = fork.compact(None, &context()).await.unwrap();
	let result = outcome(&chat, compact).await;
	let placed = submission(&chat, compaction_submission(&result)).await.wait(&context()).await.unwrap();
	assert_eq!(placed.status, SubmissionStatus::Done);
	assert!(fork.context(&context()).await.unwrap().head.as_ref().is_some_and(|head| head.kind == "pi.compaction"));
	assert!(!kinds(&chat.root).await.iter().any(|kind| kind == "pi.compaction"));

	let (release, reached) = chat.script.gate_summary(answer("LATE"));
	// Upstream "settles stale in a fork reset while it summarizes" forks
	// fresh from the parent: the already-compacted fork above has no cut
	// left, so its second compaction would request nothing.
	let fork = Arc::new(chat.root.fork(entries.last().unwrap().id, ConversationCreateOptions { ownership: ConversationOwnership::Ownerless, agent: None, init: None }, &context()).await.unwrap());
	let compact = fork.compact(None, &context()).await.unwrap();
	reached.await.unwrap();
	fork.reset(None, &context()).await.unwrap();
	release.send(()).unwrap();
	let result = outcome(&chat, compact).await;
	assert_eq!(submission(&chat, compaction_submission(&result)).await.wait(&context()).await.unwrap().reason.as_deref(), Some("stale"));
	assert_eq!(fork.context(&context()).await.unwrap().head.as_ref().unwrap().kind, "pi.reset");
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn application_replacement_is_summarized_and_hook_receives_replaced_contribution() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let first = chat.root.context(&context()).await.unwrap().entries[0].clone();
	chat.root.submit(SubmissionDraft::Write(WriteSubmissionDraft {
		request_id: None,
		entry: EntryDraft {
			kind: Some("app.redact".into()),
			edits: Some(vec![crate::types::ContextEdit {
				target: first.id,
				action: crate::types::ContextEditAction::Replace,
				messages: Some(vec![user_message("REDACTED")]),
			}]),
			..Default::default()
		},
	}), &context()).await.unwrap().wait(&context()).await.unwrap();
	chat.script.push_summary(answer("SUMMARY"));
	assert_eq!(manual_result(&chat, None).await.status, TaskOutcomeStatus::Completed);
	assert!(message_text(&chat.script.requests(true)[0].messages[1]).contains("[User]: REDACTED"));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn late_summary_is_stale_when_newer_compaction_already_cut_further() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (release, reached) = chat.script.gate_summary(answer("FIRST"));
	let early = chat.root.compact(None, &context()).await.unwrap();
	reached.await.unwrap();
	turn(&chat, text("u4", 100), text("a4", 100)).await;
	chat.script.push_summary(answer("SECOND"));
	let later = chat.root.compact(None, &context()).await.unwrap();
	assert_eq!(outcome(&chat, later).await.status, TaskOutcomeStatus::Completed);
	release.send(()).unwrap();
	let early_outcome = outcome(&chat, early).await;
	assert_eq!(submission(&chat, compaction_submission(&early_outcome)).await.status(&context()).await.unwrap().reason.as_deref(), Some("stale"));
	assert!(message_text(&chat.root.context(&context()).await.unwrap().messages[0]).contains("SECOND"));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn compaction_reopens_without_repeating_a_placed_summary() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("compaction.sqlite");
	let open = || async {
		Arc::new(crate::storage::sqlite::SqliteStorage::open_path(
			&path,
			&crate::storage::sqlite::NodeSqliteStorageOptions::default(),
		).await.expect("open sqlite storage")) as Arc<dyn Storage>
	};
	let chat = open_chat_on(open().await, WINDOW, policy(false, 1000, 150, 0), 1, None, None).await;
	history(&chat).await;
	chat.script.push_summary(answer("SUMMARY"));
	let id = chat.root.compact(None, &context()).await.unwrap();
	let result = outcome(&chat, id).await;
	let placed = submission(&chat, compaction_submission(&result)).await.wait(&context()).await.unwrap();
	assert_eq!(placed.status, SubmissionStatus::Done);
	let entries = chat.root.context(&context()).await.unwrap().entries;
	let usage_before = usage(&chat).await;
	chat.harness.close(&context()).await.unwrap();
	let reopened = open_harness(
		open().await,
		HarnessOptions {
			models: {
				let models = pi_ai::create_models(None);
				models.set_provider(chat.faux.provider.clone());
				models
			},
			registry: chat.registry.clone(),
			settings: Some(HarnessSettings {
				compaction: Some(policy(false, 1000, 150, 0)),
				retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(1), ..Default::default() }),
				..Default::default()
			}),
			..Default::default()
		},
		&context(),
	).await.unwrap();
	reopened.resume();
	let root = Arc::new(reopened.conversation(chat.root.id, &context()).await.unwrap().unwrap());
	reopened.wait_for_idle(&context()).await.unwrap();
	assert_eq!(chat.script.requests(true).len(), 1);
	assert_eq!(root.context(&context()).await.unwrap().entries, entries);
	let usage_after = reopened.session.snapshot(&crate::harness::usage::usage_doc(), Some(root.id.0), None, &context()).await.unwrap().unwrap();
	assert_eq!(Value::Object((*usage_after).clone()), usage_before);
	let inspection = reopened.inspect(&context()).await.unwrap();
	assert!(inspection.tasks.is_empty());
	assert!(inspection.submissions.is_empty());
	reopened.close(&context()).await.unwrap();
}

#[tokio::test]
async fn compaction_retry_checkpoint_pins_model_and_advances_attempt() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (release, reached) = chat.script.gate_summary(failure("overloaded"));
	chat.script.push_summary(answer("SUMMARY"));
	let id = chat.root.compact(None, &context()).await.unwrap();
	reached.await.unwrap();
	chat.root.configure(
		AgentChange { model: Some(SetOrClear::Set(ModelRef { provider: "faux".into(), model_id: "faux-2".into() })), ..Default::default() },
		&context(),
	).await.unwrap();
	release.send(()).unwrap();
	let result = outcome(&chat, id).await;
	assert_eq!(result.status, TaskOutcomeStatus::Completed);
	let requests = chat.script.requests(true);
	assert_eq!(requests.iter().map(|request| request.model.as_str()).collect::<Vec<_>>(), ["faux-1", "faux-1"]);
	assert_eq!(chat.root.agent(&context()).await.unwrap().model.unwrap().model_id, "faux-2");
	assert_eq!(live(&chat).await.compactions, None);
	chat.harness.close(&context()).await.unwrap();
}
 
#[tokio::test]
async fn reset_after_summary_queue_order_determines_staleness() {
	for summary_first in [true, false] {
		let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
		history(&chat).await;
		let (release, reached) = chat.script.gate_agent(answer("done"));
		let busy = chat.root.submit(input("busy"), &context()).await.unwrap();
		reached.await.unwrap();
		let compact_submission = if summary_first {
			chat.script.push_summary(answer("SUMMARY"));
			let task = chat.root.compact(None, &context()).await.unwrap();
			let compact = outcome(&chat, task).await;
			let submission = submission(&chat, compaction_submission(&compact)).await;
			chat.root.reset(None, &context()).await.unwrap();
			submission
		} else {
			chat.root.reset(None, &context()).await.unwrap();
			chat.script.push_summary(answer("SUMMARY"));
			let task = chat.root.compact(None, &context()).await.unwrap();
			let compact = outcome(&chat, task).await;
			submission(&chat, compaction_submission(&compact)).await
		};
		release.send(()).unwrap();
		busy.wait(&context()).await.unwrap();
		let settlement = compact_submission.wait(&context()).await.unwrap();
		assert_eq!(settlement.status, if summary_first { SubmissionStatus::Done } else { SubmissionStatus::Unanswered });
		assert_eq!(kinds(&chat.root).await.last().map(String::as_str), Some("pi.reset"));
		chat.harness.close(&context()).await.unwrap();
	}
}


#[tokio::test]
async fn live_compaction_status_exposes_retry_and_late_joiner_state() {
	let chat = open_chat_with_retry(WINDOW, policy(false, 1000, 150, 0), 60_000, None).await;
	history(&chat).await;
	chat.script.push_summary(failure("overloaded"));
	let id = chat.root.compact(None, &context()).await.unwrap();
	tokio::time::timeout(std::time::Duration::from_secs(2), async {
		loop {
			if live(&chat).await.compactions.as_ref().is_some_and(|statuses| statuses[0].retry.is_some()) {
				break;
			}
			tokio::task::yield_now().await;
		}
	}).await.expect("retry backoff becomes observable");
	let status = live(&chat).await.compactions.unwrap().remove(0);
	let late = crate::harness::events::watch_events(&chat.harness, chat.root.id, &context()).await.unwrap();
	assert_eq!(late.snapshot.compactions.len(), 1);
	assert_eq!(late.snapshot.compactions[0].task_id, id);
	assert_eq!(late.snapshot.compactions[0].attempt, 1);
	assert_eq!(late.snapshot.compactions[0].retry.as_ref().unwrap().error, "overloaded");
	assert!(late.snapshot.compactions[0].retry.as_ref().unwrap().at > 0);
	late.stop().await;
	assert_eq!(status.reason, CompactionReason::Manual);
	assert_eq!(status.retry.as_ref().unwrap().error, "overloaded");
	chat.harness.abort_task(id, &context()).await.unwrap();
	assert_eq!(outcome(&chat, id).await.status, TaskOutcomeStatus::Aborted);
	assert!(live(&chat).await.compactions.is_none());
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn silent_or_length_assistant_responses_are_not_classified_as_context_overflow() {
	for response in [
		answer("fine"),
		faux_assistant_message("", FauxMessageOptions { stop_reason: Some(StopReason::Length), ..Default::default() }),
	] {
		let chat = open_chat(300, policy(true, 0, 100_000, 0)).await;
		history(&chat).await;
		chat.script.push_agent(response);
		let input = chat.root.submit(input("go"), &context()).await.unwrap();
		assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
		assert!(chat.script.requests(true).is_empty());
		chat.harness.close(&context()).await.unwrap();
	}
}

#[tokio::test]
async fn interrupted_summarization_restarts_from_its_durable_checkpoint() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("compaction.sqlite");
	let chat = open_chat_on(open_reopen_storage(&path).await, WINDOW, policy(false, 1000, 150, 0), 1, None, None).await;
	history(&chat).await;
	let (release, reached) = chat.script.gate_summary(answer("discarded in-flight reply"));
	let task = chat.root.compact(None, &context()).await.unwrap();
	reached.await.unwrap();
	assert_eq!(chat.harness.get_task(task, &context()).await.unwrap().unwrap().state.checkpoint.as_ref().unwrap()["phase"], "summarize");
	chat.harness.close(&context()).await.unwrap();
	drop(release);

	let models = pi_ai::create_models(None);
	models.set_provider(chat.faux.provider.clone());
	let reopened = open_harness(
		open_reopen_storage(&path).await,
		HarnessOptions {
			models,
			registry: chat.registry.clone(),
			settings: Some(HarnessSettings {
				compaction: Some(policy(false, 1000, 150, 0)),
				retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(1), ..Default::default() }),
				..Default::default()
			}),
			..Default::default()
		},
		&context(),
	).await.unwrap();
	let root = Arc::new(reopened.conversation(chat.root.id, &context()).await.unwrap().unwrap());
	chat.script.push_summary(answer("SUMMARY"));
	reopened.resume();
	let settled = reopened.wait_for_task(task, &context()).await.unwrap();
	let task_outcome = settled.record.state.outcome.unwrap();
	assert_eq!(task_outcome.status, TaskOutcomeStatus::Completed);
	assert_eq!(chat.script.requests(true).len(), 2);
	let submission_id = crate::types::SubmissionId(task_outcome.result.unwrap()["submissionId"].as_u64().unwrap());
	assert_eq!(reopened.submission(submission_id, &context()).await.unwrap().unwrap().wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert!(root.context(&context()).await.unwrap().entries.iter().any(|entry| entry.kind == "pi.compaction"));
	reopened.close(&context()).await.unwrap();
}

#[tokio::test]
async fn queued_compaction_submission_survives_reopen_and_uses_next_boundary() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("compaction.sqlite");
	let chat = open_chat_on(open_reopen_storage(&path).await, WINDOW, policy(false, 1000, 150, 0), 1, None, None).await;
	history(&chat).await;
	let (release, reached) = chat.script.gate_agent(answer("interrupted run"));
	let input_handle = chat.root.submit(input("busy"), &context()).await.unwrap();
	reached.await.unwrap();
	chat.script.push_summary(answer("SUMMARY"));
	let id = chat.root.compact(None, &context()).await.unwrap();
	let result = outcome(&chat, id).await;
	let submission_id = compaction_submission(&result);
	chat.harness.close(&context()).await.unwrap();
	drop(release);

	let models = pi_ai::create_models(None);
	models.set_provider(chat.faux.provider.clone());
	let reopened = open_harness(
		open_reopen_storage(&path).await,
		HarnessOptions {
			models,
			registry: chat.registry.clone(),
			settings: Some(HarnessSettings {
				compaction: Some(policy(false, 1000, 150, 0)),
				retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(1), ..Default::default() }),
				..Default::default()
			}),
			..Default::default()
		},
		&context(),
	).await.unwrap();
	chat.script.push_agent(answer("resumed answer"));
	reopened.resume();
	let root = Arc::new(reopened.conversation(chat.root.id, &context()).await.unwrap().unwrap());
	assert_eq!(reopened.submission(input_handle.id, &context()).await.unwrap().unwrap().wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(reopened.submission(submission_id, &context()).await.unwrap().unwrap().wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert!(root.context(&context()).await.unwrap().entries.iter().any(|entry| entry.kind == "pi.compaction"));
	reopened.close(&context()).await.unwrap();
}

#[tokio::test]
async fn compaction_context_contributions_apply_prior_head_edits() {
	let chat = open_chat(WINDOW, policy(false, 1000, 1, 0)).await;
	let conversation_id = chat.root.id;
	let ids = chat.root.commit(move |tx| {
		let a = tx.append_entry(conversation_id, EntryDraft { kind: Some("app.note".into()), model: Some(vec![user_message("a")]), ..Default::default() }).map_err(|error| error.to_string())?;
		let b = tx.append_entry(conversation_id, EntryDraft { kind: Some("app.note".into()), model: Some(vec![user_message("b")]), ..Default::default() }).map_err(|error| error.to_string())?;
		tx.append_entry(conversation_id, EntryDraft {
			kind: Some("app.head".into()), head: Some(EntryHead::Entry(a.id)),
			edits: Some(vec![crate::types::ContextEdit { target: b.id, action: crate::types::ContextEditAction::Omit, messages: None }]), ..Default::default()
		}).map_err(|error| error.to_string())?;
		let c = tx.append_entry(conversation_id, EntryDraft { kind: Some("app.note".into()), model: Some(vec![user_message("c")]), ..Default::default() }).map_err(|error| error.to_string())?;
		tx.append_entry(conversation_id, EntryDraft { kind: Some("app.head".into()), head: Some(EntryHead::Entry(a.id)), model: Some(vec![user_message("H")]), ..Default::default() }).map_err(|error| error.to_string())?;
		Ok((a.id, b.id, c.id))
	}, &context()).await.unwrap();
	let view = chat.root.context(&context()).await.unwrap();
	assert_eq!(view.entries.iter().skip(1).map(|record| record.id).collect::<Vec<_>>(), [ids.0, ids.1, ids.2]);
	assert_eq!(view.messages.iter().map(message_text).collect::<Vec<_>>(), ["H", "a", "c"]);
	chat.script.push_summary(answer("SUMMARY"));
	assert_eq!(manual_result(&chat, None).await.status, TaskOutcomeStatus::Completed);
	let prompt = message_text(&chat.script.requests(true)[0].messages[1]);
	assert!(prompt.contains("[User]: H\n\n[User]: a"));
	assert!(!prompt.contains("[User]: b"));
	chat.harness.close(&context()).await.unwrap();
}


#[tokio::test]
async fn retry_backoff_waits_across_reopen_until_the_persisted_deadline() {
	let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
	let clock_fn = {
		let clock = clock.clone();
		Arc::new(move || clock.load(std::sync::atomic::Ordering::SeqCst)) as Arc<dyn Fn() -> i64 + Send + Sync>
	};
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("compaction.sqlite");
	let chat = open_chat_on(open_reopen_storage(&path).await, WINDOW, policy(false, 1000, 150, 0), 60_000, Some(clock_fn.clone()), None).await;
	history(&chat).await;
	chat.script.push_summary(failure("overloaded"));
	let id = chat.root.compact(None, &context()).await.unwrap();
	tokio::time::timeout(std::time::Duration::from_secs(2), async {
		loop {
			if live(&chat).await.compactions.as_ref().is_some_and(|statuses| statuses[0].retry.is_some()) {
				break;
			}
			tokio::task::yield_now().await;
		}
	}).await.expect("persist retry checkpoint");
	assert_eq!(chat.harness.get_task(id, &context()).await.unwrap().unwrap().state.checkpoint.as_ref().unwrap()["phase"], "retry");
	chat.harness.close(&context()).await.unwrap();
	clock.store(120_000, std::sync::atomic::Ordering::SeqCst);

	let models = pi_ai::create_models(None);
	models.set_provider(chat.faux.provider.clone());
	let reopened = open_harness(
		open_reopen_storage(&path).await,
		HarnessOptions {
			models,
			registry: chat.registry.clone(),
			now: Some(clock_fn),
			settings: Some(HarnessSettings {
				compaction: Some(policy(false, 1000, 150, 0)),
				retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(60_000), ..Default::default() }),
				..Default::default()
			}),
			..Default::default()
		},
		&context(),
	).await.unwrap();
	chat.script.push_summary(answer("SUMMARY"));
	reopened.resume();
	let settled = reopened.wait_for_task(id, &context()).await.unwrap();
	assert_eq!(settled.record.state.outcome.unwrap().status, TaskOutcomeStatus::Completed);
	assert_eq!(chat.script.requests(true).len(), 2);
	reopened.close(&context()).await.unwrap();
}

#[tokio::test]
async fn blocking_generation_and_its_compaction_resume_together_after_reopen() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("compaction.sqlite");
	let chat = open_chat_on(open_reopen_storage(&path).await, 1000, policy(true, 300, 150, 0), 1, None, None).await;
	history(&chat).await;
	let (release, reached) = chat.script.gate_summary(answer("discarded"));
	let input_handle = chat.root.submit(input(text("u4", 200)), &context()).await.unwrap();
	reached.await.unwrap();
	let live_state = live(&chat).await;
	let generation = live_state.run.unwrap().task_id;
	let child = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
		.find(|task| task.record.kind == "pi.compaction").unwrap();
	assert_eq!(child.record.owner, Some(generation));
	chat.harness.close(&context()).await.unwrap();
	drop(release);

	let models = pi_ai::create_models(None);
	models.set_provider(chat.faux.provider.clone());
	let reopened = open_harness(
		open_reopen_storage(&path).await,
		HarnessOptions {
			models,
			registry: chat.registry.clone(),
			settings: Some(HarnessSettings {
				compaction: Some(policy(true, 300, 150, 0)),
				retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(1), ..Default::default() }),
				..Default::default()
			}),
			..Default::default()
		},
		&context(),
	).await.unwrap();
	chat.script.push_summary(answer("SUMMARY"));
	chat.script.push_agent(answer("a4"));
	reopened.resume();
	let root = Arc::new(reopened.conversation(chat.root.id, &context()).await.unwrap().unwrap());
	assert_eq!(reopened.submission(input_handle.id, &context()).await.unwrap().unwrap().wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(reopened.wait_for_task(child.record.id, &context()).await.unwrap().record.state.outcome.unwrap().status, TaskOutcomeStatus::Completed);
	assert_eq!(kinds(&root).await.iter().rev().take(3).cloned().collect::<Vec<_>>(), ["pi.assistant", "pi.system", "pi.compaction"]);
	reopened.close(&context()).await.unwrap();
}

#[tokio::test]
async fn manual_compaction_does_not_start_background_work_while_listed() {
	let current_policy = Arc::new(Mutex::new(policy(false, 500, 150, 1000)));
	let provider_policy = current_policy.clone();
	let settings_provider = Arc::new(move || HarnessSettings {
		compaction: Some(provider_policy.lock().clone()),
		..Default::default()
	});
	let chat = open_chat_on(
		Arc::new(MemoryStorage::new()),
		2000,
		policy(false, 500, 150, 1000),
		1,
		None,
		Some(settings_provider),
	).await;
	history(&chat).await;
	*current_policy.lock() = policy(true, 500, 150, 1000);
	let (release, reached) = chat.script.gate_summary(answer("MANUAL"));
	let manual = chat.root.compact(None, &context()).await.unwrap();
	reached.await.unwrap();
	turn(&chat, text("u4", 100), text("a4", 100)).await;
	let tasks = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
		.filter(|task| task.record.kind == "pi.compaction").collect::<Vec<_>>();
	assert_eq!(tasks.len(), 1);
	assert_eq!(tasks[0].record.id, manual);
	assert_eq!(chat.script.requests(true).len(), 1);
	chat.harness.abort_task(manual, &context()).await.unwrap();
	assert_eq!(outcome(&chat, manual).await.status, TaskOutcomeStatus::Aborted);
	drop(release);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn summary_queued_through_escape_lands_before_the_next_input() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (run_release, run_reached) = chat.script.gate_agent(answer("never"));
	let busy = chat.root.submit(input("busy"), &context()).await.unwrap();
	run_reached.await.unwrap();
	chat.script.push_summary(answer("SUMMARY"));
	let task = chat.root.compact(None, &context()).await.unwrap();
	let compact = outcome(&chat, task).await;
	let queued = submission(&chat, compaction_submission(&compact)).await;
	assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	chat.root.abort(&context(), None).await.unwrap();
	assert_eq!(busy.wait(&context()).await.unwrap().reason.as_deref(), Some("aborted"));
	assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	turn(&chat, "u5", "a5").await;
	assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Done);
	let request = chat.script.requests(false).last().unwrap().messages.clone();
	assert!(message_text(&request[0]).contains("SUMMARY"));
	assert!(request.iter().any(|message| message_text(message) == "u5"));
	assert_eq!(kinds(&chat.root).await.iter().rev().take(4).cloned().collect::<Vec<_>>(), ["pi.assistant", "pi.system", "pi.user", "pi.compaction"]);
	drop(run_release);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn blocking_compaction_places_before_request_and_manual_summary_joins_final_boundary() {
	let current_policy = Arc::new(Mutex::new(policy(false, 300, 150, 0)));
	let provider_policy = current_policy.clone();
	let settings_provider = Arc::new(move || HarnessSettings {
		compaction: Some(provider_policy.lock().clone()),
		..Default::default()
	});
	let chat = open_chat_on(
		Arc::new(MemoryStorage::new()),
		1000,
		policy(false, 300, 150, 0),
		1,
		None,
		Some(settings_provider),
	).await;
	history(&chat).await;
	*current_policy.lock() = policy(true, 300, 150, 0);
	let (blocking_release, blocking_reached) = chat.script.gate_summary(answer("BLOCKING"));
	let input = chat.root.submit(input(text("u4", 200)), &context()).await.unwrap();
	blocking_reached.await.unwrap();
	let blocking_id = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
		.find(|task| task.record.kind == "pi.compaction").unwrap().record.id;
	chat.script.push_summary(answer("MANUAL"));
	let manual_outcome = outcome(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
	let manual_write = submission(&chat, compaction_submission(&manual_outcome)).await;
	assert_eq!(manual_write.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	chat.script.push_agent(answer("a4"));
	blocking_release.send(()).unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert!(message_text(&chat.script.requests(false).last().unwrap().messages[0]).contains("BLOCKING"));
	assert_eq!(manual_write.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	let markers = raw_entries(&chat.root).await.into_iter().filter(|entry| entry.kind == "pi.compaction").collect::<Vec<_>>();
	assert_eq!(markers.len(), 2);
	assert_eq!(markers[1].head, markers[0].head);
	let messages = chat.root.context(&context()).await.unwrap().messages;
	assert!(message_text(&messages[0]).contains("MANUAL"));
	assert!(messages.iter().all(|message| !message_text(message).contains("BLOCKING")));
	assert_eq!(outcome(&chat, blocking_id).await.status, TaskOutcomeStatus::Completed);
	chat.harness.close(&context()).await.unwrap();
}
#[tokio::test]
async fn legacy_conversation_gets_provider_state_before_compaction_request() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let token = crate::harness::provider::provider_doc();
	let conversation_id = chat.root.id;
	let retire_token = token.clone();
	let provider_context = context();
	let read_provider = || chat.harness.session.snapshot(&token, Some(conversation_id.0), None, &provider_context);
	assert!(read_provider().await.unwrap().is_some());
	chat.root.commit(move |tx| {
		tx.retire_doc(&retire_token, Some(conversation_id.0), None).map_err(|error| error.to_string())?;
		Ok(())
	}, &context()).await.expect("retire provider state");
	assert!(read_provider().await.unwrap().is_none());
	chat.script.push_summary(answer("SUMMARY"));
	assert_eq!(manual_result(&chat, None).await.status, TaskOutcomeStatus::Completed);
	let stored = read_provider().await.unwrap().expect("new provider state");
	let request = chat.script.requests(true).remove(0);
	assert_eq!(request.options.unwrap().base.session_id.as_deref(), stored["sessionId"].as_str());
	chat.harness.close(&context()).await.unwrap();
}
#[tokio::test]
async fn concurrent_manual_compactions_remain_listed_in_task_order() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (release_a, reached_a) = chat.script.gate_summary(answer("A"));
	let (release_b, reached_b) = chat.script.gate_summary(answer("B"));
	let a = chat.root.compact(None, &context()).await.unwrap();
	let b = chat.root.compact(None, &context()).await.unwrap();
	reached_a.await.unwrap();
	reached_b.await.unwrap();
	assert_eq!(live(&chat).await.compactions.unwrap().iter().map(|status| status.task_id).collect::<Vec<_>>(), [a, b]);
	chat.root.abort(&context(), None).await.unwrap();
	assert_eq!(outcome(&chat, a).await.status, TaskOutcomeStatus::Aborted);
	assert_eq!(outcome(&chat, b).await.status, TaskOutcomeStatus::Aborted);
	assert!(live(&chat).await.compactions.is_none());
	drop((release_a, release_b));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn overflow_compaction_failure_or_abort_preserves_the_generation_error() {
	for abort_summary in [false, true] {
		let chat = open_chat(WINDOW, policy(true, 1000, 150, 0)).await;
		history(&chat).await;
		let reached = if abort_summary {
			let (release, reached) = chat.script.gate_summary(answer("SUMMARY"));
			chat.script.push_agent(failure(OVERFLOW));
			let input = chat.root.submit(input(text("u4", 100)), &context()).await.unwrap();
			reached.await.unwrap();
			let task = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
				.find(|task| task.record.kind == "pi.compaction").unwrap();
			chat.harness.abort_task(task.record.id, &context()).await.unwrap();
			assert_eq!(input.wait(&context()).await.unwrap().detail.as_ref().and_then(Value::as_str), Some(OVERFLOW));
			drop(release);
			true
		} else {
			chat.script.push_summary(failure("prompt is too long for the summary"));
			chat.script.push_agent(failure(OVERFLOW));
			let input = chat.root.submit(input(text("u4", 100)), &context()).await.unwrap();
			assert_eq!(input.wait(&context()).await.unwrap().detail.as_ref().and_then(Value::as_str), Some(OVERFLOW));
			false
		};
		assert!(live(&chat).await.compactions.is_none(), "{abort_summary}");
		chat.harness.close(&context()).await.unwrap();
	}
}

#[tokio::test]
async fn queued_summary_after_a_failed_run_precedes_the_next_input() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (run_release, run_reached) = chat.script.gate_agent(failure("bad request"));
	let failed = chat.root.submit(input("fails"), &context()).await.unwrap();
	run_reached.await.unwrap();
	chat.script.push_summary(answer("SUMMARY"));
	let compact = chat.root.compact(None, &context()).await.unwrap();
	let task = outcome(&chat, compact).await;
	let queued = submission(&chat, compaction_submission(&task)).await;
	assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	run_release.send(()).unwrap();
	assert_eq!(failed.wait(&context()).await.unwrap().status, SubmissionStatus::Unanswered);
	assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	turn(&chat, "again", "ok").await;
	assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Done);
	let request = chat.script.requests(false).last().unwrap().messages.clone();
	assert!(message_text(&request[0]).contains("SUMMARY"));
	assert!(request.iter().any(|message| message_text(message) == "again"));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn application_edit_during_summarization_does_not_rewrite_the_selected_summary() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let first = chat.root.context(&context()).await.unwrap().entries[0].clone();
	let (release, reached) = chat.script.gate_summary(answer("SUMMARY"));
	let compact = chat.root.compact(None, &context()).await.unwrap();
	reached.await.unwrap();
	let edit = crate::types::ContextEdit {
		target: first.id,
		action: crate::types::ContextEditAction::Replace,
		messages: Some(vec![user_message("REDACTED")]),
	};
	let written = chat.root.submit(
		SubmissionDraft::Write(crate::harness::types::WriteSubmissionDraft {
			request_id: None,
			entry: EntryDraft { kind: Some("app.redact".into()), edits: Some(vec![edit]), ..Default::default() },
		}),
		&context(),
	).await.unwrap();
	assert_eq!(written.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	release.send(()).unwrap();
	assert_eq!(outcome(&chat, compact).await.status, TaskOutcomeStatus::Completed);
	assert!(message_text(&chat.script.requests(true)[0].messages[1]).contains("[User]: u1 "));
	assert!(!chat.root.context(&context()).await.unwrap().messages.iter().any(|message| message_text(message) == "REDACTED"));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn compacted_entries_remain_mounted_in_the_conversation_view() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let state = chat.root.view_state(&context()).await.unwrap();
	chat.script.push_summary(answer("SUMMARY"));
	assert_eq!(manual_result(&chat, None).await.status, TaskOutcomeStatus::Completed);
	let current = chat.root.context(&context()).await.unwrap();
	let expected = serde_json::to_value(current.entries).unwrap();
	tokio::time::timeout(std::time::Duration::from_secs(2), async {
		while state.value()["entries"] != expected {
			tokio::task::yield_now().await;
		}
	}).await.expect("mounted view receives compaction");
	assert_eq!(state.value()["entries"], expected);
	state.dispose();
	chat.harness.close(&context()).await.unwrap();
}
#[tokio::test]
async fn compaction_rebaselines_system_prompt_over_kept_system_deltas() {
	let chat = open_chat(WINDOW, policy(false, 1000, 250, 0)).await;
	let mood = Arc::new(Mutex::new("cheerful".to_string()));
	let rendered = mood.clone();
	let section = crate::harness::define::section(
		"mood",
		Arc::new(move |_, _| {
			let mood = rendered.lock().clone();
			Box::pin(async move { Ok(Some(mood)) })
		}),
		Some(false),
	);
	chat.registry.install(Extension::new("mood").with_sections(vec![section]));
	turn(&chat, text("u1", 100), text("a1", 100)).await;
	turn(&chat, text("u2", 100), text("a2", 100)).await;
	*mood.lock() = "terse".into();
	turn(&chat, text("u3", 100), text("a3", 100)).await;
	chat.script.push_summary(answer("SUMMARY"));
	assert_eq!(manual_result(&chat, None).await.status, TaskOutcomeStatus::Completed);
	assert!(chat.root.context(&context()).await.unwrap().entries.iter().any(|entry| entry.kind == "pi.system"));
	turn(&chat, "next", "ok").await;
	let requests = chat.script.requests(false);
	let systems = requests.last().unwrap().messages.iter()
		.filter_map(|message| match message { Message::System(system) => Some(system), _ => None })
		.collect::<Vec<_>>();
	assert_eq!(systems.len(), 1);
	assert_eq!(systems[0].sections.as_ref().unwrap()["mood"], Some("terse".to_owned()));
	chat.harness.close(&context()).await.unwrap();
}

fn compaction_event_tag(event: &crate::harness::events::AgentEvent) -> &'static str {
	use crate::harness::events::AgentEvent;
	match event {
		AgentEvent::MessageStart { .. } => "message_start",
		AgentEvent::MessageEnd { .. } => "message_end",
		AgentEvent::CompactionStart { .. } => "compaction_start",
		AgentEvent::CompactionEnd { .. } => "compaction_end",
		AgentEvent::Submission { .. } => "submission",
		AgentEvent::UsageChanged { .. } => "usage_changed",
		_ => "other",
	}
}

#[tokio::test]
async fn compaction_events_preserve_placement_and_preparation_batch_order() {
	use crate::harness::events::{watch_events, AgentEvent};
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let mut stream = watch_events(&chat.harness, chat.root.id, &context()).await.unwrap();
	assert!(matches!(stream.next().await.unwrap().unwrap().as_slice(), [AgentEvent::Snapshot { .. }]));
	chat.script.push_summary(answer("SUMMARY"));
	assert_eq!(manual_result(&chat, None).await.status, TaskOutcomeStatus::Completed);
	let mut events = Vec::new();
	loop {
		let batch = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
			.await.expect("compaction event batch").unwrap().unwrap();
		let done = batch.iter().any(|event| matches!(event, AgentEvent::CompactionEnd { .. }));
		events.extend(batch);
		if done { break; }
	}
	let tags = events.iter().map(compaction_event_tag).collect::<Vec<_>>();
	assert!(tags.windows(5).any(|window| window == ["message_start", "message_end", "compaction_end", "submission", "usage_changed"]));
	stream.stop().await;
	chat.harness.close(&context()).await.unwrap();

	let current_policy = Arc::new(Mutex::new(policy(false, 500, 150, 1000)));
	let provider_policy = current_policy.clone();
	let settings_provider = Arc::new(move || HarnessSettings {
		compaction: Some(provider_policy.lock().clone()),
		..Default::default()
	});
	let chat = open_chat_on(
		Arc::new(MemoryStorage::new()),
		2000,
		policy(false, 500, 150, 1000),
		1,
		None,
		Some(settings_provider),
	).await;
	history(&chat).await;
	*current_policy.lock() = policy(true, 500, 150, 1000);
	let section = crate::harness::define::section(
		"extra",
		Arc::new(|_, _| Box::pin(async { Ok(Some("extra".to_string())) })),
		None,
	);
	chat.registry.install(Extension::new("extra").with_sections(vec![section]));
	let mut stream = watch_events(&chat.harness, chat.root.id, &context()).await.unwrap();
	let _ = stream.next().await.unwrap().unwrap();
	let (release, reached) = chat.script.gate_summary(answer("SUMMARY"));
	turn(&chat, text("u4", 100), text("a4", 100)).await;
	reached.await.unwrap();
	let task = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
		.find(|task| task.record.kind == "pi.compaction").unwrap();
	let start_batch = loop {
		let batch = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
			.await.expect("compaction start batch").unwrap().unwrap();
		if batch.iter().any(|event| matches!(event, AgentEvent::CompactionStart { .. })) {
			break batch;
		}
	};
	assert_eq!(start_batch.iter().map(compaction_event_tag).collect::<Vec<_>>(), ["message_start", "message_end", "compaction_start"]);
	stream.stop().await;
	chat.harness.abort_task(task.record.id, &context()).await.unwrap();
	drop(release);
	chat.harness.close(&context()).await.unwrap();
}
#[tokio::test]
async fn older_summary_with_a_later_cut_can_replace_a_newer_earlier_cut() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (release_a, reached_a) = chat.script.gate_summary(answer("A"));
	let a = chat.root.compact(None, &context()).await.unwrap();
	reached_a.await.unwrap();
	let entries = chat.root.context(&context()).await.unwrap().entries;
	let omitted = entries.iter().filter(|entry| entry.model.as_ref().is_some_and(|messages| {
		messages.first().is_some_and(|message| {
			let text = message_text(message);
			text.starts_with("u3 ") || text.starts_with("a3 ")
		})
	})).map(|entry| entry.id).collect::<Vec<_>>();
	assert_eq!(omitted.len(), 2);
	let edits = omitted.into_iter().map(|target| crate::types::ContextEdit {
		target,
		action: crate::types::ContextEditAction::Omit,
		messages: None,
	}).collect();
	let written = chat.root.submit(SubmissionDraft::Write(crate::harness::types::WriteSubmissionDraft {
		request_id: None,
		entry: EntryDraft { kind: Some("app.omit".into()), edits: Some(edits), ..Default::default() },
	}), &context()).await.unwrap();
	assert_eq!(written.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	chat.script.push_summary(answer("B"));
	let b = chat.root.compact(None, &context()).await.unwrap();
	assert_eq!(outcome(&chat, b).await.status, TaskOutcomeStatus::Completed);
	release_a.send(()).unwrap();
	let first = outcome(&chat, a).await;
	assert_eq!(submission(&chat, compaction_submission(&first)).await.status(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert!(message_text(&chat.root.context(&context()).await.unwrap().messages[0]).contains("<summary>\nA\n</summary>"));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn newer_blocked_compaction_survives_reopen_and_is_orphaned_on_abort() {
	use crate::harness::events::{watch_events, AgentEvent};
	use crate::harness::types::{AnyTask, TaskInspectionState};
	use crate::types::{TaskOptions, TaskOwnership};
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("compaction.sqlite");
	let chat = open_chat_on(open_reopen_storage(&path).await, WINDOW, policy(false, 1000, 150, 0), 1, None, None).await;
	let storage = open_reopen_storage(&path).await;
	let registry = chat.registry.clone();
	let conversation_id = chat.root.id;
	let mut definition = (*crate::harness::compaction::compaction_task().definition).clone();
	definition.version = 2;
	let newer = AnyTask { definition: Arc::new(definition) };
	let id = chat.root.commit(move |tx| {
		let id = tx.create_task(
			&newer,
			json!({"reason":"manual","instructions":null}),
			TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: Some(conversation_id), background: Some(false) },
		).map_err(|error| error.to_string())?;
		tx.doc(&live_doc(), Some(conversation_id.0), None, None)
			.map_err(|error| error.to_string())?
			.set(&[pi_chord::delta::Seg::key("compactions")], json!([{
				"taskId": id, "reason": "manual", "blocking": false, "attempt": 1
			}]))
			.map_err(|error| error.to_string())?;
		Ok(id)
	}, &context()).await.unwrap();
	chat.harness.close(&context()).await.unwrap();
	let models = pi_ai::create_models(None);
	models.set_provider(chat.faux.provider.clone());
	let harness = open_harness(
		storage.clone(),
		HarnessOptions {
			models,
			registry: registry.clone(),
			settings: Some(HarnessSettings { compaction: Some(policy(false, 1000, 150, 0)), ..Default::default() }),
			..Default::default()
		},
		&context(),
	).await.unwrap();
	harness.resume();
	let root = Arc::new(harness.conversation(conversation_id, &context()).await.unwrap().unwrap());
	let reopened = Chat { storage, registry, harness: harness.clone(), root, faux: chat.faux.clone(), script: chat.script.clone() };
	let inspection = harness.inspect(&context()).await.unwrap();
	assert!(matches!(
		&inspection.tasks.iter().find(|task| task.record.id == id).unwrap().state,
		TaskInspectionState::Blocked { reason, .. } if *reason == crate::harness::types::BlockedReason::TaskTooOld
	));
	let mut stream = watch_events(&harness, conversation_id, &context()).await.unwrap();
	let _ = stream.next().await.unwrap().unwrap();
	harness.abort_task(id, &context()).await.unwrap();
	let task = harness.wait_for_task(id, &context()).await.unwrap();
	assert_eq!(task.record.state.outcome.as_ref().unwrap().status, TaskOutcomeStatus::Orphaned);
	assert_eq!(task.record.state.outcome.as_ref().unwrap().reason.as_deref(), Some("task_too_old"));
	assert!(live(&reopened).await.compactions.is_none());
	let end = tokio::time::timeout(std::time::Duration::from_secs(2), async {
		loop {
			let events = stream.next().await.unwrap().unwrap();
			if events.iter().any(|event| matches!(event, AgentEvent::CompactionEnd { task_id, .. } if *task_id == id)) {
				break;
			}
		}
	}).await;
	assert!(end.is_ok(), "blocked-task cleanup emits compaction_end");
	stream.stop().await;
	harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn overflow_compaction_preserves_the_full_generation_retry_budget() {
	let chat = open_chat(WINDOW, policy(true, 1000, 150, 0)).await;
	history(&chat).await;
	chat.script.push_summary(answer("SUMMARY"));
	chat.script.push_agent(failure("prompt is too long"));
	chat.script.push_agent(failure("overloaded"));
	chat.script.push_agent(failure("overloaded"));
	chat.script.push_agent(answer("finally"));
	let input = chat.root.submit(input(text("u4", 100)), &context()).await.unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(chat.script.requests(false).len(), 7);
	assert_eq!(chat.script.requests(true).len(), 1);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn a_blocking_cut_supersedes_a_background_summary_still_in_flight() {
	let current_policy = Arc::new(Mutex::new(policy(false, 500, 150, 1000)));
	let provider_policy = current_policy.clone();
	let settings_provider = Arc::new(move || HarnessSettings {
		retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(1), ..Default::default() }),
		compaction: Some(provider_policy.lock().clone()),
		..Default::default()
	});
	let chat = open_chat_on(
		Arc::new(MemoryStorage::new()),
		2000,
		policy(false, 500, 150, 1000),
		1,
		None,
		Some(settings_provider),
	).await;
	history(&chat).await;
	*current_policy.lock() = policy(true, 500, 150, 1000);
	let (release_background, reached_background) = chat.script.gate_summary(answer("BACKGROUND"));
	turn(&chat, text("u4", 100), text("a4", 100)).await;
	reached_background.await.unwrap();
	let background = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
		.find(|task| task.record.kind == "pi.compaction").unwrap();
	chat.script.push_summary(answer("BLOCKING"));
	turn(&chat, text("u5", 1000), "a5").await;
	assert!(message_text(&chat.root.context(&context()).await.unwrap().messages[0]).contains("BLOCKING"));
	release_background.send(()).unwrap();
	let outcome = outcome(&chat, background.record.id).await;
	assert_eq!(outcome.status, TaskOutcomeStatus::Completed);
	assert_eq!(
		submission(&chat, compaction_submission(&outcome)).await.status(&context()).await.unwrap().reason.as_deref(),
		Some("stale"),
	);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn a_declined_blocking_cut_is_not_repeated_during_generation_retry() {
	use std::sync::atomic::{AtomicUsize, Ordering};
	let chat = open_chat_with_retry(1000, policy(true, 300, 150, 0), 1, None).await;
	let asked = Arc::new(AtomicUsize::new(0));
	let count = asked.clone();
	install_compaction_hook(&chat.registry, move |_| {
		count.fetch_add(1, Ordering::SeqCst);
		Ok(Some(BeforeCompactDecision::Decline))
	});
	history(&chat).await;
	chat.script.push_agent(failure("overloaded"));
	chat.script.push_agent(answer("a4"));
	let input = chat.root.submit(input(text("u4", 200)), &context()).await.unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(asked.load(Ordering::SeqCst), 1);
	assert_eq!(chat.script.requests(false).len(), 5);
	assert!(chat.script.requests(true).is_empty());
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn manual_compaction_after_blocking_summary_finds_no_old_prefix() {
	let chat = open_chat(1000, policy(true, 300, 150, 0)).await;
	history(&chat).await;
	let (blocking_release, blocking_reached) = chat.script.gate_summary(answer("BLOCKING"));
	let input = chat.root.submit(input(text("u4", 200)), &context()).await.unwrap();
	blocking_reached.await.unwrap();
	let (answer_release, answer_reached) = chat.script.gate_agent(answer("a4"));
	blocking_release.send(()).unwrap();
	answer_reached.await.unwrap();
	assert_eq!(manual_result(&chat, None).await.result, Some(json!({})));
	assert_eq!(chat.script.requests(true).len(), 1);
	answer_release.send(()).unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn escape_aborts_the_blocking_run_and_both_compactions_without_placement() {
	let chat = open_chat(1000, policy(true, 300, 150, 0)).await;
	history(&chat).await;
	let (blocking_release, blocking_reached) = chat.script.gate_summary(answer("BLOCKING"));
	let input = chat.root.submit(input(text("u4", 200)), &context()).await.unwrap();
	blocking_reached.await.unwrap();
	let blocking_id = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
		.find(|task| task.record.kind == "pi.compaction").unwrap().record.id;
	let (manual_release, manual_reached) = chat.script.gate_summary(answer("MANUAL"));
	let manual_id = chat.root.compact(None, &context()).await.unwrap();
	manual_reached.await.unwrap();
	assert_eq!(live(&chat).await.compactions.unwrap().iter().map(|status| status.blocking).collect::<Vec<_>>(), [true, false]);
	chat.root.abort(&context(), None).await.unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().reason.as_deref(), Some("aborted"));
	assert_eq!(outcome(&chat, blocking_id).await.status, TaskOutcomeStatus::Aborted);
	assert_eq!(outcome(&chat, manual_id).await.status, TaskOutcomeStatus::Aborted);
	assert!(live(&chat).await.compactions.is_none());
	assert!(!kinds(&chat.root).await.iter().any(|kind| kind == "pi.compaction"));
	assert!(chat.harness.inspect(&context()).await.unwrap().submissions.is_empty());
	drop((blocking_release, manual_release));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn manual_compaction_admitted_during_preparation_suppresses_background_work() {
	use std::sync::atomic::{AtomicBool, Ordering};
	let current_policy = Arc::new(Mutex::new(policy(false, 500, 150, 1000)));
	let provider_policy = current_policy.clone();
	let settings_provider = Arc::new(move || HarnessSettings {
		compaction: Some(provider_policy.lock().clone()),
		..Default::default()
	});
	let chat = open_chat_on(
		Arc::new(MemoryStorage::new()),
		2000,
		policy(false, 500, 150, 1000),
		1,
		None,
		Some(settings_provider),
	).await;
	history(&chat).await;
	*current_policy.lock() = policy(true, 500, 150, 1000);
	let (section_release, release_rx) = tokio::sync::oneshot::channel();
	let (section_reached, reached_rx) = tokio::sync::oneshot::channel();
	let release = Arc::new(Mutex::new(Some(release_rx)));
	let reached = Arc::new(Mutex::new(Some(section_reached)));
	let first = Arc::new(AtomicBool::new(true));
	let release_hook = release.clone();
	let reached_hook = reached.clone();
	let first_hook = first.clone();
	let section = crate::harness::define::section(
		"slow",
		Arc::new(move |_, _| {
			let wait = if first_hook.swap(false, Ordering::SeqCst) {
				if let Some(sender) = reached_hook.lock().take() { let _ = sender.send(()); }
				release_hook.lock().take()
			} else {
				None
			};
			Box::pin(async move {
				if let Some(wait) = wait { let _ = wait.await; }
				Ok(Some("slow".to_string()))
			})
		}),
		None,
	);
	chat.registry.install(Extension::new("slow-section").with_sections(vec![section]));
	let (summary_release, summary_reached) = chat.script.gate_summary(answer("MANUAL"));
	chat.script.push_agent(answer("a4"));
	let input = chat.root.submit(input(text("u4", 100)), &context()).await.unwrap();
	reached_rx.await.unwrap();
	let manual = chat.root.compact(None, &context()).await.unwrap();
	summary_reached.await.unwrap();
	section_release.send(()).unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	let tasks = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
		.filter(|task| task.record.kind == "pi.compaction").map(|task| task.record.id).collect::<Vec<_>>();
	assert_eq!(tasks, [manual]);
	assert_eq!(chat.script.requests(true).len(), 1);
	chat.harness.abort_task(manual, &context()).await.unwrap();
	assert_eq!(outcome(&chat, manual).await.status, TaskOutcomeStatus::Aborted);
	drop(summary_release);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn pre_summary_tool_usage_is_not_reused_after_mid_run_compaction() {
	use crate::harness::types::{ToolExecutionResult, ToolRegistration, ToolsChange};
	for fixed_clock in [false, true] {
		let now = fixed_clock.then(|| Arc::new(|| 1_000_i64) as Arc<dyn Fn() -> i64 + Send + Sync>);
		// Upstream enables the background policy after history: the three
		// turns stay under the background threshold until then.
		let current_policy = Arc::new(Mutex::new(policy(false, 500, 150, 600)));
		let provider_policy = current_policy.clone();
		let settings_provider = Arc::new(move || HarnessSettings {
			compaction: Some(provider_policy.lock().clone()),
			..Default::default()
		});
		let chat = open_chat_on(
			Arc::new(MemoryStorage::new()),
			2000,
			policy(false, 500, 150, 600),
			1,
			now,
			Some(settings_provider),
		).await;
		let (tool_release_tx, tool_wait) = tokio::sync::oneshot::channel();
		let (tool_reached, reached_wait) = tokio::sync::oneshot::channel();
		let tool_release = Arc::new(Mutex::new(Some(tool_wait)));
		let tool_reached = Arc::new(Mutex::new(Some(tool_reached)));
		let release_hook = tool_release.clone();
		let reached_hook = tool_reached.clone();
		let tool = crate::harness::define::define_tool_of(ToolRegistration {
			tool: pi_ai::types::Tool {
				name: "slow".into(),
				description: "slow".into(),
				parameters: json!({"type":"object","properties":{}}),
				constrained_sampling: None,
			},
			replay: None,
			execution_mode: None,
			prepare_arguments: None,
			output_limits: None,
			execute: Arc::new(move |_, _, _| {
				let release = release_hook.lock().take();
				if let Some(sender) = tool_reached.lock().take() { let _ = sender.send(()); }
				Box::pin(async move {
					if let Some(release) = release { let _ = release.await; }
					ToolExecutionResult {
						content: Some(serde_json::to_value(vec![UserContent::Text(TextContent::new(text("result", 200)))]).unwrap()),
						..Default::default()
					}
				})
			}),
		});
		let offered_tool = tool.clone();
		chat.registry.install(Extension::new("slow-tool").with_tools(vec![Arc::new(tool)]));
		chat.root.configure(
			AgentChange { tools: Some(SetOrClear::Set(ToolsChange::List(vec![offered_tool]))), ..Default::default() },
			&context(),
		).await.unwrap();
		for index in 1..=3 {
			turn(&chat, text(&format!("u{index}"), 220), text(&format!("a{index}"), 220)).await;
		}
		*current_policy.lock() = policy(true, 500, 150, 600);
		let call = faux_assistant_message(
			FauxContentBlock::from(faux_tool_call("slow", json!({}).as_object().cloned().unwrap_or_default(), Some("slow-call"))),
			FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
		);
		chat.script.push_agent(call);
		chat.script.push_agent(answer("done"));
		let (summary_release, summary_reached) = chat.script.gate_summary(answer("SUMMARY"));
		let input = chat.root.submit(input(text("u4", 50)), &context()).await.unwrap();
		let (tool_started, summary_started) = tokio::time::timeout(std::time::Duration::from_secs(3), async {
			tokio::join!(reached_wait, summary_reached)
		}).await.expect("tool and threshold summarization start");
		tool_started.expect("tool starts");
		summary_started.expect("threshold summarization starts");
		let compaction = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
			.find(|task| task.record.kind == "pi.compaction").unwrap();
		assert!(compaction.record.background);
		summary_release.send(()).unwrap();
		let result = outcome(&chat, compaction.record.id).await;
		let queued = submission(&chat, compaction_submission(&result)).await;
		assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
		tool_release_tx.send(()).unwrap();
		assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
		assert_eq!(chat.script.requests(true).len(), 1);
		assert!(message_text(&chat.script.requests(false).last().unwrap().messages[0]).contains("SUMMARY"));
		assert!(chat.harness.inspect(&context()).await.unwrap().tasks.iter().all(|task| task.record.kind != "pi.compaction" || task.record.id == compaction.record.id));
		chat.harness.close(&context()).await.unwrap();
	}
}

#[tokio::test]
async fn current_and_older_queued_summaries_are_both_drained_at_the_next_boundary() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (run_release, run_reached) = chat.script.gate_agent(failure("bad request"));
	let failed = chat.root.submit(input("fails"), &context()).await.unwrap();
	run_reached.await.unwrap();
	chat.script.push_summary(answer("OLDER"));
	let older_task = chat.root.compact(None, &context()).await.unwrap();
	let older_result = outcome(&chat, older_task).await;
	let older = submission(&chat, compaction_submission(&older_result)).await;
	run_release.send(()).unwrap();
	assert_eq!(failed.wait(&context()).await.unwrap().status, SubmissionStatus::Unanswered);
	assert_eq!(older.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	chat.script.push_summary(answer("CURRENT"));
	let current_task = chat.root.compact(None, &context()).await.unwrap();
	let current_result = outcome(&chat, current_task).await;
	let current = submission(&chat, compaction_submission(&current_result)).await;
	assert_eq!(older.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(current.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert!(message_text(&chat.root.context(&context()).await.unwrap().messages[0]).contains("CURRENT"));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn hook_summary_waits_for_owned_child_work_after_its_marker_is_placed() {
	use crate::harness::types::{AnyTask, HookApi};
	use crate::tasks::{NextTaskState, PhaseHandler, TaskDefinition, TaskRuntime, commit_change};
	use crate::types::{TaskOptions, TaskOwnership};
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	let child_gate = Arc::new(tokio::sync::Semaphore::new(0));
	let (child_started, mut child_started_rx) = tokio::sync::oneshot::channel();
	let child_started = Arc::new(Mutex::new(Some(child_started)));
	let child_phase: PhaseHandler = {
		let gate = child_gate.clone();
		let started = child_started.clone();
		Arc::new(move |record, runtime: Arc<dyn TaskRuntime>, task_context| {
			let gate = gate.clone();
			let started = started.clone();
			Box::pin(async move {
				if let Some(sender) = started.lock().take() { let _ = sender.send(record.id); }
				let _permit = gate.acquire().await.map_err(|error| error.to_string())?;
				runtime.commit(
					commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
					&task_context,
				).await?;
				Ok(())
			})
		})
	};
	let child_abort: PhaseHandler = Arc::new(|_, runtime: Arc<dyn TaskRuntime>, task_context| {
		Box::pin(async move {
			runtime.commit(
				commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("aborted")))),
				&task_context,
			).await?;
			Ok(())
		})
	});
	let child = AnyTask::new(TaskDefinition {
		name: "test.compaction-child".into(),
		version: 1,
		initial: Arc::new(|_| json!({"phase":"run"})),
		phases: Arc::new(std::collections::BTreeMap::from([("run".to_string(), child_phase)])),
		abort: child_abort,
		migrate: None,
		hooks: None,
	});
	let root = chat.root.clone();
	let child_for_hook = child.clone();
	let before = Arc::new(move |_payload: BeforeCompactPayload, api: Arc<dyn HookApi>, hook_context: pi_chord::context::Context| {
		let root = root.clone();
		let child = child_for_hook.clone();
		Box::pin(async move {
			let parent = api.task_id();
			let conversation_id = root.id;
			root.commit(move |tx| {
				tx.create_task(
					&child,
					Value::Null,
					TaskOptions {
						ownership: TaskOwnership::Task { task_id: parent },
						conversation_id: Some(conversation_id),
						background: Some(false),
					},
				).map_err(|error| error.to_string())?;
				Ok(())
			}, &hook_context).await?;
			Ok(Some(BeforeCompactDecision::Summary("HOOK".into())))
		}) as crate::tasks::BoxFuture<Result<Option<BeforeCompactDecision>, String>>
	});
	chat.registry.install(Extension::new("compaction-child")
		.with_tasks(vec![child])
		.with_hooks(vec![HookRegistration {
			task: "pi.compaction".into(),
			handlers: Arc::new(CompactionHooks { before_compact: Some(before) }),
		}]));
	history(&chat).await;
	let compaction = chat.root.compact(None, &context()).await.unwrap();
	let child_id = tokio::time::timeout(std::time::Duration::from_secs(2), &mut child_started_rx)
		.await.expect("hook child starts").unwrap();
	tokio::time::timeout(std::time::Duration::from_secs(2), async {
		loop {
			if chat.harness.get_task(compaction, &context()).await.unwrap().unwrap().state.status == crate::types::TaskStateStatus::Completing {
				break;
			}
			tokio::task::yield_now().await;
		}
	}).await.expect("summary waits on owned child");
	assert_eq!(chat.harness.get_task(child_id, &context()).await.unwrap().unwrap().state.status, crate::types::TaskStateStatus::Running);
	assert_eq!(kinds(&chat.root).await.last().map(String::as_str), Some("pi.compaction"));
	assert!(live(&chat).await.compactions.is_none());
	child_gate.add_permits(1);
	assert_eq!(outcome(&chat, compaction).await.status, TaskOutcomeStatus::Completed);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn compaction_classification_commit_rejection_is_atomic() {
	use crate::session::support::ControlledStorage;
	let storage = Arc::new(ControlledStorage::new());
	let chat = open_chat_on(storage.clone(), WINDOW, policy(false, 1000, 150, 0), 1, None, None).await;
	history(&chat).await;
	let usage_before = usage(&chat).await;
	let fail_storage = storage.clone();
	chat.script.push_summary_action(answer("SUMMARY"), move || fail_storage.fail_next_commit("rejected"));
	let id = chat.root.compact(None, &context()).await.unwrap();
	assert_eq!(outcome(&chat, id).await.status, TaskOutcomeStatus::Faulted);
	assert_eq!(usage(&chat).await, usage_before);
	assert!(!kinds(&chat.root).await.iter().any(|kind| kind == "pi.compaction"));
	assert!(chat.harness.inspect(&context()).await.unwrap().submissions.is_empty());
	assert!(storage.submission_by_request(chat.root.id, &format!("compaction:{id}"), &context()).await.is_none());
	assert!(live(&chat).await.compactions.is_none());
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn faulted_compaction_clears_status_and_unblocks_a_blocking_generation() {
	use crate::session::support::ControlledStorage;
	let storage = Arc::new(ControlledStorage::new());
	let chat = open_chat_on(storage.clone(), 1000, policy(true, 300, 150, 0), 1, None, None).await;
	history(&chat).await;
	let fail_storage = storage.clone();
	chat.script.push_summary_action(answer("SUMMARY"), move || fail_storage.fail_next_commit("classification rejected"));
	chat.script.push_agent(answer("a4"));
	let input = chat.root.submit(input(text("u4", 200)), &context()).await.unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	// Terminal generation-owned tasks leave the scheduler's live map (upstream
	// inspect semantics), so read the durable record from storage.
	let task = chat.storage
		.scan_tasks(
			&crate::types::TaskQuery { conversation_id: Some(chat.root.id), ..Default::default() },
			100,
			None,
			&context(),
		)
		.await
		.items
		.into_iter()
		.find(|task| task.kind == "pi.compaction")
		.expect("faulted compaction recorded");
	assert_eq!(task.state.outcome.as_ref().unwrap().status, TaskOutcomeStatus::Faulted);
	assert!(live(&chat).await.compactions.is_none());
	assert!(!kinds(&chat.root).await.iter().any(|kind| kind == "pi.compaction"));
	assert_eq!(chat.script.requests(true).len(), 1);
	assert_eq!(chat.script.requests(false).len(), 4);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn overflow_waiting_compaction_failure_after_reopen_keeps_the_original_error() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("compaction.sqlite");
	let chat = open_chat_on(open_reopen_storage(&path).await, WINDOW, policy(true, 1000, 150, 0), 1, None, None).await;
	let storage = open_reopen_storage(&path).await;
	let conversation_id = chat.root.id;
	history(&chat).await;
	let (release, reached) = chat.script.gate_summary(answer("interrupted"));
	chat.script.push_agent(failure(OVERFLOW));
	let input = chat.root.submit(input(text("u4", 100)), &context()).await.unwrap();
	reached.await.unwrap();
	let generation = live(&chat).await.run.unwrap().task_id;
	let checkpoint = chat.harness.get_task(generation, &context()).await.unwrap().unwrap().state.checkpoint.unwrap();
	assert_eq!(checkpoint["phase"], "prepare");
	assert_eq!(checkpoint["overflow"], OVERFLOW);
	chat.harness.close(&context()).await.unwrap();
	drop(release);

	let models = pi_ai::create_models(None);
	models.set_provider(chat.faux.provider.clone());
	let reopened = open_harness(
		storage,
		HarnessOptions {
			models,
			registry: chat.registry.clone(),
			settings: Some(HarnessSettings {
				compaction: Some(policy(true, 1000, 150, 0)),
				retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(1), ..Default::default() }),
				..Default::default()
			}),
			..Default::default()
		},
		&context(),
	).await.unwrap();
	chat.script.push_summary(failure("bad request"));
	reopened.resume();
	let settled = reopened.submission(input.id, &context()).await.unwrap().unwrap().wait(&context()).await.unwrap();
	assert_eq!(settled.status, SubmissionStatus::Unanswered);
	assert_eq!(settled.reason.as_deref(), Some("model_error"));
	assert_eq!(settled.detail.as_ref().and_then(Value::as_str), Some(OVERFLOW));
	let root = Arc::new(reopened.conversation(conversation_id, &context()).await.unwrap().unwrap());
	assert!(!kinds(&root).await.iter().any(|kind| kind == "pi.compaction"));
	reopened.close(&context()).await.unwrap();
}

#[tokio::test]
async fn select_phase_hook_restarts_after_a_harness_close() {
	use std::sync::atomic::{AtomicUsize, Ordering};
	use crate::harness::types::HookApi;
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("compaction.sqlite");
	let chat = open_chat_on(open_reopen_storage(&path).await, WINDOW, policy(false, 1000, 150, 0), 1, None, None).await;
	history(&chat).await;
	let calls = Arc::new(AtomicUsize::new(0));
	let observed = calls.clone();
	let reached = Arc::new(tokio::sync::Notify::new());
	let reached_hook = reached.clone();
	let before = Arc::new(move |_payload: BeforeCompactPayload, _api: Arc<dyn HookApi>, hook_context: pi_chord::context::Context| {
		let calls = observed.clone();
		let reached = reached_hook.clone();
		Box::pin(async move {
			if calls.fetch_add(1, Ordering::SeqCst) == 0 {
				reached.notify_one();
				if let Some(signal) = hook_context.abort_signal() { signal.aborted_future().await; }
				return Err("interrupted select".into());
			}
			Ok(None)
		}) as crate::tasks::BoxFuture<Result<Option<BeforeCompactDecision>, String>>
	});
	chat.registry.install(Extension::new("select-recovery").with_hooks(vec![HookRegistration {
		task: "pi.compaction".into(),
		handlers: Arc::new(CompactionHooks { before_compact: Some(before) }),
	}]));
	let id = chat.root.compact(None, &context()).await.unwrap();
	let select = chat.harness.get_task(id, &context()).await.unwrap().unwrap().state.checkpoint.unwrap();
	assert_eq!(select["phase"], "select");
	chat.harness.close(&context()).await.unwrap();
	chat.script.push_summary(answer("SUMMARY"));
	let models = pi_ai::create_models(None);
	models.set_provider(chat.faux.provider.clone());
	let reopened = open_harness(
		open_reopen_storage(&path).await,
		HarnessOptions {
			models,
			registry: chat.registry.clone(),
			settings: Some(HarnessSettings {
				compaction: Some(policy(false, 1000, 150, 0)),
				retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(1), ..Default::default() }),
				..Default::default()
			}),
			..Default::default()
		},
		&context(),
	).await.unwrap();
	reopened.resume();
	assert_eq!(reopened.wait_for_task(id, &context()).await.unwrap().record.state.outcome.unwrap().status, TaskOutcomeStatus::Completed);
	assert_eq!(calls.load(Ordering::SeqCst), 2);
	assert_eq!(chat.script.requests(true).len(), 1);
	reopened.close(&context()).await.unwrap();
}

#[tokio::test]
async fn blocking_selection_uses_policy_reread_after_generation_preparation() {
	use std::sync::atomic::{AtomicBool, Ordering};
	// The "policy" section flips the settings-provider policy on its first
	// render, mid-preparation; every later cut selection, threshold or
	// overflow, must read the changed policy and find no cut, instead of a
	// phase-start snapshot.
	let current_policy = Arc::new(Mutex::new(policy(true, 300, 150, 0)));
	let provider_policy = current_policy.clone();
	let settings_provider: crate::harness::types::HarnessSettingsProvider = Arc::new(move || HarnessSettings {
		retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(1), ..Default::default() }),
		compaction: Some(provider_policy.lock().clone()),
		..Default::default()
	});
	let policy_for_section = current_policy.clone();
	let section = crate::harness::define::section(
		"policy",
		Arc::new(move |_, _| {
			*policy_for_section.lock() = policy(true, 300, 100_000, 0);
			Box::pin(async { Ok(Some("p".to_string())) })
		}),
		None,
	);
	let registry = Arc::new(create_registry());
	registry.install(Extension::new("policy").with_sections(vec![section]));
	let chat = {
		let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
		let faux = Arc::new(faux_provider(RegisterFauxProviderOptions {
			models: vec![
				FauxModelDefinition { id: "faux-1".into(), context_window: Some(1000), max_tokens: Some(900), ..Default::default() },
				FauxModelDefinition { id: "faux-2".into(), context_window: Some(1000), max_tokens: Some(900), ..Default::default() },
			],
			..Default::default()
		}));
		let models = pi_ai::create_models(None);
		models.set_provider(faux.provider.clone());
		let script = Arc::new(Script::default());
		faux.set_responses((0..512).map(|_| FauxResponseStep::Factory(reply_factory(script.clone()))).collect());
		let report_script = script.clone();
		let on_report: crate::harness::types::ReportHook = Arc::new(move |error| report_script.reports.lock().push(error));
		let harness = open_harness(
			storage.clone(),
			HarnessOptions {
				models,
				registry: registry.clone(),
				settings_provider: Some(settings_provider),
				settings: Some(HarnessSettings {
					stream: Some(ConversationStreamOptions {
						cache_retention: Some(pi_ai::types::CacheRetention::None),
						..Default::default()
					}),
					retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(1), ..Default::default() }),
					compaction: Some(policy(true, 300, 150, 0)),
					..Default::default()
				}),
				on_report: Some(on_report),
				..Default::default()
			},
			&context(),
		).await.expect("open harness");
		let root = Arc::new(root(&harness).await);
		root.configure(
			AgentChange {
				model: Some(SetOrClear::Set(ModelRef { provider: "faux".into(), model_id: "faux-1".into() })),
				..Default::default()
			},
			&context(),
		).await.expect("select faux model");
		harness.resume();
		Chat { storage, registry, harness, root, faux, script }
	};
	history(&chat).await;
	assert_eq!(current_policy.lock().keep_recent_tokens, Some(100_000), "preparations render the policy section");
	chat.script.push_agent(failure(OVERFLOW));
	let input = chat.root.submit(input(text("u4", 200)), &context()).await.unwrap();
	assert_eq!(input.wait(&context()).await.unwrap().reason.as_deref(), Some("model_error"));
	assert!(chat.script.requests(true).is_empty());
	assert!(!kinds(&chat.root).await.iter().any(|kind| kind == "pi.compaction"));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn background_summary_queued_in_retry_backoff_loses_to_a_new_blocking_cut() {
	let current_policy = Arc::new(Mutex::new(policy(false, 500, 150, 1000)));
	let provider_policy = current_policy.clone();
	let settings_provider = Arc::new(move || HarnessSettings {
		retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(300), ..Default::default() }),
		compaction: Some(provider_policy.lock().clone()),
		..Default::default()
	});
	let chat = open_chat_on(
		Arc::new(MemoryStorage::new()),
		2000,
		policy(false, 500, 150, 1000),
		300,
		None,
		Some(settings_provider),
	).await;
	history(&chat).await;
	*current_policy.lock() = policy(true, 500, 150, 1000);
	let (background_release, background_reached) = chat.script.gate_summary(answer("BACKGROUND"));
	chat.script.push_agent(failure("overloaded"));
	let input = chat.root.submit(input(text("u4", 100)), &context()).await.unwrap();
	tokio::time::timeout(std::time::Duration::from_secs(3), async {
		background_reached.await.unwrap();
		loop {
			if live(&chat).await.generation.as_ref().is_some_and(|generation| generation.retry.is_some()) {
				break;
			}
			tokio::task::yield_now().await;
		}
	}).await.expect("background summary and generation retry overlap");
	let background = chat.harness.inspect(&context()).await.unwrap().tasks.into_iter()
		.find(|task| task.record.kind == "pi.compaction").unwrap();
	assert!(background.record.background);
	background_release.send(()).unwrap();
	let result = outcome(&chat, background.record.id).await;
	let queued = submission(&chat, compaction_submission(&result)).await;
	assert_eq!(queued.status(&context()).await.unwrap().status, SubmissionStatus::Queued);
	*current_policy.lock() = policy(true, 1500, 50, 1000);
	chat.script.push_summary(answer("BLOCKING"));
	chat.script.push_agent(answer("a4"));
	assert_eq!(input.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert!(message_text(&chat.script.requests(false).last().unwrap().messages[0]).contains("BLOCKING"));
	assert_eq!(queued.wait(&context()).await.unwrap().reason.as_deref(), Some("stale"));
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn compaction_scheduling_is_enabled_immediately_after_open() {
	let chat = open_chat(WINDOW, policy(false, 1000, 10, 0)).await;
	let id = chat.root.compact(None, &context()).await.unwrap();
	let result = tokio::time::timeout(std::time::Duration::from_secs(2), outcome(&chat, id))
		.await.expect("compaction scheduled after open");
	assert_eq!(result.status, TaskOutcomeStatus::Completed);
	assert_eq!(result.result, Some(json!({})));
	assert_eq!(chat.script.requests(true).len(), 0);
	assert_eq!(chat.harness.get_task(id, &context()).await.unwrap().unwrap().state.status, crate::types::TaskStateStatus::Terminal);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn late_joiner_snapshot_lists_a_compaction_while_summarizing() {
	let chat = open_chat(WINDOW, policy(false, 1000, 150, 0)).await;
	history(&chat).await;
	let (release, reached) = chat.script.gate_summary(answer("SUMMARY"));
	let id = chat.root.compact(None, &context()).await.unwrap();
	reached.await.unwrap();
	let late = crate::harness::events::watch_events(&chat.harness, chat.root.id, &context()).await.unwrap();
	assert_eq!(late.snapshot.compactions.len(), 1);
	assert_eq!(late.snapshot.compactions[0].task_id, id);
	assert_eq!(late.snapshot.compactions[0].reason, CompactionReason::Manual);
	assert!(!late.snapshot.compactions[0].blocking);
	assert_eq!(late.snapshot.compactions[0].attempt, 1);
	assert!(late.snapshot.compactions[0].retry.is_none());
	late.stop().await;
	chat.harness.abort_task(id, &context()).await.unwrap();
	assert_eq!(outcome(&chat, id).await.status, TaskOutcomeStatus::Aborted);
	drop(release);
	chat.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn blocking_cut_does_not_start_background_compaction_later_in_same_generation() {
	use std::sync::atomic::{AtomicBool, Ordering};
	let enabled = Arc::new(AtomicBool::new(false));
	let enabled_settings = enabled.clone();
	let settings_provider = Arc::new(move || HarnessSettings {
		retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(1), ..Default::default() }),
		compaction: Some(policy(enabled_settings.load(Ordering::SeqCst), 500, 400, 1300)),
		..Default::default()
	});
	let chat = open_chat_on(
		Arc::new(MemoryStorage::new()),
		2000,
		policy(false, 500, 400, 1300),
		1,
		None,
		Some(settings_provider),
	).await;
	history(&chat).await;
	enabled.store(true, Ordering::SeqCst);
	turn(&chat, text("u4", 1000), "a4").await;
	assert_eq!(chat.script.requests(true).len(), 1);
	assert_eq!(kinds(&chat.root).await.iter().filter(|kind| kind.as_str() == "pi.compaction").count(), 1);
	chat.harness.close(&context()).await.unwrap();
}
