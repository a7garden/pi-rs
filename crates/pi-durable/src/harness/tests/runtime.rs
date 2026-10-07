use std::sync::Arc;

use pi_ai::types::{AssistantMessage, Message, StopReason, UserContentField, UserMessage};
use serde_json::json;

use super::support::{context, open_default, root};
use crate::harness::api::ToolExecutionResult;
use crate::harness::define::define_tool_of;
use crate::harness::registry::{create_registry, RegistryImpl};
use crate::harness::types::{
	AgentChange, EntryDraft, EntryHead, Extension, HarnessOptions, HarnessSettings, InputSubmissionDraft,
	ModelRef, Registry, RegistryReader, RetryPolicyOverrides, SetOrClear, Submission, SubmissionDraft,
	SubmissionStatus, ToolRegistration,
};
use crate::storage::memory::MemoryStorage;
use crate::types::{ContextEdit, ContextEditAction};

fn user(text: &str) -> Message {
	Message::User(UserMessage { content: UserContentField::Text(text.to_string()), timestamp: 0 })
}

fn assistant(text: &str, stop_reason: StopReason) -> Message {
	Message::Assistant(AssistantMessage {
		content: vec![pi_ai::types::AssistantContent::Text(pi_ai::types::TextContent::new(text))],
		api: "test".into(),
		provider: "test".into(),
		model: "test".into(),
		response_model: None,
		response_id: None,
		provider_thinking_level: None,
		thinking_level: None,
		diagnostics: None,
		usage: Default::default(),
		stop_reason,
		deferred: None,
		error_message: None,
		raw_stop_reason: None,
		end_turn: None,
		timestamp: 0,
	})
}

#[tokio::test]
async fn context_preserves_raw_entries_but_filters_non_model_and_failed_answers() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let conversation_id = conversation.id;
	let first = conversation
		.commit(move |tx| tx.append_entry(conversation_id, EntryDraft { kind: Some("message".into()), model: Some(vec![user("question")]), ..Default::default() }).map_err(|error| error.to_string()), &context())
		.await
		.expect("append input");
	let note = conversation
		.commit(move |tx| tx.append_entry(conversation_id, EntryDraft { kind: Some("note".into()), data: Some(json!({"text":"visible only"})), ..Default::default() }).map_err(|error| error.to_string()), &context())
		.await
		.expect("append note");
	conversation
		.commit(move |tx| tx.append_entry(conversation_id, EntryDraft { kind: Some("message".into()), model: Some(vec![assistant("partial", StopReason::Aborted), assistant("failed", StopReason::Error), assistant("answer", StopReason::Stop)]), ..Default::default() }).map_err(|error| error.to_string()), &context())
		.await
		.expect("append answers");
	let view = conversation.context(&context()).await.expect("derive context");
	assert_eq!(view.entries.iter().map(|entry| entry.id).collect::<Vec<_>>(), [first.id, note.id, view.entries[2].id]);
	assert_eq!(view.messages.len(), 2, "user and completed assistant are visible; note and interrupted assistant messages are excluded");
	assert!(matches!(&view.messages[0], Message::User(message) if message.content == UserContentField::Text("question".into())));
	assert!(matches!(&view.messages[1], Message::Assistant(message) if message.stop_reason == StopReason::Stop));
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn context_uses_latest_head_and_latest_edit() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let conversation_id = conversation.id;
	let original = conversation
		.commit(move |tx| tx.append_entry(conversation_id, EntryDraft { kind: Some("message".into()), model: Some(vec![user("old")]), ..Default::default() }).map_err(|error| error.to_string()), &context())
		.await
		.expect("append original");
	let conversation_id = conversation.id;
	let reset = conversation
		.commit(move |tx| tx.append_entry(conversation_id, EntryDraft { kind: Some("reset".into()), head: Some(EntryHead::SelfMarker(crate::types::SelfMarker)), model: Some(vec![user("fresh")]), ..Default::default() }).map_err(|error| error.to_string()), &context())
		.await
		.expect("append reset");
	let conversation_id = conversation.id;
	let target = conversation
		.commit(move |tx| tx.append_entry(conversation_id, EntryDraft { kind: Some("message".into()), model: Some(vec![user("replace me")]), ..Default::default() }).map_err(|error| error.to_string()), &context())
		.await
		.expect("append target");
	let conversation_id = conversation.id;
	conversation
		.commit(move |tx| {
			tx.append_entry(conversation_id, EntryDraft {
				kind: Some("edit".into()),
				edits: Some(vec![ContextEdit { target: target.id, action: ContextEditAction::Replace, messages: Some(vec![user("replacement")]) }]),
				..Default::default()
			}).map_err(|error| error.to_string())
		}, &context())
		.await
		.expect("append edit");
	let view = conversation.context(&context()).await.expect("derive context");
	assert_eq!(view.head.as_ref().map(|entry| entry.id), Some(reset.id));
	assert_eq!(view.entries.iter().map(|entry| entry.id).collect::<Vec<_>>(), [reset.id, target.id, view.entries[2].id]);
	assert_eq!(view.messages.len(), 2);
	assert!(matches!(&view.messages[0], Message::User(message) if message.content == UserContentField::Text("fresh".into())));
	assert!(matches!(&view.messages[1], Message::User(message) if message.content == UserContentField::Text("replacement".into())));
	assert_ne!(original.id, reset.id);
	harness.close(&context()).await.expect("close harness");
}

#[test]
fn registry_replacement_is_in_place_and_old_snapshots_are_stable() {
	let registry = create_registry();
	registry.install(Extension::new("a"));
	registry.install(Extension::new("b"));
	let old = registry.snapshot();
	registry.install(Extension::new("a"));
	let installed = registry.snapshot().installed();
	assert_eq!(installed.iter().map(|extension| extension.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
	assert_eq!(old.installed().iter().map(|extension| extension.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
	registry.uninstall(&Extension::new("a"));
	registry.uninstall(&Extension::new("a"));
	registry.install(Extension::new("a"));
	assert_eq!(registry.snapshot().installed().iter().map(|extension| extension.name.as_str()).collect::<Vec<_>>(), ["b", "a"]);
}

#[test]
fn registry_publishes_nothing_for_invalid_duplicate_names() {
	let registry = create_registry();
	let notifications = Arc::new(std::sync::atomic::AtomicUsize::new(0));
	let observed = notifications.clone();
	let _unsubscribe = registry.subscribe(Arc::new(move || { observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst); }));
	let before = registry.snapshot();
	let duplicate = Extension::new("bad");
	// Invalid extension installs panic before publication. A duplicate section key is a consumer-visible rejection.
	let malformed = Extension { sections: vec![
		crate::harness::define::section("same", Arc::new(|_, _| Box::pin(async { Ok(Some("one".to_string())) })), None),
		crate::harness::define::section("same", Arc::new(|_, _| Box::pin(async { Ok(Some("two".to_string())) })), None),
		], ..duplicate };
	let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| registry.install(malformed)));
	assert!(rejected.is_err());
	assert!(Arc::ptr_eq(&before, &registry.snapshot()));
	assert_eq!(notifications.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn empty_registry_harness_has_no_live_work_and_starts_paused() {
	let (_storage, _registry, harness) = open_default().await;
	let inspection = harness.inspect(&context()).await.expect("inspect harness");
	assert_eq!(inspection.scheduling, crate::harness::types::SchedulingState::Paused);
	assert!(inspection.tasks.is_empty());
	assert!(inspection.submissions.is_empty());
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn generation_without_a_model_settles_input_as_unanswered() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let submission = conversation
		.submit(
			SubmissionDraft::Input(InputSubmissionDraft {
				request_id: None,
				content: UserContentField::Text("hi".into()),
				when_busy: None,
			}),
			&context(),
		)
		.await
		.expect("submit input");
	let settled = submission.wait(&context()).await.expect("wait for generation");
	harness.wait_for_idle(&context()).await.expect("wait for task settlement");
	assert!(matches!(settled.status, crate::harness::types::SubmissionStatus::Unanswered));
	let inspection = harness.inspect(&context()).await.expect("inspect settled harness");
	assert!(
		inspection.tasks.iter().all(|task| matches!(task.record.state.status, crate::types::TaskStateStatus::Terminal)),
		"nonterminal tasks after submission settlement: {:#?}",
		inspection.tasks,
	);
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn model_tool_round_runs_registered_tool_and_commits_result() {
	use pi_ai::models::MutableModels;
	use pi_ai::providers::faux::{
		faux_assistant_message, faux_provider, faux_tool_call, FauxContentBlock, FauxMessageOptions,
		FauxResponseStep, RegisterFauxProviderOptions,
	};
	use crate::harness::harness::open_harness;

	let registry: Arc<RegistryImpl> = Arc::new(create_registry());
	let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
	let execution_count = completed.clone();
	let registration = define_tool_of(ToolRegistration {
		tool: pi_ai::types::Tool {
			name: "echo".into(),
			description: "Echo text".into(),
			parameters: json!({"type":"object","properties":{"text":{"type":"string"}}}),
			constrained_sampling: None,
		},
		replay: None,
		execution_mode: None,
		prepare_arguments: None,
		output_limits: None,
		execute: Arc::new(move |arguments, api, _context| {
			let execution_count = execution_count.clone();
			Box::pin(async move {
				execution_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
				api.output(format!("echo {}", arguments["text"].as_str().unwrap_or("")).as_bytes(), None);
				ToolExecutionResult::default()
			})
		}),
	});
	registry.install(Extension::new("tools").with_tools(vec![Arc::new(registration)]));
	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(vec![
		FauxResponseStep::Message(faux_assistant_message(
			FauxContentBlock::from(faux_tool_call("echo", serde_json::Map::from_iter([("text".to_string(), json!("hi"))]), Some("call-1"))),
			FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
		)),
		FauxResponseStep::Message(faux_assistant_message("done", FauxMessageOptions::default())),
	]);
	let models = pi_ai::create_models(None);
	models.set_provider(faux.provider.clone());
	let options = HarnessOptions { models, registry: registry.clone(), ..HarnessOptions::default() };
	let harness = open_harness(Arc::new(MemoryStorage::new()), options, &context())
		.await
		.expect("open harness");
	let conversation = root(&harness).await;
	conversation
		.configure(
			AgentChange {
				model: Some(SetOrClear::Set(ModelRef { provider: "faux".into(), model_id: "faux-1".into() })),
				..Default::default()
			},
			&context(),
		)
		.await
		.expect("configure model");
	let submission = conversation
		.submit(
			SubmissionDraft::Input(InputSubmissionDraft {
				request_id: None,
				content: UserContentField::Text("run echo".into()),
				when_busy: None,
			}),
			&context(),
		)
		.await
		.expect("submit turn");
	let settled = submission.wait(&context()).await.expect("wait for turn");
	assert_eq!(settled.status, SubmissionStatus::Done);
	assert_eq!(completed.load(std::sync::atomic::Ordering::SeqCst), 1);
	assert_eq!(faux.state().call_count(), 2);
	let transcript = conversation.context(&context()).await.expect("read transcript");
	let result = transcript.messages.iter().find_map(|message| match message {
		Message::ToolResult(result) => Some(result),
		_ => None,
	}).expect("tool result in model transcript");
	assert_eq!(result.tool_call_id, "call-1");
	assert!(!result.is_error);
	assert!(result.content.iter().any(|part| matches!(part, pi_ai::types::UserContent::Text(text) if text.text == "echo hi")));
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn generation_retries_retryable_model_error_and_commits_recovery_answer() {
	use pi_ai::models::MutableModels;
	use pi_ai::providers::faux::{
		faux_assistant_message, faux_provider, FauxMessageOptions, FauxResponseStep, RegisterFauxProviderOptions,
	};
	use crate::harness::harness::open_harness;

	let faux = faux_provider(RegisterFauxProviderOptions::default());
	faux.set_responses(vec![
		FauxResponseStep::Message(faux_assistant_message(
			"",
			FauxMessageOptions { stop_reason: Some(StopReason::Error), error_message: Some("503 Service Unavailable".into()), ..Default::default() },
		)),
		FauxResponseStep::Message(faux_assistant_message("recovered", FauxMessageOptions::default())),
	]);
	let models = pi_ai::create_models(None);
	models.set_provider(faux.provider.clone());
	let options = HarnessOptions {
		models,
		settings: Some(HarnessSettings {
			retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(1), base_delay_ms: Some(1), ..Default::default() }),
			..Default::default()
		}),
		..HarnessOptions::default()
	};
	let harness = open_harness(Arc::new(MemoryStorage::new()), options, &context())
		.await
		.expect("open harness");
	let conversation = root(&harness).await;
	conversation.configure(AgentChange {
		model: Some(SetOrClear::Set(ModelRef { provider: "faux".into(), model_id: "faux-1".into() })),
		..Default::default()
	}, &context()).await.expect("configure model");
	let submission = conversation.submit(SubmissionDraft::Input(InputSubmissionDraft {
		request_id: None,
		content: UserContentField::Text("hi".into()),
		when_busy: None,
	}), &context()).await.expect("submit input");
	assert_eq!(submission.wait(&context()).await.expect("wait for retry").status, SubmissionStatus::Done);
	assert_eq!(faux.state().call_count(), 2);
	assert!(conversation.context(&context()).await.expect("read transcript").messages.iter().any(
		|message| matches!(message, Message::Assistant(answer) if answer.stop_reason == StopReason::Stop && answer.content.iter().any(|part| matches!(part, pi_ai::types::AssistantContent::Text(text) if text.text == "recovered")))
	));
	harness.close(&context()).await.expect("close harness");
}

#[test]
fn agent_resolution_selects_installed_extensions_by_name_and_skips_missing_names() {
	use crate::harness::agent::{resolve_agent, resolve_settings};
	use crate::harness::types::{AgentState, ExtensionSelection};

	let registry = create_registry();
	registry.install(Extension::new("alpha"));
	registry.install(Extension::new("beta"));
	let snapshot = registry.snapshot();
	let report = |_| {};
	let default = resolve_agent(None, snapshot.as_ref(), &resolve_settings(None), &report);
	assert_eq!(default.extensions.iter().map(|extension| extension.name.as_str()).collect::<Vec<_>>(), ["alpha", "beta"]);
	let selected = AgentState {
		extensions: Some(ExtensionSelection::List(vec!["beta".into(), "missing".into(), "alpha".into()])),
		..Default::default()
	};
	let agent = resolve_agent(Some(&selected), snapshot.as_ref(), &resolve_settings(None), &report);
	assert_eq!(agent.extensions.iter().map(|extension| extension.name.as_str()).collect::<Vec<_>>(), ["beta", "alpha"]);
	assert_eq!(agent.thinking_level, pi_ai::types::ModelThinkingLevel::Off);
}

#[test]
fn agent_resolution_preserves_extension_composition_and_filter_order() {
	use crate::harness::agent::{resolve_agent, resolve_settings};
	use crate::harness::types::{AgentState, ExtensionSelection, ToolsSelection};

	let tool = |name: &str, description: &str| {
		Arc::new(define_tool_of(ToolRegistration {
			tool: pi_ai::types::Tool {
				name: name.to_string(),
				description: description.to_string(),
				parameters: json!({"type":"object","properties":{}}),
				constrained_sampling: None,
			},
			replay: None,
			execution_mode: None,
			prepare_arguments: None,
			output_limits: None,
			execute: Arc::new(|_, _, _| Box::pin(async { ToolExecutionResult::default() })),
		}))
	};
	let section = |key: &str, text: &str, tag| {
		let text = text.to_string();
		crate::harness::define::section(
			key,
			Arc::new(move |_, _| {
				let text = text.clone();
				Box::pin(async move { Ok(Some(text)) })
			}),
			tag,
		)
	};

	let second = Extension::new("second")
		.with_tools(vec![tool("z", "second z"), tool("shared", "second shared")])
		.with_sections(vec![section("z", "second z", None), section("shared", "second shared", Some(false))]);
	let shared = tool("shared", "first shared");
	let first = Extension::new("first")
		.with_tools(vec![tool("a", "first a"), shared.clone()])
		.with_sections(vec![section("a", "first a", None), section("shared", "first shared", None)])
		.with_wraps(vec![
			crate::harness::define::wrap_tool(shared.as_ref(), |tool| {
				let mut wrapped = tool.as_ref().clone();
				wrapped.tool.description = "wrapped shared".to_string();
				Arc::new(wrapped)
			}),
			crate::harness::define::wrap_section("shared", |mut section| {
				section.tag = Some(true);
				section
			}),
		]);
	let registry = create_registry();
	registry.install(second);
	registry.install(first);
	let snapshot = registry.snapshot();
	let extensions = ExtensionSelection::List(vec!["second".into(), "missing".into(), "first".into(), "second".into()]);
	let selected = AgentState { extensions: Some(extensions.clone()), ..Default::default() };
	let report = |_| {};
	let composed = resolve_agent(Some(&selected), snapshot.as_ref(), &resolve_settings(None), &report);
	assert_eq!(
		composed.extensions.iter().map(|extension| extension.name.as_str()).collect::<Vec<_>>(),
		["second", "first"]
	);
	assert_eq!(
		composed.tools.iter().map(|tool| tool.tool.name.as_str()).collect::<Vec<_>>(),
		["z", "shared", "a"]
	);
	assert_eq!(composed.tools[1].tool.description, "wrapped shared");
	assert_eq!(
		composed.sections.iter().map(|section| section.key.as_str()).collect::<Vec<_>>(),
		["z", "shared", "a"]
	);
	assert_eq!(composed.sections[1].tag, Some(true));

	let filtered = AgentState {
		extensions: Some(extensions),
		tools: Some(ToolsSelection::List(vec!["a".into(), "z".into(), "a".into(), "shared".into(), "shared".into()])),
		..Default::default()
	};
	let filtered = resolve_agent(Some(&filtered), snapshot.as_ref(), &resolve_settings(None), &report);
	assert_eq!(
		filtered.tools.iter().map(|tool| tool.tool.name.as_str()).collect::<Vec<_>>(),
		["a", "z", "shared"]
	);
}

#[test]
fn compaction_keeps_recent_turns_and_cuts_only_at_a_message_entry() {
	use crate::session::ContextView;
	use crate::types::{ConversationId, EntryId, EntryRecord};

	let messages = vec![
		user("question one"),
		assistant("answer one", StopReason::Stop),
		user("question two"),
		assistant("answer two", StopReason::Stop),
	];
	let entries: Vec<EntryRecord> = messages
		.iter()
		.enumerate()
		.map(|(index, message)| EntryRecord {
			id: EntryId(index as u64 + 1),
			conversation_id: ConversationId(1),
			kind: if index % 2 == 0 { "pi.user" } else { "pi.assistant" }.into(),
			model: Some(vec![message.clone()]),
			data: None,
			head: None,
			edits: None,
			by_task_id: None,
		})
		.collect();
	let view = ContextView {
		head: None,
		entries,
		contributions: messages.iter().cloned().map(|message| vec![message]).collect(),
		messages,
	};
	let cut = crate::harness::compaction::select_cut(&view, 1).expect("old history can be compacted");
	assert_eq!(cut, 3);
	assert!(matches!(view.contributions[cut].first(), Some(Message::Assistant(_))));
	assert!(crate::harness::compaction::summarized_messages(&view, cut).iter().any(
		|message| matches!(message, Message::User(user) if user.content == UserContentField::Text("question one".into()))
	));
}

#[tokio::test]
async fn context_orders_tool_results_by_the_assistant_call_order() {
	use pi_ai::types::{TextContent, ToolResultMessage, UserContent};

	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let conversation_id = conversation.id;
	let first = user("run tools");
	conversation.commit(move |tx| tx.append_entry(conversation_id, EntryDraft {
		kind: Some("pi.user".into()), model: Some(vec![first]), ..Default::default()
	}).map_err(|error| error.to_string()), &context()).await.expect("append user");
	let call = Message::Assistant(AssistantMessage {
		content: vec![
			pi_ai::types::AssistantContent::ToolCall(pi_ai::types::ToolCall {
				id: "call-b".into(), name: "b".into(), arguments: Default::default(), thought_signature: None, namespace: None,
			}),
			pi_ai::types::AssistantContent::ToolCall(pi_ai::types::ToolCall {
				id: "call-a".into(), name: "a".into(), arguments: Default::default(), thought_signature: None, namespace: None,
			}),
		],
		api: "test".into(), provider: "test".into(), model: "test".into(), response_model: None, response_id: None,
		provider_thinking_level: None, thinking_level: None, diagnostics: None, usage: Default::default(),
		stop_reason: StopReason::ToolUse, deferred: None, error_message: None, raw_stop_reason: None,
		end_turn: None, timestamp: 0,
	});
	let conversation_id = conversation.id;
	conversation.commit(move |tx| tx.append_entry(conversation_id, EntryDraft {
		kind: Some("pi.assistant".into()), model: Some(vec![call]), ..Default::default()
	}).map_err(|error| error.to_string()), &context()).await.expect("append calls");
	for (id, name) in [("call-a", "a"), ("call-b", "b")] {
		let result = Message::ToolResult(ToolResultMessage {
			tool_call_id: id.into(), tool_name: name.into(), content: vec![UserContent::Text(TextContent::new(name))],
			details: None, usage: None, nested_calls: None, is_error: false, timestamp: 0,
		});
		let conversation_id = conversation.id;
		conversation.commit(move |tx| tx.append_entry(conversation_id, EntryDraft {
			kind: Some("pi.tool-result".into()), model: Some(vec![result]), ..Default::default()
		}).map_err(|error| error.to_string()), &context()).await.expect("append tool result");
	}
	let view = conversation.context(&context()).await.expect("derive transcript");
	let ordered: Vec<_> = view.messages.iter().filter_map(|message| match message {
		Message::ToolResult(result) => Some(result.tool_call_id.as_str()),
		_ => None,
	}).collect();
	assert_eq!(ordered, ["call-b", "call-a"]);
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn generation_polls_deferred_faux_response_until_answer_is_ready() {
	use pi_ai::models::MutableModels;
	use pi_ai::providers::faux::{
		faux_assistant_message, faux_provider, FauxDeferredOptions, FauxMessageOptions,
		FauxResponseStep, RegisterFauxProviderOptions,
	};
	use crate::harness::harness::open_harness;

	let faux = faux_provider(RegisterFauxProviderOptions {
		deferred: Some(FauxDeferredOptions { pending_fetches: Some(1), poll_after_ms: Some(1) }),
		..Default::default()
	});
	faux.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
		"deferred answer", FauxMessageOptions::default(),
	))]);
	let models = pi_ai::create_models(None);
	models.set_provider(faux.provider.clone());
	let options = HarnessOptions {
		models,
		settings: Some(HarnessSettings {
			stream: Some(crate::harness::types::ConversationStreamOptions { deferred: Some(json!(true)), ..Default::default() }),
			..Default::default()
		}),
		..HarnessOptions::default()
	};
	let harness = open_harness(Arc::new(MemoryStorage::new()), options, &context())
		.await
		.expect("open harness");
	let conversation = root(&harness).await;
	conversation.configure(AgentChange {
		model: Some(SetOrClear::Set(ModelRef { provider: "faux".into(), model_id: "faux-1".into() })),
		..Default::default()
	}, &context()).await.expect("configure model");
	let submission = conversation.submit(SubmissionDraft::Input(InputSubmissionDraft {
		request_id: None, content: UserContentField::Text("hi".into()), when_busy: None,
	}), &context()).await.expect("submit input");
	assert_eq!(submission.wait(&context()).await.expect("wait for deferred response").status, SubmissionStatus::Done);
	assert_eq!(faux.state().deferred_fetch_count(), 2);
	assert!(conversation.context(&context()).await.expect("read answer").messages.iter().any(
		|message| matches!(message, Message::Assistant(answer) if answer.content.iter().any(|part| matches!(part, pi_ai::types::AssistantContent::Text(text) if text.text == "deferred answer")))
	));
	harness.close(&context()).await.expect("close harness");
}
