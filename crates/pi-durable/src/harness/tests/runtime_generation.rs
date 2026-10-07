use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::time::Duration;

use parking_lot::{Mutex, RwLock};
use pi_ai::models::{ApiDispatch, CreateProviderOptions, MutableModels, Provider, ProviderStreams, create_provider};
use pi_ai::providers::faux::{
	FauxCore, FauxDeferredOptions, FauxMessageOptions, FauxResponseStep, FauxTokenSize, RegisterFauxProviderOptions,
	faux_assistant_message,
};
use pi_ai::types::{
	AssistantContent, AssistantMessage, AssistantMessageEvent, DeferredCancelOptions, DeferredFetchOptions,
	Message, Model, ModelThinkingLevel, SimpleStreamOptions, StopReason, StreamDoneReason, StreamOptions,
	ThinkingLevel, TranscriptContext, UserContentField,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::thrown::ThrownValue;
use serde_json::{Value, json};

use super::support::{context, open, root};
use crate::documents::{DocDefinition, DocToken};
use crate::harness::agent::resolve_settings;
use crate::harness::define::{section, wrap_section};
use crate::harness::harness::{Conversation, Harness};
use crate::harness::live::{LiveState, live_doc};
use crate::harness::registry::{RegistryImpl, create_registry};
use crate::harness::types::{
	AgentChange, AnyTask, ConversationCreateOptions, ConversationOwnership, ConversationRetryPolicy,
	ConversationStreamOptions, EntryDraft, Extension, HarnessOptions, HarnessSettings, HarnessSettingsProvider, InputSubmissionDraft,
	ModelRef, ProgressPolicy, ProgressPolicyOverrides, Registry, RegistryListener, RegistryReader,
	RegistrySnapshot, RetryPolicyOverrides, SetOrClear, Submission, SubmissionDraft, SubmissionStatus, Unsubscribe, WhenBusy,
};
use crate::session::support::ControlledStorage;
use crate::storage::memory::MemoryStorage;
use crate::storage::sqlite::{NodeSqliteStorageOptions, SqliteStorage};
use crate::types::{
	CommitChange, ConversationId, DocumentAddress, DocumentCommitChange, DocumentContent, DocumentFork,
	DocumentHistory, DocumentPoint, DocumentScope, EntryRecord, Storage, StorageWrite, SubmissionSettlement,
	SubmissionType, TaskOptions, TaskOutcomeStatus, TaskOwnership, TaskQuery, TaskStateStatus,
};

struct Fixture {
	core: FauxCore,
	models: Arc<pi_ai::models::ModelsImpl>,
	provider: Arc<dyn Provider>,
	registry: Arc<RegistryImpl>,
}

#[derive(Clone)]
struct PartialGate {
	started: Arc<tokio::sync::Notify>,
	release: Arc<tokio::sync::Notify>,
}

#[derive(Clone)]
struct TestApi {
	core: FauxCore,
	partial: Option<PartialGate>,
	fail_cancel: bool,
}

impl ProviderStreams for TestApi {
	fn stream(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
		self.core.stream(model, context, Some(&SimpleStreamOptions::from(options.clone())))
	}

	fn stream_simple(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: &SimpleStreamOptions,
	) -> AssistantMessageEventStream {
		let Some(gate) = self.partial.clone() else {
			return self.core.stream(model, context, Some(options));
		};
		let events = AssistantMessageEventStream::new();
		let partial = faux_assistant_message(
			"partial",
			FauxMessageOptions { stop_reason: Some(StopReason::Pending), ..Default::default() },
		);
		gate.started.notify_one();
		events.push(AssistantMessageEvent::Start { partial });
		let producer = events.clone();
		tokio::spawn(async move {
			gate.release.notified().await;
			let final_message = faux_assistant_message("final", FauxMessageOptions::default());
			producer.push(AssistantMessageEvent::Done {
				reason: StreamDoneReason::Stop,
				message: final_message.clone(),
			});
			producer.end(Some(final_message));
		});
		events
	}

	fn has_fetch_deferred(&self) -> bool {
		true
	}

	fn fetch_deferred(
		&self,
		model: &Model,
		handle: &pi_ai::types::DeferredHandle,
		options: &DeferredFetchOptions,
	) -> Option<AssistantMessageEventStream> {
		Some(self.core.fetch_deferred(model, handle, Some(options)))
	}

	fn has_cancel_deferred(&self) -> bool {
		true
	}

	fn cancel_deferred(
		&self,
		model: &Model,
		handle: &pi_ai::types::DeferredHandle,
		options: &DeferredCancelOptions,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		if self.fail_cancel {
			return Some(Box::pin(async { Err(ThrownValue::message("cancel failed")) }));
		}
		let core = self.core.clone();
		let model = model.clone();
		let handle = handle.clone();
		let options = options.clone();
		Some(Box::pin(async move {
			core.cancel_deferred(&model, &handle, Some(&options)).await;
			Ok(())
		}))
	}
}

/// Ambient-only faux api-key `resolve` (upstream `{ auth: {} }`): without it
/// auth resolution returns `None` and every request fails with
/// "Provider is not configured".
fn ambient_faux_resolve<'a>(
	_input: pi_ai::auth::types::ApiKeyAuthInput<'a>,
) -> pi_ai::auth::types::BoxFuture<'a, Result<Option<pi_ai::auth::types::AuthResult>, ThrownValue>> {
	Box::pin(async move { Ok(Some(pi_ai::auth::types::AuthResult::default())) })
}

fn fixture(options: RegisterFauxProviderOptions, partial: Option<PartialGate>, fail_cancel: bool) -> Fixture {
	let core = FauxCore::new(options);
	let provider = create_provider(CreateProviderOptions {
		id: core.provider().to_string(),
		name: None,
		base_url: None,
		headers: None,
		auth: pi_ai::auth::types::ProviderAuth {
			api_key: Some(pi_ai::auth::types::ApiKeyAuth {
				name: "Faux".to_string(),
				login: None,
				check: None,
				resolve: Box::new(ambient_faux_resolve),
			}),
			oauth: None,
		},
		models: core.models().into_iter().map(pi_ai::types::AnyModel::Chat).collect(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(Arc::new(TestApi { core: core.clone(), partial, fail_cancel }))),
		images: BTreeMap::new(),
		classifiers: BTreeMap::new(),
	})
	.expect("construct test provider");
	let models = pi_ai::create_models(None);
	models.set_provider(provider.clone());
	Fixture { core, models, provider, registry: Arc::new(create_registry()) }
}

fn default_fixture() -> Fixture {
	fixture(RegisterFauxProviderOptions::default(), None, false)
}

fn retry_error(message: &str) -> AssistantMessage {
	faux_assistant_message(
		"",
		FauxMessageOptions {
			stop_reason: Some(StopReason::Error),
			error_message: Some(message.to_string()),
			..Default::default()
		},
	)
}

fn model_change(model_id: &str) -> AgentChange {
	AgentChange {
		model: Some(SetOrClear::Set(ModelRef { provider: "faux".into(), model_id: model_id.into() })),
		..Default::default()
	}
}

async fn open_fixture(
	storage: Arc<dyn Storage>,
	fixture: &Fixture,
	settings: Option<HarnessSettings>,
	now: Option<Arc<dyn Fn() -> i64 + Send + Sync>>,
	report: Option<Arc<dyn Fn(String) + Send + Sync>>,
) -> Arc<Harness> {
	open(
		storage,
		HarnessOptions {
			models: fixture.models.clone(),
			registry: fixture.registry.clone(),
			settings,
			now,
			on_report: report,
			..HarnessOptions::default()
		},
	)
	.await
}

async fn open_root(
	storage: Arc<dyn Storage>,
	fixture: &Fixture,
	settings: Option<HarnessSettings>,
	now: Option<Arc<dyn Fn() -> i64 + Send + Sync>>,
	report: Option<Arc<dyn Fn(String) + Send + Sync>>,
	configure: bool,
) -> (Arc<Harness>, Arc<Conversation>) {
	let harness = open_fixture(storage, fixture, settings, now, report).await;
	let conversation = Arc::new(root(&harness).await);
	if configure {
		conversation.configure(model_change("faux-1"), &context()).await.expect("configure faux model");
	}
	(harness, conversation)
}

async fn submit(conversation: &Arc<Conversation>, text: &str) -> crate::harness::submissions::SubmissionHandle {
	conversation
		.submit(
			SubmissionDraft::Input(InputSubmissionDraft {
				request_id: None,
				content: UserContentField::Text(text.to_string()),
				when_busy: None,
			}),
			&context(),
		)
		.await
		.expect("submit input")
}

async fn live(harness: &Harness, conversation_id: ConversationId) -> LiveState {
	let value = harness
		.session
		.snapshot(&live_doc(), Some(conversation_id.0), None, &context())
		.await
		.expect("read live document");
	value
		.map(|value| serde_json::from_value(Value::Object((*value).clone())).expect("decode live document"))
		.unwrap_or_default()
}

async fn wait_live(
	harness: &Harness,
	conversation_id: ConversationId,
	predicate: impl Fn(&LiveState) -> bool,
) -> LiveState {
	tokio::time::timeout(Duration::from_secs(8), async {
		loop {
			let state = live(harness, conversation_id).await;
			if predicate(&state) {
				return state;
			}
			tokio::time::sleep(Duration::from_millis(2)).await;
		}
	})
	.await
	.expect("live state did not reach the expected boundary")
}

async fn all_entries(conversation: &Conversation) -> Vec<EntryRecord> {
	let mut entries = conversation
		.entries(crate::harness::types::ConversationEntryQuery::default(), 1_000, None, &context())
		.await
		.expect("read entries")
		.items;
	entries.sort_by_key(|entry| entry.id);
	entries
}

fn message_text(message: &Message) -> String {
	match message {
		Message::Assistant(assistant) => assistant
			.content
			.iter()
			.filter_map(|part| match part {
				AssistantContent::Text(text) => Some(text.text.as_str()),
				_ => None,
			})
			.collect::<Vec<_>>()
			.join(""),
		_ => String::new(),
	}
}

fn live_partial_text(state: &LiveState) -> String {
	state
		.generation
		.as_ref()
		.and_then(|generation| generation.message.as_ref())
		.and_then(|value| serde_json::from_value::<AssistantMessage>(value.clone()).ok())
		.map(|message| message_text(&Message::Assistant(message)))
		.unwrap_or_default()
}

fn entry_kinds(entries: &[EntryRecord]) -> Vec<&str> {
	entries.iter().map(|entry| entry.kind.as_str()).collect()
}

fn message_role(message: &Message) -> &'static str {
	match message {
		Message::System(_) => "system",
		Message::User(_) => "user",
		Message::Assistant(_) => "assistant",
		Message::ToolResult(_) => "toolResult",
	}
}

fn last_user_text(context: &TranscriptContext) -> String {
	context
		.messages()
		.iter()
		.rev()
		.find_map(|message| match message {
			Message::User(user) => match &user.content {
				UserContentField::Text(text) => Some(text.clone()),
				_ => None,
			},
			_ => None,
		})
		.unwrap_or_default()
}

fn live_message_observer(harness: &Arc<Harness>) -> (Arc<Mutex<Vec<LiveState>>>, crate::session::Unsubscribe) {
	let states = Arc::new(Mutex::new(Vec::new()));
	let captured = states.clone();
	let unsubscribe = harness.session.subscribe_commits(Arc::new(move |publication, _| {
		for change in &publication.changes {
			if let CommitChange::Document(DocumentCommitChange::Document { record, value: Some(value), .. }) = change {
				if record.kind == "pi.live" {
					if let Ok(state) = serde_json::from_value(Value::Object(value.clone())) {
						captured.lock().push(state);
					}
				}
			}
		}
	}));
	(states, unsubscribe)
}

fn sqlite_options() -> NodeSqliteStorageOptions {
	NodeSqliteStorageOptions::default()
}

async fn open_sqlite(path: &std::path::Path) -> Arc<dyn Storage> {
	Arc::new(SqliteStorage::open_path(path, &sqlite_options()).await.expect("open sqlite storage"))
}

fn live_address(conversation_id: ConversationId) -> DocumentAddress {
	DocumentAddress {
		kind: "pi.live".into(),
		scope: DocumentScope::Conversation { conversation_id },
		key: None,
	}
}

#[tokio::test]
async fn simple_answer_persists_the_prompt_and_attributes_generation_entries() {
	let fixture = default_fixture();
	fixture.registry.install(Extension::new("prompt").with_sections(vec![section(
		"preamble",
		Arc::new(|_, _| Box::pin(async { Ok(Some("You are helpful.".to_string())) })),
		Some(false),
	)]));
	fixture.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("Hello there", FauxMessageOptions::default()))]);
	let storage = Arc::new(MemoryStorage::new());
	let (harness, conversation) = open_root(storage.clone(), &fixture, None, None, None, true).await;
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	let settled = submission.wait(&context()).await.expect("answer settles");
	assert_eq!(settled.status, SubmissionStatus::Done);
	let entries = all_entries(&conversation).await;
	assert_eq!(entry_kinds(&entries), ["pi.user", "pi.system", "pi.assistant"]);
	assert_eq!(entries[0].id, settled.entry.expect("placed input entry"));
	let system = match entries[1].model.as_ref().and_then(|messages| messages.first()) {
		Some(Message::System(system)) => system,
		other => panic!("expected rendered system message, got {other:?}"),
	};
	assert_eq!(system.content, pi_ai::types::SystemContent::Text(String::new()));
	assert_eq!(system.sections.as_ref().and_then(|sections| sections.get("preamble")).and_then(Option::as_deref), Some("You are helpful."));
	assert!(system.timestamp >= 0);
	assert_eq!(message_text(&entries[2].model.as_ref().unwrap()[0]), "Hello there");
	let tasks = storage
		.scan_tasks(&TaskQuery { conversation_id: Some(conversation.id), ..Default::default() }, 10, None, &context())
		.await;
	let task = tasks.items.first().expect("generation task");
	assert_eq!(task.kind, "pi.generation");
	assert_eq!(entries[0].by_task_id, None);
	assert_eq!(entries[1].by_task_id, Some(task.id));
	assert_eq!(entries[2].by_task_id, Some(task.id));
	assert_eq!(task.state.status, TaskStateStatus::Terminal);
	let outcome = task.state.outcome.as_ref().expect("terminal task outcome");
	assert_eq!(outcome.status, TaskOutcomeStatus::Completed);
	assert_eq!(outcome.result, Some(json!({"entryId": settled.answer.expect("answer entry")})));
	assert_eq!(live(&harness, conversation.id).await, LiveState::default());
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn streaming_partials_are_deltas_and_settlement_writes_a_complete_base() {
	let fixture = fixture(
		RegisterFauxProviderOptions {
			tokens_per_second: Some(10_000.0),
			token_size: Some(FauxTokenSize { min: Some(1), max: Some(1) }),
			..Default::default()
		},
		None,
		false,
	);
	fixture.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("w".repeat(80), FauxMessageOptions::default()))]);
	let storage = Arc::new(ControlledStorage::new());
	let (harness, conversation) = open_root(storage.clone(), &fixture, None, None, None, true).await;
	let live_record = storage
		.memory()
		.find_document(&live_address(conversation.id), DocumentPoint::Current, &context())
		.await
		.expect("live document record");
	harness.resume();
	assert_eq!(submit(&conversation, "hi").await.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	let contents: Vec<_> = storage
		.commits()
		.into_iter()
		.flatten()
		.filter_map(|write| match write {
			StorageWrite::DocumentChange { id, content } if id == live_record.id => Some(content),
			_ => None,
		})
		.collect();
	assert!(contents.iter().any(|content| matches!(content, DocumentContent::Delta { .. })));
	assert!(matches!(contents.last(), Some(DocumentContent::Base { .. })));
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn an_already_settled_input_does_not_keep_an_aborted_run_live() {
	let fixture = default_fixture();
	let reached = Arc::new(tokio::sync::Notify::new());
	let reached_request = reached.clone();
	fixture.core.set_responses(vec![FauxResponseStep::Factory(Arc::new(move |_transcript, options, _state, _model| {
		let signal = options.and_then(|options| options.base.base.signal.clone());
		let reached = reached_request.clone();
		Box::pin(async move {
			reached.notify_one();
			if let Some(signal) = signal {
				signal.aborted_future().await;
			}
			Err(ThrownValue::message("request stopped"))
		})
	}))]);
	let storage = Arc::new(MemoryStorage::new());
	let (harness, conversation) = open_root(storage, &fixture, None, None, None, true).await;
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	tokio::time::timeout(Duration::from_secs(5), reached.notified()).await.expect("request reached faux provider");
	conversation
		.commit(
			move |tx| {
				tx.settle_submission(
					submission.id,
					SubmissionSettlement::Unanswered { reason: "withdrawn".into(), detail: None },
				)
				.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("settle input independently");
	let task_id = wait_live(&harness, conversation.id, |state| state.run.is_some())
		.await
		.run
		.expect("run").task_id;
	assert_eq!(harness.abort_task(task_id, &context()).await.unwrap(), "marked");
	assert_eq!(harness.wait_for_task(task_id, &context()).await.unwrap().state.outcome.as_ref().unwrap().status, TaskOutcomeStatus::Aborted);
	let status = submission.status(&context()).await.unwrap();
	assert_eq!((status.status, status.reason.as_deref()), (SubmissionStatus::Unanswered, Some("withdrawn")));
	assert_eq!(live(&harness, conversation.id).await, LiveState::default());
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn an_unknown_configured_model_settles_no_model_without_writing_a_prompt() {
	let fixture = default_fixture();
	let storage = Arc::new(MemoryStorage::new());
	let (harness, conversation) = open_root(storage.clone(), &fixture, None, None, None, true).await;
	conversation.configure(model_change("missing"), &context()).await.expect("configure unknown model");
	harness.resume();
	let settled = submit(&conversation, "hi").await.wait(&context()).await.unwrap();
	assert_eq!((settled.status, settled.reason.as_deref()), (SubmissionStatus::Unanswered, Some("no_model")));
	let entries = all_entries(&conversation).await;
	assert_eq!(entry_kinds(&entries), ["pi.user"]);
	let tasks = storage
		.scan_tasks(&TaskQuery { conversation_id: Some(conversation.id), ..Default::default() }, 10, None, &context())
		.await;
	assert_eq!(tasks.items[0].state.outcome.as_ref().unwrap().status, TaskOutcomeStatus::Failed);
	assert_eq!(tasks.items[0].state.outcome.as_ref().unwrap().error.as_ref().unwrap().message, "Model faux/missing is not available");
	assert_eq!(live(&harness, conversation.id).await, LiveState::default());
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn retry_exhaustion_settles_model_error_and_leaves_later_responses_unused() {
	let fixture = default_fixture();
	fixture.core.set_responses(vec![
		FauxResponseStep::from(retry_error("503 Service Unavailable")),
		FauxResponseStep::from(retry_error("503 Service Unavailable")),
		FauxResponseStep::from(faux_assistant_message("never", FauxMessageOptions::default())),
	]);
	let storage = Arc::new(MemoryStorage::new());
	let settings = HarnessSettings {
		retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(1), base_delay_ms: Some(1), ..Default::default() }),
		..Default::default()
	};
	let (harness, conversation) = open_root(storage, &fixture, Some(settings), None, None, true).await;
	harness.resume();
	let settled = submit(&conversation, "hi").await.wait(&context()).await.unwrap();
	assert_eq!((settled.status, settled.reason.as_deref(), settled.detail.as_ref().and_then(Value::as_str)),
		(SubmissionStatus::Unanswered, Some("model_error"), Some("503 Service Unavailable")));
	assert_eq!(entry_kinds(&all_entries(&conversation).await), ["pi.user", "pi.assistant", "pi.assistant"]);
	assert_eq!(fixture.core.get_pending_response_count(), 1);
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn disabled_retry_policy_does_not_retry_a_retryable_provider_error() {
	let fixture = default_fixture();
	fixture.core.set_responses(vec![
		FauxResponseStep::from(retry_error("503 Service Unavailable")),
		FauxResponseStep::from(faux_assistant_message("never", FauxMessageOptions::default())),
	]);
	let settings = HarnessSettings {
		retry: Some(RetryPolicyOverrides { enabled: Some(false), ..Default::default() }),
		..Default::default()
	};
	let (harness, conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, Some(settings), None, None, true).await;
	harness.resume();
	let settled = submit(&conversation, "hi").await.wait(&context()).await.unwrap();
	assert_eq!((settled.status, settled.reason.as_deref()), (SubmissionStatus::Unanswered, Some("model_error")));
	assert_eq!(fixture.core.state().call_count(), 1);
	assert_eq!(fixture.core.get_pending_response_count(), 1);
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn non_retryable_error_fails_on_its_first_attempt() {
	let fixture = default_fixture();
	fixture.core.set_responses(vec![
		FauxResponseStep::from(retry_error("Invalid request")),
		FauxResponseStep::from(faux_assistant_message("never", FauxMessageOptions::default())),
	]);
	let storage = Arc::new(MemoryStorage::new());
	let (harness, conversation) = open_root(storage.clone(), &fixture, None, None, None, true).await;
	harness.resume();
	let settled = submit(&conversation, "hi").await.wait(&context()).await.unwrap();
	assert_eq!((settled.status, settled.reason.as_deref(), settled.detail.as_ref().and_then(Value::as_str)),
		(SubmissionStatus::Unanswered, Some("model_error"), Some("Invalid request")));
	assert_eq!(fixture.core.state().call_count(), 1);
	let tasks = storage
		.scan_tasks(&TaskQuery { conversation_id: Some(conversation.id), ..Default::default() }, 10, None, &context())
		.await;
	let outcome = tasks.items[0].state.outcome.as_ref().unwrap();
	assert_eq!(outcome.status, TaskOutcomeStatus::Failed);
	assert_eq!(outcome.error.as_ref().unwrap().message, "Invalid request");
	assert_eq!(outcome.error.as_ref().unwrap().detail, Some(json!({"reason":"model_error"})));
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn a_section_wrapper_failure_is_reported_and_the_failed_section_is_omitted() {
	let fixture = default_fixture();
	fixture.registry.install(Extension::new("source").with_sections(vec![section(
		"cwd",
		Arc::new(|_, _| Box::pin(async { Ok(Some("/repo".to_string())) })),
		None,
	)]));
	fixture.registry.install(Extension::new("broken").with_wraps(vec![wrap_section("cwd", |_| panic!("wrapper failed"))]));
	fixture.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("ok", FauxMessageOptions::default()))]);
	let reports = Arc::new(Mutex::new(Vec::<String>::new()));
	let reported = reports.clone();
	let report: Arc<dyn Fn(String) + Send + Sync> = Arc::new(move |error| reported.lock().push(error));
	let (harness, conversation) = open_root(
		Arc::new(MemoryStorage::new()), &fixture, None, None, Some(report), true,
	)
	.await;
	harness.resume();
	assert_eq!(submit(&conversation, "hi").await.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert!(reports.lock().iter().any(|error| error.contains("wrapper failed")));
	assert_eq!(entry_kinds(&all_entries(&conversation).await), ["pi.user", "pi.assistant"]);
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn aborting_streaming_converts_the_committed_partial_into_an_aborted_entry() {
	let fixture = fixture(
		RegisterFauxProviderOptions {
			tokens_per_second: Some(20.0),
			token_size: Some(FauxTokenSize { min: Some(1), max: Some(1) }),
			..Default::default()
		},
		None,
		false,
	);
	fixture.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("x".repeat(400), FauxMessageOptions::default()))]);
	let (harness, conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, None, None, None, true).await;
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	let state = wait_live(&harness, conversation.id, |state| !live_partial_text(state).is_empty()).await;
	let partial = state.generation.unwrap().message.expect("committed partial");
	let partial: AssistantMessage = serde_json::from_value(partial).expect("decode partial assistant");
	let task_id = state.run.expect("active run").task_id;
	assert_eq!(harness.abort_task(task_id, &context()).await.unwrap(), "marked");
	let settled = submission.wait(&context()).await.unwrap();
	assert_eq!((settled.status, settled.reason.as_deref()), (SubmissionStatus::Unanswered, Some("aborted")));
	let entries = all_entries(&conversation).await;
	assert_eq!(entry_kinds(&entries), ["pi.user", "pi.assistant"]);
	let converted = match &entries[1].model.as_ref().unwrap()[0] {
		Message::Assistant(message) => message,
		other => panic!("expected aborted assistant entry, got {other:?}"),
	};
	assert_eq!(converted.stop_reason, StopReason::Aborted);
	assert!(message_text(&Message::Assistant(converted.clone())).starts_with(&message_text(&Message::Assistant(partial))));
	assert_eq!(live(&harness, conversation.id).await, LiveState::default());
	assert_eq!(harness.wait_for_task(task_id, &context()).await.unwrap().state.outcome.as_ref().unwrap().status, TaskOutcomeStatus::Aborted);
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn deferred_generation_polls_until_the_provider_returns_its_answer() {
	let fixture = fixture(
		RegisterFauxProviderOptions {
			deferred: Some(FauxDeferredOptions { pending_fetches: Some(1), poll_after_ms: Some(1) }),
			..Default::default()
		},
		None,
		false,
	);
	fixture.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("deferred answer", FauxMessageOptions::default()))]);
	let settings = HarnessSettings {
		stream: Some(ConversationStreamOptions { deferred: Some(json!(true)), ..Default::default() }),
		..Default::default()
	};
	let (harness, conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, Some(settings), None, None, true).await;
	let (states, _unsubscribe) = live_message_observer(&harness);
	harness.resume();
	let settled = submit(&conversation, "hi").await.wait(&context()).await.unwrap();
	assert_eq!(settled.status, SubmissionStatus::Done);
	assert_eq!(fixture.core.state().deferred_fetch_count(), 2);
	let poll_times = states
		.lock()
		.iter()
		.filter_map(|state| state.generation.as_ref()?.deferred.map(|poll| poll.poll_at))
		.collect::<Vec<_>>();
	assert!(poll_times.len() >= 2);
	assert!(poll_times.windows(2).all(|times| times[1] > times[0]));
	let answer = all_entries(&conversation)
		.await
		.into_iter()
		.find(|entry| entry.id == settled.answer.unwrap())
		.expect("deferred answer entry");
	assert_eq!(message_text(&answer.model.as_ref().unwrap()[0]), "deferred answer");
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn aborting_a_deferred_poll_cancels_provider_work_and_clears_live_state() {
	let fixture = fixture(
		RegisterFauxProviderOptions {
			deferred: Some(FauxDeferredOptions { pending_fetches: Some(100), poll_after_ms: Some(60_000) }),
			..Default::default()
		},
		None,
		false,
	);
	fixture.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("never", FauxMessageOptions::default()))]);
	let settings = HarnessSettings {
		stream: Some(ConversationStreamOptions { deferred: Some(json!(true)), ..Default::default() }),
		..Default::default()
	};
	let (harness, conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, Some(settings), None, None, true).await;
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	let state = wait_live(&harness, conversation.id, |state| state.generation.as_ref().and_then(|generation| generation.deferred).is_some()).await;
	let task_id = state.run.unwrap().task_id;
	assert_eq!(harness.abort_task(task_id, &context()).await.unwrap(), "marked");
	assert_eq!((submission.wait(&context()).await.unwrap().status, submission.status(&context()).await.unwrap().reason.as_deref()),
		(SubmissionStatus::Unanswered, Some("aborted")));
	assert_eq!(fixture.core.state().cancelled_deferred().len(), 1);
	assert_eq!(live(&harness, conversation.id).await, LiveState::default());
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn failed_deferred_cancellation_is_reported_without_preventing_abort_cleanup() {
	let fixture = fixture(
		RegisterFauxProviderOptions {
			deferred: Some(FauxDeferredOptions { pending_fetches: Some(100), poll_after_ms: Some(60_000) }),
			..Default::default()
		},
		None,
		true,
	);
	fixture.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("never", FauxMessageOptions::default()))]);
	let settings = HarnessSettings {
		stream: Some(ConversationStreamOptions { deferred: Some(json!(true)), ..Default::default() }),
		..Default::default()
	};
	let reports = Arc::new(Mutex::new(Vec::<String>::new()));
	let captured = reports.clone();
	let report: Arc<dyn Fn(String) + Send + Sync> = Arc::new(move |error| captured.lock().push(error));
	let (harness, conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, Some(settings), None, Some(report), true).await;
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	let state = wait_live(&harness, conversation.id, |state| state.generation.as_ref().and_then(|generation| generation.deferred).is_some()).await;
	let task_id = state.run.unwrap().task_id;
	assert_eq!(harness.abort_task(task_id, &context()).await.unwrap(), "marked");
	assert_eq!((submission.wait(&context()).await.unwrap().status, submission.status(&context()).await.unwrap().reason.as_deref()),
		(SubmissionStatus::Unanswered, Some("aborted")));
	assert!(reports.lock().iter().any(|error| error.contains("cancel failed")));
	assert_eq!(live(&harness, conversation.id).await, LiveState::default());
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn request_options_and_thinking_level_are_resolved_for_each_turn() {
	let fixture = default_fixture();
	let seen = Arc::new(Mutex::new(Vec::<(Option<u64>, Option<String>, Option<ThinkingLevel>, Option<String>, bool)>::new()));
	let captured = seen.clone();
	let captured_second = seen.clone();
	fixture.core.set_responses(vec![
		FauxResponseStep::sync(move |_transcript, options, _state, _model| {
			let options = options.expect("request options");
			captured.lock().push((
				options.base.timeout_ms,
				options.base.headers.as_ref().and_then(|headers| headers.get("x-test").cloned()).flatten(),
				options.reasoning,
				options.base.session_id.clone(),
				options.base.base.signal.is_some(),
			));
			faux_assistant_message("a", FauxMessageOptions::default())
		}),
		FauxResponseStep::sync(move |_transcript, options, _state, _model| {
			let options = options.expect("request options");
			captured_second.lock().push((
				options.base.timeout_ms,
				options.base.headers.as_ref().and_then(|headers| headers.get("x-test").cloned()).flatten(),
				options.reasoning,
				options.base.session_id.clone(),
				options.base.base.signal.is_some(),
			));
			faux_assistant_message("b", FauxMessageOptions::default())
		}),
	]);
	let settings = HarnessSettings {
		stream: Some(ConversationStreamOptions {
			timeout_ms: Some(1234),
			headers: Some(BTreeMap::from([("x-test".to_string(), "1".to_string())])),
			..Default::default()
		}),
		..Default::default()
	};
	let (harness, conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, Some(settings), None, None, true).await;
	conversation
		.configure(AgentChange { thinking_level: Some(SetOrClear::Set(ModelThinkingLevel::High)), ..Default::default() }, &context())
		.await
		.expect("set thinking level");
	harness.resume();
	assert_eq!(submit(&conversation, "one").await.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	conversation
		.configure(AgentChange { thinking_level: Some(SetOrClear::Clear), ..Default::default() }, &context())
		.await
		.expect("clear thinking level");
	assert_eq!(submit(&conversation, "two").await.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	let session_id = harness
		.session
		.snapshot(&crate::harness::provider::provider_doc(), Some(conversation.id.0), None, &context())
		.await
		.unwrap()
		.unwrap()["sessionId"]
		.as_str()
		.unwrap()
		.to_string();
	let seen = seen.lock();
	assert_eq!(seen.len(), 2);
	assert_eq!(seen[0], (Some(1234), Some("1".into()), Some(ThinkingLevel::High), Some(session_id.clone()), true));
	assert_eq!(seen[1], (Some(1234), Some("1".into()), None, Some(session_id), true));
	harness.close(&context()).await.expect("close");
}


#[tokio::test]
async fn a_legacy_conversation_persists_provider_state_before_sending_its_request() {
	let fixture = default_fixture();
	let sent = Arc::new(Mutex::new(None::<String>));
	let captured = sent.clone();
	fixture.core.set_responses(vec![FauxResponseStep::sync(move |_transcript, options, _state, _model| {
		*captured.lock() = options.and_then(|options| options.base.session_id.clone());
		faux_assistant_message("ok", FauxMessageOptions::default())
	})]);
	let (harness, conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, None, None, None, true).await;
	let provider_doc = crate::harness::provider::provider_doc();
	let id = conversation.id;
	conversation
		.commit(move |tx| tx.retire_doc(&provider_doc, Some(id.0), None).map_err(|error| error.to_string()), &context())
		.await
		.expect("retire legacy provider document");
	assert!(harness
		.session
		.snapshot(&crate::harness::provider::provider_doc(), Some(conversation.id.0), None, &context())
		.await
		.unwrap()
		.is_none());
	harness.resume();
	assert_eq!(submit(&conversation, "legacy").await.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	let stored = harness
		.session
		.snapshot(&crate::harness::provider::provider_doc(), Some(conversation.id.0), None, &context())
		.await
		.unwrap()
		.unwrap();
	let session_id = stored["sessionId"].as_str().unwrap();
	let parts: Vec<_> = session_id.split('-').collect();
	assert_eq!(parts.len(), 5);
	assert_eq!(parts[2].as_bytes().first(), Some(&b'7'));
	assert!(matches!(parts[3].as_bytes().first(), Some(b'8' | b'9' | b'a' | b'b')));
	assert_eq!(sent.lock().as_deref(), Some(session_id));
	harness.close(&context()).await.expect("close");
}

#[test]
fn resolved_settings_merge_host_overrides_with_all_builtin_defaults() {
	let defaults = resolve_settings(None);
	assert_eq!(defaults.stream, ConversationStreamOptions::default());
	assert_eq!(defaults.retry, ConversationRetryPolicy {
		enabled: true,
		max_retries: 3,
		base_delay_ms: 2_000,
		max_agent_delay_ms: Some(60_000),
	});
	assert_eq!(defaults.compaction, crate::harness::types::CompactionPolicy {
		enabled: true,
		reserve_tokens: 16_384,
		keep_recent_tokens: 20_000,
		background_tokens: 32_768,
	});
	assert_eq!(defaults.progress, ProgressPolicy { partial_interval_ms: 100, output_interval_ms: 100 });
	assert_eq!(defaults.tool_execution, crate::harness::api::ToolExecutionMode::Parallel);
	assert_eq!(defaults.steering_mode, crate::harness::api::QueueMode::OneAtATime);
	assert_eq!(defaults.follow_up_mode, crate::harness::api::QueueMode::OneAtATime);
	let overridden = resolve_settings(Some(&HarnessSettings {
		retry: Some(RetryPolicyOverrides { enabled: Some(false), ..Default::default() }),
		compaction: Some(crate::harness::types::CompactionPolicyOverrides { background_tokens: Some(0), ..Default::default() }),
		progress: Some(ProgressPolicyOverrides { output_interval_ms: Some(500), ..Default::default() }),
		..Default::default()
	}));
	assert_eq!(overridden.retry.enabled, false);
	assert_eq!(overridden.retry.max_retries, 3);
	assert_eq!(overridden.retry.base_delay_ms, 2_000);
	assert_eq!(overridden.compaction.background_tokens, 0);
	assert_eq!(overridden.compaction.enabled, true);
	assert_eq!(overridden.progress, ProgressPolicy { partial_interval_ms: 100, output_interval_ms: 500 });
}

async fn publishes_partial_for_interval(interval_ms: u64) -> bool {
	let gate = PartialGate { started: Arc::new(tokio::sync::Notify::new()), release: Arc::new(tokio::sync::Notify::new()) };
	let started = gate.started.clone();
	let release = gate.release.clone();
	let fixture = fixture(RegisterFauxProviderOptions::default(), Some(gate), false);
	let settings = HarnessSettings {
		progress: Some(ProgressPolicyOverrides { partial_interval_ms: Some(interval_ms), ..Default::default() }),
		..Default::default()
	};
	let (harness, conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, Some(settings), None, None, true).await;
	let (states, _unsubscribe) = live_message_observer(&harness);
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	tokio::time::timeout(Duration::from_secs(5), started.notified()).await.expect("partial stream started");
	tokio::time::sleep(Duration::from_millis(300)).await;
	release.notify_one();
	assert_eq!(submission.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	let published = states.lock().iter().any(|state| state.generation.as_ref().and_then(|generation| generation.message.as_ref()).is_some());
	harness.close(&context()).await.expect("close");
	published
}

#[tokio::test]
async fn partial_publication_obeys_the_configured_progress_interval() {
	assert!(publishes_partial_for_interval(100).await);
	assert!(!publishes_partial_for_interval(5_000).await);
}

#[tokio::test]
async fn sections_read_conversation_documents_from_their_own_conversation() {
	static AGENT_DOC: std::sync::LazyLock<DocToken> = std::sync::LazyLock::new(|| {
		DocToken::new(DocDefinition::conversation(
			"test.agent",
			1,
			DocumentHistory::Latest,
			DocumentFork::Initial,
			|| serde_json::Map::new(),
		))
	});
	let fixture = default_fixture();
	let cwd_doc = AGENT_DOC.clone();
	let agents_doc = AGENT_DOC.clone();
	fixture.registry.install(Extension::new("sections").with_sections(vec![
		section("cwd", Arc::new(move |input, context| {
			let doc = cwd_doc.clone();
			Box::pin(async move {
				Ok(input
					.read
					.snapshot(&doc, Some(input.conversation_id.0), None, &context)
					.await?
					.and_then(|value| value.get("cwd").and_then(Value::as_str).map(str::to_string)))
			})
		}), None),
		section("agents", Arc::new(move |input, context| {
			let doc = agents_doc.clone();
			Box::pin(async move {
				let state = input.read.snapshot(&doc, Some(input.conversation_id.0), None, &context).await?;
				Ok((state.as_ref().and_then(|value| value.get("kind")).and_then(Value::as_str) != Some("sub"))
					.then(|| "Read AGENTS.md".to_string()))
			})
		}), None),
	]));
	fixture.core.set_responses(vec![
		FauxResponseStep::from(faux_assistant_message("a", FauxMessageOptions::default())),
		FauxResponseStep::from(faux_assistant_message("b", FauxMessageOptions::default())),
	]);
	let (harness, root_conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, None, None, None, true).await;
	let root_id = root_conversation.id;
	let doc = AGENT_DOC.clone();
	root_conversation
		.commit(
			move |tx| {
				tx.doc(&doc, Some(root_id.0), None, None)
					.map_err(|error| error.to_string())?
					.set(&[pi_chord::delta::Seg::key("cwd")], json!("/repo"))
					.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("set root agent document");
	let doc = AGENT_DOC.clone();
	let child = Arc::new(
		harness
			.create_conversation(
				ConversationCreateOptions {
					ownership: ConversationOwnership::Ownerless,
					agent: Some(model_change("faux-1")),
					init: Some(Arc::new(move |tx, id| {
						let agent = tx.doc(&doc, Some(id.0), None, None).map_err(|error| error.to_string())?;
						agent.set(&[pi_chord::delta::Seg::key("kind")], json!("sub")).map_err(|error| error.to_string())?;
						agent.set(&[pi_chord::delta::Seg::key("cwd")], json!("/sub")).map_err(|error| error.to_string())
					})),
				},
				&context(),
			)
			.await
			.expect("create child conversation"),
	);
	harness.resume();
	assert_eq!(submit(&root_conversation, "one").await.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(submit(&child, "two").await.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	let system_for = |entries: &[EntryRecord]| {
		entries.iter().find_map(|entry| entry.model.as_ref()?.iter().find_map(|message| match message {
			Message::System(system) => Some(system.clone()),
			_ => None,
		}))
	};
	let root_system = system_for(&all_entries(&root_conversation).await);
	let child_system = system_for(&all_entries(&child).await);
	let root_sections = root_system.unwrap().sections.unwrap();
	assert_eq!(root_sections.get("cwd").and_then(Option::as_deref), Some("<cwd>\n/repo\n</cwd>"));
	assert_eq!(root_sections.get("agents").and_then(Option::as_deref), Some("<agents>\nRead AGENTS.md\n</agents>"));
	let child_system = child_system.unwrap();
	let child_sections = child_system.sections.unwrap();
	assert_eq!(child_sections.get("cwd").and_then(Option::as_deref), Some("<cwd>\n/sub\n</cwd>"));
	assert!(!child_sections.contains_key("agents"));
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn an_empty_deferred_start_never_publishes_a_partial_message() {
	let fixture = fixture(
		RegisterFauxProviderOptions {
			deferred: Some(FauxDeferredOptions { pending_fetches: Some(100), poll_after_ms: Some(60_000) }),
			..Default::default()
		},
		None,
		false,
	);
	fixture.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("never", FauxMessageOptions::default()))]);
	let settings = HarnessSettings {
		stream: Some(ConversationStreamOptions { deferred: Some(json!(true)), ..Default::default() }),
		..Default::default()
	};
	let (harness, conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, Some(settings), None, None, true).await;
	let (states, _unsubscribe) = live_message_observer(&harness);
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	let state = wait_live(&harness, conversation.id, |state| state.generation.as_ref().and_then(|generation| generation.deferred).is_some()).await;
	assert!(state.generation.as_ref().unwrap().message.is_none());
	assert!(!states.lock().iter().any(|state| state.generation.as_ref().and_then(|generation| generation.message.as_ref()).is_some()));
	let task_id = state.run.unwrap().task_id;
	assert_eq!(harness.abort_task(task_id, &context()).await.unwrap(), "marked");
	assert_eq!(submission.wait(&context()).await.unwrap().reason.as_deref(), Some("aborted"));
	assert_eq!(entry_kinds(&all_entries(&conversation).await), ["pi.user"]);
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn a_fault_after_a_committed_partial_settles_inputs_and_converts_that_partial() {
	let gate = PartialGate { started: Arc::new(tokio::sync::Notify::new()), release: Arc::new(tokio::sync::Notify::new()) };
	let release = gate.release.clone();
	let fixture = fixture(RegisterFauxProviderOptions::default(), Some(gate), false);
	let storage = Arc::new(ControlledStorage::new());
	let (harness, conversation) = open_root(storage.clone(), &fixture, None, None, None, true).await;
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	let state = wait_live(&harness, conversation.id, |state| state.generation.as_ref().and_then(|generation| generation.message.as_ref()).is_some()).await;
	let task_id = state.run.as_ref().unwrap().task_id;
	storage.fail_next_commit("classification commit failed");
	release.notify_one();
	let settled = submission.wait(&context()).await.unwrap();
	assert_eq!((settled.status, settled.reason.as_deref()), (SubmissionStatus::Unanswered, Some("faulted")));
	assert!(settled.detail.as_ref().and_then(Value::as_str).is_some_and(|detail| detail.contains("classification commit failed")));
	let entries = all_entries(&conversation).await;
	assert_eq!(entry_kinds(&entries), ["pi.user", "pi.assistant"]);
	let converted = match &entries[1].model.as_ref().unwrap()[0] {
		Message::Assistant(message) => message,
		other => panic!("expected aborted partial, got {other:?}"),
	};
	assert_eq!((converted.stop_reason, message_text(&Message::Assistant(converted.clone())).as_str()), (StopReason::Aborted, "partial"));
	assert_eq!(live(&harness, conversation.id).await, LiveState::default());
	assert_eq!(harness.wait_for_task(task_id, &context()).await.unwrap().state.outcome.as_ref().unwrap().status, TaskOutcomeStatus::Faulted);
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn interrupted_retry_uses_stream_settings_from_the_next_preparation() {
	let fixture = default_fixture();
	let now = Arc::new(AtomicI64::new(1_000));
	let clock = now.clone();
	let now_fn: Arc<dyn Fn() -> i64 + Send + Sync> = Arc::new(move || clock.load(Ordering::SeqCst));
	let seen = Arc::new(Mutex::new(Vec::<Option<u64>>::new()));
	let first_seen = seen.clone();
	let second_seen = seen.clone();
	fixture.core.set_responses(vec![
		FauxResponseStep::sync(move |_context, options, _state, _model| {
			first_seen.lock().push(options.map(|options| options.base.timeout_ms).flatten());
			retry_error("503 Service Unavailable")
		}),
		FauxResponseStep::sync(move |_context, options, _state, _model| {
			second_seen.lock().push(options.map(|options| options.base.timeout_ms).flatten());
			faux_assistant_message("ok", FauxMessageOptions::default())
		}),
	]);
	let settings = |timeout_ms| HarnessSettings {
		stream: Some(ConversationStreamOptions { timeout_ms: Some(timeout_ms), ..Default::default() }),
		retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(60_000), ..Default::default() }),
		..Default::default()
	};
	// Closing a harness closes the MemoryStorage it opened with; the reopen
	// needs file-backed storage (upstream uses sqlite for reopen tests too).
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("generation.sqlite");
	let open = || async {
		Arc::new(crate::storage::sqlite::SqliteStorage::open_path(
			&path,
			&crate::storage::sqlite::NodeSqliteStorageOptions::default(),
		).await.expect("open sqlite storage")) as Arc<dyn Storage>
	};
	let (harness, conversation) = open_root(open().await, &fixture, Some(settings(111)), Some(now_fn.clone()), None, true).await;
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	let retrying = wait_live(&harness, conversation.id, |state| state.generation.as_ref().and_then(|generation| generation.retry.as_ref()).is_some()).await;
	let task_id = retrying.run.unwrap().task_id;
	harness.close(&context()).await.expect("close before retry");
	let (reopened, reopened_root) = open_root(open().await, &fixture, Some(settings(222)), Some(now_fn), None, false).await;
	let reopened_task = reopened.get_task(task_id, &context()).await.unwrap().unwrap();
	assert_eq!(reopened_task.state.checkpoint.as_ref().unwrap()["phase"], "retry");
	now.store(61_000, Ordering::SeqCst);
	reopened.resume();
	let settled = reopened
		.submission(submission.id, &context())
		.await
		.unwrap()
		.unwrap()
		.wait(&context())
		.await
		.unwrap();
	assert_eq!(settled.status, SubmissionStatus::Done);
	assert_eq!(*seen.lock(), [Some(111), Some(222)]);
	assert_eq!(entry_kinds(&all_entries(&reopened_root).await), ["pi.user", "pi.assistant", "pi.assistant"]);
	reopened.close(&context()).await.expect("close reopened harness");
}


#[tokio::test]
async fn live_settings_provider_is_reread_between_requests_on_one_harness() {
	let fixture = default_fixture();
	let settings = Arc::new(RwLock::new(HarnessSettings {
		stream: Some(ConversationStreamOptions { timeout_ms: Some(111), ..Default::default() }),
		..Default::default()
	}));
	let provider_state = settings.clone();
	let settings_provider: HarnessSettingsProvider = Arc::new(move || provider_state.read().clone());
	let seen = Arc::new(Mutex::new(Vec::<Option<u64>>::new()));
	let first_seen = seen.clone();
	let second_seen = seen.clone();
	fixture.core.set_responses(vec![
		FauxResponseStep::sync(move |_context, options, _state, _model| {
			first_seen.lock().push(options.and_then(|options| options.base.timeout_ms));
			faux_assistant_message("first answer", FauxMessageOptions::default())
		}),
		FauxResponseStep::sync(move |_context, options, _state, _model| {
			second_seen.lock().push(options.and_then(|options| options.base.timeout_ms));
			faux_assistant_message("second answer", FauxMessageOptions::default())
		}),
	]);
	let harness = open(
		Arc::new(MemoryStorage::new()),
		HarnessOptions {
			models: fixture.models.clone(),
			registry: fixture.registry.clone(),
			settings_provider: Some(settings_provider),
			..Default::default()
		},
	)
	.await;
	let conversation = Arc::new(root(&harness).await);
	conversation.configure(model_change("faux-1"), &context()).await.expect("configure faux model");
	harness.resume();
	assert_eq!(submit(&conversation, "first").await.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	settings.write().stream = Some(ConversationStreamOptions { timeout_ms: Some(222), ..Default::default() });
	assert_eq!(submit(&conversation, "second").await.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(*seen.lock(), [Some(111), Some(222)]);
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn concurrent_conversations_keep_provider_session_ids_request_local() {
	let fixture = default_fixture();
	let seen = Arc::new(Mutex::new(HashMap::<String, Vec<String>>::new()));
	let captured = seen.clone();
	let response = FauxResponseStep::sync(move |transcript, options, _state, _model| {
		captured
			.lock()
			.entry(last_user_text(transcript))
			.or_default()
			.push(options.and_then(|options| options.base.session_id.clone()).unwrap_or_default());
		faux_assistant_message("ok", FauxMessageOptions::default())
	});
	fixture.core.set_responses(vec![response.clone(), response.clone(), response.clone(), response]);
	let (harness, root_conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, None, None, None, true).await;
	let child = Arc::new(
		harness
			.create_conversation(
				ConversationCreateOptions { ownership: ConversationOwnership::Ownerless, agent: Some(model_change("faux-1")), init: None },
				&context(),
			)
			.await
			.expect("create child"),
	);
	harness.resume();
	let first = tokio::join!(submit(&root_conversation, "first-root"), submit(&child, "first-child"));
	let first_wait_context = context();
	let first_wait = tokio::join!(first.0.wait(&first_wait_context), first.1.wait(&first_wait_context));
	assert_eq!((first_wait.0.unwrap().status, first_wait.1.unwrap().status), (SubmissionStatus::Done, SubmissionStatus::Done));
	let second = tokio::join!(submit(&root_conversation, "second-root"), submit(&child, "second-child"));
	let second_wait_context = context();
	let second_wait = tokio::join!(second.0.wait(&second_wait_context), second.1.wait(&second_wait_context));
	assert_eq!((second_wait.0.unwrap().status, second_wait.1.unwrap().status), (SubmissionStatus::Done, SubmissionStatus::Done));
	let provider_doc = crate::harness::provider::provider_doc();
	let root_id = harness.session.snapshot(&provider_doc, Some(root_conversation.id.0), None, &context()).await.unwrap().unwrap()["sessionId"].as_str().unwrap().to_string();
	let child_id = harness.session.snapshot(&provider_doc, Some(child.id.0), None, &context()).await.unwrap().unwrap()["sessionId"].as_str().unwrap().to_string();
	assert_ne!(root_id, child_id);
	let seen = seen.lock();
	assert_eq!(seen.get("first-root").unwrap(), &[root_id.clone()]);
	assert_eq!(seen.get("second-root").unwrap(), &[root_id]);
	assert_eq!(seen.get("first-child").unwrap(), &[child_id.clone()]);
	assert_eq!(seen.get("second-child").unwrap(), &[child_id]);
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn run_orphans_an_incompatible_generation_task_and_cleans_up_its_input() {
	let fixture = default_fixture();
	let (harness, conversation) = open_root(Arc::new(MemoryStorage::new()), &fixture, None, None, None, true).await;
	let conversation_id = conversation.id;
	let generation = crate::harness::generation::generation_task();
	let mut definition = (*generation.definition).clone();
	definition.version = 2;
	let unsupported = AnyTask { definition: Arc::new(definition) };
	let (task_id, submission_id) = conversation
		.commit(move |tx| {
			let mut draft = EntryDraft { kind: Some("pi.user".into()), ..Default::default() };
			draft.model = Some(vec![Message::User(pi_ai::types::UserMessage {
				content: UserContentField::Text("hi".into()), timestamp: 1,
			})]);
			let entry = tx.append_entry(conversation_id, draft).map_err(|error| error.to_string())?;
			let submission = tx.create_submission(crate::types::SubmissionCreate {
				conversation_id,
				request_id: None,
				submission_type: Some(SubmissionType::Input),
				status: Some(SubmissionStatus::Placed),
				entry: Some(entry.id),
				answer: None,
				reason: None,
				detail: None,
			}).map_err(|error| error.to_string())?;
			let task_id = tx.create_task(
				&unsupported,
				Value::Null,
				TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: Some(conversation_id), background: None },
			).map_err(|error| error.to_string())?;
			tx.doc(&live_doc(), Some(conversation_id.0), None, None)
				.map_err(|error| error.to_string())?
				.set(&[pi_chord::delta::Seg::key("run")], json!({"taskId":task_id,"inputs":[submission.id]}))
				.map_err(|error| error.to_string())?;
			Ok((task_id, submission.id))
		}, &context())
		.await
		.expect("seed unsupported run");
	harness.resume();
	let busy = match conversation
		.submit(
			SubmissionDraft::Input(InputSubmissionDraft {
				request_id: None,
				content: UserContentField::Text("busy".into()),
				when_busy: Some(WhenBusy::Reject),
			}),
			&context(),
		)
		.await
	{
		Ok(_) => panic!("busy conversation accepts whenBusy=reject"),
		Err(error) => error,
	};
	assert!(busy.contains("busy"));
	assert_eq!(harness.abort_task(task_id, &context()).await.unwrap(), "marked");
	let task = harness.wait_for_task(task_id, &context()).await.unwrap();
	let outcome = task.state.outcome.as_ref().unwrap();
	assert_eq!(outcome.status, TaskOutcomeStatus::Orphaned);
	assert_eq!(outcome.reason.as_deref(), Some("task_too_old"));
	let submission = harness.submission(submission_id, &context()).await.unwrap().unwrap();
	let status = submission.status(&context()).await.unwrap();
	assert_eq!((status.status, status.reason.as_deref()), (SubmissionStatus::Unanswered, Some("task_too_old")));
	assert_eq!(live(&harness, conversation.id).await, LiveState::default());
	harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn harness_rejects_a_registry_missing_the_generation_task() {
	let registry = Arc::new(create_registry());
	let filtered: Arc<dyn RegistryReader> = Arc::new(WithoutGenerationRegistry { inner: registry });
	let error = match crate::harness::harness::open_harness(
		Arc::new(MemoryStorage::new()),
		HarnessOptions { models: pi_ai::create_models(None), registry: filtered, ..Default::default() },
		&context(),
	)
	.await
	{
		Err(error) => error,
		Ok(harness) => {
			harness.close(&context()).await.expect("close unexpectedly opened harness");
			panic!("registry without the generation task opened");
		}
	};
	assert!(error.contains("Registry lacks built-in tasks pi.generation"));
}

struct WithoutGenerationRegistry {
	inner: Arc<RegistryImpl>,
}

struct WithoutGenerationSnapshot {
	inner: Arc<dyn RegistrySnapshot>,
}

impl RegistryReader for WithoutGenerationRegistry {
	fn snapshot(&self) -> Arc<dyn RegistrySnapshot> {
		Arc::new(WithoutGenerationSnapshot { inner: self.inner.snapshot() })
	}

	fn subscribe(&self, listener: RegistryListener) -> Unsubscribe {
		self.inner.subscribe(listener)
	}
}

impl RegistrySnapshot for WithoutGenerationSnapshot {
	fn installed(&self) -> Vec<Arc<Extension>> { self.inner.installed() }
	fn extension(&self, name: &str) -> Option<Arc<Extension>> { self.inner.extension(name) }
	fn tools(&self) -> Vec<(Arc<Extension>, Arc<crate::harness::types::ToolRegistration>)> { self.inner.tools() }
	fn sections(&self) -> Vec<(Arc<Extension>, crate::harness::types::PromptSection)> { self.inner.sections() }
	fn tasks(&self) -> Vec<AnyTask> { self.inner.tasks().into_iter().filter(|task| task.name() != "pi.generation").collect() }
	fn task(&self, name: &str) -> Option<AnyTask> {
		if name == "pi.generation" { None } else { self.inner.task(name) }
	}
}

#[tokio::test]
async fn preparation_interrupted_before_commit_is_replayed_after_reopen() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("session.sqlite");
	let fixture = default_fixture();
	let reached = Arc::new(tokio::sync::Notify::new());
	let should_block = Arc::new(AtomicBool::new(true));
	let section_calls = Arc::new(AtomicUsize::new(0));
	let reached_section = reached.clone();
	let block = should_block.clone();
	let calls = section_calls.clone();
	fixture.registry.install(Extension::new("prep").with_sections(vec![section(
		"preamble",
		Arc::new(move |_, context| {
			let reached = reached_section.clone();
			let block = block.clone();
			let calls = calls.clone();
			Box::pin(async move {
				calls.fetch_add(1, Ordering::SeqCst);
				if block.swap(false, Ordering::SeqCst) {
					reached.notify_one();
					if let Some(signal) = context.abort_signal() { signal.aborted_future().await; }
				}
				Ok(Some("p".into()))
			})
		}),
		Some(false),
	)]));
	fixture.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("answer", FauxMessageOptions::default()))]);
	let storage = open_sqlite(&path).await;
	let (harness, conversation) = open_root(storage, &fixture, None, None, None, true).await;
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	tokio::time::timeout(Duration::from_secs(5), reached.notified()).await.expect("section preparation blocked");
	let state = wait_live(&harness, conversation.id, |state| state.run.is_some()).await;
	let task_id = state.run.unwrap().task_id;
	harness.close(&context()).await.expect("close interrupted harness");
	let storage = open_sqlite(&path).await;
	let (reopened, reopened_root) = open_root(storage, &fixture, None, None, None, false).await;
	let task = reopened.get_task(task_id, &context()).await.unwrap().unwrap();
	assert_eq!(task.state.checkpoint, Some(json!({"phase":"prepare","attempt":1})));
	reopened.resume();
	let submission = reopened.submission(submission.id, &context()).await.unwrap().unwrap();
	assert_eq!(submission.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(section_calls.load(Ordering::SeqCst), 2);
	reopened.close(&context()).await.expect("close reopened harness");
}

#[tokio::test]
async fn request_interrupted_before_partials_reuses_its_durable_preparation_on_reopen() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("session.sqlite");
	let fixture = default_fixture();
	let section_calls = Arc::new(AtomicUsize::new(0));
	let calls = section_calls.clone();
	fixture.registry.install(Extension::new("prep").with_sections(vec![section(
		"preamble",
		Arc::new(move |_, _| {
			let calls = calls.clone();
			Box::pin(async move {
				calls.fetch_add(1, Ordering::SeqCst);
				Ok(Some("p".into()))
			})
		}),
		Some(false),
	)]));
	let reached = Arc::new(tokio::sync::Notify::new());
	let reached_request = reached.clone();
	let sent = Arc::new(Mutex::new(Vec::<(Vec<&'static str>, Option<u64>)>::new()));
	let captured = sent.clone();
	fixture.core.set_responses(vec![
		FauxResponseStep::Factory(Arc::new(move |_context, options, _state, _model| {
			let signal = options.and_then(|options| options.base.base.signal.clone());
			let reached = reached_request.clone();
			Box::pin(async move {
				reached.notify_one();
				if let Some(signal) = signal { signal.aborted_future().await; }
				Err(ThrownValue::message("request interrupted"))
			})
		})),
		FauxResponseStep::sync(move |transcript, options, _state, _model| {
			captured.lock().push((
				transcript.messages().iter().map(message_role).collect(),
				options.and_then(|options| options.base.timeout_ms),
			));
			faux_assistant_message("answer", FauxMessageOptions::default())
		}),
	]);
	let storage = open_sqlite(&path).await;
	let settings = |timeout_ms| HarnessSettings {
		stream: Some(ConversationStreamOptions { timeout_ms: Some(timeout_ms), ..Default::default() }),
		..Default::default()
	};
	let (harness, conversation) = open_root(storage, &fixture, Some(settings(1234)), None, None, true).await;
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	tokio::time::timeout(Duration::from_secs(5), reached.notified()).await.expect("model request started");
	let task_id = wait_live(&harness, conversation.id, |state| state.run.is_some()).await.run.unwrap().task_id;
	harness.close(&context()).await.expect("close interrupted request");
	let storage = open_sqlite(&path).await;
	let (reopened, reopened_root) = open_root(storage, &fixture, Some(settings(999)), None, None, false).await;
	let task = reopened.get_task(task_id, &context()).await.unwrap().unwrap();
	assert_eq!(task.state.checkpoint.as_ref().unwrap()["phase"], "request");
	assert_eq!(task.state.checkpoint.as_ref().unwrap()["attempt"], 1);
	assert_eq!(task.state.checkpoint.as_ref().unwrap()["thinkingLevel"], "off");
	assert_eq!(task.state.checkpoint.as_ref().unwrap()["streamOptions"]["timeoutMs"], 1234);
	let before_resume = live(&reopened, reopened_root.id).await;
	assert_eq!(before_resume.generation.unwrap().attempt, 1);
	reopened.resume();
	let submission = reopened.submission(submission.id, &context()).await.unwrap().unwrap();
	assert_eq!(submission.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(*sent.lock(), [(vec!["user", "system"], Some(1234))]);
	assert_eq!(section_calls.load(Ordering::SeqCst), 1);
	assert_eq!(fixture.core.state().call_count(), 2);
	reopened.close(&context()).await.expect("close reopened harness");
}

#[tokio::test]
async fn a_committed_partial_is_converted_before_reopened_generation_resends_the_request() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("session.sqlite");
	let slow = fixture(
		RegisterFauxProviderOptions {
			tokens_per_second: Some(20.0),
			token_size: Some(FauxTokenSize { min: Some(1), max: Some(1) }),
			..Default::default()
		},
		None,
		false,
	);
	slow.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("z".repeat(400), FauxMessageOptions::default()))]);
	let storage = open_sqlite(&path).await;
	let (harness, conversation) = open_root(storage, &slow, None, None, None, true).await;
	let (states, _unsubscribe) = live_message_observer(&harness);
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	let initial = wait_live(&harness, conversation.id, |state| !live_partial_text(state).is_empty()).await;
	let task_id = initial.run.as_ref().unwrap().task_id;
	let visible = message_text(&Message::Assistant(
		serde_json::from_value(initial.generation.as_ref().unwrap().message.clone().unwrap()).expect("partial message"),
	));
	assert!(!visible.is_empty());
	harness.close(&context()).await.expect("close interrupted stream");
	let published = states.lock().iter().filter_map(|state| state.generation.as_ref()?.message.as_ref()).filter_map(|value| {
		serde_json::from_value::<AssistantMessage>(value.clone()).ok().map(|message| message_text(&Message::Assistant(message)))
	}).collect::<Vec<_>>();
	assert!(published.iter().any(|partial| !partial.is_empty()));
	let fixture = default_fixture();
	let sent = Arc::new(Mutex::new(Vec::<Vec<&'static str>>::new()));
	let captured = sent.clone();
	fixture.core.set_responses(vec![FauxResponseStep::sync(move |transcript, _options, _state, _model| {
		captured.lock().push(transcript.messages().iter().map(message_role).collect());
		faux_assistant_message("answer", FauxMessageOptions::default())
	})]);
	let storage = open_sqlite(&path).await;
	let (reopened, reopened_root) = open_root(storage, &fixture, None, None, None, false).await;
	let durable = live(&reopened, reopened_root.id).await;
	let durable_partial: AssistantMessage = serde_json::from_value(durable.generation.as_ref().unwrap().message.clone().unwrap()).unwrap();
	let durable_text = message_text(&Message::Assistant(durable_partial.clone()));
	assert!(durable_text.starts_with(&visible));
	reopened.resume();
	let submission = reopened.submission(submission.id, &context()).await.unwrap().unwrap();
	assert_eq!(submission.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(*sent.lock(), [vec!["user"]]);
	let entries = all_entries(&reopened_root).await;
	assert_eq!(entry_kinds(&entries), ["pi.user", "pi.assistant", "pi.assistant"]);
	match &entries[1].model.as_ref().unwrap()[0] {
		Message::Assistant(message) => {
			assert_eq!(message.stop_reason, StopReason::Aborted);
			assert_eq!(message_text(&Message::Assistant(message.clone())), durable_text);
		}
		other => panic!("expected converted partial, got {other:?}"),
	}
	assert_eq!(live(&reopened, reopened_root.id).await, LiveState::default());
	assert_eq!(reopened.wait_for_task(task_id, &context()).await.unwrap().state.outcome.as_ref().unwrap().status, TaskOutcomeStatus::Completed);
	reopened.close(&context()).await.expect("close reopened harness");
}

#[tokio::test]
async fn retry_backoff_checkpoint_resumes_after_reopen_at_its_durable_deadline() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("session.sqlite");
	let fixture = default_fixture();
	let now = Arc::new(AtomicI64::new(1_000));
	let clock = now.clone();
	let now_fn: Arc<dyn Fn() -> i64 + Send + Sync> = Arc::new(move || clock.load(Ordering::SeqCst));
	fixture.core.set_responses(vec![
		FauxResponseStep::from(retry_error("503 Service Unavailable")),
		FauxResponseStep::from(faux_assistant_message("recovered", FauxMessageOptions::default())),
	]);
	let settings = HarnessSettings {
		retry: Some(RetryPolicyOverrides { enabled: Some(true), max_retries: Some(2), base_delay_ms: Some(60_000), ..Default::default() }),
		..Default::default()
	};
	let storage = open_sqlite(&path).await;
	let (harness, conversation) = open_root(storage, &fixture, Some(settings.clone()), Some(now_fn.clone()), None, true).await;
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	let state = wait_live(&harness, conversation.id, |state| state.generation.as_ref().and_then(|generation| generation.retry.as_ref()).is_some()).await;
	let task_id = state.run.as_ref().unwrap().task_id;
	harness.close(&context()).await.expect("close during backoff");
	let storage = open_sqlite(&path).await;
	let (reopened, reopened_root) = open_root(storage, &fixture, Some(settings), Some(now_fn), None, false).await;
	let task = reopened.get_task(task_id, &context()).await.unwrap().unwrap();
	assert_eq!(task.state.checkpoint, Some(json!({"phase":"retry","attempt":1,"until":61_000})));
	assert_eq!(live(&reopened, reopened_root.id).await.generation.unwrap().retry.unwrap().at, 61_000);
	now.store(61_000, Ordering::SeqCst);
	reopened.resume();
	let submission = reopened.submission(submission.id, &context()).await.unwrap().unwrap();
	assert_eq!(submission.wait(&context()).await.unwrap().status, SubmissionStatus::Done);
	assert_eq!(entry_kinds(&all_entries(&reopened_root).await), ["pi.user", "pi.assistant", "pi.assistant"]);
	reopened.close(&context()).await.expect("close reopened harness");
}

#[tokio::test]
async fn deferred_poll_checkpoint_resumes_after_reopen_and_reads_the_durable_answer() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("session.sqlite");
	let fixture = fixture(
		RegisterFauxProviderOptions { deferred: Some(FauxDeferredOptions { poll_after_ms: Some(60_000), ..Default::default() }), ..Default::default() },
		None,
		false,
	);
	let now = Arc::new(AtomicI64::new(1_000));
	let clock = now.clone();
	let now_fn: Arc<dyn Fn() -> i64 + Send + Sync> = Arc::new(move || clock.load(Ordering::SeqCst));
	fixture.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("deferred answer", FauxMessageOptions::default()))]);
	let settings = HarnessSettings {
		stream: Some(ConversationStreamOptions { deferred: Some(json!(true)), ..Default::default() }),
		..Default::default()
	};
	let storage = open_sqlite(&path).await;
	let (harness, conversation) = open_root(storage, &fixture, Some(settings.clone()), Some(now_fn.clone()), None, true).await;
	harness.resume();
	let submission = submit(&conversation, "hi").await;
	let state = wait_live(&harness, conversation.id, |state| state.generation.as_ref().and_then(|generation| generation.deferred).is_some()).await;
	let task_id = state.run.as_ref().unwrap().task_id;
	harness.close(&context()).await.expect("close while polling");
	let storage = open_sqlite(&path).await;
	let (reopened, reopened_root) = open_root(storage, &fixture, Some(settings), Some(now_fn), None, false).await;
	assert_eq!(reopened.get_task(task_id, &context()).await.unwrap().unwrap().state.checkpoint.as_ref().unwrap()["phase"], "poll");
	assert_eq!(reopened.get_task(task_id, &context()).await.unwrap().unwrap().state.checkpoint.as_ref().unwrap()["pollAt"], 61_000);
	now.store(61_000, Ordering::SeqCst);
	reopened.resume();
	let submission = reopened.submission(submission.id, &context()).await.unwrap().unwrap();
	let settled = submission.wait(&context()).await.unwrap();
	assert_eq!(settled.status, SubmissionStatus::Done);
	let entries = all_entries(&reopened_root).await;
	let answer = entries.iter().find(|entry| entry.kind == "pi.assistant" && entry.model.as_ref().is_some_and(|messages| messages.iter().any(|message| message_text(message) == "deferred answer"))).unwrap();
	assert_eq!(message_text(&answer.model.as_ref().unwrap()[0]), "deferred answer");
	assert_eq!(fixture.core.state().deferred_fetch_count(), 1);
	reopened.close(&context()).await.expect("close reopened harness");
}

#[tokio::test]
async fn missing_pinned_model_after_reopen_settles_request_and_poll_as_no_model() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("session.sqlite");
	let fixture = fixture(
		RegisterFauxProviderOptions { deferred: Some(FauxDeferredOptions { poll_after_ms: Some(60_000), ..Default::default() }), ..Default::default() },
		None,
		false,
	);
	let reached = Arc::new(tokio::sync::Notify::new());
	let reached_request = reached.clone();
	fixture.core.set_responses(vec![
		FauxResponseStep::Factory(Arc::new(move |_context, options, _state, _model| {
			let signal = options.and_then(|options| options.base.base.signal.clone());
			let reached = reached_request.clone();
			Box::pin(async move {
				reached.notify_one();
				if let Some(signal) = signal { signal.aborted_future().await; }
				Err(ThrownValue::message("request interrupted"))
			})
		})),
		FauxResponseStep::from(faux_assistant_message("deferred", FauxMessageOptions::default())),
	]);
	let storage = open_sqlite(&path).await;
	let (harness, conversation) = open_root(storage, &fixture, None, None, None, true).await;
	harness.resume();
	let requesting = submit(&conversation, "one").await;
	tokio::time::timeout(Duration::from_secs(5), reached.notified()).await.expect("request entered provider");
	harness.close(&context()).await.expect("close request");
	fixture.models.delete_provider("faux");
	let storage = open_sqlite(&path).await;
	let (reopened, reopened_root) = open_root(storage, &fixture, None, None, None, false).await;
	reopened.resume();
	let requesting = reopened.submission(requesting.id, &context()).await.unwrap().unwrap();
	let settled = requesting.wait(&context()).await.unwrap();
	assert_eq!((settled.status, settled.reason.as_deref()), (SubmissionStatus::Unanswered, Some("no_model")));
	reopened.close(&context()).await.expect("close missing request model");

	fixture.models.set_provider(fixture.provider.clone());
	let deferred_settings = HarnessSettings {
		stream: Some(ConversationStreamOptions { deferred: Some(json!(true)), ..Default::default() }),
		..Default::default()
	};
	let storage = open_sqlite(&path).await;
	let (polling_harness, polling_root) = open_root(storage, &fixture, Some(deferred_settings.clone()), None, None, false).await;
	polling_harness.resume();
	let polling = submit(&polling_root, "two").await;
	wait_live(&polling_harness, polling_root.id, |state| state.generation.as_ref().and_then(|generation| generation.deferred).is_some()).await;
	polling_harness.close(&context()).await.expect("close deferred request");
	fixture.models.delete_provider("faux");
	let storage = open_sqlite(&path).await;
	let (polling_reopened, _) = open_root(storage, &fixture, Some(deferred_settings), None, None, false).await;
	polling_reopened.resume();
	let polling = polling_reopened.submission(polling.id, &context()).await.unwrap().unwrap();
	let settled = polling.wait(&context()).await.unwrap();
	assert_eq!((settled.status, settled.reason.as_deref()), (SubmissionStatus::Unanswered, Some("no_model")));
	polling_reopened.close(&context()).await.expect("close missing poll model");
}

#[tokio::test]
async fn a_print_style_answer_and_settled_record_survive_harness_reopen() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let path = directory.path().join("session.sqlite");
	let fixture = default_fixture();
	fixture.registry.install(Extension::new("prompt").with_sections(vec![section(
		"preamble",
		Arc::new(|_, _| Box::pin(async { Ok(Some("You are terse.".into())) })),
		Some(false),
	)]));
	fixture.core.set_responses(vec![FauxResponseStep::from(faux_assistant_message("42", FauxMessageOptions::default()))]);
	let storage = open_sqlite(&path).await;
	let (harness, conversation) = open_root(storage, &fixture, None, None, None, true).await;
	harness.resume();
	let submission = submit(&conversation, "answer?").await;
	let settled = submission.wait(&context()).await.unwrap();
	assert_eq!(settled.status, SubmissionStatus::Done);
	let answer = all_entries(&conversation).await.into_iter().find(|entry| Some(entry.id) == settled.answer).expect("answer entry");
	assert_eq!(message_text(&answer.model.as_ref().unwrap()[0]), "42");
	harness.close(&context()).await.expect("close first harness");
	let storage = open_sqlite(&path).await;
	let (reopened, reopened_root) = open_root(storage, &fixture, None, None, None, false).await;
	let reopened_submission = reopened.submission(submission.id, &context()).await.unwrap().unwrap();
	assert_eq!(reopened_submission.status(&context()).await.unwrap(), settled);
	assert!(all_entries(&reopened_root).await.contains(&answer));
	reopened.close(&context()).await.expect("close reopened harness");
}
