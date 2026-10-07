use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use pi_ai::models::MutableModels;
use pi_ai::providers::faux::{
	faux_assistant_message, faux_provider, faux_tool_call, FauxContentBlock, FauxMessageOptions, FauxResponseStep,
	FauxTokenSize, RegisterFauxProviderOptions,
};
use pi_ai::types::{Message, StopReason};
use pi_chord::context::Context;
use pi_chord::delta::{apply_immutable_batches, Op, Seg};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use super::support::context;
use crate::errors::StorageRejected;
use crate::harness::api::{ToolDiagnostic, ToolDiagnosticSeverity, ToolExecute, ToolExecutionResult, ToolOutputLimits};
use crate::harness::api::define_tool;
use crate::harness::harness::{open_harness, Conversation, Harness};
use crate::harness::live::{live_doc, LiveState, SlotStatus};
use crate::harness::registry::{create_registry, RegistryImpl};
use crate::harness::submissions::SubmissionHandle;
use crate::harness::types::{
	AgentChange, Extension, HarnessOptions, HarnessSettings, InputSubmissionDraft, ModelRef, Registry, SetOrClear,
	Submission, SubmissionDraft, SubmissionStatus, ToolExecutionMode, ToolRegistration,
};
use crate::storage::memory::MemoryStorage;
use crate::types::{
	CommitChange, CommitPublication, ConversationId, ConversationQuery, ConversationRecord, Cursor, DocumentAddress,
	DocumentCommitChange, DocumentContent, DocumentId, DocumentPoint, DocumentQuery, DocumentRecord, EntryId,
	EntryQuery, EntryRecord, Page, Seq, Storage, StorageWrite, StoredDocument, SubmissionId, SubmissionQuery,
	SubmissionRecord, TaskId, TaskQuery, TaskRecord,
};

#[derive(Clone, Debug)]
struct LiveChange {
	seq: Seq,
	ops: Vec<Op>,
	state: LiveState,
}

#[derive(Clone, Default)]
struct LiveCapture {
	changes: Arc<Mutex<Vec<LiveChange>>>,
}

impl LiveCapture {
	fn snapshot(&self) -> Vec<LiveChange> {
		self.changes.lock().clone()
	}

	fn len(&self) -> usize {
		self.changes.lock().len()
	}
}

fn capture_live(harness: &Harness) -> LiveCapture {
	let capture = LiveCapture::default();
	let changes = capture.changes.clone();
	let _unsubscribe = harness.session.subscribe_commits(Arc::new(move |publication, _context| {
		for change in &publication.changes {
			let CommitChange::Document(DocumentCommitChange::Document { record, value, ops, .. }) = change else {
				continue;
			};
			if record.kind != "pi.live" {
				continue;
			}
			let Some(value) = value else { continue };
			let state = serde_json::from_value(Value::Object(value.clone())).expect("pi.live publication is valid");
			changes.lock().push(LiveChange { seq: publication.seq, ops: ops.clone(), state });
		}
	}));
	capture
}

fn assert_replayable(changes: &[LiveChange]) {
	let mut value = json!({});
	for change in changes {
		if !change.ops.is_empty() {
			value = apply_immutable_batches(&value, [change.ops.as_slice()]).expect("published pi.live delta replays");
		}
		assert_eq!(value, serde_json::to_value(&change.state).expect("pi.live state serializes"), "seq {}", change.seq);
	}
}

fn path(keys: &[&str]) -> Vec<Seg> {
	keys.iter().map(|key| Seg::key(*key)).collect()
}

fn slot_path(index: usize, field: &str) -> Vec<Seg> {
	vec![Seg::key("tools"), Seg::index(index), Seg::key(field)]
}

fn output_path() -> Vec<Seg> {
	slot_path(0, "output")
}

fn set(path: Vec<Seg>, value: Value) -> Op {
	Op::Set { path, value }
}

fn append(path: Vec<Seg>, text: &str) -> Op {
	Op::Append { path, text: text.to_string() }
}

fn truncate(path: Vec<Seg>, count: usize) -> Op {
	Op::Truncate { path, count }
}

fn delete(path: Vec<Seg>) -> Op {
	Op::Delete { path }
}

fn operation_path(op: &Op) -> Option<&[Seg]> {
	match op {
		Op::Replace(_) => None,
		Op::Set { path, .. }
		| Op::Delete { path }
		| Op::Append { path, .. }
		| Op::Truncate { path, .. }
		| Op::Splice { path, .. }
		| Op::Permute { path, .. } => Some(path),
	}
}

fn tool(name: &str, output_limits: Option<ToolOutputLimits>, execute: ToolExecute) -> Arc<ToolRegistration> {
	Arc::new(define_tool(ToolRegistration {
		tool: pi_ai::types::Tool {
			name: name.to_string(),
			description: name.to_string(),
			parameters: json!({ "type": "object", "properties": {} }),
			constrained_sampling: None,
		},
		replay: None,
		execution_mode: None,
		prepare_arguments: None,
		output_limits,
		execute,
	}))
}

fn empty_tool(name: &str) -> Arc<ToolRegistration> {
	tool(name, None, Arc::new(|_, _, _| Box::pin(async { ToolExecutionResult::default() })))
}

fn tool_call_step(calls: &[(&str, &str)]) -> FauxResponseStep {
	let blocks = calls
		.iter()
		.map(|(name, call_id)| FauxContentBlock::from(faux_tool_call(*name, Default::default(), Some(call_id))))
		.collect::<Vec<_>>();
	FauxResponseStep::Message(faux_assistant_message(
		blocks,
		FauxMessageOptions { stop_reason: Some(StopReason::ToolUse), ..Default::default() },
	))
}

fn answer_step(text: &str) -> FauxResponseStep {
	FauxResponseStep::Message(faux_assistant_message(text, FauxMessageOptions::default()))
}

struct Chat {
	harness: Arc<Harness>,
	root: Arc<Conversation>,
}

async fn open_chat(
	storage: Arc<dyn Storage>,
	tools: Vec<Arc<ToolRegistration>>,
	responses: Vec<FauxResponseStep>,
	faux_options: RegisterFauxProviderOptions,
	settings: HarnessSettings,
) -> Chat {
	let registry: Arc<RegistryImpl> = Arc::new(create_registry());
	registry.install(Extension::new("live-deltas").with_tools(tools));
	let faux = faux_provider(faux_options);
	faux.set_responses(responses);
	let models = pi_ai::create_models(None);
	models.set_provider(faux.provider.clone());
	let options = HarnessOptions { models, registry, settings: Some(settings), ..HarnessOptions::default() };
	let harness = open_harness(storage, options, &context()).await.expect("open harness");
	let root = Arc::new(harness.root(&context(), None).await.expect("create root conversation"));
	root.configure(
		AgentChange {
			model: Some(SetOrClear::Set(ModelRef { provider: "faux".into(), model_id: "faux-1".into() })),
			..Default::default()
		},
		&context(),
	)
	.await
	.expect("configure faux model");
	Chat { harness, root }
}

async fn submit(chat: &Chat) -> SubmissionHandle {
	chat.root
		.submit(
			SubmissionDraft::Input(InputSubmissionDraft {
				request_id: None,
				content: pi_ai::types::UserContentField::Text("go".to_string()),
				when_busy: None,
			}),
			&context(),
		)
		.await
		.expect("submit input")
}

async fn live(chat: &Chat) -> LiveState {
	let value = chat
		.harness
		.session
		.snapshot(&live_doc(), Some(chat.root.id.0), None, &context())
		.await
		.expect("read pi.live")
		.expect("pi.live document exists");
	serde_json::from_value(Value::Object((*value).clone())).expect("pi.live state is valid")
}

async fn wait_for_running_tool(chat: &Chat) {
	tokio::time::timeout(Duration::from_secs(30), async {
		loop {
			let state = live(chat).await;
			if state.tools.as_ref().is_some_and(|tools| tools.iter().any(|slot| slot.status == SlotStatus::Running)) {
				return;
			}
			tokio::time::sleep(Duration::from_millis(5)).await;
		}
	})
	.await
	.expect("tool entered running state");
}

#[derive(Debug)]
enum CommandKind {
	Output(String),
	Details(Value),
	Diagnostic(ToolDiagnostic),
	Finish,
}

struct ToolCommand {
	kind: CommandKind,
	ack: oneshot::Sender<()>,
}

struct Driven {
	chat: Chat,
	capture: LiveCapture,
	sender: mpsc::UnboundedSender<ToolCommand>,
	submission: SubmissionHandle,
}

impl Driven {
	async fn step(&self, kind: CommandKind) -> Vec<Op> {
		let before = self.capture.len();
		let (ack, received) = oneshot::channel();
		self.sender.send(ToolCommand { kind, ack }).expect("tool is waiting for a command");
		received.await.expect("tool finished the command");
		wait_for_change(&self.capture, before).await.ops
	}

	async fn finish(&self) -> Vec<LiveChange> {
		let before = self.capture.len();
		let (ack, received) = oneshot::channel();
		self.sender.send(ToolCommand { kind: CommandKind::Finish, ack }).expect("tool is waiting for a command");
		received.await.expect("tool received finish command");
		self.submission.wait(&context()).await.expect("wait for run");
		self.capture.snapshot().into_iter().skip(before).collect()
	}
}

async fn wait_for_change(capture: &LiveCapture, before: usize) -> LiveChange {
	tokio::time::timeout(Duration::from_secs(30), async {
		loop {
			let changes = capture.snapshot();
			if changes.len() > before {
				return changes[before].clone();
			}
			tokio::time::sleep(Duration::from_millis(2)).await;
		}
	})
	.await
	.expect("next pi.live commit arrives")
}

async fn drive(output_limits: Option<ToolOutputLimits>) -> Driven {
	let (sender, receiver) = mpsc::unbounded_channel::<ToolCommand>();
	let receiver = Arc::new(tokio::sync::Mutex::new(receiver));
	let execute: ToolExecute = Arc::new(move |_args, api, call_context| {
		let receiver = receiver.clone();
		Box::pin(async move {
			loop {
				let signal = call_context.abort_signal();
				let command = match signal.as_ref() {
					Some(signal) => {
						let mut receiver = receiver.lock().await;
						tokio::select! {
							command = receiver.recv() => command,
							_ = signal.aborted_future() => return ToolExecutionResult::default(),
						}
					},
					None => receiver.lock().await.recv().await,
				};
				let Some(command) = command else { return ToolExecutionResult::default() };
				let finish = matches!(&command.kind, CommandKind::Finish);
				match command.kind {
					CommandKind::Output(text) => api.output(text.as_bytes(), None),
					CommandKind::Details(value) => api.details(value, &call_context).await.expect("commit tool details"),
					CommandKind::Diagnostic(diagnostic) => api.diagnostic(diagnostic),
					CommandKind::Finish => {}
				}
				let _ = command.ack.send(());
				if finish {
					return ToolExecutionResult::default();
				}
			}
		})
	});
	let mut settings = HarnessSettings::default();
	settings.progress = Some(crate::harness::types::ProgressPolicyOverrides { output_interval_ms: Some(0), ..Default::default() });
	let chat = open_chat(
		Arc::new(MemoryStorage::new()),
		vec![tool("drive", output_limits, execute)],
		vec![tool_call_step(&[("drive", "c1")]), answer_step("done")],
		RegisterFauxProviderOptions::default(),
		settings,
	)
	.await;
	let capture = capture_live(&chat.harness);
	let submission = submit(&chat).await;
	wait_for_running_tool(&chat).await;
	Driven { chat, capture, sender, submission }
}

async fn close(chat: &Chat) {
	chat.harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn generation_hands_off_to_tools_and_back_with_field_level_live_deltas() {
	let chat = open_chat(
		Arc::new(MemoryStorage::new()),
		vec![empty_tool("noop")],
		vec![tool_call_step(&[("noop", "c1")]), answer_step("done")],
		RegisterFauxProviderOptions::default(),
		HarnessSettings::default(),
	)
	.await;
	let capture = capture_live(&chat.harness);
	let submission = submit(&chat).await;
	assert_eq!(submission.wait(&context()).await.expect("wait for run").status, SubmissionStatus::Done);
	let changes = capture.snapshot();
	assert_replayable(&changes);
	assert_eq!(changes.len(), 8, "live commit sequence: {changes:#?}");

	let first_generation = changes[0].state.run.as_ref().expect("started run").task_id;
	assert_eq!(
		changes[0].ops,
		vec![set(path(&["run"]), json!({"taskId": first_generation, "inputs": [submission.id]}))]
	);
	assert_eq!(changes[1].ops, vec![set(path(&["generation"]), json!({"attempt": 1}))]);

	let tool_task = changes[2].state.tools.as_ref().expect("first tool round")[0].task_id.expect("created tool task");
	assert_eq!(changes[2].ops.len(), 2);
	assert!(changes[2].ops.contains(&delete(path(&["generation"]))));
	assert!(changes[2].ops.contains(&set(
		path(&["tools"]),
		json!([{"callId":"c1","name":"noop","taskId":tool_task,"status":"pending"}])
	)));
	assert_eq!(changes[3].ops, vec![set(slot_path(0, "status"), json!("running"))]);

	let result = changes[4].state.tools.as_ref().expect("tool result state")[0].entry.expect("tool result entry");
	assert_eq!(changes[4].ops.len(), 2);
	assert!(changes[4].ops.contains(&set(slot_path(0, "status"), json!("done"))));
	assert!(changes[4].ops.contains(&set(slot_path(0, "entry"), json!(result))));

	let next_generation = changes[5].state.run.as_ref().expect("run handed to next generation").task_id;
	assert_eq!(changes[5].ops.len(), 2);
	assert!(changes[5].ops.contains(&delete(path(&["tools"]))));
	assert!(changes[5].ops.contains(&set(path(&["run", "taskId"]), json!(next_generation))));
	assert_eq!(changes[6].ops, vec![set(path(&["generation"]), json!({"attempt": 1}))]);
	assert_eq!(changes[7].ops.len(), 2);
	assert!(changes[7].ops.contains(&delete(path(&["run"]))));
	assert!(changes[7].ops.contains(&delete(path(&["generation"]))));
	close(&chat).await;
}

#[tokio::test]
async fn head_output_appends_until_full_then_only_updates_dropped_counts() {
	let run = drive(Some(ToolOutputLimits { max_bytes: None, max_lines: Some(2), retain: None })).await;
	assert_eq!(run.step(CommandKind::Output("one\n".into())).await, vec![set(output_path(), json!("one\n"))]);
	assert_eq!(run.step(CommandKind::Output("two\n".into())).await, vec![append(output_path(), "two\n")]);
	assert_eq!(
		run.step(CommandKind::Output("three\n".into())).await,
		vec![set(slot_path(0, "droppedBytes"), json!(6)), set(slot_path(0, "droppedLines"), json!(1))]
	);
	assert_eq!(
		run.step(CommandKind::Output("four\n".into())).await,
		vec![set(slot_path(0, "droppedBytes"), json!(11)), set(slot_path(0, "droppedLines"), json!(2))]
	);
	let _ = run.finish().await;
	assert_replayable(&run.capture.snapshot());
	close(&run.chat).await;
}

#[tokio::test]
async fn tail_output_slides_as_front_trims_and_appends() {
	let run = drive(Some(ToolOutputLimits { max_bytes: None, max_lines: Some(3), retain: Some(crate::harness::api::OutputRetain::Tail) })).await;
	assert_eq!(
		run.step(CommandKind::Output("line 1\nline 2\nline 3\n".into())).await,
		vec![set(output_path(), json!("line 1\nline 2\nline 3\n"))]
	);
	assert_eq!(
		run.step(CommandKind::Output("line 4\n".into())).await,
		vec![
			truncate(output_path(), 7),
			append(output_path(), "line 4\n"),
			set(slot_path(0, "droppedBytes"), json!(7)),
			set(slot_path(0, "droppedLines"), json!(1)),
		]
	);
	assert_eq!(
		run.step(CommandKind::Output("line 5\nline 6\n".into())).await,
		vec![
			truncate(output_path(), 14),
			append(output_path(), "line 5\nline 6\n"),
			set(slot_path(0, "droppedBytes"), json!(21)),
			set(slot_path(0, "droppedLines"), json!(3)),
		]
	);
	assert_eq!(live(&run.chat).await.tools.as_ref().unwrap()[0].output.as_deref(), Some("line 4\nline 5\nline 6\n"));
	let _ = run.finish().await;
	assert_replayable(&run.capture.snapshot());
	close(&run.chat).await;
}

#[tokio::test]
async fn bounded_overlap_search_replaces_wide_windows_and_preserves_repetitive_overlap() {
	let wide = drive(Some(ToolOutputLimits {
		max_bytes: Some(100 * 1024),
		max_lines: Some(1_000_000),
		retain: Some(crate::harness::api::OutputRetain::Tail),
	}))
	.await;
	let line = |index: usize| format!("{index:010} {}\n", "x".repeat(989));
	let text = (0..100).map(line).collect::<String>();
	let _ = wide.step(CommandKind::Output(text)).await;
	let sliding = (100..104).map(line).collect::<String>();
	let wide_delta = wide.step(CommandKind::Output(sliding)).await;
	let output_delta: Vec<_> = wide_delta
		.iter()
		.filter(|op| operation_path(op) == Some(output_path().as_slice()))
		.cloned()
		.collect();
	assert!(matches!(output_delta.as_slice(), [Op::Set { .. }]));
	let _ = wide.finish().await;
	assert_replayable(&wide.capture.snapshot());
	close(&wide.chat).await;

	let repetitive = drive(Some(ToolOutputLimits {
		max_bytes: None,
		max_lines: Some(50),
		retain: Some(crate::harness::api::OutputRetain::Tail),
	}))
	.await;
	let _ = repetitive.step(CommandKind::Output("y\n".repeat(50))).await;
	let repeated = repetitive.step(CommandKind::Output("z\n".into())).await;
	let output_delta = repeated
		.iter()
		.filter(|op| operation_path(op) == Some(output_path().as_slice()))
		.cloned()
		.collect::<Vec<_>>();
	assert_eq!(output_delta, vec![truncate(output_path(), 2), append(output_path(), "z\n")]);
	let _ = repetitive.finish().await;
	assert_replayable(&repetitive.capture.snapshot());
	close(&repetitive.chat).await;
}

#[tokio::test]
async fn details_diff_leaf_by_leaf_and_diagnostics_append_until_settlement() {
	let run = drive(None).await;
	let details = slot_path(0, "details");
	assert_eq!(
		run.step(CommandKind::Details(json!({"step": 1, "log": "a"}))).await,
		vec![set(details.clone(), json!({"step": 1, "log": "a"}))]
	);
	let second_details = run.step(CommandKind::Details(json!({"step": 2, "log": "ab"}))).await;
	let mut step = details.clone();
	step.push(Seg::key("step"));
	let mut log = details.clone();
	log.push(Seg::key("log"));
	assert!(second_details.contains(&set(step, json!(2))), "detail deltas: {second_details:?}");
	assert!(second_details.contains(&append(log.clone(), "b")));
	assert_eq!(run.step(CommandKind::Details(json!({"step": 2}))).await, vec![delete(log)]);

	let diagnostics = slot_path(0, "diagnostics");
	let first = ToolDiagnostic { severity: ToolDiagnosticSeverity::Info, message: "first".into(), code: None };
	let second = ToolDiagnostic { severity: ToolDiagnosticSeverity::Warn, message: "second".into(), code: None };
	assert_eq!(
		run.step(CommandKind::Diagnostic(first.clone())).await,
		vec![set(diagnostics.clone(), json!([first]))]
	);
	assert_eq!(
		run.step(CommandKind::Diagnostic(second.clone())).await,
		vec![Op::Splice { path: diagnostics, start: 1, remove: 0, items: vec![json!(second)] }]
	);
	let settlement = run.finish().await;
	let settled = settlement
		.iter()
		.find(|change| change.state.tools.as_ref().is_some_and(|slots| slots[0].status == SlotStatus::Done))
		.expect("tool settlement pi.live update");
	assert!(settled.ops.contains(&set(slot_path(0, "status"), json!("done"))));
	assert!(settled.ops.contains(&delete(slot_path(0, "details"))));
	assert!(settled.ops.contains(&delete(slot_path(0, "diagnostics"))));
	assert_replayable(&run.capture.snapshot());
	close(&run.chat).await;
}

#[tokio::test]
async fn generation_partials_start_with_a_message_write_then_append_text() {
	let text = "word ".repeat(250);
	let chat = open_chat(
		Arc::new(MemoryStorage::new()),
		vec![],
		vec![FauxResponseStep::Message(faux_assistant_message(text.as_str(), FauxMessageOptions::default()))],
		RegisterFauxProviderOptions {
			tokens_per_second: Some(400.0),
			token_size: Some(FauxTokenSize { min: Some(4), max: Some(4) }),
			..Default::default()
		},
		HarnessSettings::default(),
	)
	.await;
	let capture = capture_live(&chat.harness);
	let receipt = submit(&chat).await;
	assert_eq!(receipt.wait(&context()).await.expect("wait for generation").status, SubmissionStatus::Done);
	let changes = capture.snapshot();
	let message_path = path(&["generation", "message"]);
	let text_path = [
		Seg::key("generation"),
		Seg::key("message"),
		Seg::key("content"),
		Seg::index(0),
		Seg::key("text"),
	];
	let partials: Vec<_> = changes
		.iter()
		.filter(|change| change.ops.iter().any(|op| {
			operation_path(op) == Some(message_path.as_slice()) || operation_path(op) == Some(text_path.as_slice())
		}))
		.collect();
	assert!(partials.len() > 2, "partial deltas: {partials:#?}");
	assert!(matches!(partials[0].ops.as_slice(), [Op::Set { path, value }] if path == &message_path && value.is_object()));
	for change in &partials[1..] {
		assert!(matches!(change.ops.as_slice(), [Op::Append { path, text }] if path.as_slice() == text_path.as_slice() && !text.is_empty()));
	}
	assert_replayable(&changes);
	close(&chat).await;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RevisionKind {
	Base,
	Delta,
}

struct RecordingStorage {
	inner: Arc<MemoryStorage>,
	live_id: Mutex<Option<DocumentId>>,
	revisions: Mutex<Vec<(Seq, RevisionKind)>>,
	reject_tool_call: Option<String>,
	rejected_tool_calls: Mutex<Vec<String>>,
}

impl RecordingStorage {
	fn new(reject_tool_call: Option<&str>) -> Self {
		Self {
			inner: Arc::new(MemoryStorage::new()),
			live_id: Mutex::new(None),
			revisions: Mutex::new(Vec::new()),
			reject_tool_call: reject_tool_call.map(str::to_string),
			rejected_tool_calls: Mutex::new(Vec::new()),
		}
	}

	fn clear_revisions(&self) {
		self.revisions.lock().clear();
	}

	fn revisions(&self) -> Vec<(Seq, RevisionKind)> {
		self.revisions.lock().clone()
	}

	fn rejected_tool_calls(&self) -> Vec<String> {
		self.rejected_tool_calls.lock().clone()
	}

	fn is_rejected_tool_result(&self, writes: &[StorageWrite]) -> Option<String> {
		let rejected = self.reject_tool_call.as_ref()?;
		writes.iter().find_map(|write| {
			let StorageWrite::Entry { value } = write else { return None };
			if value.kind != "pi.tool-result" {
				return None;
			}
			value.model.as_ref()?.iter().find_map(|message| match message {
				Message::ToolResult(result) if result.tool_call_id.as_str() == rejected.as_str() => Some(rejected.clone()),
				_ => None,
			})
		})
	}
}

#[async_trait::async_trait]
impl Storage for RecordingStorage {
	async fn commit(&self, writes: &[StorageWrite], context: &Context) -> Result<Seq, StorageRejected> {
		if let Some(call_id) = self.is_rejected_tool_result(writes) {
			self.rejected_tool_calls.lock().push(call_id);
			return Err(StorageRejected::new("test rejected tool result commit"));
		}
		let mut live_id = *self.live_id.lock();
		let mut kind = None;
		for write in writes {
			match write {
				StorageWrite::DocumentCreate { record, .. } if record.kind == "pi.live" => {
					live_id = Some(record.id);
					kind = Some(RevisionKind::Base);
				}
				StorageWrite::DocumentChange { id, content } if Some(*id) == live_id => {
					kind = Some(match content {
						DocumentContent::Base { .. } => RevisionKind::Base,
						DocumentContent::Delta { .. } => RevisionKind::Delta,
					});
				}
				_ => {}
			}
		}
		*self.live_id.lock() = live_id;
		let seq = self.inner.commit(writes, context).await?;
		if let Some(kind) = kind {
			self.revisions.lock().push((seq, kind));
		}
		Ok(seq)
	}

	async fn mint_id(&self) -> u64 {
		self.inner.mint_id().await
	}

	async fn conversation(&self, id: ConversationId, context: &Context) -> Option<ConversationRecord> {
		self.inner.conversation(id, context).await
	}

	async fn scan_conversations(
		&self,
		query: &ConversationQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<ConversationRecord> {
		self.inner.scan_conversations(query, limit, cursor, context).await
	}

	async fn entry(&self, id: EntryId, context: &Context) -> Option<(EntryRecord, Seq)> {
		self.inner.entry(id, context).await
	}

	async fn entry_in(&self, conversation_id: ConversationId, id: EntryId, context: &Context) -> Option<(EntryRecord, Seq)> {
		self.inner.entry_in(conversation_id, id, context).await
	}

	async fn find_latest_head_marker(
		&self,
		conversation_id: ConversationId,
		at_or_before_entry_id: Option<EntryId>,
		context: &Context,
	) -> Option<EntryRecord> {
		self.inner.find_latest_head_marker(conversation_id, at_or_before_entry_id, context).await
	}

	async fn scan_entries(
		&self,
		query: &EntryQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<EntryRecord> {
		self.inner.scan_entries(query, limit, cursor, context).await
	}

	async fn task(&self, id: TaskId, context: &Context) -> Option<TaskRecord> {
		self.inner.task(id, context).await
	}

	async fn scan_tasks(&self, query: &TaskQuery, limit: u64, cursor: Option<&Cursor>, context: &Context) -> Page<TaskRecord> {
		self.inner.scan_tasks(query, limit, cursor, context).await
	}

	async fn submission(&self, id: SubmissionId, context: &Context) -> Option<SubmissionRecord> {
		self.inner.submission(id, context).await
	}

	async fn scan_submissions(
		&self,
		query: &SubmissionQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<SubmissionRecord> {
		self.inner.scan_submissions(query, limit, cursor, context).await
	}

	async fn submission_by_request(&self, conversation_id: ConversationId, request_id: &str, context: &Context) -> Option<SubmissionRecord> {
		self.inner.submission_by_request(conversation_id, request_id, context).await
	}

	async fn find_document(&self, address: &DocumentAddress, at: DocumentPoint, context: &Context) -> Option<DocumentRecord> {
		self.inner.find_document(address, at, context).await
	}

	async fn document(&self, id: DocumentId, at: DocumentPoint, context: &Context) -> Option<StoredDocument> {
		self.inner.document(id, at, context).await
	}

	async fn scan_documents(
		&self,
		query: &DocumentQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<DocumentRecord> {
		self.inner.scan_documents(query, limit, cursor, context).await
	}

	async fn close(&self, context: &Context) {
		self.inner.close(context).await
	}
}

#[tokio::test]
async fn live_document_uses_complete_bases_exactly_when_nothing_runs() {
	let storage = Arc::new(RecordingStorage::new(None));
	let output_tool = |name: &'static str| {
		let execute: ToolExecute = Arc::new(move |_args, api, _context| {
			Box::pin(async move {
				api.output(format!("{name} output\n").as_bytes(), None);
				tokio::time::sleep(Duration::from_millis(150)).await;
				api.output(format!("{name} more\n").as_bytes(), None);
				ToolExecutionResult::default()
			})
		});
		tool(name, None, execute)
	};
	let mut settings = HarnessSettings::default();
	settings.tool_execution = Some(ToolExecutionMode::Sequential);
	settings.progress = Some(crate::harness::types::ProgressPolicyOverrides { output_interval_ms: Some(0), ..Default::default() });
	let chat = open_chat(
		storage.clone(),
		vec![output_tool("first"), output_tool("second")],
		vec![tool_call_step(&[("first", "a"), ("second", "b")]), answer_step("done")],
		RegisterFauxProviderOptions::default(),
		settings,
	)
	.await;
	storage.clear_revisions();
	let capture = capture_live(&chat.harness);
	let receipt = submit(&chat).await;
	assert_eq!(receipt.wait(&context()).await.expect("wait for sequential tools").status, SubmissionStatus::Done);
	let changes = capture.snapshot();
	assert_replayable(&changes);
	let states: HashMap<_, _> = changes.iter().map(|change| (change.seq, change.state.clone())).collect();
	let revisions = storage.revisions();
	let mut bases = 0;
	let mut saw_delta = false;
	for (seq, kind) in &revisions {
		let state = states.get(seq).expect("published live value for persisted revision");
		let nothing_runs = state.generation.is_none()
			&& !state.tools.as_ref().is_some_and(|slots| slots.iter().any(|slot| slot.status == SlotStatus::Running));
		assert_eq!(*kind == RevisionKind::Base, nothing_runs, "revision at {seq} was {kind:?}: {state:#?}");
		bases += usize::from(*kind == RevisionKind::Base);
		saw_delta |= *kind == RevisionKind::Delta;
	}
	assert!(bases >= 5, "expected handover, between-tool, round, and idle bases; got {revisions:#?}");
	assert!(saw_delta, "running tool output must be stored as a delta");
	close(&chat).await;
}

#[tokio::test]
async fn unoffered_tool_is_done_and_faulted_tool_cleanup_only_marks_its_slot_done() {
	let storage = Arc::new(RecordingStorage::new(Some("b")));
	let chat = open_chat(
		storage.clone(),
		vec![empty_tool("bad")],
		vec![tool_call_step(&[("ghost", "g"), ("bad", "b")]), answer_step("done")],
		RegisterFauxProviderOptions::default(),
		HarnessSettings::default(),
	)
	.await;
	let capture = capture_live(&chat.harness);
	let receipt = submit(&chat).await;
	assert_eq!(receipt.wait(&context()).await.expect("wait for faulted tool round").status, SubmissionStatus::Done);
	assert_eq!(storage.rejected_tool_calls(), vec![String::from("b")]);
	let entries = chat.root.entries(Default::default(), 20, None, &context()).await.expect("read tool result entries");
	let ghost = entries.items.iter().find(|entry| entry.kind == "pi.tool-result").expect("unoffered call result");
	let ghost_call_id = ghost.model.as_ref().and_then(|messages| messages.iter().find_map(|message| match message {
		Message::ToolResult(result) => Some(result.tool_call_id.as_str()),
		_ => None,
	}));
	assert_eq!(ghost_call_id, Some("g"));
	assert_eq!(entries.items.iter().filter(|entry| entry.kind == "pi.tool-result").count(), 1);

	let changes = capture.snapshot();
	assert_replayable(&changes);
	let handover = changes
		.iter()
		.find(|change| change.state.tools.as_ref().is_some_and(|slots| slots.len() == 2))
		.expect("tool round handover");
	let slots = handover.state.tools.as_ref().unwrap();
	assert_eq!((slots[0].call_id.as_str(), slots[0].name.as_str(), slots[0].status), ("g", "ghost", SlotStatus::Done));
	assert_eq!(slots[0].entry, Some(ghost.id));
	assert!(slots[0].task_id.is_none());
	assert_eq!((slots[1].call_id.as_str(), slots[1].name.as_str(), slots[1].status), ("b", "bad", SlotStatus::Pending));
	assert!(slots[1].task_id.is_some());
	assert!(handover.ops.contains(&set(path(&["tools"]), serde_json::to_value(slots).expect("tool slots serialize"))));

	let cleanup = changes
		.iter()
		.find(|change| change.ops == vec![set(slot_path(1, "status"), json!("done"))])
		.expect("fault cleanup writes only the slot status");
	let faulted = &cleanup.state.tools.as_ref().expect("live tool round remains")[1];
	assert_eq!(faulted.status, SlotStatus::Done);
	assert!(faulted.entry.is_none());
	close(&chat).await;
}

#[tokio::test]
async fn handover_publishes_assistant_generation_and_both_tool_tasks_atomically() {
	let chat = open_chat(
		Arc::new(MemoryStorage::new()),
		vec![empty_tool("noop")],
		vec![tool_call_step(&[("noop", "a"), ("noop", "b")]), answer_step("done")],
		RegisterFauxProviderOptions::default(),
		HarnessSettings::default(),
	)
	.await;
	let handover: Arc<Mutex<Option<CommitPublication>>> = Arc::default();
	let handover_capture = handover.clone();
	let _unsubscribe = chat.harness.session.subscribe_commits(Arc::new(move |publication, _context| {
		for change in &publication.changes {
			let CommitChange::Document(DocumentCommitChange::Document { record, value, .. }) = change else { continue };
			if record.kind != "pi.live" {
				continue;
			}
			let Some(value) = value else { continue };
			let state: LiveState = serde_json::from_value(Value::Object(value.clone())).expect("pi.live state");
			if state.tools.as_ref().is_some_and(|tools| tools.len() == 2) {
				let mut slot = handover_capture.lock();
				if slot.is_none() {
					*slot = Some(publication.clone());
				}
			}
		}
	}));
	let receipt = submit(&chat).await;
	assert_eq!(receipt.wait(&context()).await.expect("wait for tool round").status, SubmissionStatus::Done);
	let publication = handover.lock().clone().expect("atomic handover publication");
	let mut kinds: Vec<String> = publication
		.changes
		.iter()
		.filter_map(|change| match change {
			CommitChange::Table(crate::types::TableCommitChange::Entry { value }) => Some(value.kind.clone()),
			CommitChange::Table(crate::types::TableCommitChange::Task { value }) => Some(value.kind.clone()),
			_ => None,
		})
		.collect();
	kinds.sort();
	assert_eq!(kinds, ["pi.assistant", "pi.generation", "pi.tool", "pi.tool"]);
	close(&chat).await;
}

#[tokio::test]
async fn aborting_a_running_tool_commits_its_slot_with_field_level_ops() {
	let run = drive(None).await;
	assert_eq!(run.step(CommandKind::Output("partial\n".into())).await, vec![set(output_path(), json!("partial\n"))]);
	let details = slot_path(0, "details");
	assert_eq!(run.step(CommandKind::Details(json!({"n": 1}))).await, vec![set(details.clone(), json!({"n": 1}))]);
	let task_id = live(&run.chat).await.tools.as_ref().unwrap()[0].task_id.expect("running tool task");
	let before = run.capture.len();
	assert_eq!(run.chat.harness.abort_task(task_id, &context()).await.expect("mark task aborted"), "marked");
	let abort_commit = tokio::time::timeout(Duration::from_secs(30), async {
		loop {
			if let Some(change) = run.capture.snapshot().into_iter().skip(before).find(|change| {
				change.ops.iter().any(|op| op == &set(slot_path(0, "status"), json!("done")))
			}) {
				return change;
			}
			tokio::time::sleep(Duration::from_millis(2)).await;
		}
	})
	.await
	.expect("abort commit marks tool slot done");
	assert_eq!(abort_commit.ops.len(), 4);
	assert!(abort_commit.ops.contains(&set(slot_path(0, "status"), json!("done"))));
	assert!(abort_commit.ops.contains(&set(slot_path(0, "entry"), json!(abort_commit.state.tools.as_ref().unwrap()[0].entry.unwrap()))));
	assert!(abort_commit.ops.contains(&delete(output_path())));
	assert!(abort_commit.ops.contains(&delete(details)));
	assert_replayable(&run.capture.snapshot());
	close(&run.chat).await;
}
