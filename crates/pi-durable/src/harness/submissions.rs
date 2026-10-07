//! Port of `packages/durable/src/harness/submissions.ts`.
//!
//! Deviations: D1 — `Submissions` holds `Arc`s of its collaborators (no
//! class-by-reference cycles); `SubmissionHandle` is an owned value.
//! D2 — `wait` registers through [`crate::harness::util::Waiters`]; the
//! settled record is delivered through a oneshot channel (see util D1).

use std::sync::Arc;

use pi_chord::context::Context;
use serde_json::Value;

use crate::entries::USER_ENTRY;
use crate::errors::ConversationBusy;
use crate::harness::generation::start_run;
use crate::harness::inbox::{
	apply_boundary, is_stale, prepare_boundary, remove_inbox_item, read_inbox, BoundaryAt, QueueModes,
};
use crate::harness::live::{live_doc, store_live};
use crate::harness::types::{
	ConversationId, InputSubmissionDraft, SettledSubmissionRecord, Submission, SubmissionAbort, SubmissionDraft,
	SubmissionId, SubmissionRecord, SubmissionStatus, SubmissionType, WriteSubmissionDraft,
};
use crate::harness::util::{closed_error, Waiters};
use crate::session::Session;
use crate::types::{CommitPublication, EntryDraft, Storage, SubmissionCreate};

/// Admission, waits, and withdrawal of the durable submissions of one
/// Harness.
pub struct Submissions {
	session: Arc<Session>,
	storage: Arc<dyn Storage>,
	now: Box<dyn Fn() -> i64 + Send + Sync>,
	/// Read at each admission, on the Session line.
	queue_modes: Box<dyn Fn() -> QueueModes + Send + Sync>,
	/// Enable task scheduling; submitting or waiting asks for progress.
	resume: Box<dyn Fn() + Send + Sync>,
	waiters: Waiters<SubmissionId, SettledSubmissionRecord>,
	closed: std::sync::atomic::AtomicBool,
}

/// Result of aborting a queued or placed submission.
pub type AbortResult = SubmissionAbort;
type SubmissionWaiter =
	std::pin::Pin<Box<dyn std::future::Future<Output = Result<SettledSubmissionRecord, String>> + Send + 'static>>;

enum SubmissionWait {
	Settled(SettledSubmissionRecord),
	Pending(SubmissionWaiter),
}

impl Submissions {
	pub fn new(
		session: Arc<Session>,
		storage: Arc<dyn Storage>,
		now: Box<dyn Fn() -> i64 + Send + Sync>,
		queue_modes: Box<dyn Fn() -> QueueModes + Send + Sync>,
		resume: Box<dyn Fn() + Send + Sync>,
	) -> Arc<Self> {
		let submissions = Arc::new(Self {
			session,
			storage,
			now,
			queue_modes,
			resume,
			waiters: Waiters::default(),
			closed: std::sync::atomic::AtomicBool::new(false),
		});
		let observer = submissions.clone();
		submissions.session.subscribe_commits(Arc::new(move |publication: &CommitPublication, _context: &Context| {
			observer.observe(publication);
		}));
		let closer = submissions.clone();
		submissions.session.subscribe_close(Arc::new(move || {
			closer.closed.store(true, std::sync::atomic::Ordering::SeqCst);
			closer.waiters.reject_all(&closed_error());
		}));
		submissions
	}

	/// Admit a submission in one commit; see [`admit_submission`].
	pub async fn submit(
		self: &Arc<Self>,
		conversation_id: ConversationId,
		draft: SubmissionDraft,
		context: &Context,
	) -> Result<SubmissionHandle, String> {
		(self.resume)();
		let now = (self.now)();
		let modes = (self.queue_modes)();
		let id = self
			.session
			.commit(
				move |tx| admit_submission(tx, conversation_id, &draft, now, modes),
				context,
			)
			.await
			.map_err(|e| e.to_string())?;
		Ok(SubmissionHandle { id, submissions: self.clone() })
	}

	/// Handle for an existing submission, or `None`.
	pub async fn get(
		self: &Arc<Self>,
		id: SubmissionId,
		context: &Context,
	) -> Result<Option<SubmissionHandle>, String> {
		let record = self
			.session
			.read_on_line(async move {
				self.storage
					.submission(id, context)
					.await
					.ok_or_else(|| format!("Submission {id} does not exist"))
			})
			.await?;
		Ok(Some(SubmissionHandle { id: record.id, submissions: self.clone() }))
	}

	pub async fn status(&self, id: SubmissionId, context: &Context) -> Result<SubmissionRecord, String> {
		self.session
			.read_on_line(async move {
				self.storage
					.submission(id, context)
					.await
					.ok_or_else(|| format!("Submission {id} does not exist"))
			})
			.await
	}

	pub async fn wait(
		self: &Arc<Self>,
		id: SubmissionId,
		context: &Context,
	) -> Result<SettledSubmissionRecord, String> {
		(self.resume)();
		// Check and register under the Session line; await after releasing it.
		let submissions = self.clone();
		let wait = self
			.session
			.read_on_line(async move {
				match submissions.storage.submission(id, context).await {
					None => Err(format!("Submission {id} does not exist")),
					Some(record) if crate::harness::types::is_settled(&record) => Ok(SubmissionWait::Settled(record)),
					Some(_) => {
						if submissions.closed.load(std::sync::atomic::Ordering::SeqCst) {
							return Err(closed_error());
						}
						Ok(SubmissionWait::Pending(submissions.waiters.add(id, context)))
					}
				}
			})
			.await?;
		match wait {
			SubmissionWait::Settled(record) => Ok(record),
			SubmissionWait::Pending(waiter) => waiter.await,
		}
	}

	/// Withdraw a queued submission and remove its inbox item; placed inputs
	/// and settled submissions are reported.
	pub async fn abort(
		self: &Arc<Self>,
		id: SubmissionId,
		context: &Context,
		conversation_id: Option<ConversationId>,
	) -> Result<Option<AbortResult>, String> {
		self.session
			.commit(
				move |tx| {
					let record = tx.submission(id).map_err(|e| e.to_string())?;
					let Some(record) = record else { return Ok(None) };
					if conversation_id.is_some_and(|at| record.conversation_id != at) {
						return Ok(None);
					}
					if record.status == SubmissionStatus::Queued {
						tx.settle_submission(
							id,
							crate::types::SubmissionSettlement::Unanswered {
								reason: "aborted".to_string(),
								detail: None,
							},
						)
						.map_err(|e| e.to_string())?;
						remove_inbox_item_sync(tx, record.conversation_id, id)?;
						return Ok(Some(SubmissionAbort::Aborted));
					}
					Ok(Some(match record.status {
						SubmissionStatus::Placed => SubmissionAbort::AlreadyPlaced,
						_ => SubmissionAbort::Settled,
					}))
				},
				context,
			)
			.await
			.map_err(|e| e.to_string())
	}

	fn observe(&self, publication: &CommitPublication) {
		for change in &publication.changes {
			let crate::types::CommitChange::Table(crate::types::TableCommitChange::Submission { value }) = change
			else {
				continue;
			};
			if !crate::harness::types::is_settled(value) {
				continue;
			}
			self.waiters.resolve(&value.id, value.clone());
		}
	}
}

/// Synchronous form of `remove_inbox_item` used inside commits.
fn remove_inbox_item_sync(
	tx: &mut crate::session::Tx,
	conversation_id: ConversationId,
	id: SubmissionId,
) -> Result<(), String> {
	crate::harness::inbox::remove_inbox_item(tx, conversation_id, id)
}

/// Handle of one durable submission (upstream `SubmissionHandle`).
#[derive(Clone)]
pub struct SubmissionHandle {
	pub id: SubmissionId,
	submissions: Arc<Submissions>,
}

impl std::fmt::Debug for SubmissionHandle {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.debug_struct("SubmissionHandle").field("id", &self.id).finish_non_exhaustive()
	}
}

#[async_trait::async_trait]
impl Submission for SubmissionHandle {
	fn id(&self) -> SubmissionId {
		self.id
	}

	async fn status(&self, context: &Context) -> Result<SubmissionRecord, String> {
		self.submissions.status(self.id, context).await
	}

	async fn wait(&self, context: &Context) -> Result<SettledSubmissionRecord, String> {
		self.submissions.wait(self.id, context).await
	}

	async fn abort(&self, context: &Context) -> Result<AbortResult, String> {
		self.submissions
			.abort(self.id, context, None)
			.await?
			.ok_or_else(|| format!("Submission {} does not exist", self.id))
	}
}

/// Admit a submission inside a commit (spec §6); `Conversation.submit()` and
/// conversation-owned compactions share it. A known request ID returns its
/// existing submission without writing. A busy conversation queues it in
/// `pi.inbox`, or rejects `whenBusy: reject` input with `ConversationBusy`.
/// An idle conversation with queued items queues it behind them and runs a
/// final boundary. Otherwise idle input places a user entry and starts a
/// run, and an idle write appends its entry and settles `done`, or `stale`
/// when its head reaches before the active range.
pub fn admit_submission(
	tx: &mut crate::session::Tx,
	conversation_id: ConversationId,
	draft: &SubmissionDraft,
	now: i64,
	queue_modes: QueueModes,
) -> Result<SubmissionId, String> {
	if let Some(request_id) = draft.request_id() {
		let existing = tx.submission_by_request(conversation_id, request_id).map_err(|e| e.to_string())?;
		if let Some(existing) = existing {
			let existing_type = existing.submission_type;
			let draft_type = match draft {
				SubmissionDraft::Input(_) => SubmissionType::Input,
				SubmissionDraft::Write(_) => SubmissionType::Write,
			};
			if existing_type != draft_type {
				return Err(format!(
					"Request {request_id} already identifies a submission of type {}",
					submission_type_str(existing_type)
				));
			}
			return Ok(existing.id);
		}
	}
	let live_draft = tx.doc(&live_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
	let mut live = crate::harness::live::read_live(&live_draft);
	let busy = live.run.is_some();
	if busy {
		if let SubmissionDraft::Input(input) = draft {
			if input.when_busy == Some(crate::harness::types::WhenBusy::Reject) {
				return Err(ConversationBusy::new(conversation_id).to_string());
			}
		}
	}
	// A boundary reads the table, so it is prepared before the first table
	// write; a busy one needs none.
	let mut boundary = if busy {
		None
	} else {
		Some(prepare_boundary(tx, conversation_id, queue_modes)?)
	};
	if boundary.as_ref().map(|b| b.state.items.is_empty()) != Some(true) {
		let queued = tx
			.create_submission(SubmissionCreate {
				conversation_id,
				request_id: draft.request_id().map(|s| s.to_string()),
				submission_type: Some(match draft {
					SubmissionDraft::Input(_) => SubmissionType::Input,
					SubmissionDraft::Write(_) => SubmissionType::Write,
				}),
				status: Some(SubmissionStatus::Queued),
				entry: None,
				answer: None,
				reason: None,
				detail: None,
			})
			.map_err(|e| e.to_string())?;
		let id = queued.id;
		// Hosts may leave optional fields `undefined`; drafts take strict
		// JSON.
		let item = match draft {
			SubmissionDraft::Write(WriteSubmissionDraft { entry, .. }) => {
				crate::harness::inbox::InboxItem::Write {
					id,
					entry: entry.clone(),
				}
			}
			SubmissionDraft::Input(InputSubmissionDraft { content, when_busy, .. }) => {
				let mode_content = serde_json::to_value(content).expect("UserInput serializes");
				match when_busy {
					Some(crate::harness::types::WhenBusy::Steer) => {
						crate::harness::inbox::InboxItem::Steer { id, content: mode_content }
					}
					_ => crate::harness::inbox::InboxItem::FollowUp { id, content: mode_content },
				}
			}
		};
		match &mut boundary {
			Some(boundary) => {
				boundary.state.items.push(item);
				let result = apply_boundary(tx, boundary, BoundaryAt::Final, now)?;
				if !result.users.is_empty() {
					start_run(tx, conversation_id, &mut live, result.users)?;
					store_live(&live_draft, &live)?;
				}
			}
			None => {
				let inbox = tx.doc(&crate::harness::inbox::inbox_doc(), Some(conversation_id.0), None, None).map_err(|e| e.to_string())?;
				let mut state = read_inbox(&inbox);
				state.items.push(item);
				crate::harness::inbox::store_inbox(&inbox, &state)?;
			}
		}
		return Ok(id);
	}
	if let SubmissionDraft::Write(WriteSubmissionDraft { entry, .. }) = draft {
		if is_stale(boundary.as_ref().expect("checked"), entry) {
			let stale = tx
				.create_submission(SubmissionCreate {
					conversation_id,
					request_id: draft.request_id().map(|s| s.to_string()),
					submission_type: Some(SubmissionType::Write),
					status: Some(SubmissionStatus::Unanswered),
					entry: None,
					answer: None,
					reason: Some("stale".to_string()),
					detail: None,
				})
				.map_err(|e| e.to_string())?;
			return Ok(stale.id);
		}
		let entry_record = tx.append_entry(conversation_id, entry.clone()).map_err(|e| e.to_string())?;
		let write = tx
			.create_submission(SubmissionCreate {
				conversation_id,
				request_id: draft.request_id().map(|s| s.to_string()),
				submission_type: Some(SubmissionType::Write),
				status: Some(SubmissionStatus::Done),
				entry: Some(entry_record.id),
				answer: None,
				reason: None,
				detail: None,
			})
			.map_err(|e| e.to_string())?;
		return Ok(write.id);
	}
	let content = match draft {
		SubmissionDraft::Input(input) => input.content.clone(),
		SubmissionDraft::Write(_) => unreachable!("write handled above"),
	};
	let message = pi_ai::types::UserMessage { content, timestamp: now };
	let mut entry_draft = EntryDraft { kind: Some(USER_ENTRY.kind.clone()), ..Default::default() };
	entry_draft.model = Some(vec![pi_ai::Message::User(message)]);
	let entry = tx.append_entry(conversation_id, entry_draft).map_err(|e| e.to_string())?;
	let input = tx
		.create_submission(SubmissionCreate {
			conversation_id,
			request_id: draft.request_id().map(|s| s.to_string()),
			submission_type: Some(SubmissionType::Input),
			status: Some(SubmissionStatus::Placed),
			entry: Some(entry.id),
			answer: None,
			reason: None,
			detail: None,
		})
		.map_err(|e| e.to_string())?;
	start_run(tx, conversation_id, &mut live, vec![input.id])?;
	store_live(&live_draft, &live)?;
	Ok(input.id)
}

fn submission_type_str(t: SubmissionType) -> &'static str {
	match t {
		SubmissionType::Input => "input",
		SubmissionType::Write => "write",
	}
}

// Unused import guards; Value is used by the inbox item JSON path.
#[allow(unused)]
fn _value_marker(_: Value) {}
