//! Synchronous storage access for the session mutation line.
//!
//! Upstream transaction reads/writes `await` the async `Storage` directly on the
//! JS line. The Rust session runs commit callbacks as synchronous closures
//! (`crate::tasks` D3, `crate::harness::api` seam), so [`StorageBridge`] pumps
//! each storage call through one dedicated worker thread with its own tokio
//! runtime. Read-style calls block on the reply channel: their replies never
//! park, so the wait is safe from any executor context. [`StorageBridge::commit`]
//! is awaited by the caller instead — a storage backend may legitimately park a
//! commit (test gates), and blocking the caller's executor thread for it
//! deadlocks single-thread runtimes.
//!
//! Deviations: D1 — the bridge is a mechanical detail of the sync-Tx facade
//! (session module D1); call order is exactly the caller's, so no upstream
//! ordering semantics change.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};

use pi_chord::context::Context;

use crate::errors::StorageRejected;
use crate::types::{
	ConversationId, ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentId, DocumentPoint,
	DocumentQuery, DocumentRecord, EntryId, EntryQuery, EntryRecord, Page, Seq, StoredDocument, Storage,
	StorageWrite, SubmissionId, SubmissionQuery, SubmissionRecord, TaskId, TaskQuery, TaskRecord,
};

enum Job {
	Commit {
		writes: Vec<StorageWrite>,
		context: Context,
		// Awaited by the caller (settlement runs in async context), so a
		// storage backend that parks the commit (test gates) must not stall
		// the caller's executor thread.
		reply: tokio::sync::oneshot::Sender<Result<Seq, StorageRejected>>,
	},
	MintId {
		reply: Sender<u64>,
	},
	Conversation {
		id: ConversationId,
		context: Context,
		reply: Sender<Option<ConversationRecord>>,
	},
	ScanConversations {
		query: ConversationQuery,
		limit: u64,
		cursor: Option<Cursor>,
		context: Context,
		reply: Sender<Page<ConversationRecord>>,
	},
	Entry {
		id: EntryId,
		context: Context,
		reply: Sender<Option<(EntryRecord, Seq)>>,
	},
	EntryIn {
		conversation_id: ConversationId,
		id: EntryId,
		context: Context,
		reply: Sender<Option<(EntryRecord, Seq)>>,
	},
	FindLatestHeadMarker {
		conversation_id: ConversationId,
		at_or_before: Option<EntryId>,
		context: Context,
		reply: Sender<Option<EntryRecord>>,
	},
	ScanEntries {
		query: EntryQuery,
		limit: u64,
		cursor: Option<Cursor>,
		context: Context,
		reply: Sender<Page<EntryRecord>>,
	},
	Task {
		id: TaskId,
		context: Context,
		reply: Sender<Option<TaskRecord>>,
	},
	ScanTasks {
		query: TaskQuery,
		limit: u64,
		cursor: Option<Cursor>,
		context: Context,
		reply: Sender<Page<TaskRecord>>,
	},
	Submission {
		id: SubmissionId,
		context: Context,
		reply: Sender<Option<SubmissionRecord>>,
	},
	ScanSubmissions {
		query: SubmissionQuery,
		limit: u64,
		cursor: Option<Cursor>,
		context: Context,
		reply: Sender<Page<SubmissionRecord>>,
	},
	SubmissionByRequest {
		conversation_id: ConversationId,
		request_id: String,
		context: Context,
		reply: Sender<Option<SubmissionRecord>>,
	},
	FindDocument {
		address: DocumentAddress,
		at: DocumentPoint,
		context: Context,
		reply: Sender<Option<DocumentRecord>>,
	},
	Document {
		id: DocumentId,
		at: DocumentPoint,
		context: Context,
		reply: Sender<Option<StoredDocument>>,
	},
	ScanDocuments {
		query: DocumentQuery,
		limit: u64,
		cursor: Option<Cursor>,
		context: Context,
		reply: Sender<Page<DocumentRecord>>,
	},
	Close {
		context: Context,
		reply: Sender<()>,
	},
}

/// Blocking facade over one [`Storage`] backend (see module docs).
pub(crate) struct StorageBridge {
	sender: Sender<Job>,
}

impl StorageBridge {
	/// Start the worker thread for `storage`.
	pub(crate) fn start(storage: Arc<dyn Storage>) -> Self {
		let (sender, receiver) = std::sync::mpsc::channel::<Job>();
		std::thread::Builder::new()
			.name("pi-durable-storage".to_string())
			.spawn(move || worker(storage, receiver))
			.expect("spawn pi-durable storage worker");
		Self { sender }
	}

	fn call<T: Send + 'static>(&self, build: impl FnOnce(Sender<T>) -> Job) -> T {
		let (reply, receiver) = std::sync::mpsc::channel();
		self.sender
			.send(build(reply))
			.expect("pi-durable storage worker is alive");
		receiver.recv().expect("pi-durable storage worker replied")
	}

	pub(crate) async fn commit(&self, writes: &[StorageWrite], context: &Context) -> Result<Seq, StorageRejected> {
		let writes = writes.to_vec();
		let context = context.clone();
		let (reply, receiver) = tokio::sync::oneshot::channel();
		self.sender
			.send(Job::Commit { writes, context, reply })
			.expect("pi-durable storage worker is alive");
		receiver.await.expect("pi-durable storage worker replied")
	}

	pub(crate) fn mint_id(&self) -> u64 {
		self.call(|reply| Job::MintId { reply })
	}

	pub(crate) fn conversation(&self, id: ConversationId, context: &Context) -> Option<ConversationRecord> {
		let context = context.clone();
		self.call(|reply| Job::Conversation { id, context, reply })
	}

	pub(crate) fn scan_conversations(
		&self,
		query: &ConversationQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<ConversationRecord> {
		let query = query.clone();
		let cursor = cursor.cloned();
		let context = context.clone();
		self.call(|reply| Job::ScanConversations { query, limit, cursor, context, reply })
	}

	pub(crate) fn entry(&self, id: EntryId, context: &Context) -> Option<(EntryRecord, Seq)> {
		let context = context.clone();
		self.call(|reply| Job::Entry { id, context, reply })
	}

	pub(crate) fn entry_in(
		&self,
		conversation_id: ConversationId,
		id: EntryId,
		context: &Context,
	) -> Option<(EntryRecord, Seq)> {
		let context = context.clone();
		self.call(|reply| Job::EntryIn { conversation_id, id, context, reply })
	}

	pub(crate) fn find_latest_head_marker(
		&self,
		conversation_id: ConversationId,
		at_or_before: Option<EntryId>,
		context: &Context,
	) -> Option<EntryRecord> {
		let context = context.clone();
		self.call(|reply| Job::FindLatestHeadMarker { conversation_id, at_or_before, context, reply })
	}

	pub(crate) fn scan_entries(
		&self,
		query: &EntryQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<EntryRecord> {
		let query = query.clone();
		let cursor = cursor.cloned();
		let context = context.clone();
		self.call(|reply| Job::ScanEntries { query, limit, cursor, context, reply })
	}

	pub(crate) fn task(&self, id: TaskId, context: &Context) -> Option<TaskRecord> {
		let context = context.clone();
		self.call(|reply| Job::Task { id, context, reply })
	}

	pub(crate) fn scan_tasks(
		&self,
		query: &TaskQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<TaskRecord> {
		let query = query.clone();
		let cursor = cursor.cloned();
		let context = context.clone();
		self.call(|reply| Job::ScanTasks { query, limit, cursor, context, reply })
	}

	pub(crate) fn submission(&self, id: SubmissionId, context: &Context) -> Option<SubmissionRecord> {
		let context = context.clone();
		self.call(|reply| Job::Submission { id, context, reply })
	}

	#[allow(dead_code)]
	pub(crate) fn scan_submissions(
		&self,
		query: &SubmissionQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<SubmissionRecord> {
		let query = query.clone();
		let cursor = cursor.cloned();
		let context = context.clone();
		self.call(|reply| Job::ScanSubmissions { query, limit, cursor, context, reply })
	}

	pub(crate) fn submission_by_request(
		&self,
		conversation_id: ConversationId,
		request_id: &str,
		context: &Context,
	) -> Option<SubmissionRecord> {
		let context = context.clone();
		let request_id = request_id.to_string();
		self.call(|reply| Job::SubmissionByRequest { conversation_id, request_id, context, reply })
	}

	pub(crate) fn find_document(
		&self,
		address: &DocumentAddress,
		at: DocumentPoint,
		context: &Context,
	) -> Option<DocumentRecord> {
		let address = address.clone();
		let context = context.clone();
		self.call(|reply| Job::FindDocument { address, at, context, reply })
	}

	pub(crate) fn document(&self, id: DocumentId, at: DocumentPoint, context: &Context) -> Option<StoredDocument> {
		let context = context.clone();
		self.call(|reply| Job::Document { id, at, context, reply })
	}

	pub(crate) fn scan_documents(
		&self,
		query: &DocumentQuery,
		limit: u64,
		cursor: Option<&Cursor>,
		context: &Context,
	) -> Page<DocumentRecord> {
		let query = query.clone();
		let cursor = cursor.cloned();
		let context = context.clone();
		self.call(|reply| Job::ScanDocuments { query, limit, cursor, context, reply })
	}

	pub(crate) fn close(&self, context: &Context) {
		let context = context.clone();
		self.call(|reply| Job::Close { context, reply });
	}
}

fn worker(storage: Arc<dyn Storage>, receiver: Receiver<Job>) {
	let runtime = tokio::runtime::Builder::new_multi_thread()
		.worker_threads(1)
		.enable_all()
		.build()
		.expect("pi-durable storage worker runtime");
	for job in receiver {
		runtime.block_on(run_job(&*storage, job));
	}
}

async fn run_job(storage: &dyn Storage, job: Job) {
	match job {
		Job::Commit { writes, context, reply } => {
			let _ = reply.send(storage.commit(&writes, &context).await);
		}

		Job::MintId { reply } => {
			let _ = reply.send(storage.mint_id().await);
		}
		Job::Conversation { id, context, reply } => {
			let _ = reply.send(storage.conversation(id, &context).await);
		}
		Job::ScanConversations { query, limit, cursor, context, reply } => {
			let _ = reply.send(storage.scan_conversations(&query, limit, cursor.as_ref(), &context).await);
		}
		Job::Entry { id, context, reply } => {
			let _ = reply.send(storage.entry(id, &context).await);
		}
		Job::EntryIn { conversation_id, id, context, reply } => {
			let _ = reply.send(storage.entry_in(conversation_id, id, &context).await);
		}
		Job::FindLatestHeadMarker { conversation_id, at_or_before, context, reply } => {
			let _ = reply.send(storage.find_latest_head_marker(conversation_id, at_or_before, &context).await);
		}
		Job::ScanEntries { query, limit, cursor, context, reply } => {
			let _ = reply.send(storage.scan_entries(&query, limit, cursor.as_ref(), &context).await);
		}
		Job::Task { id, context, reply } => {
			let _ = reply.send(storage.task(id, &context).await);
		}
		Job::ScanTasks { query, limit, cursor, context, reply } => {
			let _ = reply.send(storage.scan_tasks(&query, limit, cursor.as_ref(), &context).await);
		}
		Job::Submission { id, context, reply } => {
			let _ = reply.send(storage.submission(id, &context).await);
		}
		Job::ScanSubmissions { query, limit, cursor, context, reply } => {
			let _ = reply.send(storage.scan_submissions(&query, limit, cursor.as_ref(), &context).await);
		}
		Job::SubmissionByRequest { conversation_id, request_id, context, reply } => {
			let _ = reply.send(storage.submission_by_request(conversation_id, &request_id, &context).await);
		}
		Job::FindDocument { address, at, context, reply } => {
			let _ = reply.send(storage.find_document(&address, at, &context).await);
		}
		Job::Document { id, at, context, reply } => {
			let _ = reply.send(storage.document(id, at, &context).await);
		}
		Job::ScanDocuments { query, limit, cursor, context, reply } => {
			let _ = reply.send(storage.scan_documents(&query, limit, cursor.as_ref(), &context).await);
		}
		Job::Close { context, reply } => {
			storage.close(&context).await;
			let _ = reply.send(());
		}
	}
}
