//! Port of `packages/durable/test/session-support.ts` (the session test
//! fixture): memory storage with observable commits, held calls, injected
//! commit failures, and the publication recorder.

use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::context::Context;
use pi_chord::context::BACKGROUND_CONTEXT;

use crate::session::session::{CommitListener, Session};
use crate::storage::memory::MemoryStorage;
use crate::types::{
    CommitPublication, ConversationId, ConversationOwnership, DocumentCommitChange, DocumentPoint,
    DocumentRecord, EntryId, Seq, Storage, StorageWrite,
};

/// The shared test context (upstream `context = BACKGROUND_CONTEXT`).
pub(crate) fn context() -> Context {
    BACKGROUND_CONTEXT.clone()
}

/// A gate that holds calls until released and reports when the first held call
/// arrives (upstream `Gate`).
pub(crate) struct Gate {
    entered: tokio::sync::oneshot::Receiver<()>,
    release: Option<Arc<tokio::sync::Semaphore>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateKind {
    Commit,
    Find,
}

impl Gate {
    /// Resolves when the first held call arrives.
    pub async fn entered(&mut self) {
        let _ = (&mut self.entered).await;
    }

    /// Release the held call.
    pub fn release(mut self) {
        if let Some(release) = self.release.take() {
            release.add_permits(1);
        }
    }
}

struct HeldGate {
    entered: tokio::sync::oneshot::Sender<()>,
    release: Arc<tokio::sync::Semaphore>,
}

/// Memory storage with observable commits, held calls, and injected commit
/// failures (upstream `ControlledStorage`).
pub(crate) struct ControlledStorage {
    inner: MemoryStorage,
    state: Mutex<ControlledState>,
}

#[derive(Default)]
struct ControlledState {
    /// Exact borrowed batches admitted by Session.
    admitted_commits: Vec<Vec<StorageWrite>>,
    /// Detached batches for value assertions.
    commits: Vec<Vec<StorageWrite>>,
    mint_count: u64,
    document_read_count: u64,
    commit_gate: Option<HeldGate>,
    find_gate: Option<HeldGate>,
    commit_failure: Option<String>,
    closed: bool,
}

impl ControlledStorage {
    pub(crate) fn new() -> Self {
        Self {
            inner: MemoryStorage::new(),
            state: Mutex::new(ControlledState::default()),
        }
    }

    pub(crate) fn hold_commits(&self) -> Gate {
        self.hold(GateKind::Commit)
    }

    pub(crate) fn hold_find_document(&self) -> Gate {
        self.hold(GateKind::Find)
    }

    fn hold(&self, kind: GateKind) -> Gate {
        let mut state = self.state.lock();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let held = HeldGate {
            entered: entered_tx,
            release: release.clone(),
        };
        if kind == GateKind::Commit {
            state.commit_gate = Some(held);
        } else {
            state.find_gate = Some(held);
        }
        Gate {
            entered: entered_rx,
            release: Some(release),
        }
    }

    /// Simulate a crash during the held commit: it never reaches storage, and
    /// later commits proceed.
    pub(crate) fn crash(&self) {
        self.state.lock().commit_gate = None;
    }

    /// Inject one failure into the next admitted commit.
    pub(crate) fn fail_next_commit(&self, message: impl Into<String>) {
        self.state.lock().commit_failure = Some(message.into());
    }

    /// Committed document read count (upstream `documentReadCount`).
    pub(crate) fn document_read_count(&self) -> u64 {
        self.state.lock().document_read_count
    }

    /// Mint count (upstream `mintCount`).
    pub(crate) fn mint_count(&self) -> u64 {
        self.state.lock().mint_count
    }
    pub(crate) fn is_closed(&self) -> bool {
        self.state.lock().closed
    }

    /// Detached admitted batches (upstream `commits`).
    pub(crate) fn commits(&self) -> Vec<Vec<StorageWrite>> {
        self.state.lock().commits.clone()
    }

    /// Admitted batches (upstream `admittedCommits`).
    pub(crate) fn admitted_commits(&self) -> Vec<Vec<StorageWrite>> {
        self.state.lock().admitted_commits.clone()
    }

    /// The inner storage for direct assertions.
    pub(crate) fn memory(&self) -> &MemoryStorage {
        &self.inner
    }
}

#[async_trait::async_trait]
impl Storage for ControlledStorage {
    async fn commit(
        &self,
        writes: &[StorageWrite],
        context: &Context,
    ) -> Result<Seq, crate::errors::StorageRejected> {
        let held = {
            let mut state = self.state.lock();
            state.admitted_commits.push(writes.to_vec());
            state.commits.push(writes.to_vec());
            state.commit_gate.take()
        };
        if let Some(held) = held {
            let _ = held.entered.send(());
            let _ = held.release.acquire().await;
        }
        let failure = self.state.lock().commit_failure.take();
        if let Some(message) = failure {
            return Err(crate::errors::StorageRejected::new(message));
        }
        self.inner.commit(writes, context).await
    }

    async fn mint_id(&self) -> u64 {
        self.state.lock().mint_count += 1;
        self.inner.mint_id().await
    }

    async fn conversation(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Option<crate::types::ConversationRecord> {
        self.inner.conversation(id, context).await
    }

    async fn scan_conversations(
        &self,
        query: &crate::types::ConversationQuery,
        limit: u64,
        cursor: Option<&crate::types::Cursor>,
        context: &Context,
    ) -> crate::types::Page<crate::types::ConversationRecord> {
        self.inner
            .scan_conversations(query, limit, cursor, context)
            .await
    }

    async fn entry(
        &self,
        id: EntryId,
        context: &Context,
    ) -> Option<(crate::types::EntryRecord, Seq)> {
        self.inner.entry(id, context).await
    }

    async fn entry_in(
        &self,
        conversation_id: ConversationId,
        id: EntryId,
        context: &Context,
    ) -> Option<(crate::types::EntryRecord, Seq)> {
        self.inner.entry_in(conversation_id, id, context).await
    }

    async fn find_latest_head_marker(
        &self,
        conversation_id: ConversationId,
        at_or_before_entry_id: Option<EntryId>,
        context: &Context,
    ) -> Option<crate::types::EntryRecord> {
        self.inner
            .find_latest_head_marker(conversation_id, at_or_before_entry_id, context)
            .await
    }

    async fn scan_entries(
        &self,
        query: &crate::types::EntryQuery,
        limit: u64,
        cursor: Option<&crate::types::Cursor>,
        context: &Context,
    ) -> crate::types::Page<crate::types::EntryRecord> {
        self.inner.scan_entries(query, limit, cursor, context).await
    }

    async fn task(
        &self,
        id: crate::types::TaskId,
        context: &Context,
    ) -> Option<crate::types::TaskRecord> {
        self.inner.task(id, context).await
    }

    async fn scan_tasks(
        &self,
        query: &crate::types::TaskQuery,
        limit: u64,
        cursor: Option<&crate::types::Cursor>,
        context: &Context,
    ) -> crate::types::Page<crate::types::TaskRecord> {
        self.inner.scan_tasks(query, limit, cursor, context).await
    }

    async fn submission(
        &self,
        id: crate::types::SubmissionId,
        context: &Context,
    ) -> Option<crate::types::SubmissionRecord> {
        self.inner.submission(id, context).await
    }

    async fn scan_submissions(
        &self,
        query: &crate::types::SubmissionQuery,
        limit: u64,
        cursor: Option<&crate::types::Cursor>,
        context: &Context,
    ) -> crate::types::Page<crate::types::SubmissionRecord> {
        self.inner
            .scan_submissions(query, limit, cursor, context)
            .await
    }

    async fn submission_by_request(
        &self,
        conversation_id: ConversationId,
        request_id: &str,
        context: &Context,
    ) -> Option<crate::types::SubmissionRecord> {
        self.inner
            .submission_by_request(conversation_id, request_id, context)
            .await
    }

    async fn find_document(
        &self,
        address: &crate::types::DocumentAddress,
        at: DocumentPoint,
        context: &Context,
    ) -> Option<DocumentRecord> {
        let held = {
            let mut state = self.state.lock();
            state.find_gate.take()
        };
        if let Some(held) = held {
            let _ = held.entered.send(());
            let _ = held.release.acquire().await;
        }
        self.inner.find_document(address, at, context).await
    }

    async fn document(
        &self,
        id: crate::types::DocumentId,
        at: DocumentPoint,
        context: &Context,
    ) -> Option<crate::types::StoredDocument> {
        self.state.lock().document_read_count += 1;
        self.inner.document(id, at, context).await
    }

    async fn scan_documents(
        &self,
        query: &crate::types::DocumentQuery,
        limit: u64,
        cursor: Option<&crate::types::Cursor>,
        context: &Context,
    ) -> crate::types::Page<DocumentRecord> {
        self.inner
            .scan_documents(query, limit, cursor, context)
            .await
    }

    async fn close(&self, context: &Context) {
        self.state.lock().closed = true;
        self.inner.close(context).await;
    }
}

/// Session kernel plus its controlled storage and every committed publication
/// (upstream `openTestSession`).
pub(crate) struct TestSession {
    pub(crate) storage: Arc<ControlledStorage>,
    pub(crate) session: Session,
    pub(crate) publications: Arc<Mutex<Vec<CommitPublication>>>,
}

pub(crate) fn open_test_session() -> TestSession {
    let storage = Arc::new(ControlledStorage::new());
    let session = crate::session::create_session(storage.clone(), None);
    let publications = Arc::new(Mutex::new(Vec::new()));
    let recorder: CommitListener = {
        let publications = publications.clone();
        Arc::new(move |publication: &CommitPublication, _context: &Context| {
            publications.lock().push(publication.clone());
        })
    };
    session.subscribe_commits(recorder);
    TestSession {
        storage,
        session,
        publications,
    }
}

/// `document` changes of one publication (upstream `documentChanges`).
pub(crate) fn document_changes(publication: &CommitPublication) -> Vec<&DocumentCommitChange> {
    publication
        .changes
        .iter()
        .filter_map(|change| match change {
            crate::types::CommitChange::Document(
                change @ DocumentCommitChange::Document { .. },
            ) => Some(change),
            _ => None,
        })
        .collect()
}

/// `document.copy` changes of one publication (upstream `documentCopyChanges`).
pub(crate) fn document_copy_changes(publication: &CommitPublication) -> Vec<&DocumentCommitChange> {
    publication
        .changes
        .iter()
        .filter_map(|change| match change {
            crate::types::CommitChange::Document(
                change @ DocumentCommitChange::DocumentCopy { .. },
            ) => Some(change),
            _ => None,
        })
        .collect()
}

/// Create one conversation and return its ID (upstream `createConversation`).
pub(crate) async fn create_conversation(session: &Session) -> ConversationId {
    session
        .commit(
            |tx| {
                tx.create_conversation(ConversationOwnership::Ownerless)
                    .map(|record| record.id)
                    .map_err(|e| e.to_string())
            },
            &context(),
        )
        .await
        .expect("conversation created")
}

/// Resolve after pending microtasks and one macrotask turn (upstream `flush`);
/// the Rust pumps run as spawned tasks, so a timer tick suffices.
pub(crate) async fn flush() {
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
}
