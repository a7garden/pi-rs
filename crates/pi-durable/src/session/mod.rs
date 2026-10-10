//! Port of `packages/durable/src/session` (the Session kernel).
//!
//! Layout mirrors upstream 1:1: [`forks`] (fork document selection),
//! [`observation`] (`CommittedStateSource`/`CommittedWatch` bridges),
//! [`session`] (`createSession`/`SessionImpl` + the session-owned harness
//! surfaces), [`transaction`] (`Tx`). [`bridge`] is new: the synchronous
//! storage access the sync-line facade requires (see `session` D1).
//!
//! Deviations are ledged per module. Cross-module summary:
//! - D1 (session/transaction): the upstream promise-chain mutation line is a
//!   tokio async mutex held for each job; commit callbacks are synchronous
//!   closures over `&mut Tx` (the `crate::tasks` D3 convention) with
//!   [`Session::commit_async`] for async jobs, and storage calls run through a
//!   blocking bridge thread.
//! - D2 (transaction): document drafts record a mutation log replayed through
//!   one chord overlay at settlement — prepared values/ops match upstream.
//! - D4 (session): session poison is reachable only through adoption failures.
//! - Strict-JSON rejections are unrepresentable (`serde_json::Value`).

pub(crate) mod bridge;
mod forks;
pub(crate) mod observation;
mod session;
mod transaction;

#[cfg(test)]
pub(crate) mod support;

#[cfg(test)]
mod checkpoints_tests;
#[cfg(test)]
mod definitions_tests;
#[cfg(test)]
mod documents_tests;
#[cfg(test)]
mod forks_tests;
#[cfg(test)]
mod states_tests;
#[cfg(test)]
mod tables_tests;
#[cfg(test)]
mod watches_tests;

pub use observation::{
    new_view, retirement_operations, CommittedStateSource, CommittedWatch, DocumentStateView,
    DocumentWatch, ObservedDocumentValue, ObservedValue, ViewSource, WatchError,
};
pub use session::{
    create_session, ContextView, ConversationAbortOptions, ConversationDocumentSnapshot,
    ConversationHandle, Session, SessionError, SettledTask, Unsubscribe,
};

pub(crate) use session::SessionHooks;
pub use transaction::{DocDraft, TransactionScope, Tx, TxError};
/// Borrowed synchronous transaction callbacks accepted by the Session line.
pub type CommitValueCallback<'callback> = dyn for<'borrow, 'tx> FnMut(&'borrow mut Tx<'tx>) -> Result<serde_json::Value, String>
    + Send
    + 'callback;
pub type CommitGuardCallback<'callback> =
    dyn for<'borrow, 'tx> FnMut(&'borrow mut Tx<'tx>) -> Result<(), String> + Send + 'callback;
