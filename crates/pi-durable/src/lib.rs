//! Port of `@earendil-works/pi-durable` (upstream: `packages/durable/src/index.ts`).
//!
//! Durable conversations, documents, tasks, and submissions over pluggable
//! storage (memory, JSONL, SQLite). Public re-exports mirror upstream
//! `src/index.ts`; the erased wire model is documented in [`types`].
//!
//! Deviations are ledged per module (see module docs) and summarized in the
//! PORT_MAP.md status ledger.

pub mod documents;
pub mod entries;
pub mod env;
pub mod errors;
pub mod harness;
pub mod ids;
pub mod session;
pub mod storage;
pub mod tasks;
pub mod testing;
pub mod tools;
pub mod truncate;
pub mod types;

#[cfg(test)]
mod truncate_tests;

#[cfg(test)]
mod types_tests;
pub use harness::*;
pub use session::{create_session, Session, Tx};
pub use storage::MemoryStorage;
pub use tasks::{
    define_task, BoxFuture, DocumentWatch, HookHandler, HookInvokeFn, HookRunner, NextTaskState,
    PhaseHandler, RunningTask, Task, TaskDefinition, TaskMigration, TaskRuntime,
};

// Mirrors upstream `src/index.ts` re-exports (erased forms; see `types` D-notes).
pub use documents::*;
pub use entries::*;
pub use errors::{ConversationBusy, ReadAfterWrite, StorageRejected};
pub use truncate::{
    format_size, truncate_head, truncate_head_of, utf8_byte_length, TruncationLimit,
    TruncationOptions, TruncationResult, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES,
};
pub use types::*;
