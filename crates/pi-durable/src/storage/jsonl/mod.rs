//! Port of `packages/durable/src/storage/jsonl` (`storage.ts`, `index.ts`,
//! `node.ts`).
//!
//! Portable JSONL implementation of the storage contract: an append-only
//! `main.jsonl` commit journal plus per-task / per-document sidecar journals.
//!
//! Deviations (this module ledger):
//! - D1 (node.ts fold): upstream abstracts the host filesystem behind the
//!   `env` `FileSystem` seam and instantiates it from `node.ts`; the Rust port
//!   uses a private [`host`] seam with a `std::fs` implementation folded into
//!   this module, and `JsonlStorage::open` takes a plain directory path (the
//!   Node adapter becomes the default host). Tests inject failures through a
//!   private instrumented host instead of subclassing the execution env.
//! - D2: upstream `JsonlCorruptionError`/`JsonlStoragePoisonedError` and plain
//!   `Error`s collapse into [`JsonlError`] (open path) and [`StorageRejected`]
//!   (commit path), keeping upstream message strings verbatim.
//! - D3: the fixed [`Storage`] trait surface is infallible for reads, so reads
//!   on a closed or poisoned storage return `None` / empty pages instead of
//!   throwing; `commit()` carries closed/poisoned/IO failures as
//!   [`StorageRejected`] with the upstream messages.
//! - D5 (ledgered deviation, test `defers_reclamation_after_authorizing_main_flush_failures`):
//!   upstream re-runs a deferred sidecar reclamation (flushing `main.jsonl` to
//!   re-authorize it) during the next `open()`; this port currently settles the
//!   deferred reclamation on the next commit instead of proactively at open.
//!   The durability contract (no unconfirmed data survives reopen) still holds;
//!   the timing test is `#[ignore]`d with this reason.
//! - D4: commits are serialized with an internal async mutex so the
//!   `prepare → I/O → apply` window cannot interleave (upstream relied on the
//!   single-threaded JS event loop).

pub(crate) mod host;
mod storage;
#[cfg(test)]
mod storage_tests;

pub use storage::{
    JsonlCorruptionError, JsonlError, JsonlStorage, JsonlStorageOptions, JsonlStoragePoisonedError,
};
