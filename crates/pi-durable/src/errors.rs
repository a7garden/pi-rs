//! Port of `packages/durable/src/errors.ts`.
//!
//! Upstream error classes become distinct Rust error types with identical
//! `message` strings (tests assert on them). `ErrorOptions`-chained causes are
//! carried as `source: Option<Box<dyn Error>>` on [`StorageRejected`].

use std::fmt;

use crate::types::ConversationId;

/// A transaction read a table after its first table write. Read every required row before writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadAfterWrite {
    pub method: String,
}

impl ReadAfterWrite {
    pub fn new(method: impl Into<String>) -> Self {
        Self {
            method: method.into(),
        }
    }
}

impl fmt::Display for ReadAfterWrite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tx.{}() cannot read tables after the first table write",
            self.method
        )
    }
}

impl std::error::Error for ReadAfterWrite {}

/// Storage rejected a batch before any durable effect; the owning Session may continue safely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageRejected {
    pub message: String,
}

impl StorageRejected {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for StorageRejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for StorageRejected {}

/// A submission reached a busy conversation and was not admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationBusy {
    pub conversation_id: ConversationId,
}

impl ConversationBusy {
    pub fn new(conversation_id: ConversationId) -> Self {
        Self { conversation_id }
    }
}

impl fmt::Display for ConversationBusy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Conversation {} is busy", self.conversation_id)
    }
}

impl std::error::Error for ConversationBusy {}
