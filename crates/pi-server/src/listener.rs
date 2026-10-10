//! Port of `packages/server/src/listener.ts`.

use std::sync::Arc;

use async_trait::async_trait;

use crate::connection::ByteConnectionAcceptor;
use crate::types::AnyError;

/// Supplies established byte connections after any required transport
/// authentication (upstream `ServerListener`).
#[async_trait]
pub trait ServerListener: Send + Sync {
    /// Starts listening and passes authorized connections to accept.
    async fn start(&self, accept: ByteConnectionAcceptor) -> Result<(), AnyError>;
    async fn close(&self) -> Result<(), AnyError>;
}

/// Convenience alias for boxed listener collections.
pub type SharedServerListener = Arc<dyn ServerListener>;
