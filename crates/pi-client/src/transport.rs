//! Port of `packages/client/src/transport.ts`.
//!
//! The TS callback interfaces become Rust traits/closures over byte vectors.
//! The factory returns a boxed `Send` future because the client spawns
//! connection attempts on the tokio runtime.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use pi_chord::services::errors::AnyError;

/// Delivered inbound byte chunks (upstream `ByteTransportHandlers.onData`).
pub type OnData = Arc<dyn Fn(Vec<u8>) + Send + Sync>;
/// Orderly terminal close (upstream `onClose`).
pub type OnClose = Arc<dyn Fn() + Send + Sync>;
/// Terminal transport failure (upstream `onError`).
pub type OnError = Arc<dyn Fn(AnyError) + Send + Sync>;

/// Upstream `ByteTransportHandlers`.
#[derive(Clone)]
pub struct ByteTransportHandlers {
    /// Delivers an arbitrary inbound byte chunk.
    pub on_data: OnData,
    /// Reports an orderly terminal close.
    pub on_close: OnClose,
    /// Reports a terminal transport failure.
    pub on_error: OnError,
}

/// Upstream `ByteTransport`.
#[async_trait::async_trait]
pub trait ByteTransport: Send + Sync {
    /// Sends one byte chunk. Calls must be delivered in invocation order.
    async fn send(&self, chunk: Vec<u8>) -> Result<(), AnyError>;
    /// Closes the transport. Implementations must make repeated calls harmless.
    fn close(&self);
}

pub type TransportFuture =
    Pin<Box<dyn Future<Output = Result<Box<dyn ByteTransport>, AnyError>> + Send>>;

/// Upstream `ByteTransportFactory`: creates a fresh connected, authenticated
/// transport. Exactly one terminal handler is expected.
pub type ByteTransportFactory = Arc<dyn Fn(ByteTransportHandlers) -> TransportFuture + Send + Sync>;
