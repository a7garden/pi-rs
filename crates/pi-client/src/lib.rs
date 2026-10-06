//! Port of `@earendil-works/pi-client` (upstream: `packages/client`).
//!
//! Client-side connection, RPC, subscription, and local Unix-socket discovery
//! over the [`pi_protocol`] framing and [`pi_chord`] service wire types.
//! Public re-exports mirror upstream `src/index.ts`.
//!
//! Upstream re-exports with no Rust counterpart (documented deviation):
//! `Symbol.asyncDispose` — call [`Client::dispose`] explicitly.
//!
//! Module docs list per-file deviations; see `PORT_MAP.md` for the ledger.

pub mod client;
pub mod connection;
pub mod errors;
pub mod promise;
pub mod transport;
pub mod types;
pub mod unix;

#[cfg(test)]
pub(crate) mod test_support;

// Mirrors upstream `src/index.ts` re-exports.
pub use client::{connect_client, create_client_service_transport, Client};
pub use connection::ServerHello;
pub use errors::{ClientDisposedError, DisconnectedError, ServerError};
pub use transport::{ByteTransport, ByteTransportFactory, ByteTransportHandlers};
pub use types::{
	AttachmentChangeListener, ClientOptions, ConnectionChangeListener, ConnectionState, ConnectionStateChange,
	ListenerErrorHandler, RpcTarget, ServiceSubscription, SessionTarget, Unsubscribe,
};
