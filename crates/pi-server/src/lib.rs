//! Port of @earendil-works/pi-server (upstream: packages/server).
//!
//! The pi server accepts framed protocol connections (pi-protocol codec), routes
//! requests to application-hosted server services and durable Sessions
//! (session-router), and ships a Unix-domain-socket transport.
//!
//! # Deviations from upstream (v1.0.3)
//!
//! - **D1 — async runtime.** Upstream is Node-evented with callback handlers
//!   (`ByteConnectionHandler.onData` etc.) and `void`-prefixed promises. The
//!   Rust port keeps the synchronous handler surface but spawns tokio tasks
//!   internally; per-client operation ordering is preserved via FIFO
//!   `tokio::sync::Mutex` queues (upstream `runForClient` promise chaining).
//! - **D2 — metadata generics.** `ServerHost<TMetadata>` collapses to
//!   `ServerHost` + `Arc<dyn SessionMetadata>`; concrete metadata identity is
//!   preserved by the `Arc` (conformance test relies on it).
//! - **D3 — error classes.** `ServerError` subclasses become constructors on
//!   one struct carrying identical wire codes/messages; `AggregateError`
//!   reuse from pi-chord; error identity across `close()`/`closed()` awaiters
//!   uses `Arc<dyn Error>` cloning.
//! - **D4 — required option fields.** Upstream throws when `listeners` or
//!   `serverId` are missing; Rust struct fields are required by construction
//!   (only value validation remains, with identical messages).
//! - **D5 — timers.** `NodeJS.Timeout` handshake/graceful-close timers become
//!   tokio sleeps; `unref()` has no counterpart (tests await completion
//!   explicitly). Handshake-timeout vs completion races are arbitrated under
//!   the connection state lock (`handshake_settled`).
//! - **D6 — Unix transport sockets.** Node `Socket` events collapse into an
//!   internal `RawSocket` seam; the connection unit test drives a controlled
//!   fake (upstream `ControlledSocket`). `dev`/`ino` file-identity checks use
//!   `MetadataExt`; the stale-socket fixture child process is replaced by
//!   creating and dropping a local `std` listener in-test (identical final
//!   state: socket file present, nothing listening).
//! - **D7 — testing module.** `ProtocolTestClient` predicates operate on the
//!   pi-protocol `JsonValue` message shape; vitest `expect.poll` becomes a
//!   bounded poll helper.
//!
//! Public re-exports mirror upstream `src/index.ts`; the test doubles mirror
//! `src/testing/index.ts` (see [`testing`]).

#[cfg(test)]
mod tests;

pub mod connection;
pub mod errors;
pub mod listener;
pub mod server;
pub mod session_router;
pub mod testing;
pub mod transports;
pub mod types;
pub(crate) mod wire;

// Mirrors upstream `src/index.ts` re-exports (errors, listener, server, types).
pub use errors::{
	INTERNAL_SERVER_ERROR_MESSAGE, ServerError, ServerOperationErrorCode,
};
pub use listener::ServerListener;
pub use server::Server;
pub use types::{
	AnyError, BasicSessionMetadata, ConnectionCountCallback, ErrorCallback, MaybeFuture, RoutedServerPresentation,
	RoutedServerServiceAttachment, RoutedServerServiceHost, RoutedSessionAttachment, RoutedSessionHandle, ServerHost,
	ServerOptions, ServicePublish, SessionMetadata,
};
