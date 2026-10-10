//! Port of `packages/server/src/transports/unix/preset.ts`.

use std::sync::Arc;

use crate::server::Server;
use crate::transports::unix::listener::create_unix_listener;
use crate::transports::unix::types::UnixServerOptions;
use crate::types::{AnyError, ServerHost};

/// Compose Server with one Unix-domain socket listener.
pub fn create_unix_server(
    host: Arc<dyn ServerHost>,
    options: UnixServerOptions,
) -> Result<Server, AnyError> {
    // Validate Unix-listener options up front (upstream resolveUnixListenerOptions
    // throws synchronously during composition); Server::new revalidates the
    // shared options.
    validate_unix_options(&options)?;
    let listener = create_unix_listener(options.listener_options());
    Server::new(host, options.server_options(vec![listener]))
}

fn validate_unix_options(options: &UnixServerOptions) -> Result<(), AnyError> {
    let type_error = |message: &str| -> AnyError {
        crate::types::into_any_error(pi_chord::services::errors::type_error(message))
    };
    if options.path.is_empty() {
        return Err(type_error("Server Unix socket path must not be empty"));
    }
    let mode = options.mode.unwrap_or(0o600);
    if mode > 0o777 {
        return Err(type_error(
            "Server Unix socket mode must be an integer between 0 and 0o777",
        ));
    }
    let max_frame_length = options
        .max_frame_length
        .unwrap_or(pi_protocol::DEFAULT_MAX_FRAME_LENGTH);
    if max_frame_length == 0 || max_frame_length > crate::server::MAX_UINT32 {
        return Err(type_error(&format!(
            "Server maxFrameLength must be an integer between 1 and {}",
            crate::server::MAX_UINT32
        )));
    }
    let max_pending_bytes = options.max_pending_bytes.unwrap_or(max_frame_length * 4);
    if max_pending_bytes < max_frame_length + 4 {
        return Err(type_error(
            "Server maxPendingBytes must be a safe integer at least maxFrameLength + 4",
        ));
    }
    let graceful_close_timeout_ms = options.graceful_close_timeout_ms.unwrap_or(5_000);
    if graceful_close_timeout_ms == 0
        || graceful_close_timeout_ms > crate::server::MAX_TIMER_DELAY_MS
    {
        return Err(type_error(&format!(
            "Server gracefulCloseTimeoutMs must be an integer between 1 and {}",
            crate::server::MAX_TIMER_DELAY_MS
        )));
    }
    Ok(())
}
