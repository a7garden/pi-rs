//! Port of `packages/server/src/transports/unix/address.ts`.

use std::path::PathBuf;

/// Derive the local Unix socket path for one logical server identity.
pub fn get_unix_socket_path(server_id: &str, server_directory: &str) -> Result<PathBuf, crate::types::AnyError> {
	if !pi_protocol::is_server_id(server_id) {
		return Err(crate::types::into_any_error(pi_chord::services::errors::type_error(
			"Unix serverId must be a canonical lowercase UUIDv4",
		)));
	}
	Ok(PathBuf::from(server_directory).join(format!("{server_id}.sock")))
}
