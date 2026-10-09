//! Port of `experimental/durable/sessions.ts` — one session directory holding
//! `session.sqlite`, locked by this process.
//!
//! Deviations:
//! - **D170 — the shared `selectSession` body lives here.** Upstream
//!   `durable/sessions.ts` and `vacation/sessions.ts` are byte-identical
//!   except for the session-space directory (`durable-sessions` /
//!   `vacation-sessions`) and the "no session" error text; the port
//!   parameterizes them through [`select_session_in`] and both modules keep
//!   their upstream surface ([`select_session`], and the vacation
//!   re-export).
//! - `proper-lockfile` maps to [`crate::experimental::lockfile::FileLock`]
//!   (D149), which locks `<directory>.lock` next to the session directory;
//!   upstream `realpath: false` + `retries: {12 × 1000ms}` maps to a 1 s
//!   retry over a 12 s budget. `release()` is the lock value being dropped
//!   or explicitly released.
//! - `crypto.createHash("sha256")`/`randomUUID` map to `sha2` and the crate
//!   `randomUUID` seam ([`crate::utils::clipboard::uuid`]).

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::config::get_agent_dir;
use crate::experimental::lockfile::FileLock;
use crate::utils::clipboard::uuid as random_uuid;

/// One session directory holding `session.sqlite`, locked by this process
/// (upstream `SessionLocation`).
pub struct SessionLocation {
	/// Basename of the session directory (`<13-digit-ms>-<uuid>`).
	pub id: String,
	pub directory: PathBuf,
	pub database: PathBuf,
	pub cwd: String,
	pub created: bool,
	/// The process's lock on the directory; dropping or releasing it frees
	/// the session for another process.
	pub release: FileLock,
}

/// `selectSession(cwd, continueSession)` for one session space
/// (`durable-sessions` / `vacation-sessions`).
///
/// A new session for `cwd`, or its newest one with `continueSession`.
pub async fn select_session_in(
	space: &str,
	cwd_input: &str,
	continue_session: bool,
) -> Result<SessionLocation, String> {
	let cwd = tokio::fs::canonicalize(cwd_input)
		.await
		.map_err(|error| format!("Could not resolve {cwd_input}: {error}"))?
		.to_string_lossy()
		.into_owned();
	let digest = format!("{:x}", Sha256::digest(cwd.as_bytes()));
	let root = Path::new(&get_agent_dir())
		.join("experimental")
		.join(space)
		.join(&digest[..24]);
	tokio::fs::create_dir_all(&root)
		.await
		.map_err(|error| format!("Could not create {root:?}: {error}"))?;

	let directory;
	let created;
	if continue_session {
		let mut newest: Option<String> = None;
		let mut entries = tokio::fs::read_dir(&root)
			.await
			.map_err(|error| format!("Could not list {root:?}: {error}"))?;
		while let Some(entry) = entries
			.next_entry()
			.await
			.map_err(|error| format!("Could not list {root:?}: {error}"))?
		{
			let is_dir = entry
				.metadata()
				.await
				.map(|metadata| metadata.is_dir())
				.unwrap_or(false);
			let name = entry.file_name().to_string_lossy().into_owned();
			if is_dir && is_session_directory(&name) && newest.as_deref().is_none_or(|best| name.as_str() > best) {
				newest = Some(name);
			}
		}
		let Some(newest) = newest else {
			return Err(format!("No {space} session exists for {cwd}"));
		};
		directory = root.join(newest);
		created = false;
	} else {
		let name = format!("{:013}-{}", millis_now(), random_uuid());
		directory = root.join(name);
		tokio::fs::create_dir(&directory)
			.await
			.map_err(|error| format!("Could not create {directory:?}: {error}"))?;
		created = true;
	}

	// A lock left by a crashed process goes stale after 10 s; wait that long
	// before giving up (upstream `retries: {12 × 1000ms}`).
	let release = FileLock::acquire(
		&lock_path_for(&directory),
		crate::experimental::lockfile::LockOptions { retry_ms: 1_000, wait_ms: 12_000 },
	)
	.await
	.map_err(|error| format!("Session is already open in another process: {directory:?}: {error}"))?;
	Ok(SessionLocation {
		id: directory
			.file_name()
			.map(|name| name.to_string_lossy().into_owned())
			.unwrap_or_default(),
		directory: directory.clone(),
		database: directory.join("session.sqlite"),
		cwd,
		created,
		release,
	})
}

/// `selectSession` for the durable space.
pub async fn select_session(cwd_input: &str, continue_session: bool) -> Result<SessionLocation, String> {
	select_session_in("durable-sessions", cwd_input, continue_session).await
}

/// Upstream `^\d{13}-[0-9a-f-]{36}$`.
fn is_session_directory(name: &str) -> bool {
	let Some((stamp, uuid)) = name.split_once('-') else {
		return false;
	};
	stamp.len() == 13 && stamp.bytes().all(|byte| byte.is_ascii_digit()) && is_uuid_shape(uuid)
}

/// `[0-9a-f-]{36}` — four hyphens at the canonical positions, hex elsewhere.
fn is_uuid_shape(value: &str) -> bool {
	value.len() == 36
		&& value.bytes().enumerate().all(|(index, byte)| {
			if matches!(index, 8 | 13 | 18 | 23) {
				byte == b'-'
			} else {
				byte.is_ascii_hexdigit()
			}
		})
}

fn millis_now() -> u128 {
	std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|duration| duration.as_millis())
		.unwrap_or(0)
}

/// D149: the port locks `<path>.lock` next to the locked directory.
fn lock_path_for(directory: &Path) -> PathBuf {
	PathBuf::from(format!("{}.lock", directory.to_string_lossy()))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn session_directory_names_match_upstream_pattern() {
		assert!(is_session_directory("1700000000000-3f2a7c1e-8b4d-4c2a-9f01-5d6e7f8a9b0c"));
		assert!(!is_session_directory("170000000000-3f2a7c1e-8b4d-4c2a-9f01-5d6e7f8a9b0c"));
		assert!(!is_session_directory("1700000000000-3f2a7c1e8b4d4c2a9f015d6e7f8a9b0c"));
		assert!(!is_session_directory("nope"));
	}

	#[test]
	fn uuid_shape_requires_canonical_hyphens() {
		assert!(is_uuid_shape("3f2a7c1e-8b4d-4c2a-9f01-5d6e7f8a9b0c"));
		assert!(!is_uuid_shape("3f2a7c1e_8b4d-4c2a-9f01-5d6e7f8a9b0c"));
		assert!(!is_uuid_shape("3f2a7c1e-8b4d-4c2a-9f01-5d6e7f8a9b0"));
	}
}
