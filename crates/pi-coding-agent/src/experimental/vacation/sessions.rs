//! Port of `experimental/vacation/sessions.ts` — the vacation session
//! space.
//!
//! Deviations: **D170** — the shared `selectSession` body lives in
//! `durable::sessions` (`select_session_in`); this module keeps the upstream
//! surface with the vacation space name.

pub use crate::experimental::durable::sessions::SessionLocation;

/// `selectSession(cwd, continueSession)` for the vacation space: a new
/// session for `cwd`, or its newest one with `continueSession`.
pub async fn select_session(
    cwd_input: &str,
    continue_session: bool,
) -> Result<SessionLocation, String> {
    crate::experimental::durable::sessions::select_session_in(
        "vacation-sessions",
        cwd_input,
        continue_session,
    )
    .await
}
