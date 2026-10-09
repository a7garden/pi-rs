//! `cli/setup.ts` — process-level CLI setup.
//!
//! Process title and `process.emitWarning` suppression are Node-specific;
//! the env markers and early http-dispatcher configuration carry over
//! (the dispatcher itself is deferred — D83).

pub fn setup_cli() {
	std::env::set_var("PI_CODING_AGENT", "true");
	std::env::set_var("AI_AGENT", "pi");
}
