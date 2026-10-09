//! Port of `experimental/services/slash-commands.ts` — the process-local slash
//! command contribution registry contract.
//!
//! Deviations (D144):
//! - The structural `SlashCommandContribution` becomes a closure-based struct;
//!   `getArgumentCompletions` and `run` are boxed futures.
//! - `SlashCommandRunResult = AgentOperationResponse | AgentQueueResponse |
//!   undefined` becomes [`SlashCommandRunResult`] inside `Option` (`None` ==
//!   `undefined`).
//! - The upstream `SlashCommands` service interface is typed by the concrete
//!   [`crate::experimental::services::slash_commands_provider::SlashCommandRegistry`]
//!   implementation the local token carries.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};

use pi_chord::api::define_service;
use pi_chord::context::Context;

use super::agent_controller::{AgentOperationResponse, AgentQueueResponse};

/// Upstream `SlashCommandCompletion`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashCommandCompletion {
	pub value: String,
	pub label: String,
	pub description: Option<String>,
}

/// Upstream `SlashCommandRunResult`.
#[derive(Debug, Clone, PartialEq)]
pub enum SlashCommandRunResult {
	Operation(AgentOperationResponse),
	Queue(AgentQueueResponse),
}

/// Completion lookup: upstream `getArgumentCompletions` (sync array or promise;
/// both collapse into one async signature, D144). `Ok(None)` is upstream `null`.
pub type ArgumentCompletionsFn = Arc<
	dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<Option<Vec<SlashCommandCompletion>>, String>> + Send>>
		+ Send
		+ Sync,
>;

/// Command body: upstream `run(args, context)`; the result future carries the
/// run result, and `Err` propagates upstream throws.
pub type SlashCommandRunFn = Arc<
	dyn Fn(String, Context) -> Pin<Box<dyn Future<Output = Result<Option<SlashCommandRunResult>, String>> + Send>>
		+ Send
		+ Sync,
>;

/// Upstream `SlashCommandContribution`.
#[derive(Clone)]
pub struct SlashCommandContribution {
	pub name: String,
	pub description: Option<String>,
	pub argument_hint: Option<String>,
	pub get_argument_completions: Option<ArgumentCompletionsFn>,
	pub run: SlashCommandRunFn,
}

impl std::fmt::Debug for SlashCommandContribution {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("SlashCommandContribution")
			.field("name", &self.name)
			.field("description", &self.description)
			.field("argumentHint", &self.argument_hint)
			.field("getArgumentCompletions", &self.get_argument_completions.is_some())
			.finish_non_exhaustive()
	}
}

/// `defineService<SlashCommands>("pi.local.slash-commands")`; local, typed by
/// the registry implementation (D144).
pub struct SlashCommandsToken;

pub static SLASH_COMMANDS: LazyLock<pi_chord::Service<super::slash_commands_provider::SlashCommandRegistry>> =
	LazyLock::new(|| define_service("pi.local.slash-commands", true).expect("service id is valid"));

/// Upstream's exported name-pattern validator (shared with the registry).
pub fn is_valid_command_name(name: &str) -> bool {
	let mut bytes = name.bytes();
	match bytes.next() {
		Some(first) if first.is_ascii_lowercase() || first.is_ascii_digit() => {}
		_ => return false,
	}
	bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b':' || byte == b'-')
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn command_name_pattern_matches_upstream_regex() {
		// Upstream: /^[a-z0-9][a-z0-9:-]*$/u
		for valid in ["model", "a", "a1", "a:b", "a-b", "1"] {
			assert!(is_valid_command_name(valid), "{valid} should be valid");
		}
		for invalid in ["", "Model", "-a", ":a", "a b", "a/b", "ä"] {
			assert!(!is_valid_command_name(invalid), "{invalid} should be invalid");
		}
	}
}
