//! Port of `packages/coding-agent/src/core/agent-session*.ts` — the pieces
//! that do not require the unported model/system-prompt/extension layers
//! (crate docs: deferred `agent-session` surface).
//!
//! Upstream files:
//! - `agent-session.ts` (4349 lines) — the `AgentSession` class. Every
//!   constructor argument and nearly every method routes through modules that
//!   are not ported yet, so the class itself is deferred as a unit:
//!   `Agent`, `SessionManager`, `SettingsManager`, `compaction/`, and
//!   `messages.ts` are available, but `ResourceLoader`, `ModelRuntime`,
//!   `ModelRegistry`, `ExtensionRunner`/`extensions/*`, `tools/*`,
//!   `bug-report.ts`,
//!   `export-html/`, `system-prompt.ts`, `slash-commands.ts`,
//!   `source-info.ts`, `virtual-models.ts`, `prompt-templates.ts`
//!   (`session-export.ts` and `auth-guidance.ts` have since landed; see
//!   [`core::session_export`] and [`core::auth_guidance`]) are not. Porting a
//!   subset would change the class's observable contract (extension hooks
//!   fire in prompt/steer/follow-up/compaction/tree-navigation paths).
//! - `agent-session-runtime.ts` — `AgentSessionRuntime` +
//!   `SessionImportFileNotFoundError`; depends on `AgentSession`,
//!   `AgentSessionServices`, `sdk.ts` (`createAgentSession`),
//!   `extensions/runner.ts`, and `session-cwd.ts`.
//! - `agent-session-services.ts` — `AgentSessionServices` +
//!   `createAgentSessionServices`; depends on `ModelRuntime`,
//!   `resource-loader.ts`, and `sdk.ts`.
//!
//! Deferred with them: the upstream suites
//! `test/agent-session-{auto-compaction-queue,branching,compaction,concurrent,dynamic-provider,dynamic-tools,retry,runtime-events,stats,tree-navigation}.test.ts`
//! all construct the full `AgentSession`.
//!
//! What is portable without those layers lives here: the standalone skill
//! block parser, which upstream exports from `agent-session.ts` and consumes
//! from `interactive-mode.ts`, `index.ts`, and `export-html/`.

/// Parsed skill block from a user message (`ParsedSkillBlock`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSkillBlock {
	pub name: String,
	pub location: String,
	pub content: String,
	pub user_message: Option<String>,
}

/// Parse a skill block from message text (`parseSkillBlock`). Returns `None`
/// if the text doesn't contain a skill block.
pub fn parse_skill_block(text: &str) -> Option<ParsedSkillBlock> {
	// Upstream regex:
	// /^<skill name="([^"]+)" location="([^"]+)">\n([\s\S]*?)\n<\/skill>(?:\n\n([\s\S]+))?$/
	let rest = text.strip_prefix("<skill ")?;
	let (name, rest) = match_attr(rest, "name")?;
	let rest = rest.strip_prefix(' ')?;
	let (location, rest) = match_attr(rest, "location")?;
	let rest = rest.strip_prefix(">\n")?;
	let (content, tail) = split_until_close(rest)?;
	let user_message = tail
		.strip_prefix("\n\n")
		.map(str::trim)
		.filter(|value| !value.is_empty())
		.map(str::to_string);
	Some(ParsedSkillBlock { name, location, content: content.to_string(), user_message })
}

/// Match `key="value"` at the start of `input`, returning the value and the
/// remainder (`([^"]+)` capture — at least one character).
fn match_attr<'a>(input: &'a str, key: &str) -> Option<(String, &'a str)> {
	let rest = input.strip_prefix(key)?.strip_prefix("=\"")?;
	let end = rest.find('"')?;
	if end == 0 {
		return None;
	}
	Some((rest[..end].to_string(), &rest[end + 1..]))
}

/// Match the `([\s\S]*?)\n<\/skill>(?:\n\n([\s\S]+))?$` tail: the shortest
/// content terminated by `\n</skill>` at end-of-input, with an optional
/// trailing user message.
fn split_until_close(input: &str) -> Option<(&str, &str)> {
	let mut search_from = 0;
	while let Some(offset) = input[search_from..].find("\n</skill>") {
		let content_end = search_from + offset;
		let tail = &input[content_end + "\n</skill>".len()..];
		// The regex anchors `</skill>` to the end of the message: only an
		// optional `\n\n<user message>` may follow.
		if tail.is_empty() || tail.starts_with("\n\n") {
			return Some((&input[..content_end], tail));
		}
		search_from = content_end + 1;
	}
	None
}

#[cfg(test)]
mod agent_session_tests {
	//! Upstream has no direct `parseSkillBlock` unit tests (the
	//! `export-html-skill-block.test.ts` cases grep the bundled renderer
	//! source); these cases pin the ported regex semantics on the documented
	//! message shape.
	use super::*;

	#[test]
	fn parses_skill_block_without_user_message() {
		let parsed = parse_skill_block(
			"<skill name=\"commit\" location=\"/repo/.pi/skills/commit.md\">\nbody text\n</skill>",
		)
		.expect("skill block");
		assert_eq!(parsed.name, "commit");
		assert_eq!(parsed.location, "/repo/.pi/skills/commit.md");
		assert_eq!(parsed.content, "body text");
		assert_eq!(parsed.user_message, None);
	}

	#[test]
	fn parses_skill_block_with_user_message() {
		let parsed = parse_skill_block(
			"<skill name=\"review\" location=\"x.md\">\ncontent\n</skill>\n\nplease review my diff",
		)
		.expect("skill block");
		assert_eq!(parsed.content, "content");
		assert_eq!(parsed.user_message.as_deref(), Some("please review my diff"));
	}

	#[test]
	fn rejects_non_skill_text_and_unterminated_blocks() {
		assert!(parse_skill_block("plain user message").is_none());
		assert!(parse_skill_block("<skill name=\"a\" location=\"b\">\nno close").is_none());
		// Trailing garbage after </skill> (not a blank-line user message)
		// fails the anchored regex.
		assert!(parse_skill_block("<skill name=\"a\" location=\"b\">\nc\n</skill>trailing").is_none());
	}
}
