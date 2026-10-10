//! `cli/initial-message.ts` — combine stdin, @file text, and the first CLI
//! message into the initial non-interactive prompt.

use super::args::Args;

/// Input for [`build_initial_message`] (`InitialMessageInput`).
pub struct InitialMessageInput<'a> {
    /// Parsed arguments; the first message is consumed from it.
    pub parsed: &'a mut Args,
    /// Text produced by [`super::file_processor::process_file_arguments`].
    pub file_text: Option<String>,
    /// Images produced by [`super::file_processor::process_file_arguments`].
    pub file_images: Vec<pi_ai::types::ImageContent>,
    /// Piped stdin content, when any.
    pub stdin_content: Option<String>,
}

/// Result (`InitialMessageResult`).
#[derive(Debug, Default, PartialEq)]
pub struct InitialMessageResult {
    pub initial_message: Option<String>,
    pub initial_images: Option<Vec<pi_ai::types::ImageContent>>,
}

/// Combine stdin content, @file text, and the first CLI message into a
/// single initial prompt for non-interactive mode (`buildInitialMessage`).
pub fn build_initial_message(input: InitialMessageInput<'_>) -> InitialMessageResult {
    let mut parts: Vec<String> = Vec::new();
    if let Some(stdin_content) = input.stdin_content {
        parts.push(stdin_content);
    }
    if let Some(file_text) = input.file_text {
        parts.push(file_text);
    }

    if !input.parsed.messages.is_empty() {
        parts.push(input.parsed.messages.remove(0));
    }

    InitialMessageResult {
        initial_message: if parts.is_empty() {
            None
        } else {
            Some(parts.join(""))
        },
        initial_images: if !input.file_images.is_empty() {
            Some(input.file_images)
        } else {
            None
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn create_args(messages: &[&str]) -> Args {
        Args {
            messages: messages.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn run(
        parsed: &mut Args,
        file_text: Option<&str>,
        stdin_content: Option<&str>,
    ) -> InitialMessageResult {
        build_initial_message(InitialMessageInput {
            parsed,
            file_text: file_text.map(str::to_string),
            file_images: Vec::new(),
            stdin_content: stdin_content.map(str::to_string),
        })
    }

    #[test]
    fn merges_piped_stdin_with_the_first_cli_message_into_one_prompt() {
        let mut parsed = create_args(&["Summarize the text given"]);
        let result = run(&mut parsed, None, Some("README contents\n"));

        assert_eq!(
            result.initial_message.as_deref(),
            Some("README contents\nSummarize the text given")
        );
        assert!(parsed.messages.is_empty());
    }

    #[test]
    fn uses_stdin_as_the_initial_prompt_when_no_cli_message_is_present() {
        let mut parsed = create_args(&[]);
        let result = run(&mut parsed, None, Some("README contents"));

        assert_eq!(result.initial_message.as_deref(), Some("README contents"));
        assert!(parsed.messages.is_empty());
    }

    #[test]
    fn combines_stdin_file_text_and_first_cli_message_in_one_prompt() {
        let mut parsed = create_args(&["Explain it", "Second message"]);
        let result = run(&mut parsed, Some("file\n"), Some("stdin\n"));

        assert_eq!(
            result.initial_message.as_deref(),
            Some("stdin\nfile\nExplain it")
        );
        assert_eq!(parsed.messages, vec!["Second message".to_string()]);
    }

    #[test]
    fn empty_inputs_yield_no_initial_message() {
        let mut parsed = create_args(&[]);
        let result = run(&mut parsed, None, None);
        assert_eq!(result.initial_message, None);
        assert_eq!(result.initial_images, None);
        let _ = BTreeMap::<String, String>::new();
    }
}
