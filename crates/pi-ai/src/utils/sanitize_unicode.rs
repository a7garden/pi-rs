//! Port of `packages/ai/src/utils/sanitize-unicode.ts`.
//!
//! Upstream removes unpaired UTF-16 surrogate code units, which cannot be
//! encoded in UTF-8 and break provider JSON serialization. Rust `&str` is
//! valid UTF-8 by construction and `serde_json` rejects lone-surrogate escape
//! sequences at parse time, so the input can never carry unpaired surrogates;
//! the function is kept for API parity and is the identity on valid input
//! (crate docs, deviation D5).

/// Remove unpaired surrogate code units from `text`. Identity on valid UTF-8.
pub fn sanitize_surrogates(text: &str) -> String {
    text.to_string()
}

#[cfg(test)]
mod tests {
    //! Port of `packages/ai/test/unicode-surrogate.test.ts` — only the assertions
    //! that still apply when strings cannot hold unpaired surrogates (crate docs,
    //! deviation D5). The upstream file is otherwise live-provider end-to-end
    //! (emoji tool results against Anthropic/Bedrock/Azure/Cloudflare); the
    //! Rust-side guarantee is that `&str` is valid UTF-8 and `serde_json` rejects
    //! lone-surrogate escapes at parse time, so `sanitize_surrogates` is the
    //! identity. Skipped: `testEmojiInToolResults`, `testRealWorldLinkedInData`,
    //! `testUnpairedHighSurrogate` and every `describe` case in the file — each
    //! requires live provider credentials and a full provider stack.

    use super::*;

    #[test]
    fn keeps_valid_emoji_and_astral_characters_unchanged() {
        assert_eq!(sanitize_surrogates("Hello 🙈 World"), "Hello 🙈 World");
        assert_eq!(sanitize_surrogates("🤖🎉 math ∑ = 42"), "🤖🎉 math ∑ = 42");
    }

    #[test]
    fn is_the_identity_on_valid_utf8() {
        assert_eq!(sanitize_surrogates("plain text"), "plain text");
        assert_eq!(sanitize_surrogates(""), "");
        assert_eq!(sanitize_surrogates("line\nbreak\ttab"), "line\nbreak\ttab");
    }

    #[test]
    fn serde_json_rejects_unpaired_surrogate_escapes_at_parse_time() {
        // The upstream sanitizer removes unpaired surrogates that break provider
        // JSON serialization ("no low surrogate in string"). Rust cannot construct
        // such a string; serde_json rejects the escape instead, which is the
        // guarantee that makes the sanitizer an identity (deviation D5).
        assert!(serde_json::from_str::<serde_json::Value>("\"\\ud83d\"").is_err());
        assert!(serde_json::from_str::<serde_json::Value>("\"\\ude0a\"").is_err());
        assert!(serde_json::from_str::<serde_json::Value>("\"\\ud83d\\ude0a\"").is_ok());
    }
}
