//! Port of `packages/ai/src/api/openai-prompt-cache.ts`.

/// Maximum length of the OpenAI Responses `prompt_cache_key` field.
pub const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH: usize = 64;

/// Clamp a session id to the wire limit, counting Unicode scalars like the
/// upstream `Array.from` + `slice` (`clampOpenAIPromptCacheKey`).
pub fn clamp_openai_prompt_cache_key(key: Option<&str>) -> Option<String> {
	let key = key?;
	let chars: Vec<char> = key.chars().collect();
	if chars.len() <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH {
		return Some(key.to_string());
	}
	Some(chars[..OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH].iter().collect())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn keeps_short_keys() {
		assert_eq!(clamp_openai_prompt_cache_key(Some("abc")), Some("abc".to_string()));
		assert_eq!(clamp_openai_prompt_cache_key(None), None);
	}

	#[test]
	fn truncates_long_keys_by_characters() {
		let long = "🦀".repeat(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH + 1);
		let clamped = clamp_openai_prompt_cache_key(Some(&long)).expect("some");
		assert_eq!(clamped.chars().count(), OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH);
	}
}
