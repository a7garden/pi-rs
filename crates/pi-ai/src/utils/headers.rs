//! Port of `packages/ai/src/utils/headers.ts`.

use crate::types::ProviderHeaders;

/// Collect HTTP headers into a plain name→value record (`headersToRecord`).
/// Headers already arrive as `(name, value)` pairs in our HTTP seam, so this
/// preserves casing and later duplicates win, matching `Headers` iteration.
pub fn headers_to_record(headers: &[(String, String)]) -> std::collections::BTreeMap<String, String> {
	let mut result = std::collections::BTreeMap::new();
	for (key, value) in headers {
		result.insert(key.clone(), value.clone());
	}
	result
}

/// Merge provider header sources case-insensitively
/// (`providerHeadersToRecord`). A `None` value drops the header; later sources
/// win. Returns `None` when nothing remains, matching upstream.
pub fn provider_headers_to_record(
	sources: &[Option<&ProviderHeaders>],
) -> Option<std::collections::BTreeMap<String, String>> {
	let mut merged: std::collections::BTreeMap<String, (String, String)> = Default::default();
	for source in sources {
		let source: &ProviderHeaders = match source {
			Some(source) => source,
			None => continue,
		};
		for (name, value) in source {
			let normalized_name = name.to_lowercase();
			merged.remove(&normalized_name);
			if let Some(value) = value {
				merged.insert(normalized_name, (name.clone(), value.clone()));
			}
		}
	}
	if merged.is_empty() {
		None
	} else {
		Some(merged.into_values().collect())
	}
}
