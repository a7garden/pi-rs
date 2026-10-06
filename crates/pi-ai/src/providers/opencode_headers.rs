//! Port of `packages/ai/src/providers/opencode-headers.ts`.
//!
//! Deviations:
//! - D25: upstream spreads `{ ...streams, stream, streamSimple }` to wrap a
//!   `ProviderStreams` object; Rust wraps an `Arc<dyn ProviderStreams>` in a
//!   delegating struct that forwards every method (including the deferred
//!   capability probes, which upstream's spread preserves implicitly).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::models::ProviderStreams;
use crate::types::{DeferredCancelOptions, DeferredFetchOptions, DeferredHandle};
use crate::types::{
	AssistantMessageEventStream, Model, SimpleStreamOptions, StreamOptions, TranscriptContext,
};
use crate::utils::thrown::ThrownValue;

/// Upstream `OPENCODE_SESSION_HEADER`.
const OPENCODE_SESSION_HEADER: &str = "x-opencode-session";

fn has_header(headers: Option<&crate::types::ProviderHeaders>, name: &str) -> bool {
	let Some(headers) = headers else { return false };
	headers.iter().any(|(key, _)| key.eq_ignore_ascii_case(name))
}

fn with_session_header(options: &StreamOptions) -> StreamOptions {
	let has_session = options.session_id.is_some();
	let already_set = has_header(options.base.headers.as_ref(), OPENCODE_SESSION_HEADER);
	if !has_session || already_set {
		return options.clone();
	}
	let mut next = options.clone();
	let session_id = next.session_id.clone().expect("checked above");
	next.base
		.headers
		.get_or_insert_with(Default::default)
		.insert(OPENCODE_SESSION_HEADER.to_string(), Some(session_id));
	next
}

/// Adds OpenCode's required per-conversation routing header before API
/// dispatch (`withOpenCodeSessionHeader`).
pub fn with_opencode_session_header(streams: Arc<dyn ProviderStreams>) -> Arc<dyn ProviderStreams> {
	Arc::new(WithOpenCodeSessionHeader { inner: streams })
}

struct WithOpenCodeSessionHeader {
	inner: Arc<dyn ProviderStreams>,
}

impl ProviderStreams for WithOpenCodeSessionHeader {
	fn stream(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: &StreamOptions,
	) -> AssistantMessageEventStream {
		self.inner.stream(model, context, &with_session_header(options))
	}

	fn stream_simple(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: &SimpleStreamOptions,
	) -> AssistantMessageEventStream {
		self.inner.stream_simple(model, context, &SimpleStreamOptions { base: with_session_header(&options.base), ..options.clone() })
	}

	fn has_fetch_deferred(&self) -> bool {
		self.inner.has_fetch_deferred()
	}

	fn has_cancel_deferred(&self) -> bool {
		self.inner.has_cancel_deferred()
	}

	fn fetch_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: &DeferredFetchOptions,
	) -> Option<AssistantMessageEventStream> {
		self.inner.fetch_deferred(model, handle, options)
	}

	fn cancel_deferred(
		&self,
		model: &Model,
		handle: &DeferredHandle,
		options: &DeferredCancelOptions,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		self.inner.cancel_deferred(model, handle, options)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::providers::faux::{faux_assistant_message, FauxMessageOptions};
	use crate::types::{CacheRetention, Context, Message, SimpleStreamOptions, StreamOptions, UserContentField, UserMessage};
	use crate::utils::transcript::normalize_context;
	use std::sync::{Arc};
	use {parking_lot::Mutex};

	fn test_model() -> Model {
		serde_json::from_value(serde_json::json!({
			"id": "test-model",
			"name": "Test model",
			"api": "test-api",
			"provider": "opencode",
			"baseUrl": "https://opencode.ai/zen/v1",
			"reasoning": false,
			"input": ["text"],
			"cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
			"contextWindow": 1000,
			"maxTokens": 100,
		}))
		.expect("test model parses")
	}

	fn test_context() -> TranscriptContext {
		normalize_context(&Context {
			system_prompt: None,
			messages: vec![Message::User(UserMessage {
				content: UserContentField::Text("hi".to_string()),
				timestamp: 0,
			})],
			tools: None,
		})
	}

	struct RecordingStreams {
		captured: Mutex<Vec<String>>,
	}

	/// Records the outgoing headers JSON of each call.
	impl ProviderStreams for RecordingStreams {
		fn stream(
			&self,
			_model: &Model,
			_context: &TranscriptContext,
			options: &StreamOptions,
		) -> AssistantMessageEventStream {
			self.captured
				.lock()
				
				.push(serde_json::to_string(&options.base.headers).unwrap());
			let stream = AssistantMessageEventStream::new();
			let message = faux_assistant_message("ok", FauxMessageOptions::default());
			stream.push(crate::types::AssistantMessageEvent::Start { partial: message.clone() });
			stream.push(crate::utils::event_stream::done_event(message.clone(), crate::types::StreamDoneReason::Stop));
			stream.end(Some(message));
			stream
		}

		fn stream_simple(
			&self,
			model: &Model,
			context: &TranscriptContext,
			options: &SimpleStreamOptions,
		) -> AssistantMessageEventStream {
			self.stream(model, context, &options.base)
		}
	}

	fn captured_headers(capture: &RecordingStreams) -> serde_json::Value {
		let raw = capture
			.captured
			.lock()
			
			.pop()
			.expect("captured call");
		serde_json::from_str(&raw).expect("headers serialize")
	}

	// Regression test for https://github.com/earendil-works/pi/issues/9326
	#[test]
	fn maps_session_id_even_without_cache_retention() {
		let capture = Arc::new(RecordingStreams { captured: Mutex::new(Vec::new()) });
		let streams = with_opencode_session_header(capture.clone());
		let mut options = StreamOptions::default();
		options.session_id = Some("conversation-1".to_string());
		options.cache_retention = Some(CacheRetention::None);
		let _ = streams.stream(&test_model(), &test_context(), &options);
		assert_eq!(
			captured_headers(&capture),
			serde_json::json!({ "x-opencode-session": "conversation-1" })
		);

		let mut options = SimpleStreamOptions::default();
		options.base.session_id = Some("conversation-1".to_string());
		options.base.cache_retention = Some(CacheRetention::None);
		let _ = streams.stream_simple(&test_model(), &test_context(), &options);
		assert_eq!(
			captured_headers(&capture),
			serde_json::json!({ "x-opencode-session": "conversation-1" })
		);
	}

	#[test]
	fn preserves_case_insensitive_caller_override() {
		for value in [serde_json::json!("caller-value"), serde_json::json!(null)] {
			let capture = Arc::new(RecordingStreams { captured: Mutex::new(Vec::new()) });
			let streams = with_opencode_session_header(capture.clone());
			let mut options = SimpleStreamOptions::default();
			options.base.session_id = Some("generated-value".to_string());
			options.base.headers = Some(
				[("X-OpenCode-Session".to_string(), serde_json::from_value(value.clone()).unwrap())]
					.into_iter()
					.collect(),
			);
			let _ = streams.stream_simple(&test_model(), &test_context(), &options);
			let headers = captured_headers(&capture);
			// Exact header-map equality, like upstream.
			assert_eq!(headers, serde_json::json!({ "X-OpenCode-Session": value.clone() }));
		}
	}

	#[test]
	fn does_not_fabricate_session_header_when_absent() {
		let capture = Arc::new(RecordingStreams { captured: Mutex::new(Vec::new()) });
		let streams = with_opencode_session_header(capture.clone());
		let mut options = SimpleStreamOptions::default();
		options.base.headers =
			Some([("x-custom".to_string(), Some("value".to_string()))].into_iter().collect());
		let _ = streams.stream_simple(&test_model(), &test_context(), &options);
		assert_eq!(captured_headers(&capture), serde_json::json!({ "x-custom": "value" }));
	}
}
