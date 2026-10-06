//! Port of `packages/ai/src/legacy-api-aliases.ts` — the deprecated per-API
//! stream wrappers from the pre-registry global surface.
//!
//! Upstream builds each lazy API's streams object once at module load and
//! aliases its `stream`/`streamSimple`; the Rust port keeps the same
//! once-only construction with `LazyLock` statics (deviation D41 — the lazy
//! import itself is static, per D10).

use std::sync::{Arc, LazyLock};

use crate::models::ProviderStreams;
use crate::types::{Context, Model, SimpleStreamOptions, StreamOptions};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::transcript::normalize_context;

/// Generates the streams singleton + deprecated `stream*`/`streamSimple*`
/// alias pair for one API (`module::api_fn()`).
macro_rules! legacy_api {
	($module:ident, $api_fn:ident, $streams_fn:ident, $stream_fn:ident, $stream_simple_fn:ident, $upstream:literal) => {
		fn $streams_fn() -> &'static Arc<dyn ProviderStreams> {
			static STREAMS: LazyLock<Arc<dyn ProviderStreams>> = LazyLock::new(|| crate::api::$module::$api_fn());
			&STREAMS
		}

		#[doc = concat!("Upstream `", $upstream, "` (deprecated).")]
		pub fn $stream_fn(model: &Model, context: &Context, options: Option<&StreamOptions>) -> AssistantMessageEventStream {
			let default_options = StreamOptions::default();
			let options = options.unwrap_or(&default_options);
			$streams_fn().stream(model, &normalize_context(context), options)
		}

		#[doc = concat!("Upstream `streamSimple", $upstream, "` (deprecated).")]
		pub fn $stream_simple_fn(
			model: &Model,
			context: &Context,
			options: Option<&SimpleStreamOptions>,
		) -> AssistantMessageEventStream {
			let default_options = SimpleStreamOptions::default();
			let options = options.unwrap_or(&default_options);
			$streams_fn().stream_simple(model, &normalize_context(context), options)
		}
	};
}

legacy_api!(
	anthropic_messages,
	anthropic_messages_api,
	anthropic_streams,
	stream_anthropic,
	stream_simple_anthropic,
	"streamAnthropic"
);
legacy_api!(
	openai_completions,
	openai_completions_api,
	openai_completions_streams,
	stream_openai_completions,
	stream_simple_openai_completions,
	"streamOpenAICompletions"
);
legacy_api!(
	openai_responses,
	openai_responses_api,
	openai_responses_streams,
	stream_openai_responses,
	stream_simple_openai_responses,
	"streamOpenAIResponses"
);
legacy_api!(
	openai_codex_responses,
	openai_codex_responses_api,
	openai_codex_responses_streams,
	stream_openai_codex_responses,
	stream_simple_openai_codex_responses,
	"streamOpenAICodexResponses"
);
legacy_api!(
	azure_openai_responses,
	azure_openai_responses_api,
	azure_openai_responses_streams,
	stream_azure_openai_responses,
	stream_simple_azure_openai_responses,
	"streamAzureOpenAIResponses"
);
legacy_api!(
	google_generative_ai,
	google_generative_ai_api,
	google_generative_ai_streams,
	stream_google_generative_ai,
	stream_simple_google_generative_ai,
	"streamGoogleGenerativeAI"
);
legacy_api!(
	google_vertex,
	google_vertex_api,
	google_vertex_streams,
	stream_google_vertex,
	stream_simple_google_vertex,
	"streamGoogleVertex"
);
legacy_api!(
	mistral_conversations,
	mistral_conversations_api,
	mistral_conversations_streams,
	stream_mistral_conversations,
	stream_simple_mistral_conversations,
	"streamMistralConversations"
);
