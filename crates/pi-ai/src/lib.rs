//! Port of @earendil-works/pi-ai (upstream: packages/ai) — CORE wave.
//!
//! pi-ai is the model-facing substrate: message/context types, the provider
//! abstraction, the API implementation contracts, auth resolution, model
//! catalogs, and the first four provider factories. Public re-exports mirror
//! upstream `src/index.ts`.
//!
//! # The Provider contract (read this before implementing a provider)
//!
//! Upstream `Provider` is an object literal with optional methods; Rust
//! expresses it as the [`Provider`](models::Provider) trait where every
//! optional upstream method maps to a method returning `Option`
//! (`refresh_models`, `fetch_deferred`, `cancel_deferred`,
//! `generate_images`, `classify`, `filter_models`, `filter_all_models`).
//! Hand-rolled providers should be built with
//! [`create_provider`](models::create_provider), which reproduces upstream
//! dispatch and error strings; only special providers (like faux) implement
//! the trait directly.
//!
//! ## Streaming and event model
//!
//! * Upstream returns `AssistantMessageEventStream` — an `AsyncIterable` of
//!   `AssistantMessageEvent` carrying a shared `partial` AssistantMessage,
//!   terminating with `done`/`error`, plus an independent `.result()`
//!   promise. The Rust port is the concrete cloneable
//!   [`AssistantMessageEventStream`] (utils/event_stream.rs):
//!   producers `push(event)` / `end(result)`; consumers await it as a
//!   `futures_core::Stream` (single consumer, FIFO) and/or await `.result()`
//!   (`EventResultFuture`) independently. Clone the handle when both
//!   producer and consumer need it — this mirrors upstream, where the same
//!   object is pushed to and iterated.
//! * Event ordering invariants are unchanged: `start` precedes updates;
//!   `done`/`error` terminate; updates never precede `start`. `partial` is
//!   the shared live object, mutated in place by adapters between events
//!   (upstream passes the same object reference; Rust adapters clone per
//!   event).
//! * Long-running work happens inside `tokio::spawn`ed tasks owned by the
//!   adapter; request setup failures (auth missing, unknown api) are
//!   converted by [`lazy_stream`](api::lazy::lazy_stream) into an `error`
//!   event whose `error` AssistantMessage carries the upstream
//!   `errorMessage` string. Adapters encode request/model/runtime failures
//!   in the stream (never panics, never `Err` returns at the trait
//!   boundary). `stream_simple` missing-auth failures upstream throw
//!   synchronously; in Rust they surface as the same stream error event
//!   (deviation D1).
//! * Abort: upstream `AbortSignal` is the pure-Rust
//!   [`AbortSignal`](utils::abort::AbortSignal)/[`AbortController`](utils::abort::AbortController)
//!   (pi-chord pattern, deviation D3). Adapters race provider reads against
//!   `signal.aborted_future()` and terminate with `stopReason: "aborted"`.
//! * Retry: adapter-level request retries go through
//!   `utils/provider_retry.rs` (`retryProviderRequest`), honoring
//!   `maxRetries`/`maxRetryDelayMs`/`signal`; auth-missing requests are not
//!   retried.
//! * HTTP: upstream `FetchFunction` is the injectable
//!   [`FetchFn`](http::FetchFn) seam (`options.fetch`), defaulting to a
//!   reqwest/rustls adapter (`http.rs`). SSE bodies are byte streams decoded
//!   per API module exactly as upstream decodes them.
//! * Per-API option interfaces (`AnthropicOptions`,
//!   `OpenAICompletionsOptions`, `OpenAIResponsesOptions`) become nested
//!   groups on [`StreamOptions`] (`options.anthropic.*`,
//!   `options.openai_completions.*`, `options.openai_responses.*`) so a
//!   single options type flows through the object-safe provider trait
//!   (deviation D2).
//!
//! ## Deferred responses
//!
//! `ProviderStreams::fetch_deferred`/`cancel_deferred` return `Option`;
//! presence probes `has_fetch_deferred`/`has_cancel_deferred` mirror
//! upstream's `!== undefined` checks. [`lazy_api`](api::lazy::lazy_api)
//! declares the capabilities of wrapped implementations.
//!
//! # Module map (upstream `packages/ai/src` → this crate)
//!
//! Ported this wave: `types.ts` → `types.rs`; `models.ts` → `models.rs`;
//! `models-store.ts` → `models_store.rs`; `utils/*` → `utils/*`;
//! `auth/{types,context,helpers,resolve,credential-store}.ts` → `auth/*`;
//! `api/{lazy,anthropic-messages,openai-completions,openai-responses,openai-responses-shared,openai-prompt-cache,typesafe-system-one,system-one-shared,simple-options,transform-messages,constrained-sampling}.ts`
//! → `api/*`; `providers/{anthropic,openai,typesafe,meta,faux}.ts` +
//! `*.models.ts` → `providers/*`; `model-catalog.ts` → `model_catalog.rs`;
//! `models.generated.ts` → embedded data (`models_generated.rs`).
//!
//! Deferred to later waves (deliberately absent, do not stub):
//! `api/google*`, `api/bedrock*`, `api/cloudflare*`, `api/mistral*`,
//! `api/github-copilot-headers.ts`, `api/azure-openai-*`,
//! `api/openai-codex-responses.ts`, `api/pi-messages.ts`,
//! `api/openrouter-images.ts`, `api/llama-cpp-classify.ts` and their
//! `.lazy.ts` wrappers; `auth/oauth/*`; `oauth.ts`; `images*.ts`; `compat*`;
//! `env-api-keys.ts`; `bun-oauth.ts`; `cli.ts`; `session-resources.ts`;
//! `legacy-api-aliases.ts`; `bedrock-provider.ts`; `providers/*` other than
//! the five above (including `providers/all.ts` registration);
//! `utils/node-http-proxy.ts`, `utils/oauth-page.ts` (Node-specific);
//! `image-models.ts`.
//!
//! # Deviations from upstream
//!
//! - **D1 — synchronous throws become stream errors.** Rust has no
//!   exceptions; upstream `streamSimple` throwing when auth is missing
//!   surfaces as the standard `error` event with the same message
//!   (`No API key for provider: <provider>`).
//! - **D2 — per-API options are nested groups** on `StreamOptions`
//!   (`options.anthropic` etc.), replacing TypeScript conditional option
//!   types. Field names and defaults are unchanged.
//! - **D3 — DOM AbortSignal re-implemented** in pure Rust following the
//!   pi-chord port pattern (same observable semantics; self-contained copy,
//!   no dependency on pi-chord).
//! - **D4 — `globalThis.fetch` is an explicit seam** (`http.rs::FetchFn`)
//!   with a reqwest/rustls default adapter; `options.fetch` overrides it and
//!   tests inject fakes (fetch-option test port).
//! - **D5 — `sanitizeSurrogates` is the identity** on valid UTF-8; Rust
//!   strings cannot hold unpaired surrogates and serde_json rejects them at
//!   parse time.
//! - **D6 — TypeBox schemas are JSON Schema `serde_json::Value`.** Tool
//!   parameters carry the identical wire JSON upstream TypeBox produces;
//!   `utils/typebox_helpers.rs` rebuilds the helpers (`StringEnum`).
//! - **D7 — `Model.compat` is one merged struct** (`ModelCompat`)
//!   representing the union of the five upstream per-API compat interfaces
//!   (the runtime shape is already a single object; unknown-API providers
//!   could not be typed upstream either).
//! - **D8 — Bun sandbox env fallback dropped** (`/proc/self/environ`
//!   probing is Node/Bun-specific); provider env resolves from scoped
//!   overrides then the process environment.
//! - **D9 — `partial-json` replaced** by a native suffix-completion parser
//!   in `utils/json_parse.rs` with the same externally visible behavior for
//!   streaming tool-call arguments (valid partial object in → object out,
//!   never a rejection).
//! - **D10 — dynamic imports are static.** `lazyApi`/`lazyStream` keep
//!   their stream-error and capability semantics; module loading itself is
//!   a no-op (Rust links statically), so `test/lazy-module-load.test.ts` has
//!   no runtime assertions left to port.
//! - **D11 — upstream `AnthropicOptions.client` is not carried.** It exists
//!   to inject alternative Anthropic SDK clients (AnthropicVertex); the Rust
//!   seam is `FetchFn` + headers on the model instead. Only the
//!   out-of-scope google-vertex provider used it.
//! - **D12 — hook/callback arguments are owned** (`Model` clones) where
//!   upstream passes live object references; the HTTP seam and options are
//!   `Clone` so request pipelines stay straightforward.

pub mod api;
pub mod auth;
pub mod http;
pub mod model_catalog;
pub mod models;
pub mod models_generated;
pub mod models_store;
pub mod providers;
pub mod types;
pub mod utils;

// Mirrors upstream `src/index.ts` re-exports (in-scope modules; per-API option
// types carry the D2 `*ApiOptions` names, deferred modules land with later waves).
pub use api::lazy::{lazy_api, lazy_stream, LazyApiCapabilities};
pub use models::{
	calculate_cost, clamp_thinking_level, create_models, create_provider, get_supported_thinking_levels, has_api,
	models_are_equal, CreateModelsOptions, CreateProviderOptions, Models, ModelsPublication, ModelsRefreshOptions,
	ModelsRefreshResult, ModelsRequestTransforms, MutableModels, Provider, ProviderClassifier, ProviderImages,
	ProviderStreams, RefreshModelsContext,
};
pub use utils::model_operations::{get_model_type, is_chat_model};
pub use models_store::{in_memory_models_store, InMemoryModelsStore, ModelsStore, ModelsStoreEntry};
pub use providers::faux::{
	faux_assistant_message, faux_provider, faux_text, faux_thinking, faux_tool_call, FauxContent, FauxContentBlock,
	FauxCore, FauxDeferredOptions, FauxMessageOptions, FauxModelDefinition, FauxProviderHandle, FauxProviderState,
	FauxResponseFactory, FauxResponseStep, FauxTokenSize, RegisterFauxProviderOptions,
};
pub use types::{
	AssistantImages, AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent, ClassifierAnswer,
	ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierQuestion, ClassifierResult, ClassifierStopReason,
	Context, DeferredHandle, DeferredRequest, ImageContent, ImagesContext, ImagesOptions, ImagesStopReason, Message,
	Model, ModelCompat, ModelCost, ModelType, StopReason, TextContent, ThinkingContent, Tool, ToolCall, ToolChoice,
	ToolResultMessage, TranscriptContext, Usage, UserMessage, AnthropicApiOptions, AnthropicEffort,
	AnthropicThinkingDisplay, OpenAICompletionsApiOptions, OpenAIResponsesApiOptions,
};
pub use auth::context::default_provider_auth_context;
pub use auth::credential_store::InMemoryCredentialStore;
pub use auth::helpers::{env_api_key_auth, lazy_oauth, LazyOAuthInput};
pub use auth::types::{
	ApiKeyAuth, ApiKeyCredential, AuthCheck, AuthContext,
	AuthEvent, AuthInfoLink, AuthInteraction, AuthOperationOptions, AuthPrompt, AuthResult, AuthType, Credential,
	CredentialInfo, CredentialStore, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential, OAuthCredentials,
	ProviderAuth, ProviderAuthInteraction,
};
pub use utils::assistant_message_frame::{
	reduce_assistant_message_frames, AssistantMessageFrame, AssistantMessageFrameEncoder, FrameError,
};
pub use utils::diagnostics::{
	append_assistant_message_diagnostic, create_assistant_message_diagnostic, extract_diagnostic_error,
	format_thrown_value,
};
pub use utils::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream, EventStream};
pub use utils::json_parse::{parse_json_with_repair, parse_streaming_json, repair_json};
pub use utils::models_error::{ModelsError, ModelsErrorCode};
pub use utils::overflow::{get_overflow_patterns, is_context_overflow, is_recoverable_length};
pub use utils::retry::{
	is_retryable_assistant_error, retry_assistant_call, retry_delay_ms, RetryCallbacks, RetryPolicy,
	DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
};
pub use utils::text::{get_system_message_text, render_system_message_update, system_content_text, user_content_text};
pub use utils::transcript::{
	collapse_system_messages, create_initial_system_message, declarations_equal, get_current_system_message,
	get_current_system_prompt, get_current_tools, get_declared_tools, get_initial_system_message, get_tool_state_changes,
	has_non_additive_tool_changes, has_tool_redefinitions, resolve_transcript, resolve_transcript_tools,
	to_tool_declaration, without_initial_system_message, ToolStateChanges, TranscriptTools,
};
pub use utils::typebox_helpers::{object_schema, string_enum, StringEnumOptions};
pub use utils::uuid::uuidv7;
pub use utils::validation::{validate_tool_arguments, validate_tool_call, ValidationError};
