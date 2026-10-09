//! Port of `packages/coding-agent/src/extensions/llama/` — the llama.cpp
//! router extension: management client, Hugging Face search, native
//! provider, and the `/llama` command entry.
//!
//! | Upstream | Port | Status |
//! |---|---|---|
//! | `llama/index.ts` | [`self`] ([`llama_extension`]) | ported (factory + provider/command registration); the command's TUI catalog flow defers with `ui.ts` — D99 |
//! | `llama/client.ts` | [`client`] | ported (router client, SSE watch, progress parsing, URL/byte helpers) |
//! | `llama/huggingface.ts` | [`huggingface`] | ported (search, details, token discovery; regexes hand-rolled — D98a) |
//! | `llama/provider.ts` | [`provider`] | ported (native `pi_ai::Provider` — D98d) |
//! | `llama/ui.ts` | [`ui`] | **partial**: pure string helpers (`contextLabel`, `modelDescription`, `compactCount`) ported + tested (D160); the interactive TUI surface (search widget, menus, progress screens) stays deferred — D99 |
//! | `test/llama-extension.test.ts` | inline `#[cfg(test)]` modules + [`test_support`] | ported 1:1; every upstream case drives a local mock server, none need real network (see Deferred tests) |
//!
//! # Entry point (index.ts, D40 seam)
//!
//! [`llama_extension`] is the native port of the JS default export: it
//! registers the provider via `registerProvider(provider)` (landing on the
//! native provider queue) and the `llama` command. The command handler
//! performs the upstream mode guard (`/llama` is interactive-only) and the
//! client configuration check; the interactive catalog UI itself is the
//! skipped `ui.ts` surface (D99).
//!
//! # Deviations
//!
//! - **D97**/**D97a**–**D97c** — client-layer deviations: abort/timeout
//!   modeling, unknown status fallback, typed catalog parsing, byte
//!   formatting (see [`client`]).
//! - **D98**–**D98f** — provider/huggingface deviations: hand-rolled
//!   quantization regexes, byte-wise name sort, `HOME`-based homedir,
//!   native-provider mapping, `setCatalog` options, catalog storage (see
//!   [`provider`], [`huggingface`]).
//! - **D99** — `ui.ts` is skipped (TUI-only). The `/llama` command keeps the
//!   upstream non-TUI behavior (a warning notification) and the client
//!   configuration path; in TUI mode it reports the deferral instead of
//!   opening the skipped catalog UI. The UI-flow helpers that upstream
//!   defines in `index.ts` ([`model_is_loaded`],
//!   [`is_connection_error`], [`connection_error_message`],
//!   [`parse_hugging_face_model`]) are ported verbatim as the shared logic
//!   of that deferred flow.
//! - **D160 — `ui.ts` partial port.** The pure, TUI-free string helpers the
//!   view uses to compose model rows (`contextLabel`, `modelDescription`,
//!   `compactCount`) are ported in [`ui`] with unit tests; everything
//!   interactive (the `HuggingFaceSearch` widget, menus, the progress
//!   screens, `showLlamaUi`, `runWithProgress`) stays deferred (D99).
//!   Upstream pushes the raw `status.value` string into the description;
//!   the D97a enum erases unknown status strings, so the label collapses
//!   to `"unknown"`.
//!
//! # Deferred tests
//!
//! None of the upstream `test/llama-extension.test.ts` cases reach the real
//! network — they all serve local mocks, so all 12 cases are ported against
//! [`test_support`]. (Upstream has no test for the `/llama` command's
//! interactive flow itself, which is the D99-deferred surface.)
//!
//! Error strings: JS `Error` rejections become `Err(String)` with the same
//! messages; [`is_connection_error`] therefore matches on the message text
//! alone (upstream includes `error.name`).

pub mod client;
pub mod huggingface;
pub mod provider;
pub mod ui;
#[cfg(test)]
pub(crate) mod test_support;

use std::sync::Arc;

use pi_ai::models::Provider;

use crate::core::extensions::types::{
	ExtensionApiHandle, ExtensionCommandContext, ExtensionFactory, ExtensionMode, NotifyKind,
	RegisteredCommandOptions,
};

/// Whether a catalog model is ready to serve (`modelIsLoaded`).
pub fn model_is_loaded(model: &client::LlamaModelInfo) -> bool {
	matches!(model.status.value, client::LlamaModelStatus::Loaded | client::LlamaModelStatus::Sleeping)
}

/// Upstream `isConnectionError` (D97: message-only matching).
pub fn is_connection_error(error: &str) -> bool {
	let message = error.to_lowercase();
	message.contains("fetch failed") || message.contains("timeout") || message.contains("network")
}

/// Upstream `connectionErrorMessage`.
pub fn connection_error_message(error: &str) -> String {
	if is_connection_error(error) {
		return "Could not connect to the server.".to_string();
	}
	error.to_string()
}

/// Upstream `parseHuggingFaceModel`: `owner/repo:quantization`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedHuggingFaceModel {
	pub repository: String,
	pub quantization: Option<String>,
}

pub fn parse_hugging_face_model(value: &str) -> ParsedHuggingFaceModel {
	let search_start = value.find('/').map(|index| index + 1).unwrap_or(0);
	match value[search_start..].find(':') {
		None => ParsedHuggingFaceModel { repository: value.to_string(), quantization: None },
		Some(offset) => {
			let colon = search_start + offset;
			ParsedHuggingFaceModel {
				repository: value[..colon].to_string(),
				quantization: Some(value[colon + 1..].to_string()),
			}
		}
	}
}

/// Upstream `configuredClient`: resolve the stored llama.cpp credential into
/// a management client, warning when unconfigured.
pub async fn configured_client(context: &ExtensionCommandContext) -> Result<Option<client::LlamaClient>, String> {
	let result = context
		.model_registry
		.get_provider_auth(provider::LLAMA_PROVIDER_ID)
		.await
		.map_err(|error| provider::thrown_text(&error))?;
	let Some(result) = result else {
		context.ui.notify(
			format!("Configure llama.cpp with /login {}", provider::LLAMA_PROVIDER_ID),
			Some(NotifyKind::Warning),
		);
		return Ok(None);
	};
	let configured_url = result
		.env
		.as_ref()
		.and_then(|env| env.get("LLAMA_BASE_URL"))
		.filter(|value| !value.is_empty())
		.cloned();
	let server_url = client::normalize_llama_server_url(&match configured_url {
		Some(configured_url) => configured_url,
		None => result.auth.base_url.clone().unwrap_or_default(),
	})?;
	Ok(Some(client::LlamaClient::new(&server_url, result.auth.api_key.as_deref())?))
}

/// The `/llama` command handler (upstream registers the full TUI catalog
/// loop; D99 defers that surface).
async fn handle_llama_command(context: ExtensionCommandContext) {
	if context.mode != ExtensionMode::Tui {
		context.ui.notify("/llama is available in interactive mode".to_string(), Some(NotifyKind::Warning));
		return;
	}
	match configured_client(&context).await {
		Ok(Some(_client)) => {
			// D99: `showLlamaUi` and the `LlamaUi` flows (ui.ts) are not ported.
			context.ui.notify(
				"The llama.cpp catalog UI is not available in this build".to_string(),
				Some(NotifyKind::Warning),
			);
		}
		Ok(None) => {}
		Err(error) => context.ui.notify(error, Some(NotifyKind::Error)),
	}
}

/// The llama extension entry (upstream default export).
pub fn llama_extension() -> ExtensionFactory {
	Arc::new(move |pi: ExtensionApiHandle| {
		Box::pin(async move {
			let controller = provider::create_llama_provider();
			let native: Arc<dyn Provider> = controller.provider.clone();
			pi.register_native_provider(native);
			pi.register_command(
				"llama",
				RegisteredCommandOptions {
					description: Some("Manage llama.cpp router models".to_string()),
					get_argument_completions: None,
					handler: Arc::new(|_args: String, context: ExtensionCommandContext| {
						Box::pin(async move { handle_llama_command(context).await })
					}),
				},
			);
			Ok(())
		})
	})
}

// ============================================================================
// Tests — port of `test/llama-extension.test.ts` (registration case; the
// client/huggingface/provider cases live next to their modules).
// ============================================================================

#[cfg(test)]
mod index_tests {
	use super::*;
	use crate::core::extensions::loader::{create_extension_runtime, load_extension_from_factory};
	use crate::core::extensions::types::{EventBus as EventBusTrait, EventBusHandler, Unsubscribe};
	use serde_json::Value as JsonValue;

	struct ControllerEventBus;

	impl EventBusTrait for ControllerEventBus {
		fn emit(&self, _channel: &str, _data: JsonValue) {}

		fn on(&self, _channel: &str, _handler: EventBusHandler) -> Unsubscribe {
			Box::new(|| {})
		}
	}

	#[tokio::test]
	async fn registers_a_native_provider_and_llama_command() {
		let runtime = create_extension_runtime();
		let extension = load_extension_from_factory(
			llama_extension(),
			".",
			Arc::new(ControllerEventBus),
			Some(runtime.clone()),
			"builtin:llama.cpp",
		)
		.await
		.expect("extension loads");

		assert_eq!(
			extension.commands.get("llama").and_then(|command| command.description.clone()).as_deref(),
			Some("Manage llama.cpp router models")
		);
		let registrations = runtime.pending_native_provider_registrations.read();
		assert_eq!(registrations.len(), 1);
		assert_eq!(
			registrations[0].provider.as_ref().map(|provider| provider.id().to_string()).as_deref(),
			Some(provider::LLAMA_PROVIDER_ID)
		);
	}

	#[test]
	fn parses_hugging_face_model_references() {
		assert_eq!(
			parse_hugging_face_model("owner/repo:Q4_K_M"),
			ParsedHuggingFaceModel { repository: "owner/repo".to_string(), quantization: Some("Q4_K_M".to_string()) }
		);
		assert_eq!(
			parse_hugging_face_model("owner/repo"),
			ParsedHuggingFaceModel { repository: "owner/repo".to_string(), quantization: None }
		);
		assert_eq!(
			parse_hugging_face_model("repo:Q4_K_M"),
			ParsedHuggingFaceModel { repository: "repo".to_string(), quantization: Some("Q4_K_M".to_string()) }
		);
	}

	#[test]
	fn connection_errors_collapse_to_a_friendly_message() {
		assert_eq!(connection_error_message("TimeoutError: the request timed out"), "Could not connect to the server.");
		assert_eq!(connection_error_message("llama.cpp returned HTTP 500"), "llama.cpp returned HTTP 500");
		assert!(is_connection_error("fetch failed"));
	}
}
