//! Port of `packages/ai/src/api/lazy.ts`.
//!
//! Upstream wraps dynamically imported API modules so the module loads on
//! first stream call and load failures surface as stream errors. Rust links
//! statically, so there is no import step; the wrapper is kept because it
//! still (a) converts construction/setup failures into stream errors and
//! (b) records which deferred capabilities an implementation exposes
//! (crate docs, deviation D10).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::types::{
	DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, Model, SimpleStreamOptions, StreamOptions,
	TranscriptContext,
};
use crate::utils::diagnostics::now_ms;
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::thrown::ThrownValue;

pub use crate::models::{ProviderStreams, API_MISSING_DEFERRED_CANCEL, API_MISSING_DEFERRED_FETCH};

/// Upstream `LazyApiCapabilities`: which optional deferred methods the wrapped
/// implementation is expected to provide.
#[derive(Debug, Clone, Copy, Default)]
pub struct LazyApiCapabilities {
	pub fetch_deferred: bool,
	pub cancel_deferred: bool,
}

fn create_setup_error_message(model: &Model, error: &ThrownValue) -> crate::types::AssistantMessage {
	crate::types::AssistantMessage {
		content: Vec::new(),
		api: model.api().to_string(),
		provider: model.provider().to_string(),
		model: model.id().to_string(),
		response_model: None,
		response_id: None,
		provider_thinking_level: None,
		thinking_level: None,
		diagnostics: None,
		usage: Default::default(),
		stop_reason: crate::types::StopReason::Error,
		deferred: None,
		error_message: Some(crate::utils::diagnostics::format_thrown_value(error)),
		raw_stop_reason: None,
		end_turn: None,
		timestamp: now_ms(),
	}
}

/// Returns a stream synchronously while running async setup (auth resolution)
/// behind it. Setup failures terminate the stream with an error event
/// (`lazyStream`). The setup future runs on first poll of the returned stream
/// and its events are forwarded to the outer stream.
pub fn lazy_stream<F>(model: &Model, setup: F) -> AssistantMessageEventStream
where
	F: std::future::Future<Output = Result<AssistantMessageEventStream, ThrownValue>> + Send + 'static,
{
	let outer = AssistantMessageEventStream::new();
	let target = outer.clone();
	let model = model.clone();
	tokio::spawn(async move {
		match setup.await {
			Ok(inner) => {
				futures_util::pin_mut!(inner);
				let mut final_result = None;
				while let Some(event) = futures_util::StreamExt::next(&mut inner).await {
					if let crate::types::AssistantMessageEvent::Done { message, .. }
					| crate::types::AssistantMessageEvent::Error { error: message, .. } = &event
					{
						final_result = Some(message.clone());
					}
					target.push(event);
				}
				target.end(final_result);
			}
			Err(error) => {
				let message = create_setup_error_message(&model, &error);
				target.push(crate::types::AssistantMessageEvent::Error {
					reason: crate::types::StreamErrorReason::Error,
					error: message.clone(),
				});
				target.end(Some(message));
			}
		}
	});
	outer
}

/// Wraps an API implementation loader as `ProviderStreams` (`lazyApi`). The
/// loader runs on first use; failures terminate the returned stream with an
/// error event.
pub fn lazy_api<L>(load: L, capabilities: Option<LazyApiCapabilities>) -> Arc<dyn ProviderStreams>
where
	L: Fn() -> Result<Arc<dyn ProviderStreams>, ThrownValue> + Send + Sync + 'static,
{
	struct LazyApi<L> {
		load: Arc<L>,
		capabilities: Option<LazyApiCapabilities>,
	}

	impl<L> ProviderStreams for LazyApi<L>
	where
		L: Fn() -> Result<Arc<dyn ProviderStreams>, ThrownValue> + Send + Sync + 'static,
	{
		fn stream(&self, model: &Model, context: &TranscriptContext, options: &StreamOptions) -> AssistantMessageEventStream {
			lazy_stream(model, {
				let model = model.clone();
				let context = context.clone();
				let options = options.clone();
				let load = Arc::clone(&self.load);
				async move {
					let implementation = load()?;
					Ok(implementation.stream(&model, &context, &options))
				}
			})
		}

		fn stream_simple(
			&self,
			model: &Model,
			context: &TranscriptContext,
			options: &SimpleStreamOptions,
		) -> AssistantMessageEventStream {
			lazy_stream(model, {
				let model = model.clone();
				let context = context.clone();
				let options = options.clone();
				let load = Arc::clone(&self.load);
				async move {
					let implementation = load()?;
					Ok(implementation.stream_simple(&model, &context, &options))
				}
			})
		}

		fn fetch_deferred(
			&self,
			model: &Model,
			handle: &DeferredHandle,
			options: &DeferredFetchOptions,
		) -> Option<AssistantMessageEventStream> {
			if !self.capabilities.map(|c| c.fetch_deferred).unwrap_or(false) {
				return None;
			}
			Some(lazy_stream(model, {
				let model = model.clone();
				let handle = handle.clone();
				let options = options.clone();
				let load = Arc::clone(&self.load);
				async move {
					let implementation = load()?;
					implementation
						.fetch_deferred(&model, &handle, &options)
						.ok_or_else(|| ThrownValue::message(API_MISSING_DEFERRED_FETCH))
				}
			}))
		}

		fn cancel_deferred(
			&self,
			model: &Model,
			handle: &DeferredHandle,
			options: &DeferredCancelOptions,
		) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
			if !self.capabilities.map(|c| c.cancel_deferred).unwrap_or(false) {
				return None;
			}
			let model = model.clone();
			let handle = handle.clone();
			let options = options.clone();
			let load = Arc::clone(&self.load);
			Some(Box::pin(async move {
				let implementation = load()?;
				let operation = match implementation.cancel_deferred(&model, &handle, &options) {
					Some(operation) => operation,
					None => return Err(ThrownValue::message(API_MISSING_DEFERRED_CANCEL)),
				};
				operation.await
			}))
		}
	}

	Arc::new(LazyApi { load: Arc::new(load), capabilities })
}
