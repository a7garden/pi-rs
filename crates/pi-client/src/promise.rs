//! Port of `packages/client/src/promise.ts`.
//!
//! `PromiseResolvers<T>` becomes a shared resolver over a
//! `Result<T, AnyError>` channel: `resolve` corresponds to the `Ok` arm and
//! `reject` to `Err`. First settlement wins, like a JS promise.

use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use std::pin::Pin;
use std::task::{Context, Poll};

use pi_chord::services::errors::AnyError;
use tokio::sync::oneshot;

/// The promise side of [`create_promise_resolvers`]: awaits `Result<T, AnyError>`.
pub struct Promise<T> {
	receiver: oneshot::Receiver<Result<T, AnyError>>,
}

impl<T> Future for Promise<T> {
	type Output = Result<T, AnyError>;

	fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		match Pin::new(&mut self.receiver).poll(cx) {
			Poll::Ready(Ok(result)) => Poll::Ready(result),
			// Resolver dropped without settling: mirror upstream's impossible case
			// with a disconnected-style error instead of panicking.
			Poll::Ready(Err(_)) => Poll::Ready(Err(Box::new(crate::errors::DisconnectedError::new(
				"Promise resolver was dropped without settling",
			)))),
			Poll::Pending => Poll::Pending,
		}
	}
}

type SharedSender<T> = Arc<parking_lot::Mutex<Option<oneshot::Sender<Result<T, AnyError>>>>>;

/// Upstream `PromiseResolvers<T>`.
#[derive(Clone)]
pub struct PromiseResolvers<T> {
	sender: SharedSender<T>,
	_phantom: PhantomData<fn() -> T>,
}

/// Upstream `createPromiseResolvers<T>()`.
pub fn create_promise_resolvers<T>() -> (PromiseResolvers<T>, Promise<T>) {
	let (sender, receiver) = oneshot::channel();
	(PromiseResolvers { sender: Arc::new(parking_lot::Mutex::new(Some(sender))), _phantom: PhantomData }, Promise { receiver })
}

impl<T> PromiseResolvers<T> {
	/// Upstream `resolve(value)`.
	pub fn resolve(&self, value: T) {
		if let Some(sender) = self.sender.lock().take() {
			let _ = sender.send(Ok(value));
		}
	}

	/// Upstream `reject(reason)`.
	pub fn reject(&self, reason: AnyError) {
		if let Some(sender) = self.sender.lock().take() {
			let _ = sender.send(Err(reason));
		}
	}

	/// Rejects with a pre-built error without requiring ownership (clones the message).
	pub fn reject_with<E: std::error::Error + Send + Sync + 'static>(&self, reason: E) {
		self.reject(Box::new(reason));
	}
}
