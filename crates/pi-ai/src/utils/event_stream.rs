//! Port of `packages/ai/src/utils/event-stream.ts`.
//!
//! Upstream `EventStream<T, R>` is a push-based FIFO that is both the producer
//! handle (`push`/`end`) and the single-consumer `AsyncIterable`, plus an
//! independent `result()` promise. The Rust port keeps the same shape: a
//! cloneable handle where [`EventStream::push`]/[`EventStream::end`] produce
//! events, the [`futures_core::Stream`] impl consumes them (single consumer,
//! FIFO), and [`EventStream::result`] resolves independently of iteration.
//!
//! `AssistantMessageEventStream` is the concrete specialization used by the
//! whole provider surface: it completes on `done`/`error` events and extracts
//! the final `AssistantMessage` from either.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc};
use {parking_lot::Mutex};
use std::task::{Context as TaskContext, Poll, Waker};

use crate::types::{AssistantMessage, AssistantMessageEvent, StreamDoneReason, StreamErrorReason};

struct StreamState<T> {
	queue: VecDeque<T>,
	waiters: VecDeque<Waker>,
	done: bool,
}

struct ResultState<R> {
	value: Option<R>,
	waker: Option<Waker>,
}

struct Inner<T, R> {
	state: Mutex<StreamState<T>>,
	result: Mutex<ResultState<R>>,
	is_complete: Box<dyn Fn(&T) -> bool + Send + Sync>,
	extract_result: Box<dyn Fn(&T) -> R + Send + Sync>,
}

/// Generic event stream (upstream `EventStream<T, R = T>`). Cloneable handle;
/// `push`/`end` drive it, the `Stream` impl iterates it, `result` awaits the
/// extracted final value.
pub struct EventStream<T, R = T> {
	inner: Arc<Inner<T, R>>,
}

impl<T, R: Clone + Send + 'static> EventStream<T, R> {
	/// `new(isComplete, extractResult)` upstream.
	pub fn new(
		is_complete: impl Fn(&T) -> bool + Send + Sync + 'static,
		extract_result: impl Fn(&T) -> R + Send + Sync + 'static,
	) -> EventStream<T, R> {
		EventStream {
			inner: Arc::new(Inner {
				state: Mutex::new(StreamState { queue: VecDeque::new(), waiters: VecDeque::new(), done: false }),
				result: Mutex::new(ResultState { value: None, waker: None }),
				is_complete: Box::new(is_complete),
				extract_result: Box::new(extract_result),
			}),
		}
	}

	/// Deliver an event. After a completing event or `end`, further pushes are
	/// ignored.
	pub fn push(&self, event: T) {
		let complete = (self.inner.is_complete)(&event);
		let extracted = if complete { Some((self.inner.extract_result)(&event)) } else { None };

		let mut wakers: Vec<Waker> = Vec::new();
		{
			let mut state = self.inner.state.lock();
			if state.done {
				return;
			}
			if complete {
				state.done = true;
			}
			state.queue.push_back(event);
			// Wake the next iteration waiter (FIFO), mirroring upstream's
			// waiting-consumer dequeue.
			if let Some(waiter) = state.waiters.pop_front() {
				wakers.push(waiter);
			}
		}
		if complete {
			let result_waker = {
				let mut result = self.inner.result.lock();
				result.value = extracted;
				result.waker.take()
			};
			if let Some(result_waker) = result_waker {
				wakers.push(result_waker);
			}
		}
		for waker in wakers {
			waker.wake();
		}
	}

	/// Terminate the stream, optionally supplying the final result for callers
	/// that never saw a completing event (upstream `end(result?)`).
	pub fn end(&self, result: Option<R>) {
		let waker = {
			let mut state = self.inner.state.lock();
			state.done = true;
			if let Some(value) = result {
				let mut result_state = self.inner.result.lock();
				if result_state.value.is_none() {
					result_state.value = Some(value);
				}
				result_state.waker.take()
			} else {
				None
			}
		};
		if let Some(waker) = waker {
			waker.wake();
		}
		// Wake iteration waiters so they observe termination.
		let waiters = {
			let mut state = self.inner.state.lock();
			std::mem::take(&mut state.waiters)
		};
		for waiter in waiters {
			waiter.wake();
		}
	}

	/// Upstream `stream.result()`: resolves with the extracted final value.
	pub fn result(&self) -> EventResultFuture<T, R> {
		EventResultFuture { inner: Arc::clone(&self.inner) }
	}

	/// Whether the stream has terminated.
	pub fn is_done(&self) -> bool {
		self.inner.state.lock().done
	}
}

impl<T, R> Clone for EventStream<T, R> {
	fn clone(&self) -> Self {
		EventStream { inner: Arc::clone(&self.inner) }
	}
}

impl<T: Send + 'static, R: Send + 'static> futures_core::Stream for EventStream<T, R> {
	type Item = T;

	fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<T>> {
		let mut state = self.inner.state.lock();
		if let Some(event) = state.queue.pop_front() {
			return Poll::Ready(Some(event));
		}
		if state.done {
			return Poll::Ready(None);
		}
		state.waiters.push_back(cx.waker().clone());
		Poll::Pending
	}
}

/// Future returned by [`EventStream::result`].
pub struct EventResultFuture<T, R> {
	inner: Arc<Inner<T, R>>,
}

impl<T, R: Clone> Future for EventResultFuture<T, R> {
	type Output = R;

	fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<R> {
		let mut result = self.inner.result.lock();
		if let Some(value) = &result.value {
			return Poll::Ready(value.clone());
		}
		result.waker = Some(cx.waker().clone());
		Poll::Pending
	}
}

/// Producer/consumer handle of the assistant-message event protocol
/// (`AssistantMessageEventStream`). Completing events are `done` and `error`;
/// the final result is `message` for `done` and `error` for `error`.
#[derive(Clone)]
pub struct AssistantMessageEventStream {
	inner: EventStream<AssistantMessageEvent, AssistantMessage>,
}

impl AssistantMessageEventStream {
	/// `new AssistantMessageEventStream()` upstream.
	pub fn new() -> AssistantMessageEventStream {
		AssistantMessageEventStream {
			inner: EventStream::new(
				|event| matches!(event, AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }),
				|event| match event {
					AssistantMessageEvent::Done { message, .. } => message.clone(),
					AssistantMessageEvent::Error { error, .. } => error.clone(),
					other => unreachable!("non-completing assistant event reached extract: {other:?}"),
				},
			),
		}
	}

	pub fn push(&self, event: AssistantMessageEvent) {
		self.inner.push(event);
	}

	/// Terminate without a completing event; `result()` then resolves to
	/// `result` when supplied (used by `lazyStream` forwarding).
	pub fn end(&self, result: Option<AssistantMessage>) {
		self.inner.end(result);
	}

	/// Upstream `.result()`: the final `AssistantMessage`.
	pub fn result(&self) -> EventResultFuture<AssistantMessageEvent, AssistantMessage> {
		self.inner.result()
	}

	/// Collect only the final message, mirroring `for await` + `result()` for
	/// callers that do not need partials.
	pub async fn into_result(self) -> AssistantMessage {
		use futures_util::StreamExt;
		let mut stream = self;
		while stream.next().await.is_some() {}
		stream.result().await
	}
}

impl Default for AssistantMessageEventStream {
	fn default() -> Self {
		AssistantMessageEventStream::new()
	}
}

impl futures_core::Stream for AssistantMessageEventStream {
	type Item = AssistantMessageEvent;

	fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<AssistantMessageEvent>> {
		let this = self.get_mut();
		Pin::new(&mut this.inner).poll_next(cx)
	}
}

/// Factory function for `AssistantMessageEventStream` (for use in extensions).
pub fn create_assistant_message_event_stream() -> AssistantMessageEventStream {
	AssistantMessageEventStream::new()
}

#[cfg(test)]
mod tests {
	// Port of test/event-stream.test.ts (issue #9055 regressions).
	use super::*;
	use futures_util::StreamExt;

	fn number_stream() -> EventStream<u64, u64> {
		EventStream::new(|event| *event == 3, |event| *event)
	}

	#[tokio::test]
	async fn drains_buffered_events_in_order_and_ignores_events_pushed_after_completion() {
		let stream = number_stream();
		stream.push(1);
		stream.push(2);
		stream.push(3);
		stream.push(4);

		assert_eq!(stream.result().await, 3);

		let mut events = Vec::new();
		let mut iter = stream.clone();
		while let Some(event) = iter.next().await {
			events.push(event);
		}
		assert_eq!(events, vec![1, 2, 3]);
	}

	#[tokio::test]
	async fn preserves_order_when_events_arrive_after_buffered_draining_starts() {
		let stream = EventStream::new(|_: &u64| false, |event| *event);
		stream.push(1);
		stream.push(2);

		let mut iterator = stream.clone();
		assert_eq!(iterator.next().await, Some(1));

		stream.push(3);
		assert_eq!(iterator.next().await, Some(2));
		assert_eq!(iterator.next().await, Some(3));

		stream.end(Some(3));
		assert_eq!(iterator.next().await, None);
	}

	#[tokio::test]
	async fn delivers_events_to_waiting_consumers_in_registration_order() {
		let stream = EventStream::new(|_: &u64| false, |event| *event);
		let mut first = stream.clone();
		let mut second = stream.clone();

		let first_task = tokio::spawn(async move { first.next().await });
		let second_task = tokio::spawn(async move { second.next().await });

		// Give both pollers a chance to register their wakers before events
		// arrive.
		tokio::task::yield_now().await;
		stream.push(1);
		stream.push(2);

		assert_eq!(first_task.await.expect("task"), Some(1));
		assert_eq!(second_task.await.expect("task"), Some(2));
	}

	#[tokio::test]
	async fn drains_buffered_events_after_end_and_resolves_the_explicit_result() {
		let stream = EventStream::new(|_: &u64| false, |event| event.to_string());
		stream.push(1);
		stream.push(2);
		stream.end(Some("complete".to_string()));

		assert_eq!(stream.result().await, "complete");

		let mut events = Vec::new();
		let mut iter = stream.clone();
		while let Some(event) = iter.next().await {
			events.push(event);
		}
		assert_eq!(events, vec![1, 2]);
	}

	#[tokio::test]
	async fn wakes_all_waiting_consumers_when_ended_without_a_result() {
		let stream = EventStream::new(|_: &u64| false, |event| *event);
		let mut first = stream.clone();
		let mut second = stream.clone();

		let first_task = tokio::spawn(async move { first.next().await });
		let second_task = tokio::spawn(async move { second.next().await });
		tokio::task::yield_now().await;

		stream.end(None);

		assert_eq!(first_task.await.expect("task"), None);
		assert_eq!(second_task.await.expect("task"), None);
	}
}

/// Build a `done` event (helper shared by adapters).
pub fn done_event(message: AssistantMessage, reason: StreamDoneReason) -> AssistantMessageEvent {
	AssistantMessageEvent::Done { reason, message }
}

/// Build an `error` event (helper shared by adapters).
pub fn error_event(error: AssistantMessage, reason: StreamErrorReason) -> AssistantMessageEvent {
	AssistantMessageEvent::Error { reason, error }
}
