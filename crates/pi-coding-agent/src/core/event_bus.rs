//! Port of `packages/coding-agent/src/core/event-bus.ts` — a minimal typed
//! event bus: consumers subscribe per-channel callbacks and publishers emit
//! data to a channel.
//!
//! Deviation D57a — upstream is a thin wrapper over Node's `EventEmitter`
//! where handlers may be async and handler exceptions are logged to the
//! console. Rust handlers are synchronous callbacks taking a JSON value; the
//! upstream API surface (`createEventBus`, `emit`/`publish`, `on`/`subscribe`
//! returning an unsubscribe, `clear`) is preserved as
//! [`create_event_bus`]/[`EventBusController::publish`]/[`EventBusController::subscribe`].

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;

type Handler = std::sync::Arc<dyn Fn(&Value) + Send + Sync>;

#[derive(Default)]
struct Inner {
	handlers: HashMap<String, Vec<(u64, Handler)>>,
}

/// Handler-owner and publisher (upstream `EventBusController`); cloneable
/// handles view it as an [`EventBus`].
pub struct EventBusController {
	inner: Arc<Mutex<Inner>>,
	next_id: Arc<Mutex<u64>>,
}

impl Clone for EventBusController {
	fn clone(&self) -> Self {
		Self { inner: Arc::clone(&self.inner), next_id: Arc::clone(&self.next_id) }
	}
}

impl Default for EventBusController {
	fn default() -> Self {
		create_event_bus()
	}
}

/// Read-only view of the bus (upstream `EventBus`).
#[derive(Clone)]
pub struct EventBus {
	inner: Arc<Mutex<Inner>>,
	next_id: Arc<Mutex<u64>>,
}

impl EventBus {
	/// Publish `data` on `channel` (upstream `emit`).
	pub fn publish(&self, channel: &str, data: &Value) {
		EventBusController { inner: Arc::clone(&self.inner), next_id: Arc::clone(&self.next_id) }.publish(channel, data);
	}

	/// Subscribe to `channel`; returns an unsubscribe closure (upstream `on`).
	pub fn subscribe(&mut self, channel: &str, handler: impl Fn(&Value) + Send + Sync + 'static) -> Box<dyn FnOnce() + Send> {
		EventBusController { inner: Arc::clone(&self.inner), next_id: Arc::clone(&self.next_id) }.subscribe(channel, handler)
	}
}

impl EventBusController {
	/// Publish `data` on `channel` (upstream `emit`). Handlers run in
	/// subscription order.
	pub fn publish(&self, channel: &str, data: &Value) {
		let handlers: Vec<Handler> = {
			let mut inner = self.inner.lock();
			match inner.handlers.get_mut(channel) {
				Some(entries) => entries.iter().map(|(_, handler)| Arc::clone(handler)).collect(),
				None => return,
			}
		};
		for handler in handlers {
			handler(data);
		}
	}
	/// Subscribe to `channel`; returns an unsubscribe closure (upstream `on`).
	pub fn subscribe(&mut self, channel: &str, handler: impl Fn(&Value) + Send + Sync + 'static) -> Box<dyn FnOnce() + Send> {
		let id = {
			let mut next_id = self.next_id.lock();
			let id = *next_id;
			*next_id += 1;
			id
		};
		self.inner.lock().handlers.entry(channel.to_string()).or_default().push((id, Arc::new(handler) as Handler));
		let inner = Arc::clone(&self.inner);
		let channel = channel.to_string();
		Box::new(move || {
			inner.lock().handlers.get_mut(&channel).and_then(|entries| {
				entries.iter().position(|(entry_id, _)| *entry_id == id).map(|index| entries.remove(index))
			});
		})
	}

	/// Remove all listeners on all channels (upstream `clear`).
	pub fn clear(&self) {
		self.inner.lock().handlers.clear();
	}

	/// Borrow this controller as an [`EventBus`] view.
	pub fn as_event_bus(&self) -> EventBus {
		EventBus { inner: Arc::clone(&self.inner), next_id: Arc::clone(&self.next_id) }
	}
}

/// Create a new event bus controller (upstream `createEventBus`).
pub fn create_event_bus() -> EventBusController {
	EventBusController { inner: Arc::default(), next_id: Arc::new(Mutex::new(0)) }
}

/// Bridge to the erased extension-side event-bus trait
/// (`extensions/types.rs`): emit publishes, on subscribes through the
/// controller.
impl crate::core::extensions::types::EventBus for EventBusController {
	fn emit(&self, channel: &str, data: Value) {
		self.publish(channel, &data);
	}

	fn on(
		&self,
		channel: &str,
		handler: crate::core::extensions::types::EventBusHandler,
	) -> crate::core::extensions::types::Unsubscribe {
		let controller = EventBusController {
			inner: Arc::clone(&self.inner),
			next_id: Arc::clone(&self.next_id),
		};
		let mut controller = controller;
		controller.subscribe(channel, move |value| handler(value.clone()))
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;
	use std::sync::atomic::{AtomicUsize, Ordering};

	#[test]
	fn publishes_to_channel_subscribers() {
		let mut bus = create_event_bus();
		let count = Arc::new(AtomicUsize::new(0));
		let count2 = Arc::clone(&count);
		let unsubscribe = bus.subscribe("test", move |data| {
			assert_eq!(data, &json!({"value": 1}));
			count2.fetch_add(1, Ordering::SeqCst);
		});

		bus.publish("test", &json!({"value": 1}));
		assert_eq!(count.load(Ordering::SeqCst), 1);

		unsubscribe();
		bus.publish("test", &json!({"value": 1}));
		assert_eq!(count.load(Ordering::SeqCst), 1);
	}

	#[test]
	fn channels_are_isolated_and_clear_removes_all() {
		let mut bus = create_event_bus();
		let count = Arc::new(AtomicUsize::new(0));
		let count2 = Arc::clone(&count);
		let _unsubscribe = bus.subscribe("a", move |_| {
			count2.fetch_add(1, Ordering::SeqCst);
		});
		let count3 = Arc::clone(&count);
		let _unsubscribe_b = bus.subscribe("b", move |_| {
			count3.fetch_add(10, Ordering::SeqCst);
		});

		bus.publish("a", &Value::Null);
		assert_eq!(count.load(Ordering::SeqCst), 1);
		bus.publish("b", &Value::Null);
		assert_eq!(count.load(Ordering::SeqCst), 11);

		bus.clear();
		bus.publish("a", &Value::Null);
		bus.publish("b", &Value::Null);
		assert_eq!(count.load(Ordering::SeqCst), 11);
	}
}
