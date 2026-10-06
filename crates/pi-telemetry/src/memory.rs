//! Port of packages/telemetry/src/memory.ts.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc};
use {parking_lot::Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::noop::NoopTelemetryContext;
use crate::{
	BoxSpanFuture, SharedError, SpanAttributes, SpanCallback, SpanError, SpanErrorDetail,
	SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan,
};

/// `RecordedTelemetryEvent` upstream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedTelemetryEvent {
	pub name: String,
	pub attributes: SpanAttributes,
}

/// `RecordedTelemetrySpan` upstream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedTelemetrySpan {
	pub id: u64,
	pub parent_id: Option<u64>,
	pub name: String,
	pub attributes: SpanAttributes,
	pub events: Vec<RecordedTelemetryEvent>,
	pub status: SpanStatus,
	pub settled: bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub end_sequence: Option<u64>,
}

struct MutableRecordedTelemetryEvent {
	name: String,
	attributes: SpanAttributes,
}

struct MutableRecordedTelemetrySpan {
	id: u64,
	parent_id: Option<u64>,
	name: String,
	attributes: SpanAttributes,
	events: Vec<MutableRecordedTelemetryEvent>,
	status: SpanStatus,
	explicit_status: bool,
	settled: bool,
	end_sequence: Option<u64>,
}

struct InMemoryTelemetryState {
	spans: Mutex<Vec<SharedSpan>>,
	next_span_id: AtomicU64,
	next_end_sequence: AtomicU64,
}

impl InMemoryTelemetryState {
	fn new() -> Self {
		InMemoryTelemetryState {
			spans: Mutex::new(Vec::new()),
			next_span_id: AtomicU64::new(1),
			next_end_sequence: AtomicU64::new(1),
		}
	}
}

type SharedSpan = Arc<Mutex<MutableRecordedTelemetrySpan>>;

fn copy_attributes(attributes: Option<&SpanAttributes>) -> SpanAttributes {
	// Attribute values are cloned; upstream copies array values element-wise.
	attributes.cloned().unwrap_or_default()
}

fn merge_attributes(current: &SpanAttributes, attributes: &SpanAttributes) -> SpanAttributes {
	let mut merged = current.clone();
	for (name, value) in attributes {
		merged.insert(name.clone(), value.clone());
	}
	merged
}

fn copy_status(status: &SpanStatus) -> SpanStatus {
	status.clone()
}

fn automatic_error_status(error: &SharedError) -> SpanStatus {
	SpanStatus::Error {
		error: Some(SpanErrorDetail {
			name: error.telemetry_name(),
			message: error.telemetry_message(),
		}),
	}
}

fn settle_span(state: &InMemoryTelemetryState, span: &SharedSpan, failed: bool, error: Option<&SharedError>) {
	let mut inner = span.lock();
	if inner.settled {
		return;
	}
	if failed && !inner.explicit_status {
		inner.status = match error {
			Some(error) => automatic_error_status(error),
			// Upstream records a detail-less error status when nothing else is
			// derivable from the rejection value.
			None => SpanStatus::error(),
		};
	}
	inner.settled = true;
	inner.end_sequence = Some(state.next_end_sequence.fetch_add(1, Ordering::SeqCst));
}

fn create_span(state: &InMemoryTelemetryState, parent: Option<&SharedSpan>, options: SpanOptions) -> SharedSpan {
	let name = options.name;
	let attributes = copy_attributes(options.attributes.as_ref());
	let id = state.next_span_id.fetch_add(1, Ordering::SeqCst);
	let parent_id = parent.map(|p| p.lock().id);
	Arc::new(Mutex::new(MutableRecordedTelemetrySpan {
		id,
		parent_id,
		name,
		attributes,
		events: Vec::new(),
		status: SpanStatus::Ok,
		explicit_status: false,
		settled: false,
		end_sequence: None,
	}))
}

fn start_in_memory_span<'a>(
	state: Arc<InMemoryTelemetryState>,
	parent: Option<SharedSpan>,
	options: SpanOptions,
	callback: SpanCallback<'a>,
) -> BoxSpanFuture<'a> {
	let parent_settled = parent
		.as_ref()
		.map(|p| p.lock().settled)
		.unwrap_or(false);
	if parent_settled {
		return NoopTelemetryContext.start_span_box(options, callback);
	}

	let recorded_span = create_span(&state, parent.as_ref(), options);
	state
		.spans
		.lock()
		.push(Arc::clone(&recorded_span));

	let span: Arc<dyn TelemetrySpan> = Arc::new(MemoryTelemetrySpan {
		state: Arc::clone(&state),
		span: Arc::clone(&recorded_span),
	});

	Box::pin(async move {
		let result = callback(span).await;
		let err = result.as_ref().err().cloned();
		settle_span(&state, &recorded_span, result.is_err(), err.as_ref());
		result
	})
}

struct MemoryTelemetrySpan {
	state: Arc<InMemoryTelemetryState>,
	span: SharedSpan,
}

impl TelemetrySpan for MemoryTelemetrySpan {
	fn start_span_box<'a>(&'a self, options: SpanOptions, callback: SpanCallback<'a>) -> BoxSpanFuture<'a> {
		start_in_memory_span(Arc::clone(&self.state), Some(Arc::clone(&self.span)), options, callback)
	}

	fn add_event(&self, name: &str, attributes: Option<SpanAttributes>) {
		let mut inner = self.span.lock();
		if inner.settled {
			return;
		}
		// Recording is passive: malformed payloads cannot fail here because the
		// typed attribute map is cloned atomically.
		inner.events.push(MutableRecordedTelemetryEvent {
			name: name.to_string(),
			attributes: copy_attributes(attributes.as_ref()),
		});
	}

	fn set_attributes(&self, attributes: SpanAttributes) {
		let mut inner = self.span.lock();
		if inner.settled {
			return;
		}
		inner.attributes = merge_attributes(&inner.attributes, &attributes);
	}

	fn set_status(&self, status: SpanStatus) {
		let mut inner = self.span.lock();
		if inner.settled {
			return;
		}
		inner.status = copy_status(&status);
		inner.explicit_status = true;
	}
}

/// Backend-neutral reference implementation that records spans in process
/// memory (`InMemoryTelemetryContext` upstream). Create a fresh instance to
/// isolate tests or independent recording scopes.
pub struct InMemoryTelemetryContext {
	state: Arc<InMemoryTelemetryState>,
}

impl Default for InMemoryTelemetryContext {
	fn default() -> Self {
		Self::new()
	}
}

impl InMemoryTelemetryContext {
	pub fn new() -> Self {
		InMemoryTelemetryContext { state: Arc::new(InMemoryTelemetryState::new()) }
	}

	/// Returns detached snapshots in span-start order.
	pub fn get_spans(&self) -> Vec<RecordedTelemetrySpan> {
		self.state
			.spans
			.lock()
			.iter()
			.map(|span| {
				let inner = span.lock();
				RecordedTelemetrySpan {
					id: inner.id,
					parent_id: inner.parent_id,
					name: inner.name.clone(),
					attributes: copy_attributes(Some(&inner.attributes)),
					events: inner
						.events
						.iter()
						.map(|event| RecordedTelemetryEvent {
							name: event.name.clone(),
							attributes: copy_attributes(Some(&event.attributes)),
						})
						.collect(),
					status: copy_status(&inner.status),
					settled: inner.settled,
					end_sequence: inner.end_sequence,
				}
			})
			.collect()
	}

	/// Async snapshot reader mirroring the upstream fixture surface.
	pub fn get_spans_async(&self) -> Pin<Box<dyn Future<Output = Vec<RecordedTelemetrySpan>> + Send + '_>> {
		Box::pin(async move { self.get_spans() })
	}
}

impl TelemetryContext for InMemoryTelemetryContext {
	fn start_span_box<'a>(&'a self, options: SpanOptions, callback: SpanCallback<'a>) -> BoxSpanFuture<'a> {
		start_in_memory_span(Arc::clone(&self.state), None, options, callback)
	}
}

