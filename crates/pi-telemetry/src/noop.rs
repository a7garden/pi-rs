//! Port of packages/telemetry/src/noop.ts.

use std::sync::{Arc, LazyLock};

use crate::{SpanAttributes, SpanCallback, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan, BoxSpanFuture};

/// The single inert span reused by the noop context (frozen upstream; all
/// mutation is a no-op here).
pub struct NoopTelemetrySpan;

impl TelemetrySpan for NoopTelemetrySpan {
	fn start_span_box<'a>(&'a self, _options: SpanOptions, callback: SpanCallback<'a>) -> BoxSpanFuture<'a> {
		callback(noop_telemetry_span())
	}

	fn add_event(&self, _name: &str, _attributes: Option<SpanAttributes>) {}

	fn set_attributes(&self, _attributes: SpanAttributes) {}

	fn set_status(&self, _status: SpanStatus) {}
}

static NOOP_SPAN: LazyLock<Arc<NoopTelemetrySpan>> = LazyLock::new(|| Arc::new(NoopTelemetrySpan));

/// Shared inert span instance, so noop children are the same span as their
/// parent (upstream reuses one frozen span object).
pub fn noop_telemetry_span() -> Arc<dyn TelemetrySpan> {
	Arc::clone(&*NOOP_SPAN) as Arc<dyn TelemetrySpan>
}

/// Shared telemetry context used when an application does not provide one
/// (`NOOP_TELEMETRY_CONTEXT` upstream).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopTelemetryContext;

pub const NOOP_TELEMETRY_CONTEXT: NoopTelemetryContext = NoopTelemetryContext;

impl TelemetryContext for NoopTelemetryContext {
	fn start_span_box<'a>(&'a self, _options: SpanOptions, callback: SpanCallback<'a>) -> BoxSpanFuture<'a> {
		callback(noop_telemetry_span())
	}
}
