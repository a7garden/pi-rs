//! A fresh adapter instance and normalized snapshot reader owned by one
//! conformance case (`testing/types.ts` upstream).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::{RecordedTelemetrySpan, TelemetryContext};

/// Async snapshot reader owned by the fixture (`getSpans` upstream).
pub type SnapshotReader =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Vec<RecordedTelemetrySpan>> + Send>> + Send + Sync>;

/// A fresh adapter instance and normalized snapshot reader owned by one
/// conformance case (`TelemetryAdapterFixture` upstream; disposal happens on
/// drop instead of `Symbol.asyncDispose`).
pub struct TelemetryAdapterFixture {
    pub context: Arc<dyn TelemetryContext>,
    pub get_spans: SnapshotReader,
}

impl TelemetryAdapterFixture {
    pub async fn get_spans(&self) -> Vec<RecordedTelemetrySpan> {
        (self.get_spans)().await
    }
}

/// Creates an isolated adapter fixture for one conformance case.
pub type TelemetryAdapterFixtureFactory =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = TelemetryAdapterFixture> + Send>> + Send + Sync>;

/// A runner-independent conformance case that can be registered with any test
/// framework (`TelemetryAdapterConformanceCase` upstream).
pub struct TelemetryAdapterConformanceCase {
    pub group: &'static str,
    pub name: &'static str,
    pub run: Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>,
}

impl TelemetryAdapterConformanceCase {
    pub async fn run(&self) {
        (self.run)().await;
    }
}
