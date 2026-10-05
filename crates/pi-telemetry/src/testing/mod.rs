//! Port of packages/telemetry/src/testing/index.ts.

pub mod conformance;
pub mod types;

pub use conformance::{create_telemetry_adapter_conformance, CaseError, case_error};
pub use types::{SnapshotReader, TelemetryAdapterConformanceCase, TelemetryAdapterFixture, TelemetryAdapterFixtureFactory};
