//! Port of packages/telemetry/src/testing/index.ts.

pub mod conformance;
pub mod types;

pub use conformance::{case_error, create_telemetry_adapter_conformance, CaseError};
pub use types::{
    SnapshotReader, TelemetryAdapterConformanceCase, TelemetryAdapterFixture,
    TelemetryAdapterFixtureFactory,
};
