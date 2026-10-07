//! Port of `packages/durable/src/testing/types.ts` — the runner-independent
//! contracts shared by the conformance suites and the benchmark seeding.
//!
//! Deviations (module ledger):
//! - D1: upstream assertion methods take `unknown` and compare structurally at
//!   runtime; Rust takes `serde_json::Value` (typed records are converted at
//!   the call site) so the trait stays object-safe.
//! - D2: upstream `rejects` receives an already-started `Promise`; Rust
//!   receives an unevaluated boxed future so the operation actually runs
//!   inside the assertion, and the error is carried as its `Display` string.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::env::ExecutionEnv;
use crate::types::Storage;

/// An operation expected to fail, carrying the error message on rejection.
pub type RejectedOperation = Pin<Box<dyn Future<Output = Result<(), String>> + Send>>;

/// Upstream `StorageConformanceAssertions` / `EnvConformanceAssertions`.
#[allow(async_fn_in_trait)]
pub trait ConformanceAssertions: Send + Sync {
	/// `expect(value).toBeTruthy()`.
	fn ok(&self, value: bool, message: &str);
	/// `expect(actual).toBe(expected)`; [`serde_json::Value`] equality is the
	/// structural comparison upstream uses on JSON records.
	fn strict_equal(&self, actual: &Value, expected: &Value);
	/// `expect(actual).toEqual(expected)`.
	fn deep_equal(&self, actual: &Value, expected: &Value);
	/// `expect(actual).toMatchObject(expected)`.
	fn partial_deep_equal(&self, actual: &Value, expected: &Value);
	/// `expect(actual).toBeGreaterThan(expected)`.
	fn greater_than(&self, actual: i64, expected: i64);
	/// `await expect(operation).rejects.toThrow(messageIncludes)`.
	fn rejects(
		&self,
		operation: RejectedOperation,
		message_includes: &str,
	) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

/// Upstream `StorageConformanceProvider`: hands each case a fresh storage.
/// `use` must be called and awaited exactly once per case.
pub type StorageUse =
	Box<dyn FnOnce(Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send>;

pub type StorageConformanceProvider =
	Arc<dyn Fn(StorageUse) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// One runner-independent storage case; the host test runner invokes `run`.
pub struct StorageConformanceCase {
	pub name: String,
	pub run: Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>,
}

pub type EnvUse =
	Box<dyn FnOnce(Arc<dyn ExecutionEnv>) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send>;

/// Upstream `EnvConformanceProvider`: `use` must be called and awaited exactly
/// once per case with an environment whose `cwd` is a fresh, empty, writable
/// directory, then cleaned up.
pub type EnvConformanceProvider =
	Arc<dyn Fn(EnvUse) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

pub struct EnvConformanceOptions {
	pub assertions: Arc<dyn ConformanceAssertions>,
	pub with_env: EnvConformanceProvider,
	/// Program and flag that run a POSIX shell script from the next argument; default `["sh", "-c"]`.
	pub shell: Option<Vec<String>>,
	/// Whether the shell's `ln -s` creates symbolic links; default `true`.
	pub symlinks: Option<bool>,
}

/// One runner-independent `ExecutionEnv` case.
pub struct EnvConformanceCase {
	pub name: String,
	/// Cases that wait for an environment's watch latency need longer than a test runner's default timeout.
	pub timeout_ms: Option<u64>,
	pub run: Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>,
}
