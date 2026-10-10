//! Port of `packages/durable/src/testing/runner.ts` — registers the
//! runner-independent cases with a Vitest/Jest-style host test runner
//! (describe/it/expect).
//!
//! Deviations (module ledger):
//! - D1: upstream `expect` returns a chainable matchers object; the Rust
//!   [`ExpectLike`] is a precomputed checker (see `assertions.rs` D1).
//! - D2: upstream's `register_storage_conformance` is deferred: the behavioral
//!   suite already lives runner-free in `crate::storage::conformance` (run by
//!   every backend's tests), and a second assertion-parameterized copy of all
//!   1583 upstream lines would duplicate maintained behavior. Revisit if a
//!   crate outside `storage` needs it through a host runner.
//! - D3: instead of upstream's `Pick<EnvConformanceOptions, "shell" |
//!   "symlinks">` object, `register_env_conformance` takes the two fields
//!   directly.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::assertions::create_expect_assertions;
use super::env_conformance::create_env_conformance;
use super::types::{EnvConformanceOptions, EnvConformanceProvider};

/// Upstream `ExpectLike` (see `assertions.rs` D1): a checker that must fail
/// the test when its condition is `false`.
pub type ExpectLike = Arc<dyn Fn(bool, String) + Send + Sync>;

/// One registered test body.
pub type ItCase = Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + Send>>>;

/// Upstream `StorageConformanceRunner`: the host runner's registration surface.
pub struct StorageConformanceRunner {
    /// Register a suite; the callback performs the nested `it` registrations.
    pub describe: Arc<dyn Fn(&str, Box<dyn FnOnce()>) + Send + Sync>,
    pub expect: ExpectLike,
    /// Register one test; the optional timeout is in milliseconds.
    pub it: Arc<dyn Fn(&str, ItCase, Option<u64>) + Send + Sync>,
}

type CaseRun = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Registers the runner-independent `ExecutionEnv` cases with the host runner.
pub fn register_env_conformance(
    runner: &StorageConformanceRunner,
    name: &str,
    with_env: EnvConformanceProvider,
    shell: Option<Vec<String>>,
    symlinks: Option<bool>,
) {
    let cases = create_env_conformance(EnvConformanceOptions {
        assertions: create_expect_assertions(runner.expect.clone()),
        with_env,
        shell,
        symlinks,
    });
    let describe = runner.describe.clone();
    let it = runner.it.clone();
    let registrations: Vec<(String, Option<u64>, CaseRun)> = cases
        .into_iter()
        .map(|case| {
            let run = case.run.clone();
            (case.name, case.timeout_ms, run)
        })
        .collect();
    describe(
        name,
        Box::new(move || {
            for (case_name, timeout_ms, run) in registrations {
                it(
                    &case_name,
                    Box::new(move || Box::pin(async move { run().await })),
                    timeout_ms,
                );
            }
        }),
    );
}
