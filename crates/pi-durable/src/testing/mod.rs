//! Port of `packages/durable/src/testing/` — upstream's runner-independent
//! test infrastructure: conformance suites, expectation adapters, host-runner
//! registration, and benchmark seeding. External crates reuse these against
//! their own `Storage` and `ExecutionEnv` implementations.
//!
//! Module ledger (upstream file → here):
//! - `types.ts` → [`types`]
//! - `assertions.ts` → [`assertions`]
//! - `runner.ts` → [`runner`]
//! - `env-conformance.ts` → [`env_conformance`]
//! - `storage-benchmark.ts` → [`storage_benchmark`]
//! - `storage-conformance.ts` → `crate::storage::conformance` (D1 below)
//!
//! Deviations (module ledger):
//! - D1: upstream `testing/storage-conformance.ts` (the full 1583-line
//!   backend suite) is ported as `crate::storage::conformance` — same cases,
//!   same names — because in this crate every storage backend consumes it
//!   directly (memory, jsonl, sqlite tests) and an earlier pass already
//!   landed a sound direct-assert port there. The assertion-parameterized
//!   creator (`createStorageConformance`) and the storage half of
//!   `runner.ts` are deferred (see `runner.rs` D2) to avoid maintaining a
//!   second full copy; revisit when an external crate needs to run the
//!   storage suite through a host runner.
//! - D2: `index.ts` re-exports targeting the deferred storage creator are
//!   omitted; everything else is re-exported below in upstream's order.

mod assertions;
mod env_conformance;
mod runner;
mod storage_benchmark;
mod types;

pub use assertions::{
    create_expect_assertions, panicking_assertions, partial_equal, to_value, ExpectLike,
};
pub use env_conformance::create_env_conformance;
pub use runner::{register_env_conformance, StorageConformanceRunner};
pub use storage_benchmark::{
    seed_storage_benchmark, seed_storage_write_benchmark, storage_benchmark_primary_record_count,
    StorageBenchmarkDataset, StorageBenchmarkScale, StorageReadBenchmark, StorageWriteBenchmark,
    STORAGE_MEMORY_SCALES, STORAGE_READ_BENCHMARKS, STORAGE_WRITE_BENCHMARKS, TIMING_SCALE,
};
pub use types::{
    ConformanceAssertions, EnvConformanceCase, EnvConformanceOptions, EnvConformanceProvider,
    RejectedOperation, StorageConformanceCase, StorageConformanceProvider, StorageUse,
};

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    use crate::env::{ExecutionEnv, NodeEnvOptions, NodeExecutionEnv};

    use super::types::{EnvConformanceOptions, EnvConformanceProvider};
    use super::{create_env_conformance, panicking_assertions};

    /// Runs the whole `ExecutionEnv` suite against the local `NodeExecutionEnv`,
    /// each case in a fresh empty directory (upstream `env-conformance.test.ts`).
    #[tokio::test(flavor = "multi_thread")]
    async fn env_conformance_against_node_env() {
        let root =
            std::env::temp_dir().join(format!("pi-durable-env-conformance-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("root creates");
        let case_index = Arc::new(AtomicU64::new(0));
        let cleanup_root = root.clone();
        let with_env: EnvConformanceProvider = Arc::new(move |use_env| {
            let root = root.clone();
            let case_index = case_index.clone();
            Box::pin(async move {
                let cwd = root.join(format!(
                    "case-{}",
                    case_index.fetch_add(1, Ordering::SeqCst)
                ));
                std::fs::create_dir_all(&cwd).expect("case dir creates");
                let env: Arc<dyn ExecutionEnv> = Arc::new(NodeExecutionEnv::new(NodeEnvOptions {
                    cwd: cwd.to_string_lossy().into_owned(),
                    ..Default::default()
                }));
                use_env(env).await;
            })
        });
        for case in create_env_conformance(EnvConformanceOptions {
            assertions: panicking_assertions(),
            with_env,
            shell: None,
            symlinks: None,
        }) {
            eprintln!("env conformance case: {}", case.name);
            (case.run)().await;
        }
        std::fs::remove_dir_all(&cleanup_root).ok();
    }
}
