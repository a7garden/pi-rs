//! Mirrors upstream `src/wasm.ts`.
//!
//! Upstream, `loadQuickJSWasm` reads `quickjs-wasi/quickjs.wasm` from disk and compiles it into
//! a `WebAssembly.Module`, caching one compiled module per path. The Rust core has no
//! WebAssembly runtime; the compiled module is therefore an opaque handle ([`CodemodeWasmModule`])
//! produced by a [`WasmLoader`] trait implementation. pi-runtime supplies the QuickJS-wasi
//! loader; the crate default is a loader that always fails, mirroring the "wasm not available"
//! sandbox error path.

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// A compiled `WebAssembly.Module` of `quickjs-wasi/quickjs.wasm`, typed opaquely: only the
/// runtime that produced it can hand it to a VM.
pub type CodemodeWasmModule = Arc<dyn Any + Send + Sync>;

/// Loads (and compiles) the QuickJS wasm. A failed load may be retried on the next call,
/// matching upstream caching semantics: successes may be cached per path, failures are not.
///
/// The async surface mirrors upstream `loadQuickJSWasm(path?): Promise<CodemodeWasmModule>`.
pub trait WasmLoader: Send + Sync {
    fn load(&self)
        -> Pin<Box<dyn Future<Output = Result<CodemodeWasmModule, String>> + Send + '_>>;
}

/// The crate-default loader: the QuickJS wasm lives in pi-runtime, not in this pure-Rust core.
/// Its error message feeds the host's `Failed to load QuickJS: …` sandbox error, exactly like
/// the upstream failure path when the wasm file cannot be read or compiled.
pub struct UnavailableWasmLoader;

impl WasmLoader for UnavailableWasmLoader {
    fn load(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<CodemodeWasmModule, String>> + Send + '_>> {
        Box::pin(async {
            Err("the QuickJS wasm is provided by a runtime implementation; configure `CodemodeSandboxOptions::wasm_loader`".to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_loader_fails_with_message() {
        let loader = UnavailableWasmLoader;
        let error = futures_block(loader.load());
        assert!(error.contains("runtime implementation"));
    }

    fn futures_block(
        fut: Pin<Box<dyn Future<Output = Result<CodemodeWasmModule, String>> + Send + '_>>,
    ) -> String {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { fut.await.unwrap_err() })
    }
}
