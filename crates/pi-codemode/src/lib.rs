//! Port of `@earendil-works/pi-codemode` (upstream: `packages/codemode`), v1.0.3.
//!
//! Codemode is sandboxed JavaScript execution where the only capability is calling injected
//! tools: each run goes into a fresh QuickJS VM (its own wasm instance) inside a worker, the
//! script sees `tools.<name>(args)`, `ALL_TOOLS`, the output helpers `text`, `image`, `exit`,
//! `console.*`, `store`/`load`, and configured globals — nothing else.
//!
//! # Public API map (mirrors upstream `src/index.ts`)
//!
//! | Upstream | Rust |
//! |---|---|
//! | `declarations.ts` | [`declarations`] |
//! | `identifier.ts` | [`identifier`] |
//! | `source.ts` | [`source`] |
//! | `types.ts` | [`types`] |
//! | `wasm.ts` | [`wasm`] |
//! | `runtime/protocol.ts` | [`runtime::protocol`] |
//! | `runtime/host.ts` (`CodemodeSandbox`) | [`runtime::host::CodemodeSandbox`] |
//! | `runtime/prelude-source.ts` | [`runtime::prelude_source`] |
//! | `runtime/worker.ts` | [`runtime::worker`] (docs + error shaping) |
//!
//! # Traits pi-runtime must implement
//!
//! The JS-runtime surface lives behind explicit traits so this crate stays pure Rust and is
//! testable against in-process fakes:
//!
//! - [`runtime::host::CodeModeWorkerFactory`] — spawns one worker per execution
//!   (upstream `new Worker(workerUrl, { workerData })`), returning a
//!   [`runtime::host::RunningWorker`] (message channels + [`runtime::host::WorkerHandle`]
//!   termination + interrupt flag).
//! - [`wasm::WasmLoader`] — loads/compiles `quickjs-wasi/quickjs.wasm` (upstream
//!   `loadQuickJSWasm`). Defaults to [`wasm::UnavailableWasmLoader`]; worker factory defaults
//!   to [`runtime::host::NoWorkerFactory`]. Both failure paths surface as sandbox errors,
//!   exactly like the upstream missing-worker/wasm failure paths.
//!
//! `execute`/`close` require a tokio runtime (they spawn tasks and await worker termination).
//!
//! # Port deviations
//!
//! - **Worker/VM surface → traits.** `runtime/worker.ts` (the worker-thread entry that boots
//!   the QuickJS VM) and `wasm.ts` (`WebAssembly.compile`) are Node/wasm behavior. Their
//!   behavioral surface is captured by the traits above; the worker-side error shaping that
//!   both protocol sides rely on (`describeException`) is portable and kept as
//!   [`runtime::worker::script_error_json`].
//! - **Prelude is embedded data.** `runtime/prelude-source.ts` interpolates constants into a
//!   template literal; the resolved string is embedded verbatim
//!   (`runtime/prelude.js` via `include_str!`). The upstream "embedded sources parse as
//!   JavaScript" test needs a JS parser and is deferred to pi-runtime.
//! - **Sandbox test scope.** Upstream `test/sandbox.test.ts` runs real JS in QuickJS. Cases
//!   asserting host-side behavior (timeouts, aborts, close, call records, store snapshot
//!   serialization, worker/wasm failure paths, registration validation, parallel execution
//!   isolation) are ported in `tests/sandbox.rs` against an in-process worker fake that
//!   reproduces the upstream host/worker protocol semantics. Cases whose assertions live
//!   inside JS semantics of the VM (script values, `image()`/`text()` validation, V8-shaped
//!   stacks and line numbers, the `tools` guard proxy messages, store size errors raised by
//!   the prelude, escape hatches like eval/`Function`/import denial, deep-recursion
//!   `RangeError`, output-limit enforcement inside the prelude) cannot be expressed without
//!   a real JS engine and are deferred to pi-runtime's QuickJS integration.
//! - **`CodemodeJsonSchema`** is `serde_json::Value` (upstream `{ [key]: unknown } | boolean`);
//!   non-object non-boolean values render as `unknown`, matching the upstream `isObject`
//!   guards.
//! - **`execute` returns `Err("Sandbox is closed")`** where upstream rejects a promise;
//!   script failures still come back as `{ ok: false }` results, never `Err`.
//! - **Worker exit code.** Upstream reports the actual worker exit code in
//!   "Worker exited with code N before the script settled"; the trait exposes
//!   [`runtime::host::WorkerHandle::exit_code`] (default 1) since exit codes are a runtime
//!   concept.
//! - **Tool `execute` errors** carry only the message (`Err(String)`), which is all the
//!   protocol transports (`errorMessage(error)` upstream).

pub mod declarations;
pub mod identifier;
pub mod runtime;
pub mod source;
pub mod types;
pub mod wasm;

// Mirrors upstream `src/index.ts` re-exports.
pub use declarations::{
    mcp_structured_content_schema, render_declarations, render_tool_output_type, render_tool_sample,
    render_tool_signature, schema_to_type, to_codemode_identifier, MCP_TYPESCRIPT_PREAMBLE,
    DEFAULT_INPUT_SCHEMA_MAX_CHARS, RenderDeclarationsOptions, SampleTool, SignatureTool,
};
pub use runtime::host::{
    CodemodeSandbox, CodemodeSandboxOptions, CodeModeWorkerFactory, NoWorkerFactory, RunningWorker,
    WorkerHandle,
};
pub use runtime::prelude_source::{
    MAX_OUTPUT_CHARS, MAX_OUTPUT_ITEMS, MAX_STORE_TOTAL_CHARS, MAX_STORE_VALUE_CHARS,
};
pub use runtime::protocol::{
    is_host_to_worker_message, is_worker_to_host_message, CallTarget, DoneOutcome,
    HostToWorkerMessage, WorkerData, WorkerToHostMessage,
};
pub use runtime::worker::script_error_json;
pub use source::{
    parse_codemode_source, CodemodeSourceError, CodemodeSourceOptions, ParsedCodemodeSource,
    CODEMODE_OPTIONS_PREFIX, CODEMODE_SOURCE_GRAMMAR,
};
pub use types::{
    AbortSignal, CodemodeCall, CodemodeCallStatus, CodemodeError, CodemodeErrorKind,
    CodemodeExecuteOptions, CodemodeJsonSchema, CodemodeOutputItem, CodemodeResult,
    CodemodeStoreWrites, CodemodeTool, CodemodeToolContext,
};
pub use wasm::{CodemodeWasmModule, UnavailableWasmLoader, WasmLoader};
