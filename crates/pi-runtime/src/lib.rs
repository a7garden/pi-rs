//! Embedded JavaScript host for the pi Rust port.
//!
//! The rest of the workspace is pure Rust; every place the upstream pi runs
//! JavaScript lands here behind an explicit trait, implemented over an
//! embedded [deno_core](https://docs.rs/deno_core) `JsRuntime` (V8). This
//! crate has no upstream counterpart — it is the JS-runtime capability layer
//! named by `PORT_MAP.md` rule 5.
//!
//! # Trait implementation map
//!
//! | Surface | Upstream behavior | This crate |
//! |---|---|---|
//! | `pi_chord::node::bundle::FacetCompiler` | esbuild bundles a TS entry into one CJS facet file | [`compiler::FacetBundleCompiler`] — deno_ast/swc graph walk + TS strip, single-file CJS artifact carrying an ESM module map |
//! | `pi_chord::node::bundle_loader::FacetBundleModuleLoader` | `node:vm` `compileFunction` + `createRequire` evaluate the verified bundle | [`facets::FacetRuntimeModuleLoader`] — CJS shell evaluation, V8-native ESM linking of the bundled modules, facet export validation via pi-chord's `facets_from_module` |
//! | `pi_codemode::runtime::host::CodeModeWorkerFactory` + `WorkerHandle` | Node `worker_thread` owning a QuickJS-wasi VM per execution | [`codemode::DenoCodeModeWorkerFactory`] — one OS thread owning a fresh `JsRuntime`, running the verbatim upstream prelude |
//! | `pi_codemode::wasm::WasmLoader` | loads `quickjs-wasi/quickjs.wasm` | [`codemode::DenoWasmLoader`] — opaque no-op token (V8 replaces QuickJS) |
//! | `extensions::ExtensionHost` (new seam, below) | jiti-imported TS extensions with an `ExtensionAPI` (`pi`) object | [`extensions::DenoExtensionHost`] — ESM import with TS/JSX transpile, factory invocation, event dispatch, callback slots |
//!
//! # `ExtensionHost` contract (for the pi-coding-agent wave)
//!
//! [`extensions::ExtensionHost`] is the seam this crate provides; the
//! coding-agent implements [`extensions::ExtensionBackend`] and drives the
//! host through [`extensions::DenoExtensionHost::new`].
//!
//! - **Loading.** `load(entry, key)` imports a `.js`/`.ts` entry as an ES
//!   module (relative imports resolve against the entry; TypeScript/JSX
//!   transpile via swc; bare specifiers resolve only to registered
//!   node-module shims), then calls its default export — the factory
//!   `ExtensionFactory` — with a `pi` object. The factory may be async; a
//!   rejected factory surfaces as a load error. `key` is chosen by the
//!   caller; re-loading a path under a new key creates a new generation
//!   without touching the old one. `reload(key, new_key)` re-runs a stored
//!   factory against a fresh `pi` object without re-importing the module
//!   (upstream's cached-factory path: module side effects run once).
//! - **The `pi` object.** `pi.on(event, handler)` subscribes a handler and
//!   returns an unsubscribe function. Every other `pi.register*` method
//!   (`registerTool`, `registerCommand`, `registerShortcut`, `registerFlag`,
//!   `registerMessageRenderer`, `registerEntryRenderer`,
//!   `registerMarkdownTransformer`, `registerVirtualModel`,
//!   `registerProvider`) reaches the backend as
//!   `ExtensionBackend::register(ext, method, payload)` where all function
//!   arguments were replaced by slot references `{"__piSlot": n}`. Any other
//!   property access produces an *action* function:
//!   `pi.<method>(...)` → `ExtensionBackend::call(ext, method, args)`.
//!   `LoadedExtension` reports the summary (`events`, `tools`, `commands`) the
//!   way the upstream `Extension` record exposes them.
//! - **Dispatch.** `dispatch_event(key, event, payload)` invokes that
//!   extension's handlers for `event` in registration order. Each handler
//!   receives `(event, ctx)`; `ctx.<method>` routes to
//!   `ExtensionBackend::call(ext, "ctx.<method>", args)` for the duration of
//!   the handler. Outcomes are reported per handler (`value` or `error`);
//!   aggregation semantics (which result wins) stay with the caller, matching
//!   upstream where each event type aggregates differently.
//! - **Callbacks.** `call_slot(key, slot, args)` invokes a registered function
//!   (a tool `execute`, command handler, renderer, …) with `(args, ctx)` and
//!   awaits its promise. Slot indices come from the registration payloads.
//! - **Lifecycle.** `dispose(key)` drops one extension's handlers.
//!   There is intentionally no per-handler timeout: upstream extensions share
//!   the host process and rely on cooperative cancellation.
//!
//! # Capability surface
//!
//! - Ops: `op_runtime_console` (console → `tracing`), `op_codemode_bridge`
//!   (worker protocol), `op_pi_register`/`op_pi_action` (extension backend).
//!   No fetch, no network, no filesystem access from JS except module loading
//!   of extension/ facet files by the host.
//! - No node/npm compatibility shims are provided. Extensions that need
//!   `node:path` etc. work only when the embedder registers a shim via
//!   `DenoExtensionHostOptions::node_modules` (JSON exports, exposed as an
//!   ESM facade with named exports), `DenoExtensionHostOptions::virtual_sources`
//!   (raw JS modules, e.g. the extension bridge's `typebox` shim), or
//!   `FacetRuntimeOptions::externals` (facet external imports). deno_core
//!   resolves bare specifiers internally before the module loader is
//!   consulted; unregistered ones load unbound, so extensions must only
//!   import registered shims.
//! - deno_core defaults apply: no permissions model, no timers beyond
//!   deno_core's built-ins, `console` routed to `tracing` as above.
//!
//! # Error and termination paths
//!
//! - Codemode: `interrupt` + `terminate()` request V8 termination
//!   (`IsolateHandle::terminate_execution` is thread-safe); the worker loop
//!   polls the interrupt flag between event-loop ticks, so spinning scripts
//!   die on timeout/abort exactly like the upstream QuickJS interrupt handler.
//!   `memory_limit_bytes` becomes a V8 near-heap-limit callback that requests
//!   termination instead of a fatal OOM. Script errors cross the wire shaped by
//!   pi-codemode's `script_error_json` (upstream `describeException`).
//! - Facets: artifact/module/ export errors carry the upstream messages
//!   (validation reuses pi-chord's `facets_from_module`); JS exceptions during
//!   evaluation surface as the chord `AnyError`.
//! - Extensions: load/factory errors surface as `ExtensionHostError`;
//!   handler exceptions are reported per handler, not fatal.
//!
//! # Deviations (ledger)
//!
//! - **D1 — Facet bundles are ESM inside.** Upstream esbuild emits CommonJS
//!   and the loader evaluates it with `node:vm`. The compiler here transpiles
//!   TypeScript to ESM and emits one CJS artifact file whose payload is an ESM
//!   module map; the loader evaluates the CJS shell and links the ESM modules
//!   with V8's native module system. Content addressing, single-`.cjs`
//!   validation, integrity hashing, and the manifest flow (all in pi-chord)
//!   are unchanged.
//! - **D2 — Compiler subset.** Static relative-import graphs only; dynamic
//!   `import()` and TypeScript namespaces fail the build (loudly). `define`
//!   applies to identifier keys only (as per-module top-level bindings);
//!   `minify` is accepted but does not minify; `target`/`platform` are
//!   accepted but do not lower syntax (the embedded V8 targets current ES);
//!   emitted source maps list sources without mappings.
//! - **D3 — JS facet environments are lifecycle-only.** A facet setup from a
//!   JS bundle sees `own`/`onActivate`/`onDeactivate`; the typed service
//!   operations (`use`, `observe`, `provide`, `provideMany`,
//!   `replicatedState`) cannot cross an untyped JS boundary in the Rust port
//!   and throw with a pointer to these docs. The chord SDK external
//!   (`@earendil-works/chord`) must be registered via
//!   `FacetRuntimeOptions::externals` for facets that import it.
//! - **D4 — QuickJS-wasi replaced by V8.** The codemode wasm is a no-op token;
//!   `memory_limit_bytes` maps to the V8 near-heap-limit + terminate semantics
//!   described above. Deep recursion raises a native V8 `RangeError` rather
//!   than a wasm stack guard.
//! - **D5 — Codemode sandbox semantics.** The prelude, protocol, and host
//!   logic are upstream-verbatim (the prelude source is embedded in
//!   pi-codemode and evaluated here unchanged); VM-specific test cases from
//!   upstream `sandbox.test.ts` are exercised in this crate's tests instead of
//!   pi-codemode's in-process fakes.
//! - **D6 — Extension module graph.** Extensions import as ES modules with
//!   swc transpile (upstream: jiti). There is no jiti virtual-module/alias
//!   machinery and no npm resolution: bare specifiers resolve only to
//!   explicitly registered shims. A missing relative import either fails the
//!   load at resolve time or — racing deno_core 0.412's module recursion,
//!   which swallows failing loads — links with the binding unbound; both
//!   outcomes leave the broken import visible to the embedder.

pub mod codemode;
pub mod compiler;
pub mod discovery;
pub mod extensions;
pub mod facets;

mod host;
mod loader;

pub use codemode::{DenoCodeModeWorkerFactory, DenoWasmLoader};
pub use compiler::{BUNDLE_FORMAT, BUNDLE_FORMAT_VERSION, FacetBundleCompiler};
pub use discovery::{dedupe_paths, discover_extensions_in_dir, is_extension_file, read_pi_manifest, resolve_extension_entries, PiManifest};
pub use extensions::{
    DenoExtensionHost, DenoExtensionHostOptions, ExtensionBackend, ExtensionHost, ExtensionHostError,
    HandlerOutcome, LoadedExtension,
};
pub use facets::{FacetRuntimeModuleLoader, FacetRuntimeOptions};
