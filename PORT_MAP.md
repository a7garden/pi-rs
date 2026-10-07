# pi-rs — Rust port of earendil-works/pi

Rust port of [pi](https://github.com/earendil-works/pi) (minimal, extensible agent harness).
TUI package is **excluded** by decision (2026-10-05). Extensions run via an embedded
**deno_core** JS runtime to stay compatible with upstream TypeScript extensions.

- **Baseline upstream tag:** `v1.0.3`
- **Upstream checkout:** `upstream/pi` (gitignored; refresh via `sync/fetch-upstream.sh <tag>`)
- **Automation:** `sync/check-release.sh` + launchd `com.garden.pi-rs-sync` (hourly) detect new
  upstream releases and run an omp session that applies the diff to this repo.

## Crate mapping

| Upstream package (`packages/`) | Crate (`crates/`) | Depends on | Notes |
|---|---|---|---|
| telemetry | `pi-telemetry` | — | |
| chord | `pi-chord` | — | |
| codemode | `pi-codemode` | — | |
| mcp | `pi-mcp` | — | |
| protocol | `pi-protocol` | pi-chord | |
| ai | `pi-ai` | pi-telemetry | ~30 providers |
| agent | `pi-agent` | pi-ai | agent loop |
| server | `pi-server` | pi-chord, pi-protocol | |
| durable | `pi-durable` | pi-chord, pi-ai | |
| client | `pi-client` | pi-chord, pi-protocol | |
| env | `pi-env` | pi-chord, pi-durable | |
| coding-agent | `pi-coding-agent` | all above + pi-runtime | CLI + core + modes |
| — | `pi-runtime` | deno_core | new: embedded JS host for extensions/skills |
| tui | *(none)* | | **Excluded.** `coding-agent` interactive mode deferred; print/JSON/RPC modes do not need it. |

## Port order (topological)

1. `pi-telemetry`, `pi-chord`, `pi-codemode`, `pi-mcp`, `pi-protocol`
2. `pi-ai`, `pi-runtime`
3. `pi-agent`, `pi-server`, `pi-durable`, `pi-client`
4. `pi-env`
5. `pi-coding-agent`

## Mapping rules (keep the sync mechanical)

1. **Module layout mirrors upstream 1:1.** `packages/<p>/src/foo/bar.ts` →
   `crates/pi-<p>/src/foo/bar.rs` (or `foo/bar/mod.rs`). `index.ts` → `lib.rs` re-exports.
2. **Types first.** Port type definitions before behavior; keep serde `#[derive]`s on wire types
   so JSON shapes stay byte-compatible with upstream where formats cross process boundaries.
3. **Behavior parity over idiom.** Port upstream logic faithfully (including edge cases and
   error strings used in tests) before "improving". Deviations must be listed in the ledger.
4. **Tests travel.** Every upstream `*.test.ts` becomes a Rust `#[test]`/`#[tokio::test]` in the
   same module, asserting the same cases. No upstream test may be dropped silently.
5. **JS-specific code** (dynamic import, eval) lands in `pi-runtime` behind an explicit trait
   (e.g. `ExtensionHost`) so the rest of the workspace stays pure Rust.
6. **Version discipline.** Workspace `[workspace.package] version` = last upstream tag fully
   ported. Never bump until the ledger says all crates are at that tag.

## Release sync workflow (automated)

Trigger: new tag/release on `earendil-works/pi` → launchd runs `sync/check-release.sh` →
it spawns an omp headless session in this repo with `sync/SYNC_PROMPT.md` as the task.

The sync session MUST:

1. `sync/fetch-upstream.sh <new-tag>`; `git -C upstream/pi diff --stat <prev>..<new> -- packages/`
2. For each changed upstream file: locate its Rust counterpart via this document, apply the
   equivalent change, honor mapping rule 5 for JS-specific surface changes.
3. Update ported tests per rule 4; run `cargo test --workspace`.
4. Update the status ledger below (crate rows: `@ <tag>` once current).
5. Bump workspace version to `<new-tag>` (rule 6); commit as `sync: upstream v<new-tag>`.
6. Report: changed crates, skipped items + reason, test results.

## Status ledger

| Crate | Status | Upstream tag ported | Deviations |
|---|---|---|---|
| pi-telemetry | ported | v1.0.3 | backend-neutral abstraction behind dyn-safe traits; rejection-identity via `Arc<dyn SpanError>` `ptr_eq`; TS type-level machinery has no runtime counterpart (documented); details in crate docs |
| pi-chord | ported | v1.0.3 | JS Proxy drafts → explicit `Draft` API; `AbortSignal` pure-Rust; TS conditional types → runtime member tables (`RemoteServiceImpl`); esbuild/CJS behind `FacetCompiler`/`FacetBundleModuleLoader` traits (pi-runtime implements); `reportErrorAsync` logs instead of crashing host; `serde_json/preserve_order` for key-order parity; details in crate docs |
| pi-codemode | ported | v1.0.3 | JS worker/QuickJS VM + wasm compile behind `CodeModeWorkerFactory`/`WorkerHandle`/`WasmLoader` traits (pi-runtime implements; default impls reproduce upstream failure paths); sandbox.test.ts ported against in-process FakeWorkerFactory — VM-semantics cases deferred to pi-runtime; prelude + MCP preamble embedded verbatim; details in crate docs |
| pi-mcp | ported | v1.0.3 | error classes → one `McpError` enum (payload structs keep upstream names/Display); platform fetch → `McpFetch` seam (reqwest/rustls default); crate-local AbortSignal/Controller; optional OAuthClientProvider methods → trait defaults with observable absence; stdio via tokio process groups + atexit safety net; node test fixtures → Rust fixture bins (CARGO_BIN_EXE); details in crate docs |
| pi-protocol | ported | v1.0.3 | CBOR input via explicit `CborValue` enum; JS dynamic-value guards (`isJsonValue`/`copyJson`) collapse into serde types; `serde_json/preserve_order` keeps CBOR map encoding byte-compatible; per-module details in crate docs |
| pi-ai | ported | v1.0.3 | Provider trait object-safe (optional upstream methods → `Option`-returning); `AssistantMessageEventStream` concrete cloneable handle = upstream AsyncIterable+`.result()`; sync throws → stream error events (D1); core per-API options as nested `StreamOptions` groups (D2); wave-B APIs keep option structs module-local with `*_with_options` factory variants (per-file ledgers); pure-Rust AbortSignal (D3); `FetchFn` HTTP seam, reqwest/rustls default (D4); TypeBox → JSON Schema `serde_json::Value` (D6); merged `ModelCompat` (D7); static linking (D10); Bedrock: hand-rolled SigV4 + AWS binary event-stream decoder + credential chain (no AWS SDK); google-vertex: RS256 service-account JWT via `rsa` (ADC); oauth flows: ambient `oauth_fetch` seam (D29), tokio-clock flow deadline (D43), callback server on tokio TCP; openai-codex: WebSocket transport + zstd body compression omitted (SSE-only path, upstream-fallback JSON; D25/D26); utils/node-http-proxy + oauth-page Node-specific parts simplified (D19/D38); Bun-only code dropped (D8/D40); full per-file ledgers D15+ in crate docs |
| pi-runtime | ported | v1.0.3 | new crate, no upstream counterpart: deno_core 0.412 host; implements pi-chord `FacetCompiler`/`FacetBundleModuleLoader`, pi-codemode `CodeModeWorkerFactory`/`WorkerHandle`/`WasmLoader`, and the `ExtensionHost` TS-transpile seam; details in crate docs |
| pi-agent | ported | v1.0.3 | agent loop + Agent state machine; provider streaming behind `StreamFn` seam; proxy streaming events; details in crate docs |
| pi-server | ported | v1.0.3 | tokio evented core: handler callbacks stay sync, async work spawned per task; `ServerHost<TMetadata>` → `Arc<dyn SessionMetadata>` (identity via `Arc`); ServerError subclasses → constructors with identical wire codes; required-by-construction option fields (missing-`listeners` check dropped); timers → tokio sleeps with lock-arbitrated handshake races; Unix transport behind internal `RawSocket` seam (test fake = upstream ControlledSocket); stale-socket fixture child process → in-test dead-listener state; aggregate errors flatten member errors to Display (aggregate message preserved); details in crate docs |
| pi-durable | ported | v1.0.3 | harness/session/storage port; all harness test suites green (runtime_compaction 70/71, 1 ignored: select_phase dispatch race; runtime_generation, runtime_tools 37, state 26, tasks_close 7, task_runtime 6, tasks_abort, task_graph, tasks_recovery 23). Fixed during bring-up: TaskScheduler::open read/write batching, view_state live observer wiring, place_summary live-state clobber, threshold selection live policy re-read, orphaned outcome settlement, scheduler spawns via captured runtime handle (bridge-thread publications), settings provider overlays static settings; storage bridge commit made awaitable (gate deadlock on single-thread executors); observe updates live before abort-signalling; abort joins only the run invocation via await_with_context; fail-fast checks re-queue the waiter (not the failed task); failed-migration block keyed to the definition; seal invokes the registry unsubscribe; create_agent creates pi.agent for ownerless conversations; task-graph watch registered as mount observer; runtime commits pass caller context ungated. Details in crate docs an…
| pi-client | ported | v1.0.3 | connection state machine, service subscriptions, unix transport discovery; details in crate docs |
| pi-env | ported | v1.0.3 | client/daemon protocol parity: request() performs the hello handshake and caches handshake failure; lexical path normalization in resolve/join_path (matches Node path.resolve); closed/aborted early guards on RemoteReader/RemoteDirReader/RemoteDirReader scans; polling-watch keys directories by identity and recent small files by sha256 content hash; deviation: daemon ERR_FS_EISDIR maps to `is_directory` (upstream leaves unknown) to keep local/remote differential equality; tests: upstream conformance (21 cases over the pipe daemon) + differential (60 seeds × 25 steps, 15 op kinds) + tool differential; details in crate docs |
| pi-coding-agent | porting (foundation) | v1.0.3 | ported: crate skeleton, core/session-manager.ts complete (all wire types, migrations v1→v3, projections, SessionManager incl. branching/labels/fork) + core/messages.ts + config.ts path subset + utils/paths subset; all 117 upstream test/session-manager cases ported 1:1 (123 Rust tests green). Deviations D1–D7 in crate docs (canonical rewrites drop unknown extra JSON fields; Date → i64 unix-ms seam, invalid parses read 0 + UTC for offset-less stamps; normalizePath does not expand file:// URLs). Remaining (dependency order): rest of utils/, config.ts remainder, core/{defaults,compaction,tools,extensions}, settings-manager, model-*/system-prompt/slash-commands, agent-session*, export-html, modes/{print,json-event,rpc} (interactive deferred — tui excluded), extensions/{mcp,tool-search,llama,codemode}, experimental/, services/, plugins/, durable/, migrations/, rpc-entry, main.ts |
