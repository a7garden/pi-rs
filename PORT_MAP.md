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
| pi-coding-agent | porting (slice 8: export-html + modes + extensions + entry) | v1.0.3 | ported: crate skeleton, core/session-manager.ts complete (wire types, migrations v1→v3, projections, SessionManager incl. branching/labels/fork) + core/messages.ts + config.ts complete + utils/ complete (all non-TUI) + core/defaults.ts + core/settings-manager.ts + core/compaction/ complete (D23) + agent-session standalone surface (parseSkillBlock) + core/diagnostics.ts + core/source-info.ts + core/skills.ts complete (D24–D26) + core/extensions/types.ts complete (erased ToolDefinition, D27–D31) + core/tools/ complete (all 16 modules + tools.test.ts 85 cases 1:1, D32–D39) + core/bash-executor.ts + core/extensions/{loader,runner,wrapper,index,virtual-modules}.ts (ExtensionModuleLoader seam until pi-runtime, D40–D46/D52–D54) + core/{model-resolver,provider-composer,model-config,resolve-config-value}.ts (D47–D51) + core/resource-loader.ts + core/{package-manager,pi-manifest,prompt-templates,event-bus,timings}.ts + theme layer (theme.ts/theme-json.ts/system-theme.ts, D55–D58) + core/{model-runtime,model-registry,runtime-credentials,auth-storage,models-store-wrapper,remote-catalog-provider,virtual-models}.ts (D60–D64, D70; ModelRuntime implements the D47 seam; 3 pi-ai bugs found+fixed en route) + core/{system-prompt,slash-commands}.ts (D59) + core/{cache-warmer,nested-tool-calls,session-cwd,session-export,auth-guidance}.ts (D65–D69) + core/agent-session.ts complete (AgentSession class, D74–D79) + core/{sdk,agent-session-runtime,agent-session-services,provider-attribution,telemetry}.ts (D71–D73) + core/export-html/ (templates embedded, D80–D81) + modes/{print-mode,json-event,rpc} (D90–D94; interactive TUI excluded) + core/output-guard.ts + extensions/{mcp,llama,codemode,tool-search} (JS entries behind the D40 seam, D106+) + cli/ + migrations.ts + rpc-entry + main dispatch minus interactive (D100–D105). Tests: 1439 lib + 3 integration green; pi-durable serial 580 green. Deviations D1–D105+ in crate docs. Remaining: pi-runtime JS extension host (D40/D43/D54 seams), export tool-renderer components (D81), deferred live-CLI rpc tests (D94), export-html, export-html, modes/{print,json-event,rpc} (interactive deferred — tui excluded), extensions/{mcp,tool-search,llama,codemode}, experimental/, services/, plugins/, durable/, migrations/, rpc-entry, main.tsnd-in loader seam; jiti lands with pi-runtime) + core/model-{config,resolver}.ts + core/resolve-config-value.ts + core/provider-composer.ts + core/package-manager.ts + core/pi-manifest.ts + core/prompt-templates.ts + core/event-bus.ts + core/timings.ts + core/footer-data-provider.ts (findGitPaths) + core/resource-loader.ts complete (D58; extension-JS/AgentSession test cases deferred with those slices) + modes/interactive/theme/ portable surface (theme/theme-json/system-theme + dark/light/schema, D56) + minimal pi-tui capability port (utils/tui_compat, D56). Tests: 957 lib + 3 integration Rust tests green; pi-durable serial suite 580 green (0 failed, 4 ignored). Deviations D1–D58 in crate docs. Remaining: extensions/{mcp,tool-search,llama,codemode}, system-prompt/slash-commands, export-html, modes/{print,json-event,rpc} (interactive deferred — tui excluded), experimental/, services/, plugins/, durable/, migrations/, rpc-entry, main.ts |
