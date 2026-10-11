# pi-rs — Rust port of earendil-works/pi

Rust port of [pi](https://github.com/earendil-works/pi) (minimal, extensible agent harness).
TUI package is **excluded** by decision (2026-10-05). Extensions run via an embedded
**deno_core** JS runtime to stay compatible with upstream TypeScript extensions.

- **Baseline upstream tag:** `v1.1.0`
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
| evals | *(none)* | | **Omitted.** Tooling-only package; no Rust runtime counterpart. |
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
6. Publish: push `main`; if the version bumped (full sync), push tag `v<new-tag>` —
   `.github/workflows/release.yml` builds the four platform binaries and publishes
   the GitHub release.
7. Report: changed crates, skipped items + reason, test results.

## Status ledger

| Crate | Status | Upstream tag ported | Deviations |
|---|---|---|---|
| pi-telemetry | ported | v1.1.0 | backend-neutral abstraction behind dyn-safe traits; rejection-identity via `Arc<dyn SpanError>` `ptr_eq`; TS type-level machinery has no runtime counterpart (documented); details in crate docs |
| pi-chord | ported | v1.1.0 | JS Proxy drafts → explicit `Draft` API; `AbortSignal` pure-Rust; TS conditional types → runtime member tables (`RemoteServiceImpl`); esbuild/CJS behind `FacetCompiler`/`FacetBundleModuleLoader` traits (pi-runtime implements); `reportErrorAsync` logs instead of crashing host; `serde_json/preserve_order` for key-order parity; details in crate docs |
| pi-codemode | ported | v1.1.0 | JS worker/QuickJS VM + wasm compile behind `CodeModeWorkerFactory`/`WorkerHandle`/`WasmLoader` traits (pi-runtime implements); sandbox tests use `FakeWorkerFactory`; v1.1 JSON worker-channel and validation parity; details in crate docs |
| pi-mcp | ported | v1.1.0 | `McpError` enum, fetch seam, crate-local abort, stdio process-group safety, and Rust fixture bins; v1.1 OAuth/client parity; details in crate docs |
| pi-protocol | ported | v1.1.0 | CBOR input via explicit `CborValue` enum; JS dynamic-value guards collapse into serde types; `serde_json/preserve_order` keeps CBOR map encoding byte-compatible; per-module details in crate docs |
| pi-ai | ported | v1.1.0 | OpenAI Decisions classifier; provider/model metadata and OAuth updates; Codex header precedence; response-duration propagation. Existing provider/API adaptations are detailed in crate docs |
| pi-runtime | ported | v1.1.0 | new crate, no upstream counterpart: deno_core 0.412 host; implements pi-chord `FacetCompiler`/`FacetBundleModuleLoader`, pi-codemode `CodeModeWorkerFactory`/`WorkerHandle`/`WasmLoader`, and the `ExtensionHost` TS-transpile seam; details in crate docs |
| pi-agent | ported | v1.1.0 | agent loop + Agent state machine; provider streaming behind `StreamFn` seam; proxy streaming events; v1.1 assistant-duration metadata propagation; details in crate docs |
| pi-server | ported | v1.1.0 | tokio evented core: handler callbacks stay sync, async work spawned per task; `ServerHost<TMetadata>` → `Arc<dyn SessionMetadata>` (identity via `Arc`); ServerError subclasses → constructors with identical wire codes; required-by-construction option fields (missing-`listeners` check dropped); timers → tokio sleeps with lock-arbitrated handshake races; Unix transport behind internal `RawSocket` seam (test fake = upstream ControlledSocket); stale-socket fixture child process → in-test dead-listener state; aggregate errors flatten member errors to Display (aggregate message preserved); details in crate docs |
| pi-durable | ported | v1.1.0 | Harness/session/storage v1.1 lifecycle, timestamps, scan/context semantics, tool execution, and PowerShell updates. Cloudflare Durable Object SQLite adapter and test intentionally omitted with user approval (rusqlite-coupled); provider-cache live E2E remains credential-gated; detailed history in crate docs |
| pi-client | ported | v1.1.0 | connection state machine, service subscriptions, unix transport discovery; details in crate docs |
| pi-env | ported | v1.1.0 | remote daemon, SSH/host-key trust, filesystem/exec/watch, path and line-scan parity; no packaged cross-target daemon assets (use explicit binary or `PI_ENV_PACKAGE_ROOT`); SSH external-host test gated by `PI_ENV_SSH_HOST`; `EISDIR` mapping remains a documented deviation; details in crate docs |
| pi-coding-agent | ported | v1.1.0 | Rust CLI/core/API synced, including `defaultTools`/`--tools` modifiers, MCP OAuth, codemode, bash/read, and session/runtime updates. TUI/interactive mode remains excluded; `packages/evals` is tooling-only. Rust harness adaptations and remaining deviations are detailed in crate docs |
