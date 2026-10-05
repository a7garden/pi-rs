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
| pi-codemode | not started | — | |
| pi-mcp | not started | — | |
| pi-protocol | ported | v1.0.3 | CBOR input via explicit `CborValue` enum; JS dynamic-value guards (`isJsonValue`/`copyJson`) collapse into serde types; `serde_json/preserve_order` keeps CBOR map encoding byte-compatible; per-module details in crate docs |
| pi-ai | not started | — | |
| pi-runtime | not started | — | new crate, no upstream counterpart |
| pi-agent | not started | — | |
| pi-server | not started | — | |
| pi-durable | not started | — | |
| pi-client | not started | — | |
| pi-env | not started | — | |
| pi-coding-agent | not started | — | interactive mode deferred (tui excluded) |
