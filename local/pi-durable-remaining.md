# pi-durable remaining work (continuation notes, 2026-10-07)

State after dbabec0+1: `runtime_compaction` 70/71 green (1 ignored),
`runtime_generation` 31/31 green.

Additional fixes landed (commit after dbabec0):
- runtime_generation test fixture builds its provider without api-key auth —
  wired `resolve` (ambient-only) so auth resolution stops returning None.
- SchedulerRuntime::commit/commit_tool now commit under
  TransactionScope{task_id} — appended entries attribute byTaskId (upstream
  runtime commits stamp entries; the admitted user entry stays unattributed).
- Staged terminal outcomes finalize in the same commit when no owned work is
  live (upstream writes `terminal` directly; the race left `completing`).
- ConversationStreamOptions serializes camelCase into durable checkpoints.
- render_diagnostics lowercases severity (`[error]`, upstream string enum).
- beforeRequest replacement messages were stored in a thread_local nobody
  read; now carried through a shared slot and applied to the request.
Full lib suite excluding `harness::tests::state::` still has failures and one
hang. Per-module failure counts from the last full run:

- runtime_generation: FIXED — 31/31 green (fixture was missing faux api-key
  auth; interrupted-retry reopen needed sqlite + reopened submission handle)
- runtime_tools: 15 FAILED (was 16; generation_hooks fixed)
- task_runtime: 2 FAILED
- tasks_abort: 1 FAILED
- task_graph: 1 FAILED
- runtime_registry: 1 FAILED
- hang: `tasks_close::close_during_progress_snapshot_prevents_the_next_phase`
  (serial and parallel; blocks the suite tail)
- `harness::tests::state::` (26 tests): suite hangs at its first test even on
  the pre-fix baseline (verified by stash) — view-watch/listener area.

Known root-cause candidates to investigate first:
1. tasks_close hang: close path waiting on a runner spawned through the new
   `runtime_handle` (current_thread test runtimes only poll when the test
   future yields; a close→join→spawn cycle can self-deadlock).
2. runtime_generation cluster: check whether failures share the faux usage
   estimate / settings overlay changes before assuming independent bugs —
   run one test with `--nocapture` and compare against the upstream
   counterpart in `upstream/pi/packages/durable/test/harness-generation.test.ts`.
3. state module: `ConversationViews::watch` (attach_watch_internal) still does
   not register an observer — likely the same missing-wiring class fixed for
   `state()` in view.rs; the failing-view-listener test is the first hang.
4. select_phase ignore: pre-close creation publication does not trigger a
   seeing drain pass; a lossless publication→kick→drain handoff would let the
   `#[ignore]` come off.

Verification commands:
- cargo test -p pi-durable --lib harness::tests::runtime_compaction:: -- --test-threads=1
- cargo test -p pi-durable --lib -- --skip harness::tests::state::

Then: pi-env, pi-coding-agent (scaffolds), ledger finalization, workspace
version bump only after all crates are at v1.0.3 (PORT_MAP rule 6).
