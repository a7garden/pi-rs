# pi-durable remaining work (continuation notes, 2026-10-08)

## State after this session's fixes (uncommitted, on top of 552c7d8)

Every previously-known failure and hang in `pi-durable` is fixed; full serial
suite run in progress at commit time (all module suites green individually:
state 26/26, tasks_close 7/7, task_runtime 6/6, tasks_abort, task_graph,
tasks_recovery 23/23, runtime_tools 37/37, runtime_compaction/generation/
registry green).

## Root causes fixed (port bugs)

1. **Storage bridge deadlock (session/bridge.rs)** — `StorageBridge::commit`
   blocked the caller's executor thread on a std mpsc reply while a storage
   backend parked the commit (ControlledStorage gates). On current_thread
   test runtimes this deadlocked structurally (the thread that must release
   the gate is blocked on the reply). Fix: Job::Commit reply is now a tokio
   oneshot and `commit` is awaited by `settle_and_store`; read-style calls
   still block (replies never park).
2. **tasks_close hang, close_during_progress_snapshot** — close was spawned
   with `tokio::spawn` from the spinning worker; it landed in the unstealable
   LIFO slot and starved. Upstream's `void (async () => close())` runs the
   close prefix synchronously. Fix (test): spawn close on a dedicated OS
   thread with `Handle::block_on` (CloseOnSnapshotRegistry).
3. **observe() ordering (scheduler.rs)** — abort-mark publication signalled
   the run invocation BEFORE `inner.live.insert`; the woken step could read a
   stale live record, miss rule 3, and fall into rule 6 ("no durable
   progress"), writing faulted→terminal before the first abort_task. Fix:
   insert into live before signalling.
4. **abort() join semantics** — unconditional mode-blind `join_invocation`
   that ignored caller cancellation. Rewritten to upstream shape: the commit
   returns ("terminal"|"marked", Option<run-done>); only the run-mode
   invocation is joined, via `await_with_context` so a cancelled caller
   rejects with its own reason; a live abort handler is never joined.
5. **decide rule 5 report guard** — `reported: Option<()>` merged the
   "nothing reported yet" and "missing definition reported" states, so the
   first `missing_task` report never fired. Now
   `reported: Option<Option<String>>` tracks the reported definition name.
6. **failed-migration retry** — `fit` blocked on `failed_migrations
   .contains_key`, so a changed definition never retried. Upstream keys the
   failure to the definition (`failed?.task === task`); the map now stores
   the AnyTask and `fit` blocks only on the same definition.
7. **seal() leaked the registry subscription** — `_unsubscribe = None`
   dropped the `Box<dyn FnOnce>` without calling it. Now invoked (upstream
   `#unsubscribeRegistry`).
8. **create_agent** — returned before creating the conversation's own agent
   doc for ownerless conversations; upstream `createAgent` creates the doc
   first. `pi.agent` now exists (empty) for every non-fork conversation.
9. **TaskGraphView::watch** — mount created with empty observers and an
   empty graph; the CommittedWatch is now registered as the mount observer
   and the mount starts from the built graph (same class as the view.rs fix).

## Test-fixture deviations fixed (tests matched upstream, not the port)

- state.rs fork history: fork sees its own entries + inherited (upstream
  [c2,c1,r2,r1] shape), not just the fork point.
- fail-fast: upstream aborts live siblings and the PARENT COMPLETES via its
  resume phase (held sibling for determinism); the old ported test expected
  the parent to fail without resuming.
- plain-session agent test: the conversation is created by a separate plain
  Session before the harness opens (upstream shape), so no hook fires.
- task_runtime close/reopen tests: sqlite storage (upstream uses sqlite for
  all close/reopen; MemoryStorage blinds reads after session.close).
- interrupted-running task: definition carries `run` + `resume` phases (the
  checkpoint names the next phase).
- migration test: migrate returns the replacement checkpoint wholesale.
- failed-open test: asserts `storage.is_closed()` (upstream expects the
  shared storage read to reject after the failed harness's close).
- all_settled waiter: initial checkpoint is `wait` (upstream), not `run`.
- runtime_tools recovery hooks/tools await `ctx.abort_signal()` (upstream
  `aborted(callContext.abortSignal)`), not unconditional `pending()`.
- close_rejects_runtime_commit: the queued commit uses the base context;
  upstream rejects via the scheduler closing check, not the invocation
  signal. SchedulerRuntime::commit/commit_tool pass the caller's context to
  the gated checks unchanged (no forced invocation-signal chaining).

## Scheduling-order hardening (port↔upstream semantics)

- `drain()` yields once before reserving: a task end that resolves a joiner
  must let the joiner's next commit land before a reservation reads the line
  (upstream's microtask ordering; tokio's LIFO slot inverts it).

## Wrap-up remaining

- pi-durable: full serial suite GREEN as of 2026-10-08 (`cargo test -p
  pi-durable --lib -- --test-threads=1` → 580 passed, 0 failed, 4 ignored;
  parallel run green too). The former known-failure set of 5 and the 11
  session failures are fixed; see "Root causes fixed (2026-10-08 afternoon)"
  below. After the 2026-10-09 deferred-item work the two agent tests run
  un-ignored: 582 passed, 0 failed, 2 ignored (the remaining two are the
  pre-existing select_phase_hook and jsonl D5 ignores). Serial time vs the
  pre-change baseline (580 passed, 56.14s): 56.33s / 56.10s / 58.98s /
  52.31s across four consecutive green serial runs — within noise; the two
  un-ignored tests themselves run in ~0ms.
- pi-durable tasks_phase_runtime deferred items — RESOLVED (2026-10-09):
  - `task_commits_entries_and_outcomes_atomically...`: the commit-composition
    assertion is deterministic now — the writer's Task write in the answer
    commit is always `terminal` with the completed outcome (the child is
    conversation-owned, so the writer owns nothing; upstream `#commitState`
    writes `terminal` directly in that case and the port's `finalize_on`
    settles the staged `completing` in the same commit; probed 40/40).
    The missing parity also landed: the runtime commit callback now runs
    upstream's terminal-branch scope load (`load_scopes_on` with the commit
    overlay, chain-known-guarded) before `finalize_on` on TERMINAL commits
    only, so the owned-live decision is chain-correct at ~zero cost
    (non-terminal commits are untouched; no storage scans were observable —
    the earlier "slow" attempt had wrapped every commit).
  - `task_agent_rejects_a_pre_cancelled_caller` +
    `task_surfaces_agent_resolution_failure...`: un-ignored. Upstream
    `awaitWithContext(agent(), context)` parity in `SchedulerRuntime::agent`:
    the phase's shared resolution is bound to the invocation's context
    (upstream passes `invocation.context` into `#agent`) and
    `await_with_context` races only the caller's wait, so a cancelled caller
    rejects with its own reason without poisoning the shared resolution and a
    failed resolution stays shared for later callers. The second test's body
    was adapted to inject the resolution's own failure (a direct-storage
    poisoned `pi.agent` document at version 2 — the port stand-in for
    upstream's throwing settings getter, which typed settings cannot
    reproduce); its old gate+abort shape encoded the pre-parity
    resolution-bound-to-caller behavior and is unrepresentable under parity.
    Timing deviation (documented at the callsite): a pre-cancelled caller
    rejects before polling the cell, so the shared resolution starts at the
    next caller rather than eagerly.
- pi-coding-agent: foundation + utils/config/defaults/settings-manager landed;
  remaining module checklist in crate docs (compaction, tools, extensions,
  export-html, modes print/json-event/rpc, agent-session, model-*/system-prompt/
  slash-commands, experimental, services, plugins, durable, migrations, main).
- Workspace version bump only after all crates are at v1.0.3 (PORT_MAP rule 6).

## Root causes fixed (2026-10-08 afternoon session)

Port bugs (impl):
1. **Invocation watches (scheduler.rs)** — upstream `#end` stops the
   invocation's watches (synchronously, settling `stopped`) before aborting
   its controller; the port never tracked them, so invocation-end watches
   died `cancelled`. Port added `Invocation.watches` + `register_watch`
   (upstream `#watchDoc` registration incl. the ended-during-acquisition
   race) and `CommittedWatch::stop_detached` (upstream `stop()` terminates
   synchronously).
2. **Submissions::get (submissions.rs)** — errored "does not exist" where
   upstream `get` returns `undefined`; only `status`/`wait` throw.
3. **Scheduler step double report (scheduler.rs)** — port's `step` reported a
   rejected commit in `map_err` AND returned `Err`, which `start`'s wrapper
   reported again; upstream `#step` catches, reports once, returns
   `undefined`. `step` now swallows after reporting (scheduler D5).
4. **Runtime now/report after end (scheduler.rs)** — no ended guard; upstream
   throws "invocation has ended". Port panics with the same message (D5).
5. **Checkpoint predicate rollback (transaction.rs, D-T5)** — predicate
   panics (the Rust stand-in for upstream throws) escaped `assemble` and
   killed the caller; drafts were not revoked on assemble failure. Now
   caught → `TxError` with the payload message, and `settle_success` revokes
   all drafts on any failure (upstream try/catch → `#abortChanges`).
6. **Fork duplicate-selection member string (forks.rs)** — `{key}/{kind}` →
   upstream's `{kind}/{key}`.

Test-fixture fixes (upstream tests are ground truth):
- forks: as-of/current copies, incarnations, duplicate selections, and
  retire-recreate tests split upstream's bundled "appendEntry + document
  writes in ONE commit" into several commits (or missed family seeds —
  upstream passes a placeholder seed for family members); re-bundled to
  match. checkpoints rewindable migration: inverse — merged two upstream
  commits into one; split back (and its current snapshot needs the
  conversation id).
- checkpoints delta-count fixture set the same value 3× where upstream
  increments (later sets were empty batches that skip the predicate); the
  version-2 base check reads `commits[len-2]` (upstream `.at(-2)`), and the
  strict-migration deviation asserts "requires migration from version 1"
  (the strict-JSON surface is unrepresentable over serde_json — documented
  in the test header).
- tables create-task fixture pushed "started" where upstream pushes
  "created" in that transaction ("started" belongs to the file's createTask
  helper).
- definitions overload test fabricated `TaskId(7)`; upstream is type-level
  only — the port now creates a real task in the commit and exercises every
  addressing overload against it.
- tasks_queue: request-id dedup returns the existing submission BEFORE the
  busy-reject check (upstream "Deduplication runs before the busy check");
  a write placed at a final boundary settles `done` (upstream
  `applySubmissionChange`: placement turns a queued input `placed` and a
  queued write `done`), and entry scans are newest-first (upstream
  `EntryQuery`), so the boundary transcript asserts [f1-user, note] with the
  write carrying the smaller entry id.

## Verification commands

- cargo test -p pi-durable --lib -- --test-threads=1
- cargo test -p pi-durable --lib harness::tests::runtime_tools:: -- --test-threads=1
- cargo test -p pi-durable --lib harness::tests::tasks_recovery:: -- --test-threads=1
