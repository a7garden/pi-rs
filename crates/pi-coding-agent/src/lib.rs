//! Port of `@earendil-works/pi-coding-agent` (upstream:
//! `packages/coding-agent`) — the pi coding agent. This pass ports the
//! foundation: the session-manager core and its direct type/util deps, with
//! the upstream session-manager test suite ported 1:1.
//!
//! # Module map (upstream `packages/coding-agent/src` → this crate)
//!
//! | Upstream | Port | Status |
//! |---|---|---|
//! | `index.ts` | [`lib`] re-exports (ported surface only) | partial |
//! | `config.ts` | [`config`] (constants, `~/.pi/agent/*` paths, share-viewer URL, install-method detection, self-update commands, package-asset paths, install-change detection) | complete (JS-runtime seams) |
//! | `utils/` (all non-TUI files) | [`utils`] (`abort`, `ansi`, `changelog`, `child_process`, `clipboard*`, `deprecation`, `exif_orientation`, `frontmatter`, `fs_watch`, `git`, `html`, `image_*`, `json`, `management_http`, `mime`, `open_browser`, `output_files`, `paths`, `photon`, `pi_user_agent`, `shell`, `sleep`, `syntax_highlight`, `text`, `time`, `tool_result_images`, `tools_manager`, `version_check`, `windows_self_update`, `wsl`, `zip`) | complete (seams, see D8–D17, D21–D22) |
//! | `utils/time` (JS `Date` seam — no upstream file) | [`utils::time`] | complete (seam) |
//! | `core/messages.ts` | [`core::messages`] | complete |
//! | `core/session-manager.ts` | [`core::session_manager`] | complete |
//! | `core/defaults.ts` | [`core::defaults`] | complete |
//! | `core/settings-manager.ts` | [`core::settings_manager`] | complete (D18–D20) |
//! | `core/usage-totals.ts` | [`core::usage_totals`] | complete |
//! | `core/compaction/` | [`core::compaction`] (`compaction`, `branch_summarization`, `utils`) | complete (D23) |
//! | `core/agent-session*.ts` | [`core::agent_session`] (portable surface: `parseSkillBlock`) | partial — `AgentSession`/`AgentSessionRuntime`/`AgentSessionServices` deferred (see [`core::agent_session`] docs) |
//! | `core/diagnostics.ts` | [`core::diagnostics`] | complete |
//! | `core/source-info.ts` | [`core::source_info`] (D25) | complete |
//! | `core/skills.ts` | [`core::skills`] (D24, D26) | complete |
//! | `core/system-prompt.ts` | [`core::system_prompt`] | complete (D59; AgentSession-driven update cases defer with AgentSession) |
//! | `core/slash-commands.ts` | [`core::slash_commands`] | complete — surface types and built-in catalog |
//! | `core/package-manager.ts` | [`core::package_manager`] | complete (D55; `isStdoutTakenOver` stand-in lands with `core/output-guard.ts`) |
//! | `core/pi-manifest.ts` | [`core::pi_manifest`] | complete |
//! | `core/prompt-templates.ts` | [`core::prompt_templates`] | complete (D57, D57b) |
//! | `core/event-bus.ts` | [`core::event_bus`] | complete (D57a) |
//! | `core/timings.ts` | [`core::timings`] | complete (D57c) |
//! | `core/footer-data-provider.ts` | [`core::footer_data_provider`] | minimal — `findGitPaths` only (D57d); the interactive `FooterDataProvider` defers with the TUI mode |
//! | `core/resource-loader.ts` | [`core::resource_loader`] | complete (D58); upstream cases executing extension JS or constructing `AgentSession`/`ExtensionRunner` defer with those slices |
//! | `modes/interactive/theme/` (`theme.ts`, `theme-json.ts`, `system-theme.ts`) | [`theme`] | complete — portable surface (D56, D56d); TUI component themes, cli-highlight bridge, and HTML-export helpers defer with the interactive mode (D56d) |
//! | `@earendil-works/pi-tui` (capability detection) | [`utils::tui_compat`] | minimal (D56); moves to a future `pi-tui` crate |
//! | `core/extensions/types.ts` | [`core::extensions::types`] | complete — type foundation (D27–D31) plus runner-driven additions (D40–D42, D46) |
//! | `core/extensions/loader.ts` | [`core::extensions::loader`] | stand-in layer complete (D40, D43, D52–D54; jiti/TS module seam lands with pi-runtime) |
//! | `core/extensions/runner.ts` | [`core::extensions::runner`] | complete — portable surface (D40–D45; session/agent integration dispatches land with AgentSession) |
//! | `core/extensions/wrapper.ts` | [`core::extensions::wrapper`] | complete |
//! | `core/extensions/index.ts` | [`core::extensions::index`] | complete (runtime re-exports; type-only re-exports stay in [`core::extensions::types`]) |
//! | `core/extensions/virtual-modules.ts` | [`core::extensions::virtual_modules`] | complete as specifier data (D40; module values are JS namespaces behind the loader seam) |
//! | `core/tools/` (all 16 files) | [`core::tools`] | complete (D32–D35, D37–D39); `renderers/*.ts` + theme-coupled `render-utils` helpers deferred with the excluded interactive mode (D32) |
//! | `core/bash-executor.ts` | [`core::bash_executor`] | complete (D35) |
//! | `core/resolve-config-value.ts` | [`core::resolve_config_value`] | complete (sync child-process seams, module docs) |
//! | `core/auth-storage.ts` | [`core::auth_storage`] (D70) | complete |
//! | `core/cache-warmer.ts` | [`core::cache_warmer`] (D67) | complete — stream/session seams + tokio timers (D67); the `ExtensionRunner.emitCacheWarmingDecision` suite is in the module tests |
//! | `core/nested-tool-calls.ts` | [`core::nested_tool_calls`] (D68) | complete |
//! | `core/session-cwd.ts` | [`core::session_cwd`] (D65) | complete — the `createAgentSessionRuntime` case defers with `AgentSessionRuntime` |
//! | `core/session-export.ts` | [`core::session_export`] (D66) | complete — no upstream direct suite; consumers defer with AgentSession/interactive (D66) |
//! | `core/auth-guidance.ts` | [`core::auth_guidance`] (D69) | complete — no upstream suite; consumers defer with sdk/AgentSession (D69) |
//! | `core/model-config.ts` | [`core::model_config`] (D51) | complete |
//! | `core/model-resolver.ts` | [`core::model_resolver`] (D47, D48) | complete |
//! | `core/provider-composer.ts` | [`core::provider_composer`] (D49, D50) | complete |
//! | `core/models-store.ts` | [`core::models_store_wrapper`] (D60) | complete — stores consumed by the model runtime; upstream has no direct suite, see module tests |
//! | `core/model-runtime.ts` | [`core::model_runtime`] (D63; deps: [`core::auth_storage`], [`core::runtime_credentials`], [`core::remote_catalog_provider`], [`core::virtual_models`]) | complete |
//! | `core/runtime-credentials.ts` | [`core::runtime_credentials`] | complete (no upstream suite; covered by [`core::model_runtime`] tests) |
//! | `core/model-registry.ts` | [`core::model_registry`] (D64) | complete — replaces the D29 `extensions/types.rs` stand-in with a re-export |
//! | `test/model-resolver.test.ts` | [`core::model_resolver::model_resolver_tests`] (portable suites 1:1; the persisted-default suite constructs `AgentSession`/`AuthStorage`/`ModelRuntime` and follows those ports) | complete |
//! | `test/resolve-config-value.test.ts` | [`core::resolve_config_value::resolve_config_value_tests`] (1:1; the platform-mock case drives the stdin-transport path through the shell-config override seam) | complete |
//! | `test/tools.test.ts` | `core::tools::tools_tests` + `tests/tools_image_tests.rs` (85 cases, 1:1; image cases split out for photon-seam process isolation, D36) | complete |
//! | `core/tools/renderers/` | — | deferred (TUI renderer layer; D32) |
//!
//! Not ported yet (next passes), in dependency order:
//! `core/agent-session.ts` (`AgentSession` class), `core/agent-session-runtime.ts`,
//! `core/agent-session-services.ts` — blocked on `core/tools/`,
//! `core/extensions/` JS module loading (jiti seam, D40) and the session/agent
//! dispatch integration,
//! `core/sdk.ts`,
//! `core/bug-report.ts`, `core/export-html/`,
//! and the
//! `modes/interactive` theme layer; then `core/tools/renderers/` (with the
//! theme layer), `core/export-html/`, `core/modes/{print,json-event,rpc,interactive}`
//! (interactive TUI excluded by decision), `core/model-runtime.ts` +
//! `core/model-registry.ts` (the registry is a facade over the runtime and
//! ports with it, D47),
//! `core/system-prompt.ts`, `core/slash-commands.ts`,
//! `extensions/{mcp,tool-search,llama,codemode}/`, `experimental/`,
//! `services/`, `plugins/`, `durable/`, `migrations/`, `rpc-entry`,
//! `main.ts`, and TUI-only utils (`modes/interactive/**`).
//!
//! # Deviations from upstream
//!
//! - **D1 — wire types are serde structs; legacy files parse with defaults.**
//!   v1 files (entries without `id`/`parentId`, compaction
//!   `firstKeptEntryIndex`) deserialize into defaults and are repaired by the
//!   ported in-place migrations, matching upstream mutate-and-rewrite
//!   behavior. Rewrites emit canonical field order; unknown extra JSON fields
//!   are not preserved across rewrites, and JSON `null` optional payload
//!   fields (`data`, `details`, `note`) read as absent.
//! - **D2 — `Date` becomes unix-epoch milliseconds.** `SessionInfo.created`/
//!   `modified`, message timestamps and `getTime()` conversions use `i64` ms
//!   via the local [`utils::time`] seam (`toISOString`/`getTime`
//!   equivalents; invalid parses read as `0` where upstream would carry a
//!   `NaN` date). Datetimes without a timezone parse as UTC, not host-local.
//! - **D3 — JS URL seam.** `normalizePath` does not expand `file://` URLs
//!   (`fileURLToPath` has no local counterpart); all other options
//!   (trim/tilde/at-prefix/unicode-spaces/windows shell paths) are ported.
//! - **D4 — custom agent messages are opaque JSON.** Upstream TS declaration
//!   merging extends `AgentMessage`; the Rust [`pi_agent::AgentMessage`]
//!   carries custom roles as [`AgentMessage::Custom`] `serde_json::Value`.
//!   [`core::messages`] provides the typed structs and (de)serialize to the
//!   exact upstream wire shapes. `convertToLlm` drops corrupted custom
//!   payloads that fail to parse instead of rendering `undefined` fields.
//! - **D5 — `undefined` vs `null` leaf selectors.** The free-function
//!   context builders take [`core::session_manager::Leaf`] to distinguish JS
//!   `undefined` (fall back to last entry) from `null` (empty path);
//!   `SessionManager` methods map its `leaf_id` state accordingly.
//! - **D6 — sync throws become `Result`.** Following pi-agent D2:
//!   constructor/statics/appends return `Result` with byte-identical error
//!   strings; session listing aborts surface as
//!   [`core::session_manager::SessionListError::Aborted`].
//! - **D7 — file-order fidelity.** Lines parse as JSON then convert to typed
//!   entries; lines that are valid JSON but not typed entries are skipped as
//!   malformed (upstream keeps them as inert objects and crashes later if
//!   projected). Listing sorts names by byte order instead of
//!   `localeCompare`.
//! - **D8 — `stripAnsi` accepts `&str` only** (upstream's `TypeError` guard
//!   for non-string input is not representable); the ported compatibility
//!   table was generated by running upstream's `stripAnsi` under Node.
//! - **D9 — `AbortSignal`/`AbortController` seam.** The DOM signal pair
//!   becomes a local `tokio::sync::Notify`-backed `AbortSignal`/
//!   `AbortController`; abort reasons surface as [`utils::abort::AbortReason`]
//!   and `raceWithAbortSignal` drops (cancels) the abandoned future.
//! - **D10 — frontmatter YAML subset.** The `yaml` package becomes a
//!   hand-rolled minimal YAML parser (block mappings/sequences, block
//!   scalars, flow collections, comments); values are `serde_json::Value`.
//! - **D11 — git URL parsing.** `hosted-git-info` becomes a reduced in-module
//!   parser in [`utils::git`] (github/gist/gitlab/bitbucket/huggingface hosts,
//!   `github:`-style shortcuts, `#ref` committish, `.git` suffix); unusual
//!   hosts return `None`.
//! - **D12 — HTTP fetch seam.** `management_http.rs` defines a `Fetcher`
//!   trait with a reqwest/rustls default backend; caller cancellation is
//!   future-drop, `timeout_ms` is an overall budget.
//! - **D13 — node-native bundle seams.** No cross-spawn Windows `.cmd`
//!   shims; `fs.watch` becomes a 500 ms polling watcher; shared-objects
//!   quarantine reads `/proc/self/maps` on Linux only; randomness via
//!   getrandom; DOS zip timestamps use UTC and zip writes are sync;
//!   `getChangelogPath` uses `PI_PACKAGE_DIR` or the executable directory;
//!   open-browser detaches via `setsid`.
//! - **D14 — photon codec seam.** `photon-node` becomes the `PhotonEngine`
//!   trait with a process-wide registry, unset by default (paths degrade
//!   exactly like upstream when the WASM module cannot load).
//! - **D15 — image-resize worker.** Node `worker_threads` becomes
//!   `tokio::task::spawn_blocking`; the worker keeps the upstream
//!   `{inputBytes, mimeType, options}` JSON contract over serde.
//! - **D16 — native clipboard / highlight.js seams.** `getNativeClipboard`
//!   becomes a `NativeClipboard` trait (unset by default); highlight.js
//!   becomes a `Highlighter` trait whose default HTML-escapes plain text;
//!   clipboard command runners expose a test override hook.
//! - **D17 — tools-manager network seam.** `fetchWithRetry` becomes the
//!   object-safe `ToolFetcher` trait with a reqwest default; error `cause`
//!   chains flatten to single messages.
//! - **D18 — config runtime seam.** Bun-binary/bundled-Node detection becomes
//!   explicit `set_runtime_context` flags; `__dirname`/`import.meta` asset
//!   anchors start from the running executable's directory; the QuickJS wasm
//!   path defaults to the package-relative location; codemode worker
//!   specifiers are plain path strings.
//! - **D19 — settings locking.** `proper-lockfile` becomes a `<path>.lock`
//!   create-new retry loop (10 attempts × 20 ms); storage errors surface as
//!   `Err`/recorded [`core::settings_manager::SettingsError`] instead of
//!   throws.
//! - **D20 — settings manager is sync.** Upstream's promise write queue
//!   collapses into synchronous `&mut self` writes (already serialized);
//!   `flush` is a no-op and `reload` is synchronous. [`Settings`] is an
//!   ordered JSON map preserving unknown fields; TUI-only capability types
//!   become local structs/enums.
//! - **D21 — HTML entity surrogates.** Lone surrogates (0xD800–0xDFFF)
//!   decode to `None` instead of throwing `RangeError`.
//! - **D22 — `getPiUserAgent` platform naming.** `process.platform`/`arch`
//!   map to `std::env::consts` with Node naming; the runtime token is
//!   `rust` instead of `bun/x.y` or `node/vX`.
//! - **D23 — compaction LLM seam and string metrics.** `completeSimple`/`streamFn`
//!   calls go through [`pi_agent::StreamFn`] or the pi-ai default transport
//!   behind [`core::compaction::compaction::complete_summarization`], which
//!   returns `Err` (exact upstream throw strings) where upstream throws or
//!   propagates a rejected promise; `generateBranchSummary` folds that error
//!   into `BranchSummaryResult.error`. `RetryPolicy`/`RetryCallbacks` and the
//!   `cacheRetention: "none"` + fresh-`sessionId` routing behavior are
//!   preserved via `retryAssistantCall`. JS string `.length` becomes
//!   `char` counts (UTF-8 vs UTF-16 differs only for astral text) and
//!   `FileOperations` sets are `BTreeSet`s (upstream sorts before use).
//!   `CompactionResult.details` is the concrete
//!   [`core::compaction::compaction::CompactionDetails`] instead of a generic
//!   payload. Upstream's API-key-gated LLM tests (`compaction.test.ts`
//!   "LLM summarization") stay unported; upstream's `completeSimple` mocks are
//!   replaced by `StreamFn` captures. `usage-totals.ts` was ported
//!   ([`core::usage_totals`]) as a required dependency of `compaction.ts`.
//! - **D24 — skills ignore matching.** The npm `ignore` matcher becomes the
//!   `ignore` crate's gitignore builder with identical gitignore semantics;
//!   per-directory rules are accumulated parent-first and the matcher is
//!   rebuilt at each directory level, matching upstream's shared-matcher
//!   ordering. Malformed patterns are dropped where upstream swallows
//!   `ig.add` failures.
//! - **D25 — `PathMetadata` location.** Upstream `createSourceInfo` takes
//!   `PathMetadata` from `core/package-manager.ts` (not yet ported); the
//!   struct lives in [`core::source_info`] and moves when package-manager
//!   lands.
//! - **D26 — skills validation message details.** Name/description length
//!   checks count Unicode scalar values where upstream uses JS
//!   `String.length` (UTF-16 units), and file read/stat failures surface
//!   `std::io::Error` display strings instead of Node error messages.
//! - **D27 — extension generics are erased.** `ToolDefinition<TParams,
//!   TDetails, TState>` ([`core::extensions::types::ToolDefinition`]) loses
//!   its type parameters, matching pi-agent's erased [`pi_agent::AgentTool`]:
//!   params become a JSON Schema `serde_json::Value` (`pi-ai` D6 precedent)
//!   and validated JSON objects at `execute`, details/state stay erased JSON
//!   (`AgentToolResult.details` is already JSON in pi-agent), and
//!   `ExtensionVirtualModel`/`ModelRouteRequest`/`ModelRoute` carry their
//!   `TState` as JSON. `defineTool` (a TS inference shim) survives as an
//!   identity function for API parity.
//! - **D28 — extension contexts are vtable structs.** `ExtensionContext` is a
//!   struct of data fields plus `Arc<dyn Fn>` action fields with forwarding
//!   methods (the runner-populated `ExtensionContextActions`);
//!   "extends" contexts (`ExtensionToolContext`,
//!   `ExtensionCommandContext`, `ReplacedSessionContext`) embed their base
//!   and `Deref` to it. `ui` becomes the object-safe
//!   [`core::extensions::types::ExtensionUi`] trait; `ReadonlySessionManager`
//!   is carried as `Arc<SessionManager>` (read-only by convention), and
//!   `ProjectTrustContext.ui` holds the full UI handle where upstream picks
//!   four methods. Handlers take owned contexts (cheap `Arc` clones) so
//!   futures stay `'static`.
//! - **D29 — not-yet-ported dependencies are stand-ins.** Types.ts imports
//!   from unported modules (`exec`, `system-prompt`, `cache-warmer`,
//!   `mcp-servers`, `model-registry`, `bash-executor`, `virtual-models`,
//!   `slash-commands`, `footer-data-provider`, `keybindings`, `event-bus`,
//!   `tools/*`, and the excluded TUI/theme layers); faithful stand-ins live in
//!   the *Dependency stand-ins* section of [`core::extensions::types`] and are
//!   replaced by re-exports when those modules land.
//! - **D30 — partial serde on extension events.** Event payload structs
//!   derive serde (camelCase, absent-`undefined` skipped) only where every
//!   member is wire-representable; structs carrying live handles
//!   (`AbortSignal`), non-serde compaction types (`CompactionPreparation`,
//!   `CompactionResult`), `Skill`, or `ProjectedSessionEntry` are plain
//!   structs the JS bridge will serialize explicitly. `ToolCallEvent`/
//!   `ToolResultEvent` unions keep the upstream flat wire shape via
//!   hand-rolled `Serialize` (constant `type` tag, `toolName` discriminator)
//!   with per-tool input/details payloads erased to JSON until the
//!   `core/tools` port introduces typed structs; they do not deserialize
//!   (the host constructs them), so the runtime guards stay name-only
//!   comparisons, exactly like upstream's.
//! - **D31 — extension API/handler surface is erased.** The `pi.on(...)`
//!   overload set collapses to one erased subscription keyed by the upstream
//!   event-name strings (`Map<string, HandlerFn[]>` storage), `void`
//!   handler results become `Option<R>`/`Option<JsonValue>`, and callback
//!   members (`streamSimple`, `refreshModels`, oauth flows,
//!   `BashOperations`, renderers) are host-side `Arc<dyn Fn>`/trait objects
//!   skipped by serde — the JS bridge adapts JS functions before constructing
//!   the structs. `registerProvider`'s two overloads become
//!   `register_provider` (config) + `register_native_provider`.
//! - **D32 — tool renderers are deferred with the interactive mode.** The
//!   per-tool renderers (`core/tools/renderers/*.ts`) and the theme-coupled
//!   `render-utils` helpers (`linkPath`, `formatToolCallWithArgs`,
//!   `invalidArgText`, `renderToolPath`, `getTextOutput`) style text through
//!   the excluded `Theme`/pi-tui layers and are not exercised by
//!   `tools.test.ts`; the theme-free helpers land in
//!   [`core::tools::render_utils`]. Built-in `ToolDefinition`s carry
//!   `render_shell: None` and no renderers (upstream spreads them in); the
//!   `BASH_UPDATE_THROTTLE_MS` constant moves into [`core::tools::bash`]
//!   (its home `renderers/bash.ts` is deferred).
//! - **D33 — file-mutation queue is a registry of sync mutexes.** The
//!   promise-chaining of `file-mutation-queue.ts` becomes a global registry
//!   keyed on realpath (resolved path on ENOENT/ENOTDIR) with a
//!   `tokio::sync::Mutex` per key and registry cleanup on last exit; other
//!   rethrowable realpath errors fall back to the resolved path so
//!   operations still serialize.
//! - **D34 — output accumulator is synchronous.** `OutputAccumulator` mirrors
//!   Node's synchronous `WriteStream.write` buffering with a `std::fs::File`
//!   (via `create_output_file_stream_sync`), so tool `onData` callbacks stay
//!   synchronous and data pushed while `operations.exec` runs is always
//!   captured before the final snapshot. The streaming `TextDecoder`
//!   becomes a hand-rolled incremental UTF-8 decoder (holds back incomplete
//!   trailing sequences, invalid bytes → U+FFFD). `readFullOutput` returns an
//!   owned-data future so callers never hold the lock across the await.
//! - **D35 — shell spawn seam and test substitutions.** Node
//!   `child_process.spawn` options (cwd/env/detached/stdin-transport) that
//!   [`crate::utils::child_process::spawn_process] does not cover are set
//!   through `tokio::process::Command` directly in [`core::tools::bash`]
//!   (process group via `process_group(0)`, `kill_on_drop(false)`,
//!   `windowsHide` via creation flags); signal exit codes reuse the
//!   `waitForChildProcess` 128+signal mapping. `getShellConfig` becomes an
//!   injected resolver closure on `create_local_shell_operations`, replacing
//!   the test suite's module-function spies; test commands using
//!   `process.execPath` become `sh` equivalents (`sleep`, `cat` stdin
//!   echo); the WSL `bash.exe` case keeps upstream's early return on
//!   non-Windows. `bash-executor.ts` ports with a sync temp-file variant of
//!   the output-file seam.
//! - **D36 — image pipeline and photon fixture.** The read tool reuses the
//!   ported `processImage`/mime seams; BMP→PNG conversion goes through the
//!   `PhotonEngine` seam (unset by default in production). The three image
//!   tests register a minimal in-test engine (BMP decode + PNG encode via
//!   `flate2`) and live in `tests/tools_image_tests.rs` because the engine
//!   registry is process-global. `ReadToolOptions.resize_options` carries the
//!   crate `ImageResizeOptions` (what `processImage` accepts); the
//!   model-provided resize profile maps field-for-field. The optional
//!   `detectImageMimeType` hook is a required trait method (local ops supply
//!   the mime seam).
//! - **D37 — npm `diff` becomes `similar`.** `generateUnifiedPatch` uses
//!   `similar`'s `UnifiedDiff` (header emits `--- path\n+++ path\n`,
//!   matching `FILE_HEADERS_ONLY`; hunks are standard unified format that
//!   the test suite verifies with a local patch applier standing in for npm
//!   `applyPatch`); `generateDiffString` groups `iter_all_changes` output
//!   into npm-`diffLines`-style parts. `normalize("NFKC")` uses the
//!   `unicode-normalization` crate. Edit `access` errors carry
//!   `Error code: ENOENT/EACCES` from the local ops and unknown errors
//!   render as `Error: {message}` (upstream `String(new Error(...))`).
//! - **D38 — rg/fd streaming.** grep/find spawn the real `rg`/`fd`
//!   binaries through the `ensureTool` seam and stream stdout line-by-line
//!   with tokio readers (upstream `readline`); `path.relative`/`basename`
//!   become lexical helpers emitting posix separators on all platforms.
//! - **D39 — localeCompare sort.** The ls entry sort becomes a
//!   case-insensitive byte compare (`a.to_lowercase().cmp(...)`) standing in
//!   for `localeCompare`.
//! - **D47 — model-resolver `ModelRuntime` is an injected seam trait.**
//!   `core/model-runtime.ts` is not ported yet; the resolver functions take
//!   [`core::model_resolver::ModelRuntimeSource`], an object-safe trait with
//!   exactly the five read methods they use (`getModels`,
//!   `getAvailableSnapshot`, `getAvailable`, `getModel`,
//!   `hasConfiguredAuth`). The future `core::model_runtime` port implements
//!   it; tests inject fakes mirroring upstream's duck-typed registries.
//! - **D48 — async signatures without awaits collapse.** Upstream
//!   `findInitialModel` and `restoreModelFromSession` are `async` but never
//!   await; the ported functions are synchronous.
//! - **D49 — `ProviderConfigInput` aliases the extension `ProviderConfig`.**
//!   Upstream declares structurally identical interfaces in
//!   `provider-composer.ts` and `extensions/types.ts`; the port keeps one
//!   type ([`core::extensions::types::ProviderConfig`], aliased in
//!   [`core::provider_composer`]). The only interface delta — composer chat
//!   model configs carry `samplingParamsByThinkingLevel` — is unreachable
//!   through the extension interface upstream too.
//! - **D50 — provider-composer throws, state, and capability presence.**
//!   Synchronous throws return `Err` with exact upstream strings and
//!   `composeModelProvider` still validates eagerly; refresh-path captured
//!   state lives in a mutex, with the `context.publish` update validating
//!   the refreshed list and dropping the publication on failure (upstream
//!   throws inside the update callback). Trait objects cannot report base
//!   provider capability overrides (`refreshModels`/`generateImages`/
//!   `classify`), so the composed provider exposes them whenever a base
//!   provider exists; extension `refreshToken` receives a crate abort signal
//!   bridged from the pi-ai signal by a watcher task; extension stream
//!   dispatch wraps `StreamOptions` into `SimpleStreamOptions` where
//!   upstream casts.
//! - **D51 — TypeBox validation becomes serde + explicit checks.**
//!   `model-config.ts` schemas compile to one validator; the port
//!   deserializes with serde, then checks `minLength: 1`/`oauth: "radius"`
//!   constraints, producing the same error bucket with serde-style messages.
//!   `Type.Number()` fields carried as `u64` (context windows, token counts,
//!   cache lifetimes, image limits) reject floats/negatives; the
//!   `ProviderCompat` union becomes pi-ai's merged `ModelCompat` (pi-ai D7).
//! - **D40 — in-place event mutation becomes explicit handler results.**
//!   Upstream handlers mutate shared event objects in place (tool-call args,
//!   provider headers, boundary entry arrays). The erased handler channel
//!   ([`core::extensions::types::HandlerFn`]) passes owned JSON, so patches
//!   flow through handler results: `emit_tool_call` reads an `input` key,
//!   header patches shallow-merge, boundary entries flow through result
//!   `entries` (pi-agent D8 precedent). `ExtensionRuntime` additionally
//!   carries a shared [`core::extensions::types::RuntimeCore`] bind cell so
//!   `bindCore` reaches runtimes extensions already hold.
//! - **D41 — handler failures are an explicit `Err`.** Upstream throws
//!   across the `await`; [`core::extensions::types::HandlerFn`] returns
//!   `Result<Option<JsonValue>, String>`, surfacing through
//!   `ExtensionRunner::on_error`.
//! - **D42 — `Extension.handlers` is a shared live registry.** Upstream the
//!   `Extension` object *is* the live handler map; the port stores
//!   `Arc<Mutex<HashMap<...>>>` so loader-API subscriptions and
//!   unsubscriptions after commit stay visible to the runner.
//! - **D43 — the loader is a stand-in until pi-runtime lands.** No TS/JS
//!   module resolution: `load_extensions_cached` walks the upstream pipeline
//!   (resolve, `(path, mtime)` cache, error records with upstream texts)
//!   without jiti, and provider/virtual-model registrations queue exactly
//!   like upstream's pre-bind runtime. Deep MCP config validation lands
//!   with the mcp-servers module port (name-shape, ownership, and
//!   namespace-clash checks run in the loader). The `operations.exec`
//!   function check of `user_bash` results is bridge work.
//! - **D44 — `buildSystemPrompt` is a renderer seam.** The runner takes a
//!   `set_system_prompt_renderer` closure; the default now calls the real
//!   `core::system_prompt` builder, and `ctx.getSystemPrompt()` re-renders
//!   per handler, so chained `before_agent_start` overrides stay in sync.
//! - **D45 — `KeybindingsConfig` is `ResolvedKeybindings`.** The runner
//!   takes `BTreeMap<String, Vec<KeyId>>` (action → keys) until
//!   `core/keybindings.ts` ports.
//! - **D46 — minimal stand-in surfaces.** `McpServerRegistry` exposes
//!   register/unregister/list with extension-path ownership (change
//!   listeners and connection lifecycle land with the module port); the
//!   loader ships an in-crate [`core::extensions::loader::ChannelEventBus`];
//!   shortcut conflict fallbacks use `eprintln!` where upstream
//!   `console.warn`s.
//! - **D52 — extension `exec` runs through the bash executor.** Upstream
//!   `execCommand` (core/exec.ts) is not ported; `pi.exec` shells out via
//!   [`core::bash_executor`] and maps [`core::extensions::types::ExecResult`]
//!   (command + args join into one command line). The timings-module
//!   `time()` probes are omitted.
//! - **D53 — `pi.extensions` package manifests are read inline.** The
//!   loader parses the `pi.extensions` field of `package.json` directly;
//!   the pi-manifest module port replaces `read_pi_manifest_extensions`.
//! - **D54 — post-bind provider registration from captured `pi` objects
//!   stays queued.** Upstream `pi.registerProvider` after `bindCore` takes
//!   effect immediately; the loader stand-in's API keeps its private queue,
//!   so immediate-effect registration lands with the pi-runtime loader seam.
//! - **D55 — package-manager dependency stand-ins and seams.**
//!   `isStdoutTakenOver` (upstream `core/output-guard.ts`, unported) is a
//!   stand-in returning `false`; test mocks of private methods become the
//!   [`core::package_manager::CommandRunner`] seam (`with_runner`); and
//!   `resolve`'s `onMissing` callback is synchronous. **D55a** — npm
//!   `minimatch` is approximated with the `glob` crate's `Pattern` (no brace
//!   expansion; `*` also matches `/` under default options). **D55b** —
//!   npm/git commands run synchronously-through-async via `tokio::process`
//!   (no `cross-spawn` `.cmd`/`.bat` shim; utils D13 rationale).
//! - **D56 — theme capability detection lives in
//!   [`utils::tui_compat`].** A minimal port of npm
//!   `@earendil-works/pi-tui` (`PI_TRUE_COLOR`/`PI_HYPERLINKS`/
//!   `PI_IMAGE_PROTOCOL` semantics identical) that lands in a future
//!   `pi-tui` crate. **D56a** — JS module statics become explicit
//!   mutex-protected statics. **D56b** — theme-json typebox validation
//!   becomes a hand-written validator with 1:1 accepted grammar and
//!   error-message shape (per-error wording approximated). **D56c** —
//!   built-in `dark.json`/`light.json` are embedded via `include_str!`.
//!   **D56d** — TUI/interactive-only theme surface (global `theme` proxy,
//!   custom-theme watcher, cli-highlight bridge, TUI component themes,
//!   HTML-export helpers, `ThinkingLevel`-typed accessors) defers with the
//!   interactive mode. **D56e** — chalk text decorators are not ported;
//!   use [`utils::ansi`].
//! - **D57 — prompt-template substitution scanner.** Upstream's substitution
//!   regex is a hand-rolled scanner with identical match semantics.
//!   **D57a** — the event bus wraps synchronous JSON callbacks (upstream
//!   `EventEmitter` allows async handlers); the API surface
//!   (`create_event_bus`, `publish`, `subscribe`) is preserved. **D57b** —
//!   diagnostics/truncation messages use Rust strings: I/O failures surface
//!   `std::io::Error` display strings and lengths use Unicode scalar counts
//!   where upstream uses UTF-16 units (interacts with D10: the minimal
//!   frontmatter parser accepts upstream's #9354 invalid-YAML fixture as a
//!   plain scalar, so that regression uses a non-mapping frontmatter). 
//!   **D57c** — `timings` namespace arguments are explicit with `*_main`
//!   convenience wrappers (JS default parameters). **D57d** — only
//!   `findGitPaths` is ported from `footer-data-provider.ts`; the
//!   interactive `FooterDataProvider` defers with the TUI mode.
//! - **D58 — resource-loader passes run synchronously.** Upstream `reload()`
//!   is async (settings/package processes, awaited module compilation); the
//!   port drives the same passes (pre-trust extension load, project-trust
//!   resolution, package resolution, resource loading) synchronously on a
//!   throwaway current-thread runtime. **D58a** — `file://` resource URLs
//!   are expanded by the loader's path resolution (`utils::paths` returns
//!   them unchanged by its JS-seam deviation). **D58b** — extension warning
//!   records carry their text in `LoadRecord::error` (upstream uses a
//!   `{path, warning}` shape); `None` warnings are treated as empty.
//! - **D61 — remote catalog provider ([`core::remote_catalog_provider`]).**
//!   Upstream `withRemoteCatalog` wraps a provider with a persisted pi.dev
//!   catalog overlay. **D61a** — `AbortSignal` during the catalog fetch
//!   becomes `race_with_abort_signal`, surfacing as an `Err` from
//!   `Provider::refresh_models` (upstream's rejected promise). **D61b** —
//!   catalog entries are strictly deserialized into `AnyModel` and
//!   unparseable/unsupported entries are dropped instead of spread loosely.
//!   **D61c** — `Last-Modified` parsing only accepts IMF-fixdate, falling
//!   back to `0` (JS `Date.parse`'s `NaN` fallback). **D61d** —
//!   `getBuiltinModelDataGeneratedAt` has no pi-ai data source (the embedded
//!   generated catalog carries no `generatedAt` field); the pi-ai accessor
//!   (`providers::all`) returns `None`, so callers pass `None`. **D61e** —
//!   the management-HTTP `HttpResponse` gained response headers so
//!   `etag`/`last-modified` revalidation ports 1:1.
//! - **D59 — system-prompt custom sections render in sorted name order.**
//!   Upstream `sections` is a JS object (insertion order); the port carries
//!   `BTreeMap` (D30), so extension-contributed sections are appended sorted
//!   by name after the built-in ones (`preamble`, `addendum`,
//!   `project_context`, `skills`, `cwd`).
//! - **D60 — models-store object semantics become `BTreeMap` + shared
//!   futures.** Provider maps sort by key (D16 rationale), so rewritten
//!   files are key-sorted rather than insertion-ordered; persisted entries
//!   keep the upstream camelCase wire format (pi-ai `ModelsStoreEntry`
//!   serialization gained the rename to match its lax reader). The
//!   single-flight reload promise is a `futures::Shared` boxed future raced
//!   per reader with `raceWithAbortSignal`; reader-count cleanup runs on
//!   lease drop, and the write/delete path folds the post-lock read-state
//!   refresh into the file lock. JSON errors carry `serde_json` messages
//!   (not V8 `SyntaxError` text) and map parsing is strict about non-object
//!   files that upstream would only trip over when read.
//! - **D62 — virtual-models router state is JSON.** Upstream's
//!   `VirtualModelDefinition<TState>`/`ModelRoute<TState>` generics
//!   ("must be JSON-serializable") collapse to `serde_json::Value` in
//!   [`core::virtual_models`], and the `route` callback (sync value or
//!   promise) becomes `Arc<dyn Fn(RouteRequest) ->
//!   BoxFuture<Result<ModelRoute, ThrownValue>>>` behind a free `route()`. **D62a** —
//!   `createVirtualModel` takes the full definition and ignores `route`
//!   (upstream `Omit<VirtualModelDefinition, "route">`). **D62b** —
//!   `isVirtualModel({ api })` splits into `is_virtual_model_api` (any api
//!   string) and `is_virtual_model` (chat model). **D62c** —
//!   `withVirtualModels` object-spread becomes an explicit `Provider`
//!   wrapper; the `providerId` argument only names the keyless provider,
//!   exactly like upstream. **D62d** — `findLatestResponse` and
//!   `getBranchSelection` treat only typed LLM assistant messages as
//!   responses (upstream's structural check would also admit custom
//!   `{role: "assistant"}` JSON values).
//! - **D63 — model-runtime ([`core::model_runtime`]).** Upstream
//!   `ModelRuntime` implements pi-ai `Models`; the port wraps a
//!   [`pi_ai::models::ModelsImpl`] and implements the same trait. **D63a** —
//!   `create()` is async solely for the initial refresh (`ModelConfig::load`
//!   is a sync file read). **D63b** — catalog `generatedAt` is always `None`
//!   (embedded pi-ai catalog carries no manifest field, D61d). **D63c** —
//!   credential operations serialize through a per-provider async mutex and
//!   check the signal after acquisition (upstream races the queued
//!   "started" promise with the signal; an already-running operation is
//!   awaited). **D63d** — the create-time network refresh timeout cancels
//!   the refresh future directly instead of aborting a linked signal.
//!   **D63e** — typing replaces runtime model-kind asserts (`stream` takes a
//!   chat `Model`, image/classifier entry points take their concrete model
//!   types). **D63f** — snapshot maps are `BTreeMap`/`BTreeSet` (D16
//!   rationale). **D63g** — `streamSimple` dispatches the physical request;
//!   virtual-model routing runs through [`core::model_runtime::ModelRuntime::resolve_model`]
//!   (the JS overload-based recursion collapses into the router seam).
//!   **D63h** — `refreshProviderAvailability` writes its provider-scoped
//!   facts (fresh check + credential read, merged into the current
//!   snapshot) even when a newer availability pass for the provider
//!   started; upstream drops the whole update on a `providerSeq` mismatch,
//!   which can lose the post-credential-change visibility guarantee when a
//!   fire-and-forget registration refresh overlaps synchronization.
//! - **D64 — model-registry ([`core::model_registry`]).** The synchronous
//!   extension facade over `ModelRuntime`; the D29 unit stand-in in
//!   `extensions/types.rs` becomes a re-export. **D64** — upstream
//!   `getAuth`/`registerProvider` argument overloads split into distinct
//!   methods (`get_provider_auth`, `register_provider` vs
//!   `register_native_provider`); `getApiKeyAndHeaders` returns
//!   [`core::model_registry::ResolvedRequestAuth`] (an enum) instead of a
//!   discriminated union.
//! - **D70 — auth-storage locks and reload coalescing
//!   ([`core::auth_storage`]).** npm `proper-lockfile` becomes std's
//!   advisory file locks (`File::try_lock` — `flock`/`LockFileEx` on
//!   `auth.json` itself), released by the kernel on process death, so the
//!   30 s staleness tracking and `onCompromised` machinery drop; sync
//!   acquisition keeps the 10 × 20 ms retry budget (sleeping, not
//!   busy-waiting) and async acquisition keeps exponential backoff with
//!   jitter to a 30 s deadline, then fails with a timeout message instead
//!   of rethrowing `ELOCKED`. `mkdirSync(mode: 0o700)` becomes
//!   `create_dir_all` (umask defaults); `0o600` still applies to
//!   `auth.json` on creation (unix). **D70a** — backend callbacks are
//!   typed: `withLock`'s `current` is `Option<String>` and the async lock
//!   closure is fallible (`Result<LockResult<T>, ThrownValue>`, so parse
//!   errors reject like JS throws); the backend interface is a
//!   non-object-safe trait and consumers hold concrete backends, so
//!   `AuthStorage.fromStorage` narrows to `from_file_backend` (which, like
//!   upstream's `authPath = undefined`, skips the shared read state).
//!   **D70b** — the coalesced reload runs as a tokio task to completion
//!   behind a watch-completed slot; upstream tracks readers and aborts the
//!   controller when the last one leaves. **D70c** — `parseStorageData`'s
//!   unchecked TS casts become strict `Credential` deserialization;
//!   invalid entries error instead of passing through as inert objects.
//!   **D70d** — the in-memory backend's promise chain serializes through a
//!   tokio mutex raced only on the lock wait (pi-ai D17 convention).
//!   **D70e** — `reload()` stays synchronous (upstream `lockSync`), so it
//!   must not run inside latency-sensitive async contexts.
//! - **D65 — session-cwd ([`core::session_cwd`]).** Upstream's structural
//!   `{ getCwd, getSessionFile }` parameter becomes the `SessionCwdSource`
//!   trait with a blanket impl for `SessionManager`
//!   (inherent accessors share the upstream names); `MissingSessionCwdError`
//!   is a `std::error::Error`. The suite's `createAgentSessionRuntime` case
//!   defers with `AgentSessionRuntime` (module docs list it).
//! - **D66 — session-export ([`core::session_export`]).** Upstream has no
//!   direct suite (consumers are `AgentSession.exportSession` and the
//!   interactive `session-share`/`bug-report` flows), so the module lands
//!   with a shape test pinning the JSONL output; the `TrailingEntries`
//!   callback is `FnOnce(Option<String>, String) -> Vec<JsonValue>` (upstream
//!   returns arbitrary objects).
//! - **D67 — cache-warmer ([`core::cache_warmer`]).** `Pick<ModelRuntime,
//!   "streamSimple">` and `Pick<SessionManager, "appendUsage" | "getBranch">`
//!   become the `CacheWarmStream`/`CacheWarmSessionStore` seams (blanket impl
//!   for `Mutex<SessionManager>`); the stream seam resolves to the final
//!   assistant message directly because upstream only consumes
//!   `streamSimple(...).result()`. **D67a** — the `decide` callback cannot
//!   fail: `ExtensionRunner::emit_cache_warming_decision` already falls back
//!   to pi's decision on handler failure, so no try/catch channel exists.
//!   **D67b** — `setTimeout` + `unref` becomes a tokio sleep task; "now" is
//!   the tokio clock, so `nextWarmAt` is monotonic rather than wall-clock
//!   epoch ms, and the fake-timer suites run under tokio's paused clock,
//!   driving the private `refresh` where upstream reaches into
//!   `warmer.run`/`warmer.refresh`. **D67c** — `transformHeaders` (no Rust
//!   counterpart yet) is carried as static `headers` in the test;
//!   `formatCacheWarmingUsage`'s regex trim is hand-rolled with identical
//!   keep-≥3-decimals semantics; abort-signal identity is asserted through
//!   abort propagation instead of reference equality. The D29
//!   `extensions/types.rs` stand-ins become re-exports.
//! - **D68 — nested-tool-calls ([`core::nested_tool_calls`]).** The
//!   structural `NestedToolCallHost` becomes a trait. **D68a** — `emit` is
//!   synchronous (upstream awaits a promise; the ported event paths are
//!   synchronous per D57a) and the update forwarder keeps the sync
//!   `AgentToolUpdateCallback` convention. **D68b** — the `queueTail`
//!   promise chain becomes a FIFO `tokio::sync::Mutex`; `holdsQueue` scopes
//!   keep nested calls of the queue holder reentrant. **D68c** —
//!   `runToolCall` returns the outcome directly (upstream's promise could
//!   reject; tool failures are `is_error` outcomes here).
//! - **D69 — auth-guidance ([`core::auth_guidance`]).** Upstream has no
//!   direct suite; consumers (`core/sdk.ts`, `core/agent-session.ts`,
//!   `main.ts`, `cli/list-models.ts`) are deferred modules, so the functions
//!   land with a string-shape test and their suites travel with those ports.

//!
//! # Tests
//!
//! Upstream `test/session-manager/*.test.ts` (8 files, 117 cases) are ported
//! in [`core::session_manager::session_manager_tests`]; the ported subset of
//! `test/paths.test.ts` lives in [`utils::paths`]; `test/settings-manager*.test.ts`
//! (incl. the bug-regression and compaction suites) live in
//! [`core::settings_manager::settings_manager_tests`]; upstream
//! `test/compaction.test.ts` + `test/branch-summarization.test.ts` +
//! `test/compaction-{nested-calls,serialization,summary-reasoning}.test.ts`
//! live in [`core::compaction::compaction::compaction_tests`] and the
//! compaction submodule test modules; the standalone
//! `getUsageCostBreakdown` case of `test/agent-session-stats.test.ts` lives
//! in [`core::usage_totals`]; per-utils upstream tests (`ansi-utils`,
//! `frontmatter`, `changelog`, `clipboard*`, `tool-result-images`,
//! `image-process`, `syntax-highlight`, …) are ported inline in their modules.
//! Upstream has no `extensions/types.test.ts`; the
//! [`core::extensions::types`] runtime functions (guards,
//! `isToolCallEventType`, `defineTool`) and the hand-rolled event wire shapes
//! are covered by `types_tests` in that module. Upstream
//! `test/tools.test.ts` (85 cases) is ported 1:1 in
//! `core::tools::tools_tests` (82 cases) plus
//! `tests/tools_image_tests.rs` (3 photon-dependent image cases, split for
//! process-global engine-registry isolation). The model layer ships upstream
//! `test/model-resolver.test.ts` (portable suites) and
//! `test/resolve-config-value.test.ts` 1:1 in
//! [`core::model_resolver::model_resolver_tests`] and
//! [`core::resolve_config_value::resolve_config_value_tests`]; upstream has
//! no direct `model-config.ts` tests (its behavior is exercised through the
//! unported `model-runtime` suites), so [`core::model_config`] carries
//! load/parse/schema/ordering cases mirroring the migration-suite
//! expectations; `provider-composer.ts` has no upstream test file of its own
//! (its coverage arrives with `model-runtime.ts`).
//!
//! Upstream `test/prompt-templates.test.ts` is ported 1:1 in
//! [`core::prompt_templates`]; `test/package-manager.test.ts` (portable,
//! network-free cases over the `CommandRunner` seam) lives in
//! `core::package_manager_tests`; and upstream `test/resource-loader.test.ts`
//! + `test/resource-loader-theme.test.ts` cases that do not execute extension
//! JS or construct an `AgentSession`/`ExtensionRunner` are ported 1:1 in
//! [`core::resource_loader`] `resource_loader_tests` module (deferred cases
//! listed there). Upstream `test/virtual-models.test.ts` portable suites
//! (`getBranchSelection`) plus direct `create_virtual_model` /
//! `with_virtual_models` catalog, filtering and unrouted-stream coverage live
//! in [`core::virtual_models::virtual_models_tests`]; its `ModelRuntime` and
//! `createAgentSession` describe blocks defer with those ports.

pub mod config;
pub mod core;
pub mod theme;
pub mod utils;

pub use config::{
	detect_install_change, detect_install_method, expand_tilde_path, find_node_package_dir, get_agent_dir,
	get_auth_path, get_bin_dir, get_changelog_path, get_custom_themes_dir, get_debug_log_path, get_docs_path,
	get_examples_path, get_export_template_dir, get_interactive_assets_dir, get_models_path, get_package_dir,
	get_package_json_path, get_prompts_dir, get_readme_path, get_sessions_dir, get_settings_path, get_share_viewer_url,
	get_themes_dir, get_tools_dir, get_update_instruction, get_self_update_command, get_self_update_unavailable_instruction,
	InstallChange, InstallMethod, SelfUpdateCommand, SelfUpdateCommandStep, SelfUpdatePackageTarget, APP_NAME,
	APP_TITLE, CONFIG_DIR_NAME, ENV_AGENT_DIR, ENV_SESSION_DIR, PACKAGE_NAME, VERSION,
};
pub use core::defaults::{DEFAULT_THINKING_LEVEL, THINKING_LEVEL_OPTIONS};
pub use core::messages::{
	bash_execution_to_text, create_branch_summary_message, create_compaction_summary_message, create_custom_message,
	custom_message_from_agent, custom_message_to_agent, convert_to_llm, BashExecutionMessage, BranchSummaryMessage,
	BranchSummaryRole, CompactionSummaryMessage, CompactionSummaryRole, CustomMessage, CustomRole,
	BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX, COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX,
};
pub use core::session_manager::{
	assert_valid_session_id, build_context_entries, build_session_context, build_session_projection,
	find_most_recent_session, get_default_session_dir, get_latest_compaction_entry, load_entries_from_file,
	migrate_session_entries, parse_session_entries, session_entry_to_context_messages, BranchSummaryEntry,
	CompactionEntry, ContextEditEntry, ContextEditReplacement, ContextEditableContent, CustomEntry,
	CustomMessageEntry, FileEntry, LabelEntry, Leaf, ModelChangeEntry, NewSessionOptions, ProjectedSessionEntry,
	SessionContext, SessionEntry, SessionEntryBase, SessionHeader, SessionInfo, SessionInfoEntry, SessionManager,
	SessionManagerError, SessionMessageEntry, SessionModelRef, SessionProjection, SessionTreeNode, UsageEntry,
	CURRENT_SESSION_VERSION,
};
pub use core::agent_session::{parse_skill_block, ParsedSkillBlock};
pub use core::compaction::{
	calculate_context_tokens, collect_entries_for_branch_summary, compact, complete_summarization,
	estimate_context_tokens, estimate_projected_context_tokens, estimate_tokens, find_cut_point, find_turn_start_index,
	generate_branch_summary, generate_summary, generate_summary_with_usage, get_last_assistant_usage,
	get_summarization_failure, prepare_branch_entries, prepare_compaction, should_compact, BranchPreparation,
	BranchSummaryDetails, BranchSummaryResult, CollectEntriesResult, CompactionDetails, CompactionPreparation,
	CompactionResult, CompactionSettings, ContextUsageEstimate, CutPointResult, FileLists, FileOperations,
	GenerateBranchSummaryOptions, SummaryWithUsage, DEFAULT_COMPACTION_SETTINGS, SUMMARIZATION_SYSTEM_PROMPT,
};
pub use core::usage_totals::{
	add_usage_to_totals, combine_usage, create_usage_totals, get_usage_cost_breakdown, UsageCostBreakdownEntry,
	UsageTotals,
};
pub use core::settings_manager::{
	parse_http_idle_timeout_ms, resolve_default_tools, CacheWarmingMode, CodemodeMode, DefaultProjectTrust,
	DoubleEscapeAction, FileSettingsStorage, FullscreenExitOutput, InMemorySettingsStorage, MermaidRenderingMode,
	PackageSource, QuietStartup, ScrollViewScrollbar, Settings, SettingsError, SettingsManager,
	SettingsManagerCreateOptions, SettingsScope, SteeringMode, TuiMode, TerminalCapabilityOverrides, TerminalImages,
	WheelScrollLines, DEFAULT_HTTP_IDLE_TIMEOUT_MS, DEFAULT_TOOL_NAMES,
};
pub use core::models_store_wrapper::{FileModelsStore, InMemoryCodingAgentModelsStore};
pub use core::auth_storage::read_stored_credential;
// Upstream `index.ts`: `export { createSyntheticSourceInfo } from "./core/source-info.ts"`.
pub use core::source_info::create_synthetic_source_info;
