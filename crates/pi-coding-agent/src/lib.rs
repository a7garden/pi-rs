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
//! | `core/agent-session*.ts` | [`core::agent_session`] | complete — `AgentSession` class 1:1; the portable `agent-session-*.test.ts` suites live in `agent_session_tests` (D74–D79 docs in [`core::agent_session`]) |
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
//! | `core/extensions/loader.ts` | [`core::extensions::loader`] | complete (D40, D43, D52–D54, D107/D108; the TS/JS module seam bridges to the embedded JS host) |
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
//! | `core/export-html/` | [`core::export_html`] (D80, D81) | complete — templates embedded verbatim; tests ported; `createToolHtmlRenderer` component pipeline deferred (D81) |
//! | `core/auth-guidance.ts` | [`core::auth_guidance`] (D69) | complete — no upstream suite; consumers defer with sdk/AgentSession (D69) |
//! | `core/sdk.ts` | [`core::sdk`] (D73) | complete — `createAgentSession` + re-export surface; upstream `sdk-stream-options`/`sdk-openrouter-attribution` suites run the full prompt loop and defer with AgentSession/interactive |
//! | `core/agent-session-runtime.ts` | [`core::agent_session_runtime`] (D72) | complete — lifecycle/new/fork/import flows; `agent-session-runtime-events` four lifecycle cases ported 1:1 in `runtime_events_tests` (faux-provider `session.prompt()` drive) |
//! | `core/agent-session-services.ts` | [`core::agent_session_services`] (D71) | complete — `createAgentSessionServices`/`FromServices`; flag-diagnostics and registration-drain behavior covered in module tests |
//! | `core/provider-attribution.ts` | [`core::provider_attribution`] | complete — no upstream direct suite (upstream covers it via the deferred `sdk-openrouter-attribution` prompt-loop suite); module tests pin the header tables |
//! | `core/telemetry.ts` | [`core::telemetry`] | complete — no upstream suite |
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
//! | `extensions/` (mcp, tool-search, llama, codemode) | [`extensions`] | see [`extensions`] module docs (D80+; remaining `ui.ts`/`worker.ts` surfaces re-audited D160 — llama helpers ported, mcp UI deferred, codemode worker covered by pi-runtime) |
//! | `src/cli.ts` (`bin: pi` entry) | `src/bin/pi.rs` binary target | complete (D109; dispatcher over print/rpc runners, interactive excluded) |
//! | `core/exec.ts` | [`core::exec`] (D116) | complete — `ExecOptions`/`ExecResult` re-exported by [`core::extensions::types`] |
//! | `core/crash-log.ts` | [`core::crash_log`] (D117) | complete — upstream crash-log.test.ts suite ported |
//! | `core/cache-stats.ts` | [`core::cache_stats`] (D118) | complete — upstream cache-stats.test.ts suite ported; misses keyed by session entry id |
//! | `core/http-dispatcher.ts` | [`core::http_dispatcher`] (D119) | complete portable surface — undici global-dispatcher swap deferred; [`core::settings_manager`] consumes its re-exports |
//! | `core/mcp-servers.ts` | [`core::mcp_servers`] (D120–D124) | complete — typed configs + full-parity registry; supersedes the `extensions/types.rs` stand-ins (migration of folded `extensions/mcp/config.rs` validation pending) |
//! | `core/trust-manager.ts` | [`core::trust_manager`] (D111–D114) | complete — upstream trust-manager.test.ts ported |
//! | `core/project-trust.ts` | [`core::project_trust`] (D115) | complete |
//! | `core/radius.ts` | [`core::radius`] | complete |
//! | `core/experimental.ts` | [`core::experimental`] | complete |
//! | `core/settings-diagnostics.ts` | [`core::settings_diagnostics`] | complete — upstream suite ported; [`cli::main_dispatch`] rewired to it |
//! | `core/bug-report.ts` | [`core::bug_report`] (D125–D127, D129) | complete |
//! | `core/bug-report-upload.ts` | [`core::bug_report_upload`] (D128) | complete — gateway seam re-exported from [`core::radius`] |
//! | `core/keybindings.ts` | — | deferred (D132 — pi-tui `KeybindingsManager`; only its migration subset is ported in [`core::keybindings_migration`]) |
//! | `cli/config-selector.ts` | [`cli::config_selector`] | deferred (D130 — TUI launcher for `pi config`, excluded with interactive mode) |
//! | `cli/session-picker.ts` | [`cli::session_picker`] | deferred (D131 — TUI `--resume` session selector, excluded with interactive mode) |
//! | `client/index.ts` | `crates/pi-client` | 불요 — 1-line `export *` re-export shell, fulfilled by the pi-client crate |
//! | `package-manager-cli.ts` | [`package_manager_cli`] (`install`/`remove`/`uninstall`/`update`/`list`/`config` subcommands, `cleanupManagedInstall`, managed self-update) | complete (D133–D137; `config` selector TUI defers per D130) |
//! | `experimental/process.ts` | [`experimental::process`] | complete (D138) |
//! | `experimental/services/*` (contracts, providers, connection, server, worker) | [`experimental::services`] (S1) | complete (D144–D148; `PresentationUI` implementations and the TUI chat client defer with interactive mode) |
//! | `experimental/plugin.ts` | [`experimental::plugin`] | complete |
//! | `experimental/coordinator.ts` | [`experimental::coordinator`] | complete (D139) |
//! | `experimental/coordinator-entry.ts` | [`experimental::coordinator_entry`] | complete (D140) |
//! | `experimental/session-catalog.ts` | [`experimental::session_catalog`] | complete (D141) |
//! | `experimental/radius-auth.ts` | [`experimental::radius_auth`] | complete (D151; `AuthInput` lives here until `cli/experimental/` ports) |
//! | `experimental/radius-relay.ts` | [`experimental::radius_relay`] | complete (D150 — undici WebSocket becomes a seam + tokio-tungstenite default factory) |
//! | `experimental/plugins/package.ts` | [`experimental::plugins::package`] | complete (D152) |
//! | `experimental/plugins/bundled.ts` | [`experimental::plugins::bundled`] | session facet loader + presentation data codec (D152; `createPresentationFacetLoaders` defers with the headless-client slice) |
//! | `experimental/session-worker.ts` | [`experimental::session_worker`] + [`experimental::lockfile`] | complete (D149/D151; the default `createCodingAgentHarness` factory landed with [`experimental::durable::harness_setup`]) |
//! | `experimental/session-worker-manager.ts` | [`experimental::session_worker_manager`] | complete (D161) |
//! | `experimental/server.ts` | [`experimental::server`] | complete (D162; internal `server`/`session-worker`/`coordinator` roles dispatch in the `pi` bin) |
//! | `experimental/durable/{harness-setup,prompt,runtime,sessions,subagent}.ts` | [`experimental::durable`] | complete (D170–D175; `tui.ts`/`main.ts` defer with interactive mode) |
//! | `experimental/vacation/{harness-setup,vacation,runtime,sessions}.ts` | [`experimental::vacation`] | complete (D170/D174–D176; `tui.ts`/`main.ts` defer with interactive mode) |
//! | `experimental/client-runtime.ts` | [`experimental::client_runtime`] | complete (D181) |
//! | `experimental/client.ts` | [`experimental::client`] | complete (D182) |
//! | `experimental/commands.ts` | [`experimental::commands`] | complete (D177–D179; portable CLI dispatch inlines `cli/experimental/{command,command-options,commands/*}.ts`, the client-tui branch defers with interactive mode) |
//! | `experimental/cli.ts` | [`experimental::cli`] | complete (D180; bin wiring stays with the `pi` bin slice) |
//! | (dependency seam) npm `proper-lockfile` | [`experimental::lockfile`] | complete (D149) |
//!
//! Not ported yet (next passes), in dependency order:
//! `core/agent-session.ts` (`AgentSession` class), `core/agent-session-runtime.ts`,
//! `core/agent-session-services.ts` — blocked on `core/tools/`,
//! `core/extensions/` JS module loading (jiti seam, D40) and the session/agent
//! dispatch integration,
//! `core/export-html/`,
//! and the
//! `modes/interactive` theme layer; then `core/tools/renderers/` (with the
//! theme layer), `core/export-html/`, `core/modes/{print,json-event,rpc,interactive}`
//! (interactive TUI excluded by decision), `core/model-runtime.ts` +
//! `core/model-registry.ts` (the registry is a facade over the runtime and
//! ports with it, D47),
//! `core/system-prompt.ts`, `core/slash-commands.ts`,
//! `extensions/{mcp,tool-search,llama,codemode}/`, `experimental/`,
//! `services/`, `plugins/`, `durable/`, `migrations/`, and TUI-only utils
//! (`modes/interactive/**`). `rpc-entry`, `main.ts`, and the `pi` binary
//! target (`src/cli.ts` upstream) have since landed (D102/D109).
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
//!   open-browser detaches via `setsid`; open-browser unit tests use a
//!   hidden override seam (`set_browser_launcher_override`) so `cargo test`
//!   never spawns a real browser on developer machines.
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
//! - **D43 — loader shell before the JS bridge.** The loader's host-side
//!   shell (runtime construction, `(path, mtime)` cache, error records with
//!   upstream texts) landed as a stand-in while pi-runtime was pending; D107
//!   bridges the module seam itself. Provider/virtual-model registrations
//!   queue exactly like upstream's pre-bind runtime. Deep MCP config
//!   validation lands with the mcp-servers module port (name-shape,
//!   ownership, and namespace-clash checks run in the loader). The
//!   `operations.exec` function check of `user_bash` results is bridge work.
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
//!   effect immediately; the loader API keeps its private queue, so
//!   immediate-effect registration lands with the runner-bind work (D107
//!   scopes).
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
//! - **D71 — agent-session-services ([`core::agent_session_services`]).**
//!   Upstream returns plain object/service handles; Rust returns
//!   [`core::agent_session_services::AgentSessionServices`] with `Arc`-shared
//!   runtime/loader handles (D58/D63 rationale). **D71a** — the optional
//!   create-time `modelRuntimeSignal` maps to
//!   `CreateAgentSessionServicesOptions::model_runtime_signal`. **D71b** —
//!   `applyExtensionFlagValues` writes through the extension runtime state's
//!   `set_flag_value` instead of a mutable map. **D71c** — the pending
//!   registration queues drain through the runtime state's RwLocks;
//!   extension-registered virtual models bridge through
//!   `ModelRuntime::register_extension_virtual_model` (D27/D31 route-shape
//!   split). **D71d** — `createAgentSessionFromServices` delegates to
//!   [`core::sdk::create_agent_session`]; the JS options-spread becomes an
//!   explicit options struct.
//! - **D72 — agent-session-runtime ([`core::agent_session_runtime`]).**
//!   The factory closure is an `Arc`-ed async
//!   `CreateAgentSessionRuntimeFactory`; teardown/apply/rebind order is
//!   upstream 1:1. **D72a** — `SessionImportFileNotFoundError` is a
//!   `std::error::Error` carrying `file_path` (message
//!   `File not found: <path>`). **D72b** — `existsSync`/`copyFileSync(COPYFILE_EXCL)`
//!   become `Path::exists` and create-new copy; the `name-N` suffix scan
//!   keeps upstream's collision avoidance. **D72c** —
//!   `extractUserMessageText` filters text parts from user message content.
//! - **D73 — sdk `createAgentSession` ([`core::sdk`]).** Upstream wires the
//!   agent loop through option-object closures; Rust uses
//!   [`pi_agent::types`] hook types with explicit `Arc` captures.
//!   **D73a** — the cache warmer consumes `SimpleStreamOptions` (D67
//!   shape), so warmed replays do not carry the live request's
//!   `transformHeaders` hook (headers are not part of the prompt-cache
//!   key). **D73b** — `setDefaultStreamFn(streamSimple)` is the pi-agent
//!   default already. **D73c** — the `process.cwd()` fallback becomes
//!   `std::env::current_dir()`. **D73d** — `ExtensionRunnerRef` and
//!   `NoToolsMode` live in [`core::agent_session`]/[`core::sdk`]
//!   (upstream: inline object literal / string union). **D73e** — the
//!   warmer's "is this request still current" check compares
//!   provider/id plus message-prefix equality instead of JS object
//!   identity (agent state may shallow-copy arrays).
//! - **D74 — shared mutable `SessionManager` ([`core::agent_session`]).**
//!   Upstream hands every consumer the same JS object; Rust
//!   `SessionManager` methods take `&mut self`, so the runner and the
//!   extension context hold `Arc<Mutex<SessionManager>>` (upstream
//!   `ReadonlySessionManager` stays a convention, D28) and
//!   `ExtensionRunner::bind_core`/`set_ui_context` take `&self` with
//!   interior-mutable UI state (ui context/mode/`hasUI` behind RwLocks;
//!   the write-only `runtime.actions` copy is dropped — readers use
//!   `BoundCore.actions`).
//! - **D75 — typed session event enum ([`core::agent_session`]).** The TS
//!   `AgentSessionEvent` union (core `AgentEvent` pass-throughs plus
//!   session-specific payloads and the `WithParentToolCallId` conditional)
//!   collapses into [`core::agent_session::AgentSessionEvent`], a Rust
//!   enum; `tool_execution_*` variants carry
//!   `parent_tool_call_id: Option<String>`.
//! - **D76 — message identity by value ([`core::agent_session`]).**
//!   Upstream keys `_entryIdsByMessage` (WeakMap) and
//!   `_boundaryDispatchedMessages` (WeakSet) by object identity.
//!   `AgentMessage` is a value type in Rust, so entries are located by
//!   structural equality (branch scan, then projection index) and
//!   dispatched boundaries are remembered by serialized fingerprint.
//! - **D77 — `message_end` replacement ([`core::agent_session`]).**
//!   Upstream mutates the finalized message object in place so agent
//!   state, listeners, and persistence see the extension replacement.
//!   Rust finds the structurally-equal message in agent state and swaps
//!   it.
//! - **D78 — auto-compaction test seam ([`core::agent_session`]).**
//!   Upstream tests monkey-patch the private `_runAutoCompaction` via
//!   `vi.spyOn`; the port routes the call through an internal override
//!   slot that only the in-module tests set
//!   (`run_auto_compaction_override` + `#[cfg(test)]` call log).
//! - **D79 — bug-report summary defer ([`core::agent_session`]).**

//!   `bug-report.ts` is not ported yet, so
//!   `AgentSession::summarize_for_bug_report` returns a fixed `Err`
//!   describing the missing module. `export-html/` has since landed (see
//!   D80/D81), so `AgentSession::export_to_html` is wired to it. Every other
//!   member is ported.
//! - **D90 — modes runtime-host seam ([`modes`]).** Upstream passes the
//!   concrete `AgentSessionRuntime` (tests cast object-literal fakes); Rust
//!   modes take an explicit [`modes::RuntimeHost`] trait object
//!   ([`modes::AgentSessionRuntimeHandle`] wraps
//!   `Arc<tokio::sync::Mutex<AgentSessionRuntime>>`).
//! - **D91 — stdio seam / output-guard port
//!   ([`core::output_guard`]).** Node's overridable `process.stdout` +
//!   ENOBUFS write-tail have no Rust counterpart: `writeRawStdout` is a
//!   locked blocking write, backpressure wait is a no-op, and modes take a
//!   [`modes::ModeStdio`] seam tests capture (upstream `vi.mock`).
//! - **D92 — bounded mode loops ([`modes::print_mode`],
//!   [`modes::rpc::rpc_mode`]).** Upstream modes end in `process.exit`; the
//!   Rust loops return the exit code and the production wrappers exit, so
//!   tests drive a mode in-process. Signal/stdin wiring rides the seams.
//! - **D93 — RPC wire projection ([`modes::rpc::rpc_types`],
//!   [`modes::json_event`]).** Typed-session events (D75) project through
//!   explicit JSON projections; command unions deserialize via an envelope;
//!   aggregates without `Serialize` ports (`SessionStats`, `SessionTreeNode`,
//!   `CompactionResult`) project manually. Unknown thinking levels / queue
//!   modes are rejected (typed deserialization).
//! - **D93b — mode command actions.** Infallible `SessionSwitchOutcome`
//!   actions log errors to stderr and report `cancelled: false`; the
//!   extension `setup` hook is not forwarded (extension seam takes
//!   `Arc<SessionManager>`, runtime seam `&mut SessionManager`). RPC `bash`
//!   honors `user_bash` `result` overrides; remote `operations` have no
//!   runtime bridge on the erased channel (D43) and run locally.
//! - **D94 — RPC client spawn ([`modes::rpc::rpc_client`]).** Spawns
//!   `node <cliPath>` by default with a configurable program; the
//!   private-`send` test seam is a crate-internal transport override. The
//!   upstream `test/rpc.test.ts` live-CLI suite (18 cases) is ported as
//!   `tests/rpc_live_cli_tests.rs` against the built `pi` binary
//!   (`CARGO_BIN_EXE_pi`; D109/D110).
//! - **D80 — export-html assets and theme export helpers
//!   ([`core::export_html`]).** Template assets (`template.html/css/js`,
//!   `vendor/marked.min.js`, `vendor/highlight.min.js`) are embedded verbatim
//!   with `include_str!` instead of read from a template directory at
//!   runtime; the theme HTML-export helpers upstream defers with the
//!   interactive mode (`getResolvedThemeColors`, `isLightTheme`,
//!   `getThemeExportColors`) land in `core::export_html` because only export
//!   consumes them; `getResolvedThemeColors` emits CSS custom properties in
//!   `BTreeMap` (alphabetical) token order rather than upstream's JS object
//!   insertion order; `state.systemPrompt` is replayed from the snapshot's
//!   transcript (the Rust `AgentStateSnapshot` carries messages, not a
//!   precomputed prompt); and base64 session-data encoding is hand-rolled to
//!   avoid a new dependency.
//! - **D81 — export-html tool-renderer component pipeline defer
//!   ([`core::export_html::tool_renderer`]).** Upstream
//!   `createToolHtmlRenderer` drives erased `ToolDefinition` TUI renderers
//!   and converts their `pi-tui` `Component` ANSI output to HTML; that
//!   renderer layer remains deferred (D27/D32). The pure text helpers
//!   (`isBlankRenderedLine`, `trimRenderedResultLines`) are ported and the
//!   `ToolHtmlRenderer` trait keeps the injection seam; custom-tool export
//!   falls back to structured result rendering until the seam lands.
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
//! + `test/resource-loader-theme.test.ts` cases that do not construct an
//! `AgentSession` are ported 1:1 in [`core::resource_loader`]
//! `resource_loader_tests` — the extension-JS-dependent cases execute real
//! `.ts` fixtures through the embedded host since D107 (only full
//! `AgentSession` constructions defer). Upstream `test/virtual-models.test.ts`
//! portable suites (`getBranchSelection`) plus direct `create_virtual_model` /
//! `with_virtual_models` catalog, filtering and unrouted-stream coverage live
//! in [`core::virtual_models::virtual_models_tests`]; its `ModelRuntime` and
//! `createAgentSession` describe blocks defer with those ports.
//!
//! Upstream `test/args.test.ts` is ported 1:1 in [`cli::args`];
//! `test/initial-message.test.ts` in [`cli::initial_message`];
//! `test/keybindings-migration.test.ts` (the `runMigrations` cases) in
//! [`core::keybindings_migration`]; the parse/validate cases of
//! `test/auth-check.test.ts` and `test/credential-print.test.ts` in
//! [`cli::auth_check`] / [`cli::auth_command`] / [`cli::credential_print`]
//! tests. Deferred: the `KeybindingsManager` in-memory case (manager port),
//! the runtime-backed OAuth-refresh cases (provider refresh injection is a
//! `pi-ai` runtime capability), `startup-session-name.test.ts` (spawns the
//! JS CLI binary), and all interactive/TUI-rendering surfaces.
//!
//! - **D100 — hand-rolled CLI parser ([`cli::args`]).** The upstream parser
//!   is a hand-written loop with idiosyncratic observable behavior (`-p`
//!   consuming one non-flag message, unknown `--flags` captured for
//!   extensions, `--mode`/`--tui-mode` diagnostics instead of hard errors);
//!   the loop is ported 1:1 rather than rebuilt on clap, which would need a
//!   shim re-implementing every quirk. `printHelp` prints uncolored text
//!   (chalk is upstream-only).
//! - **D101 — local fuzzy matcher ([`cli::list_models`]).** Upstream
//!   imports `fuzzyFilter` from pi-tui (excluded); the scoring function is
//!   ported verbatim.
//! - **D102 — mode dispatch seam ([`cli::main_dispatch`]).** Upstream ends
//!   `main()` by constructing `InteractiveMode`/`runPrintMode`/
//!   `runRpcMode`; the port consumes print/rpc runners through the
//!   `ModeDispatcher` trait installed with `set_mode_dispatcher`, and
//!   interactive mode is excluded with the TUI. `prepareInitialMessage`
//!   consumes the first message from the owned `Args` (same observable
//!   shift as upstream).
//! - **D103 — deferred startup infra.** `takeOverStdout`/`restoreStdout`,
//!   http dispatcher/proxy configuration, timing probes, theme init, and
//!   `setCapabilityOverrides` are not invoked from `main()`; project trust
//!   resolves from `--approve`/`--no-approve` (the `ProjectTrustStore`/
//!   trust-manager modules are deferred, so project-local files are trusted
//!   by default); `--resume` requires the excluded TUI session picker and
//!   reports an error; missing-session-cwd always takes the non-interactive
//!   error path; `showDeprecationWarnings` prints without the keypress.
//! - **D104 — package/config/mcp CLI commands defer.** `install`, `remove`,
//!   `uninstall`, `update`, `list`, `config`, and `mcp` print an error
//!   naming the deferred modules.
//! - **D105 — keybindings migration subset
//!   ([`core::keybindings_migration`]).** Only
//!   `migrateKeybindingsConfig` + config ordering are ported; the
//!   `KeybindingsManager` and default-key tables stay behind the pi-tui
//!   exclusion, and output ordering covers the ported key list with unknown
//!   keys sorted alphabetically afterwards.
//! - **D107 — extension module bridge
//!   ([`core::extensions::loader`]).** File-backed `.ts`/`.js` extensions
//!   load through a process-global
//!   [`pi_runtime::extensions::DenoExtensionHost`]: the host imports the
//!   entry (swc transpile), runs its default-exported factory against a JS
//!   `pi` proxy whose registration calls cross the JSON boundary as
//!   `{"__piSlot": n}` slot references and route into the loader API, so
//!   file extensions produce the same [`core::extensions::types::Extension`]
//!   records inline factories do. `ExtensionHost::load` takes a
//!   caller-chosen key (upstream jiti has no equivalent handle), and the new
//!   `ExtensionHost::reload` re-runs a stored factory without re-importing
//!   the module (upstream jiti `moduleCache: false` + path-keyed factory
//!   cache). Cache hits re-run factories through that reload path.
//!   Registration-side surface bridged: `on`, `registerTool`,
//!   `registerCommand`, `registerShortcut`, `registerFlag`,
//!   `registerProvider`/`unregisterProvider` (config overload),
//!   `registerMcpServer`/`unregisterMcpServer`, and `getFlag`; loader-API
//!   panics map to op errors so the JS factory rejects with the upstream
//!   message. Not bridgeable over JSON and erroring the load: renderer
//!   registrations (TUI `Component` pipeline, D56),
//!   `registerVirtualModel` (context-aware route), native
//!   `registerProvider`, `getArgumentCompletions`, `prepareArguments`;
//!   tool `execute` maps the upstream `AgentToolResult` JSON shape with
//!   `None` signal/update/tool-context arguments until the runner binds
//!   them; `ctx.*`/`pi.events`/action methods report the upstream pre-bind
//!   error until runner-bind work routes them to a live context. Command/
//!   shortcut handler failures log like upstream's unhandled rejections.
//!   `Extension.tools`/`commands`/`flags`/`shortcuts` became `IndexMap`s to
//!   preserve upstream `Map` registration order (invocation-name suffixing
//!   and conflict checks are order-observable).
//! - **D108 — `typebox` schema shim ([`core::extensions::loader`]).** The
//!   embedded runtime resolves no npm packages; the bridge registers a
//!   minimal virtual `typebox` module (`Type.Object`/`Array`/`String`/…)
//!   so upstream-style extensions importing TypeBox for parameter schemas
//!   load. `import type` specifiers are erased by transpilation.
//! - **D109 — CLI binary target (`src/bin/pi.rs`).** Upstream's `bin` entry is
//!   `src/cli.ts`: `setupCli(); main(process.argv.slice(2))`. The port adds
//!   the `pi` bin target, which installs a [`cli::main_dispatch::
//!   ModeDispatcher`] over [`modes::print_mode::run_print_mode`]/
//!   [`modes::rpc::run_rpc_mode`] (D102) and calls
//!   [`cli::setup::setup_cli`] + `main`. The dispatcher receives the
//!   `AgentSessionRuntime` by value so the production
//!   [`modes::AgentSessionRuntimeHandle`] can own it; interactive mode is
//!   never dispatched (excluded with the TUI). Upstream's RPC-mode
//!   background model-catalog refresh (`modelRuntime.refresh` with a 15s
//!   abort timeout) is not re-fired here — the ported `ModelRuntime`
//!   refreshes during creation (D63) — and the timing probes remain
//!   omitted with D103. The [`RpcClient`] spawns a custom `program`
//!   without a script argument when no `cli_path` is given (the live-CLI
//!   suite runs the built binary directly).
//! - **D110 — live-CLI RPC suite port ([`tests/rpc_live_cli_tests.rs`]).**
//!   Upstream `test/rpc.test.ts` (18 cases; the D94 note said 17 — the
//!   upstream checkout has grown one since) runs against the real CLI.
//!   Ported 1:1 against `CARGO_BIN_EXE_pi`: same prompts, commands,
//!   assertions, and per-case vitest timeouts (as a body timeout). The
//!   `describe.skipIf` credential gate becomes a runtime early-return, the
//!   per-test `cwd` is a fresh empty temp dir instead of the package dir
//!   (sessions still live under `PI_CODING_AGENT_DIR`), and cases within
//!   the file run sequentially behind a shared permit (vitest's in-file
//!   ordering).
//! - **D111 — trust-manager lockfile seam ([`core::trust_manager`]).**
//!   `proper-lockfile` becomes a create-exclusive `<trust.json>.lock`
//!   retry loop (10 attempts, 20ms sleeps; release unlinks; ELOCKED-parity
//!   "Lock file is already being held" message; stale locks never stolen).
//!   Store methods return `Result` instead of throwing; trust.json key
//!   order matches JS `Array.prototype.sort` (UTF-16 code units); `HOME`
//!   falls back to the [`utils::paths`] home seam; the project-trust emit
//!   seam pairs upstream `extensionsResult` with the full `ExtensionContext`
//!   the Rust runner requires (D115).
//! - **D112–D114 — trust-manager store details.** D112: parse failures keep
//!   the `Failed to read trust store {path}: ` prefix with a serde_json
//!   body. D113: key sort is UTF-16 code-unit order. D114:
//!   `has_trust_requiring_project_resources` has a `_with_home` test
//!   variant; canonicalization stays inside.
//! - **D115 — project-trust seams ([`core::project_trust`]).** Extension
//!   registration carries the full context; store failures surface as
//!   `Err`; `ui.select` follows the non-interactive `NoopExtensionUi`
//!   gating.
//! - **D116 — exec ([`core::exec`]).** The upstream 5s SIGKILL escalation
//!   is dead code in Node (`subprocess.killed` is already true right after
//!   the SIGTERM `kill()`); the port escalates for real. Non-unix
//!   SIGTERM/SIGKILL collapse to `start_kill` (Node parity). Spawn failures
//!   resolve with the catch-path `{ code: 1, killed: false }`; abort/timeout
//!   bookkeeping uses `tokio::select!` over the signal's aborted future
//!   instead of `addEventListener`.
//! - **D117 — crash-log ([`core::crash_log`]).** `recordCrash`'s
//!   `error: unknown` splits into explicit message/stack args (Rust errors
//!   carry no stack); the lenient read defaults absent optional fields;
//!   path/now become explicit args (`crash_log_default_path` + caller-
//!   supplied now).
//! - **D118 — cache-stats ([`core::cache_stats`]).** Misses are keyed by
//!   session entry id (Rust messages carry no stable identity for the JS
//!   by-reference map); `ModelPriceSource` is a minimal $/Mtok trait
//!   implemented for `ModelRuntime`; unparseable usage-entry timestamps
//!   read as epoch 0 where upstream's `Date.parse` NaN would propagate.
//! - **D119 — http-dispatcher ([`core::http_dispatcher`]).** The undici
//!   `EnvHttpProxyAgent` global-dispatcher swap, fetch reinstall, and
//!   autoSelectFamily attempt timeout have no reqwest equivalent;
//!   `configure_http_dispatcher` validates/normalizes with the exact
//!   upstream error string and proxying flows through `HTTP_PROXY`/
//!   `HTTPS_PROXY` env (set by `apply_http_proxy_settings`) that reqwest
//!   honors at client build.
//! - **D120–D124 — mcp-servers ([`core::mcp_servers`]).** D120: validation
//!   runs from `serde_json::Value` for exact alias/error parity, then
//!   produces typed structs (unknown fields are dropped, unlike the
//!   upstream spread copy). D121: tool patterns match without a regex
//!   engine (`*` → `.*`; wildcard also matches line terminators,
//!   unobservable for real tool names). D122: override lookup is
//!   own-property only (JS also sees `Object.prototype` names). D123:
//!   timeout keeps `serde_json::Number` so integer seconds keep their wire
//!   form. D124: `structuredClone` → `Clone` in `McpServerRegistry::list`;
//!   the full-parity registry supersedes the earlier
//!   `extensions/types.rs` stand-ins.
//! - **D125–D127 — bug-report ([`core::bug_report`]).** D125:
//!   `collectEnvironment.runtime` reports `"rust"`; platform/arch reuse the
//!   Node naming from [`utils::pi_user_agent`]. D126: `os.release()`/
//!   `os.version()` map to `uname(2)`, empty when unavailable. D127:
//!   `redactUrl` parses with the `url` crate — unchanged URLs return the
//!   input byte-for-byte; re-serialization of changed URLs can differ in
//!   percent-encoding corners.
//! - **D128 — bug-report-upload ([`core::bug_report_upload`]).** Status-text
//!   fallback uses `canonical_reason`, so unknown codes fall to the numeric
//!   status where JS echoes the wire reason. The gateway seam is
//!   re-exported from [`core::radius`], matching the upstream import.
//! - **D129 — bug-report crash capture.** Crash records serialize to JSON
//!   and drop `notified` (upstream rest-spread).
//! - **D130 — `cli/config-selector.ts` deferred.** `selectConfig` is pure
//!   TUI wiring (ProcessTerminal + TuiMainScreen + interactive
//!   ConfigSelectorComponent + theme init/stop); no portable data/logic
//!   exists. Deferred with the excluded interactive mode
//!   ([`cli::config_selector`]).
//! - **D131 — `cli/session-picker.ts` deferred.** `selectSession` is pure
//!   TUI wiring (startup TUI + pi-tui keybindings + interactive
//!   SessionSelectorComponent); `SessionInfo`/`SessionListProgress` already
//!   live in [`core::session_manager`]. Deferred with interactive mode
//!   ([`cli::session_picker`]).
//! - **D132 — `core/keybindings.ts` deferred.** pi-tui-coupled
//!   `KeybindingsManager`; only its migration subset is ported
//!   ([`core::keybindings_migration`]).
//! - **D133–D137 — package-manager CLI ([`package_manager_cli`]).** D133:
//!   chalk styling dropped (plain text, cf. D81). D134: project trust is an
//!   explicit `--approve`/`--no-approve` override only (default trusted per
//!   D103). D135: self-update note prints as trimmed raw text (pi-tui
//!   Markdown excluded with the TUI). D136: managed-install locking uses
//!   advisory file locks on `<managedRoot>/update.lock` (ELOCKED-parity
//!   message); "Location of pi executable" prints argv[0]. D137: `pi
//!   config` runs the upstream parse/trust/settings flow, then reports the
//!   D130 deferral and exits 1 (upstream exits 0 after the selector).
//! - **D138 — experimental/process ([`experimental::process`]).** Upstream
//!   spawns `process.execPath` with a `--import` source resolver (source
//!   runs) or a bundled `dist/bundle/{coordinator,cli}.js` entry
//!   (`isBundledNode`) and locates entries relative to `import.meta.url`;
//!   the Rust port always spawns the running executable
//!   (`std::env::current_exe`), with `InternalProcessSpawnOptions::
//!   entry_override` as an explicit program replacement, so `entryUrl`,
//!   `isBunBinary`/`isBundledNode` and `isDirectInternalProcessEntry` are
//!   not ported. Node `detached` + `unref()` becomes a fresh process group
//!   (`process_group(0)`) with `kill_on_drop(false)` and null stdio;
//!   `terminateInternalProcess` escalates for real (tokio `start_kill` is
//!   already SIGKILL). Control-line encoding/decoding is
//!   `serde_json::Value`-based with the same 128 MiB cap and error strings.
//! - **D139 — experimental/coordinator ([`experimental::coordinator`]).**
//!   Node sockets become [`tokio::net::UnixStream`]; the typebox wire
//!   schema becomes the serde [`experimental::coordinator::
//!   CoordinatorMessage`] enum (identical wire names) while the server side
//!   keeps the loose `type`-string dispatch over `serde_json::Value` (the
//!   upstream `ControlMessage`). The module-global coordinator singleton +
//!   `process.once(SIGINT/SIGTERM)` hooks become a function-local
//!   [`experimental::coordinator::CoordinatorServer`] racing tokio signal
//!   handlers with the same socket cleanup; a second concurrent
//!   `run_coordinator_process` in one process is still rejected.
//!   `emptyTimer.unref()` becomes an abortable tokio sleep with the same
//!   30 s/250 ms grace windows and emptiness predicate. Upstream's
//!   uncaught-exception crash on a control-handler error surfaces as an
//!   `Err` from `run_coordinator_process` (the entrypoint exits non-zero,
//!   mirroring the crash) with the identical error string; reader framing
//!   errors destroy only the offending connection. `endpoint` may be a
//!   Unix-socket path or `host:port`; public connections pump raw bytes
//!   with `copy_bidirectional`. Client `peerIds`/listeners return owned
//!   snapshots/guards instead of live JS references.
//! - **D140 — experimental/coordinator-entry
//!   ([`experimental::coordinator_entry`]).** The `#!/usr/bin/env node`
//!   entry script becomes [`experimental::coordinator_entry::
//!   run_coordinator_entry`]; `console.error` + `process.exit(1)` becomes a
//!   returned `Err` the host binary surfaces (same contract as the D102
//!   mode-dispatcher seam). The `isDirectInternalProcessEntry` file probe
//!   is a JS-runtime seam (D138).
//! - **D141 — experimental/session-catalog
//!   ([`experimental::session_catalog`]).** `crypto.randomUUID` becomes a
//!   getrandom-backed v4 id; `node:fs/promises` becomes `tokio::fs`;
//!   `Date.now()` becomes [`utils::time::now_ms`]. The tab-indented
//!   `JSON.stringify(meta, null, "\t")` metadata file is reproduced
//!   byte-for-byte (createdAt first, trailing newline). Read-side
//!   validation accepts any JSON object with numeric `createdAt` and
//!   string `cwd`, dropping unknown extra fields exactly like the upstream
//!   destructuring read; JSON/IO failures on a session's `meta.json` read
//!   as "no session" (upstream catches the same throws). Thrown errors
//!   surface as `Err(String)` with the upstream messages; `rm(force:
//!   true)` maps to ENOENT-swallowing deletes.
//! - **D144 — experimental/services structural tokens
//!   ([`experimental::services`]).** Upstream services are structural objects
//!   behind `defineService<T>`; Rust tokens carry a marker (remote-only
//!   services), a concrete implementation type (local services), or a
//!   trait-object handle (`PresentationUI`). Remote facets expose upstream
//!   object methods as JSON members through `RemoteServiceObject::builder`;
//!   state members use chord's untyped JSON replicated states with typed DTOs
//!   at the seam. The `PresentationUI` implementations (alternate-screen
//!   renderer integration, selection dialogs) defer with interactive mode.
//! - **D145 — services busy detection.** pi-durable Rust errors arrive
//!   stringified, so the typed `ConversationBusy → { code: "busy" }` mapping
//!   checks the stable durable display text (`… is busy`).
//! - **D146 — services timestamps.** `new Date().toISOString()` in
//!   `connection.ts` is formatted by a local RFC 3339 writer (no chrono
//!   dependency).
//! - **D147 — services provider sharing.** `FacetHost::provider_arc`
//!   (additive pi-chord default method) shares the assembled provider as an
//!   `Arc` for the worker's per-scope endpoints.
//! - **D148 — services requirement cloning.** chord's `LocalRequirement` is
//!   not `Clone`; the `/reload` command lends its SessionPlugins requirement
//!   out of a mutex cell for the synchronous handle acquisition.
//! - **D149 — proper-lockfile becomes std advisory file locks
//!   ([`experimental::lockfile`]).** Upstream's `proper-lockfile` locks map
//!   to `File::try_lock` polls with the same retry/wait budgets (cf. D136).
//!   Ownership is kernel-managed, so the mtime-based `stale` option has no
//!   counterpart; a dead holder releases automatically. The Session worker
//!   locks `<session path>.lock` instead of the directory path.
//! - **D150 — Radius relay WebSocket seam
//!   ([`experimental::radius_relay`]).** Upstream drives the npm `undici`
//!   `WebSocket` (event listeners, `binaryType`, `bufferedAmount`). The port
//!   defines a single-listener [`RadiusRelayWebSocket`] trait; the default
//!   factory is tokio-tungstenite with `Authorization` + subprotocol
//!   negotiation verified after the handshake. The `OrderedWebSocketWriter`
//!   drain loop keeps its 5 ms poll over the pending-bytes threshold, and
//!   local protocol/transport errors still close with application codes
//!   4000/4001.
//! - **D151 — session-worker seams ([`experimental::session_worker`],
//!   [`experimental::radius_auth`]).** typebox schemas become serde types
//!   with explicit validators; `WorkerOperationScope` is a serde mirror of
//!   the services' `WorkerServiceScope` with conversions at the seam; the
//!   chord/persistent abort-signal split bridges at `get_auth`. `AuthInput`
//!   (upstream `cli/experimental/command-options.ts`) is defined in
//!   `radius_auth` until the CLI experimental slice ports. The default
//!   `createCodingAgentHarness` factory landed with the durable slice
//!   ([`experimental::session_worker::create_coding_agent_harness`] over
//!   `experimental::durable::harness_setup`); the full worker machinery
//!   still runs against the `CreateSessionWorkerHarness` seam.
//! - **D152 — plugins slice ([`experimental::plugins`]).** `import.meta.url`
//!   external resolution becomes the pi-runtime externals registry (the
//!   plugin API specifier resolves to a JSON encoding of the barrel's
//!   service tokens; identity travels through chord service ids). Facet
//!   validation labels are read from the manifest at load time.
//!   `createPresentationFacetLoaders` (server-sent artifacts → client
//!   loaders) defers with the headless-client slice; the session facet
//!   loader and the presentation data codec are complete.
//! - **D161 — session-worker-manager ([`experimental::session_worker_manager`]).**
//!   Upstream `Promise`s become first-write-wins [`PromiseSlot`] cells;
//!   launch/demand errors travel as `Arc<dyn Error + Send + Sync>` while
//!   worker operation errors stay structured so routed `ServerError` codes
//!   survive. Unref'd timers become spawned tokio tasks (or
//!   `tokio::time::timeout` for demand settlement) whose late fires re-check
//!   the pending maps; the per-subscription `deliveryTail` chain becomes a
//!   tokio mutex. `workerPids` is a shared snapshot handle
//!   ([`WorkerPids`]) consumed by the running server, `process.kill`
//!   becomes `libc::kill` (ESRCH swallowed), and attached-demand timeout
//!   compensation runs inline through an acyclic
//!   apply/reconcile split instead of a self-spawning timer task.
//! - **D162 — experimental server entry ([`experimental::server`]).**
//!   `proper-lockfile` locks map to `FileLock` polls (D149) with the
//!   `default-server-id` read/`wx`-create/re-read ladder; `ServerLifetime`'s
//!   timers become one notify-driven retirement task with a
//!   generation-guarded startup grace; `runtime.closed` is a shared
//!   settled-once slot and `close()` memoizes onto it. `connect`'s Node
//!   error-code chain walk maps to `Error::source` downcasts
//!   (`DisconnectedError`, `version` server error, or the
//!   ENOENT/ECONNREFUSED/ECONNRESET/EPIPE/ETIMEDOUT io kinds read as "no
//!   server yet"). The routed `ServerHost` trait hands metadata opaquely, so
//!   `open_session` re-resolves the catalog metadata by ID. The internal
//!   `server` role entry (`runServerProcess`) dispatches from the `pi` bin,
//!   matching the upstream module-level `isDirectInternalProcessEntry`
//!   blocks.
//! - **D170 — shared `selectSession` ([`experimental::durable::sessions`]).**
//!   Upstream `durable/sessions.ts` and `vacation/sessions.ts` are clones
//!   differing only in the session-space directory and the "no session"
//!   error text; the port implements the body once
//!   (`select_session_in(space, ...)`) and both modules keep their upstream
//!   surface. `proper-lockfile` maps to `FileLock` (D149) locking
//!   `<directory>.lock` with the upstream `12 × 1000ms` retry budget;
//!   `crypto` hashing/UUID map to `sha2` and the crate `randomUUID` seam;
//!   the `^\d{13}-[0-9a-f-]{36}$` continue-filter is a validated shape
//!   predicate.
//! - **D171 — per-request prompt build cache
//!   ([`experimental::durable::prompt`]).** Upstream caches the built
//!   sections per `PromptInput` in a `WeakMap`; `PromptInput` has no weakly
//!   referenceable identity here, so each section render rebuilds the
//!   section map from the per-directory resource cache (context files and
//!   skills still load once per directory, like upstream). Tool prompt
//!   contributions read the crate's `*_TOOL_SYSTEM_PROMPT_SNIPPET`/
//!   `*_GUIDELINES` constants.
//! - **D172 — erased subagent tool ([`experimental::durable::subagent`]).**
//!   typebox parameter schemas become JSON-Schema `serde_json` values on
//!   `pi_ai::Tool`, and tool failures take the crate's erased error-result
//!   convention (`is_error` + a `tool_error` diagnostic), like the
//!   pi-durable coding tools. The upstream `Subagent` extension constant
//!   becomes `subagent_extension()`; the child's self-removal configure
//!   matches by extension name.
//! - **D173 — live harness settings
//!   ([`experimental::durable::harness_setup`]).** Upstream
//!   `createHarnessSettings` returns a getters object read at every use;
//!   the erased surface's live source is `HarnessOptions.settings_provider`
//!   (`HarnessSettingsProvider`). `stream.timeoutMs` keeps the upstream
//!   `idle === 0 ? 2147483647 : idle` mapping. `configureHarnessHttp`
//!   reuses `core::http_dispatcher` (D119), surfacing the idle-timeout
//!   getter's parse failure as its error. `findInitialAgentModel`'s CLI
//!   branch resolves through `resolve_cli_model` so the failure returns as
//!   the function's error (the crate's `find_initial_model` prints and
//!   exits there); `AgentThinkingLevel` converts to pi-ai's
//!   `ModelThinkingLevel` at the seam. `NodeExecutionEnv` memoizes per cwd
//!   behind the erased async `EnvBuilder`.
//! - **D174 — shared cluster runtime
//!   ([`experimental::durable::runtime`], [`experimental::vacation::runtime`]).**
//!   The twin upstream `openDurable` bodies differ only in registry
//!   construction, the root agent shape, and the execution-environment
//!   wiring; the port hosts the machinery once
//!   (`durable::runtime::open_harness_view`) and both modules keep their
//!   upstream surface (vacation re-exports the shared view/controller
//!   types). `DurableView.conversation`/`tasks` carry erased `serde_json`
//!   forms of `ConversationView`/`TaskGraph`; the exact-frame watches are
//!   `CommittedWatch` values (`subscribe` → `start(listener)`, `dispose` →
//!   `stop_detached`, `.value` → `value()`); `BACKGROUND_CONTEXT` maps to
//!   `TODO_CONTEXT`; the command promise chain becomes a tokio mutex;
//!   `setImmediate` notification coalescing becomes a spawned burst gate;
//!   the `closing ??=` memoization becomes a shared future.
//! - **D175 — shared harness assembly
//!   ([`experimental::vacation::harness_setup`]).** Upstream
//!   `configureHarnessHttp`/`createHarnessSettings`/`InitialModel`/
//!   `findInitialAgentModel` are byte-identical across the clusters; the
//!   port implements them once in `durable::harness_setup` and the vacation
//!   module re-exports them (only `createVacationRegistry` is local).
//! - **D176 — erased vacation task
//!   ([`experimental::vacation::vacation`]).** typebox schemas become
//!   JSON-Schema values and the `vacation.research` task's phases become
//!   erased `PhaseHandler`s over `TaskRecord`/`TaskRuntime`; `setTimeout`
//!   sleeps become `tokio::time::sleep` raced against the context's abort
//!   signal; tool failures take the D172 error-result convention.
//! - **D177 — portable experimental CLI dispatch
//!   ([`experimental::commands`]).** The upstream `cli/experimental/` tree is
//!   not ported as a separate module: the generic `Command` builder
//!   (`command.ts`) collapses into focused parsers that reproduce its
//!   observable behavior (`--` passthrough, `=` values, flag/value rules,
//!   option-errors-before-builder-errors, unknown-option cutoff), and
//!   `ClientCommand`/`ServerCommand` (`commands/{client,server}.ts`) plus
//!   `TransportAddress` (`command-options.ts`) live in the module. `AuthInput`
//!   stays in [`experimental::radius_auth`] (D151). WHATWG `--connect`
//!   validation maps onto the `url` crate plus a strict
//!   `decodeURIComponent`; `continue` renames to `continue_session`.
//! - **D178 — client-tui branch excluded
//!   ([`experimental::commands::run_client_command`]).** The
//!   `runClientTui` dispatch and the TTY sniffing that selects it defer with
//!   the interactive mode (D152); every portable invocation runs
//!   [`experimental::client::run_client`].
//! - **D179 — uncolored experimental dispatch errors
//!   ([`experimental::commands`]).** `chalk.red` error lines print plain
//!   (`Error: …`), matching the uncolored-output precedent (D133), and
//!   `process.exitCode` becomes the [`experimental::commands::
//!   ExperimentalDispatch`] result the entry applies.
//! - **D180 — development entry seam ([`experimental::cli`]).** The
//!   `#!/usr/bin/env node` script becomes a library entry returning the exit
//!   code; the client command still short-circuits via `process::exit`, and
//!   the `main(args)` fallback maps to
//!   [`cli::main_dispatch::main`]. Wiring the entry into the `pi` bin stays
//!   with the bin slice.
//! - **D181 — client runtime facades ([`experimental::client_runtime`]).**
//!   `ActivatedClientRuntimeServer extends ClientRuntimeServer` becomes a
//!   flat struct; `services.use(Service)` lookups become typed facades over
//!   chord's `ServiceClientHandle`; opened namespaces stay alive by holding
//!   their `Box<dyn RemoteServices>` and dispose by dropping. The
//!   `directory.state.value!` hydration assertion becomes an error, the
//!   `ServerError code "version"` fallback downcasts through Rust error
//!   types, and the infallible reconnector/client dispose steps contribute
//!   nothing to the upstream `Promise.allSettled` failure aggregation.
//! - **D182 — non-interactive client ([`experimental::client`]).**
//!   `localeCompare` ordering becomes UTF-8 byte ordering; the rejected
//!   prompt error carries a fallback message when the server omits
//!   `error.message`; the JS `finally { await runtime.dispose() }` masking
//!   semantics (dispose failure replaces the body outcome) are preserved.

pub mod cli;
pub mod config;
pub mod experimental;
pub mod core;
pub mod extensions;
pub mod migrations;
pub mod modes;
pub mod package_manager_cli;
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
