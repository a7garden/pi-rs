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
//! | `core/extensions/types.ts` | [`core::extensions::types`] | complete — type foundation only (D27–D31); runner/loader and `core/tools/*` port against it |
//! | `core/tools/` (all 16 files) | [`core::tools`] | complete (D32–D35, D37–D39); `renderers/*.ts` + theme-coupled `render-utils` helpers deferred with the excluded interactive mode (D32) |
//! | `core/bash-executor.ts` | [`core::bash_executor`] | complete (D35) |
//! | `test/tools.test.ts` | `core::tools::tools_tests` + `tests/tools_image_tests.rs` (85 cases, 1:1; image cases split out for photon-seam process isolation, D36) | complete |
//! | `core/tools/renderers/` | — | deferred (TUI renderer layer; D32) |
//!
//! Not ported yet (next passes), in dependency order:
//! `core/agent-session.ts` (`AgentSession` class), `core/agent-session-runtime.ts`,
//! `core/agent-session-services.ts` — blocked on `core/tools/`,
//! `core/extensions/` (loader/runner/API on top of the ported types),
//! `core/model-registry.ts`, `core/model-runtime.ts`,
//! `core/resource-loader.ts`, `core/sdk.ts`, `core/session-cwd.ts`,
//! `core/cache-warmer.ts`, `core/nested-tool-calls.ts`,
//! `core/bash-executor.ts`, `core/bug-report.ts`, `core/export-html/`,
//! `core/system-prompt.ts`, `core/slash-commands.ts`,
//! `core/virtual-models.ts`, `core/prompt-templates.ts`,
//! `core/session-export.ts`, `core/auth-guidance.ts` and the
//! `modes/interactive` theme layer; then `core/tools/renderers/` (with the
//! theme layer), `core/export-html/`, `core/modes/{print,json-event,rpc,interactive}`
//! (interactive TUI excluded by decision), `core/model-*`,
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
//! process-global engine-registry isolation).

pub mod config;
pub mod core;
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
