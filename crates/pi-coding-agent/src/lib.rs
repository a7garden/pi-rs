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
//! | `config.ts` | [`config`] (path/config subset: `APP_NAME`, `CONFIG_DIR_NAME`, `VERSION`, env names, `getAgentDir`, `getSessionsDir`, `expandTildePath`) | subset |
//! | `utils/paths.ts` | [`utils::paths`] (`normalizePath`, `resolvePath` + options) | subset |
//! | `utils/time` (JS `Date` seam — no upstream file) | [`utils::time`] | complete (seam) |
//! | `core/messages.ts` | [`core::messages`] | complete |
//! | `core/session-manager.ts` | [`core::session_manager`] | complete |
//!
//! Not ported yet (next passes), in dependency order:
//! `utils/*` (~33 remaining files), `core/defaults.ts`, `core/settings-manager.ts`,
//! `core/messages` consumers — `core/compaction/`, `core/tools/`,
//! `core/extensions/`, `core/export-html/`, `core/modes/{print,json-event,rpc,interactive}`
//! (interactive TUI excluded by decision), `core/agent-session*.ts`,
//! `core/model-*`, `core/system-prompt.ts`, `core/slash-commands.ts`,
//! `extensions/{mcp,tool-search,llama,codemode}/`, `experimental/`,
//! `services/`, `plugins/`, `durable/`, `migrations/`, `rpc-entry`,
//! `main.ts`, and the rest of `config.ts` (install-method detection,
//! self-update, package-asset paths).
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
//!
//! # Tests
//!
//! Upstream `test/session-manager/*.test.ts` (8 files, 117 cases) are ported
//! in [`core::session_manager::session_manager_tests`]; the ported subset of
//! `test/paths.test.ts` lives in [`utils::paths`].

pub mod config;
pub mod core;
pub mod utils;

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
pub use config::{expand_tilde_path, get_agent_dir, get_sessions_dir, CONFIG_DIR_NAME, VERSION};
