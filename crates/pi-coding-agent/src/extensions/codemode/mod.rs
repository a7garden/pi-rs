//! Port of `packages/coding-agent/src/extensions/codemode/` — the `codemode`
//! tool: the model writes JavaScript that calls other tools from a QuickJS
//! sandbox, with nested calls running through the session's tool pipeline.
//!
//! # Files
//!
//! | Upstream | Port | Status |
//! |---|---|---|
//! | `index.ts` (46) | [`index`] | ported — native factory over the erased `ExtensionApi`; JS module evaluation stays behind the D40 [`crate::core::extensions::loader`] seam (D90) |
//! | `tool.ts` (404) | [`tool`] | ported — schema, description catalog, loadout presentation, tool/`AgentTool` factories |
//! | `execute.ts` (672) | [`execute`] | ported — orchestration/validation surface; the QuickJS worker itself stays behind pi-codemode's [`pi_codemode::runtime::host::CodeModeWorkerFactory`]/[`pi_codemode::wasm::WasmLoader`] traits, which pi-runtime implements (D91) |
//! | `renderer.ts` (150) | [`renderer`] | theme-free string surface ported; TUI component assembly deferred with the theme layer (D92) |
//! | `worker.ts` (6) | — | folded — a 2-line re-export of `pi-codemode/worker` as a bundled worker entrypoint; the Rust counterpart is pi-codemode's worker protocol and the `get_codemode_worker_specifier`/`get_quick_js_wasm_path` config already ported in [`crate::config`] (D91). No Rust file needed. |
//! | `execute.lazy.ts` (2) | — | folded — a dynamic `import("./execute.ts")` so the sandbox loads on the first script; Rust modules load statically, [`execute::execute_codemode`] is called directly (D91). |
//!
//! # Deviations (D90–D94)
//!
//! - **D90 — entry/config surface.** `createCodemodeExtension` becomes a
//!   native [`index::create_codemode_extension`] factory;
//!   `readMode`/`readInlineBudget` parse the `codemode` settings JSON
//!   directly (the typed accessors live on [`crate::core::settings_manager::SettingsManager`],
//!   while `ExtensionApi::get_settings` returns the erased [`crate::core::settings_manager::Settings`]);
//!   the `CODEMODE_DOCS_PATH` constant becomes [`tool::codemode_docs_path`];
//!   the default export (CLI built-in registration) is D40 loader work.
//!   `isCodemodeTool` compares schemas by value (upstream: by reference
//!   identity); `localeCompare` becomes Unicode scalar ordering; fractional
//!   `inlineBudget` floors to whole tokens (identical selection, because
//!   section costs are integers).
//! - **D91 — worker/wasm/lazy-load.** `worker.ts` and `execute.lazy.ts` are
//!   folded (see the table). The sandbox is built with pi-codemode's
//!   unavailable-by-default wasm loader and worker factory — the exact
//!   upstream missing-worker/missing-wasm sandbox-error paths — and
//!   [`tool::CodemodeToolOptions`] carries `wasm_loader`/`worker_factory`
//!   injection points for pi-runtime (upstream wires
//!   `loadQuickJSWasm(getQuickJSWasmPath())` and `getCodemodeWorkerSpecifier()`,
//!   both already ported in [`crate::config`] as D18).
//! - **D92 — renderer/TUI.** pi-tui components, the interactive theme,
//!   `VisualLinePreview` wrapping, `highlightCode`, and `keyHint` styling are
//!   excluded by the crate-wide TUI port decision (`core/tools/render_utils.rs`
//!   D32 precedent); the definition's `render_call`/`render_result` stay
//!   `None` like every built-in tool. The theme-free string surface
//!   (durations, costs with `toPrecision(2)` semantics, status glyphs, call
//!   rows, header detection, spill-file note) is ported and tested.
//! - **D93 — host seams in `execute`.** Signals cross three types (tool-call
//!   signal ↔ sandbox signal ↔ session/pi-ai signals) through small
//!   forwarding tasks, matching the crate's abort-port conventions (D9);
//!   `CodemodeModelRuntime` is the real [`crate::core::model_registry::ModelRegistry`];
//!   base64 decoding uses the crate's `image_convert` helper; sandbox
//!   construction errors surface as `Err` (upstream: a thrown error → error
//!   tool result).
//! - **D94 — JS semantics carried into Rust.** Script values arrive as
//!   `serde_json::Value` (undefined crosses the sandbox as JSON `null`, like
//!   upstream's spread-argument note); text truncation slices by Unicode
//!   scalar values instead of UTF-16 code units (identical for ASCII and
//!   every BMP content pi renders); `describeValue` reports JS `typeof`
//!   shapes over JSON kinds.
//!
//! # Tests
//!
//! | Upstream | Port | Status |
//! |---|---|---|
//! | `test/codemode-renderer.test.ts` (109) | [`renderer`]'s `renderer_tests` | ported at the string level (D92); the fourth case's visual-wrap line-count assertions defer with pi-tui — its portable half (the collapsed `Full output:` footnote) is asserted |
//! | `test/codemode-worker-config.test.ts` (21) | `worker_config_tests` below | ported 1:1 against [`crate::config::resolve_codemode_worker_specifier`] (also covered where it lives, `config.rs`'s `codemode_worker_specifier_by_runtime`) |
//! | `test/suite/agent-session-codemode.test.ts` | — | deferred: AgentSession integration (end-to-end scripts, harness, real nested `executeTool`); needs the AgentSession test harness and a live sandbox worker — revisit with pi-runtime's QuickJS integration |
//!
//! The sandbox host behaviors themselves (timeouts, aborts, store writes,
//! protocol) are pinned by pi-codemode's `tests/sandbox.rs` worker fake; the
//! JS-semantics cases there defer to pi-runtime for the same reason.

pub mod execute;
pub mod index;
pub mod renderer;
pub mod tool;

// Port of `packages/coding-agent/test/codemode-worker-config.test.ts` —
// regression test for upstream #10204. The upstream home of
// `resolveCodemodeWorkerSpecifier` is `src/config.ts`, whose port
// ([`crate::config`]) already carries these assertions; they are repeated
// here so the codemode slice keeps its 1:1 test port.
#[cfg(test)]
mod worker_config_tests {
	use crate::config::resolve_codemode_worker_specifier;

	// Regression test for #10204.
	#[test]
	fn uses_a_relative_source_entrypoint_in_bun_binaries() {
		assert_eq!(
			resolve_codemode_worker_specifier("bun-binary", "file:///B:/~BUN/root/config.js"),
			Some("./src/extensions/codemode/worker.ts".to_string())
		);
	}

	#[test]
	fn uses_the_emitted_worker_beside_the_bundled_node_module() {
		assert_eq!(
			resolve_codemode_worker_specifier("bundled-node", "file:///app/chunks/config.js"),
			Some("file:///app/chunks/codemode-worker.js".to_string())
		);
	}

	#[test]
	fn uses_the_pi_codemode_worker_when_unbundled() {
		assert_eq!(resolve_codemode_worker_specifier("unbundled", "file:///x/config.js"), None);
	}
}
