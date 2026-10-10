//! Port of `packages/coding-agent/src/extensions/mcp/` — the built-in MCP
//! integration.
//!
//! | Upstream | Port | Status |
//! |---|---|---|
//! | `index.ts` | [`index`] + [`factory`] | ported (JS module eval → D40 loader, D80) |
//! | `config.ts` | [`config`] | ported |
//! | `oauth.ts` | [`oauth`] | ported (flow lives in `pi-mcp`) |
//! | `tools.ts` | [`tools`] | ported (renderers → D81) |
//! | `resources.ts` | [`resources`] | ported |
//! | `runtime.ts` | [`runtime`] | ported |
//! | `cli.ts` | [`cli`] | ported (chalk → D81) |
//! | `log.ts` | [`log`] | ported |
//! | `ui.ts` | — | deferred (TUI-only, D81; re-audited D160 — no portable logic) |
//! | `cli.lazy.ts`, `runtime.lazy.ts` | — | folded into plain imports (D81) |
//!
//! # Deviations D80–D89
//!
//! - **D80.** The `index.ts` JS module evaluation stays behind the D40
//!   [`core::extensions::loader`] seam; [`factory::create_mcp_extension`] is
//!   the native factory surface. The `before_agent_start` handler computes
//!   the `mcp_servers` section, but the runner's erased
//!   `BeforeAgentStartEventResult` cannot yet carry section mutations (only
//!   `systemPrompt`); the pure renderer ([`index::render_servers_section`])
//!   is ported and tested, publishing waits on runner support. Sibling
//!   package predicates (`isCodemodeTool`, `isToolSearchTool`) are matched by
//!   tool name to avoid cross-module coupling.
//! - **D81.** TUI-only surface defers: `ui.ts` / `showMcpManager`, the
//!   `/mcp` manager screens, the tool renderers
//!   (`createMcpToolRenderers`), and chalk styling. `cli.lazy.ts` /
//!   `runtime.lazy.ts` fold into plain module imports (Rust linking makes the
//!   lazy loader moot).
//! - **D82.** `mcp.json` rewrites use a local pretty printer over
//!   order-preserving `serde_json` (see [`config`]).
//! - **D83.** The generic, non-object-safe `AuthStorageBackend` erases behind
//!   [`runtime::McpCredentials`] and `Arc<dyn McpOAuthServerStore>`; the
//!   provider exposes [`oauth::SharedMcpAuthProvider`] for the
//!   `Arc<dyn AuthProvider>` seams.
//! - **D84.** The proper-lockfile refresh lock becomes a create-exclusive
//!   lock file with mtime staleness (same 20s / 25s / 100ms constants); the
//!   in-process single-flight refresh is a shared future.
//! - **D85.** `URL` → the `url` crate; `URL.canParse` → `Url::parse`; the
//!   `[::1]` loopback hostname is matched literally.
//! - **D86.** base64url/hex encoders are local (no `base64` dependency).
//! - **D87.** OAuth callback pages render plain text (upstream HTML helpers
//!   are TUI-adjacent).
//! - **D88.** Test fixtures: the OAuth MCP server fixture is an in-module
//!   tokio HTTP server (upstream `node:http`), and the stdio JSON-RPC fixture
//!   is a `/bin/sh` loop (upstream `node fixtures/stdio-server.mjs`);
//!   OS-reported spawn error text is asserted by state, not message.
//! - **D89.** JS string semantics: section budgets and the 64-character tool
//!   name cap count UTF-8 bytes where upstream counts UTF-16 units (ASCII
//!   equivalent); `McpToolDetails` is erased JSON (core extensions D27).
//! - **D160.** `ui.ts` re-audit: the file is entirely TUI-bound — the
//!   `McpMenu` items are pi-tui `SelectItem`s, the menu-selection
//!   persistence is interleaved with `SelectList` rendering/keybinding
//!   handling, and `redirectUrl`/`status` are component screens — so the
//!   D81 deferral stands with nothing portable to extract.
//!
//! # Deferred tests
//!
//! - `test/suite/agent-session-mcp*.test.ts` — AgentSession↔MCP integration
//!   needs the TUI/RPC session hosts; deferred with those ports.
//! - TUI manager flows (`/mcp` menus, tool listing screens, sign-in screen)
//!   — D81.
//! - `login`'s interactive paste prompt needs a terminal input channel; the
//!   sign-in completes through the browser callback (D81).

pub mod cli;
pub mod config;
pub mod factory;
pub mod index;
pub mod log;
pub mod oauth;
pub mod resources;
pub mod runtime;
pub mod tools;

#[cfg(test)]
mod cli_tests;

/// Upstream `export default createMcpExtension()`: the built-in factory with
/// default options.
pub fn create_mcp_extension() -> crate::core::extensions::types::ExtensionFactory {
    factory::create_mcp_extension(factory::McpExtensionOptions::default())
}
