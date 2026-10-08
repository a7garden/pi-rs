//! Port of `packages/coding-agent/src/core/extensions/index.ts` — the
//! extension-system module barrel. Type-only re-exports of `types.ts`
//! surface stay in [`types`]; the runtime exports mirror the upstream
//! `export` clauses.

pub use super::loader::{clear_extension_cache, create_extension_runtime, load_extension_from_factory, load_extensions_cached};
pub use super::runner::{
	emit_project_trust_event, emit_session_shutdown_event, ExtensionRunner,
};
pub use super::types::{
	define_tool, is_bash_tool_result, is_edit_tool_result, is_find_tool_result, is_grep_tool_result,
	is_ls_tool_result, is_power_shell_tool_result, is_read_tool_result, is_tool_call_event_type,
	is_write_tool_result, Extension, ExtensionError, ExtensionFactory, ExtensionRuntime, LoadExtensionsResult,
};
pub use super::wrapper::{wrap_registered_tool, wrap_registered_tools};
