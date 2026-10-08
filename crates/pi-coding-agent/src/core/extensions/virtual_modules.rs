//! Port of `packages/coding-agent/src/core/extensions/virtual-modules.ts` —
//! the bundled modules available to extensions.
//!
//! Upstream this is a `Record<string, unknown>` of live JS module namespaces
//! (typebox, the pi packages, their legacy `@mariozechner` aliases) consumed
//! by jiti's `virtualModules` option inside compiled binaries and the bundled
//! Node distribution, and by the TS-source jiti path. Module *values* are JS
//! namespaces and cannot exist in the Rust host, so the table stays behind
//! the [`super::loader::ExtensionModuleLoader`] seam (D40): the specifier
//! list below is the data the pi-runtime jiti loader must register, and each
//! entry resolves to the corresponding Rust crate or JS bundle at bridge
//! time.

/// Module specifiers available to extensions in source and compiled binary
/// runtimes (`VIRTUAL_MODULES` keys). `@earendil-works/*` and
/// `@mariozechner/*` names resolve to the same implementations; extensions
/// importing the pi-ai root get the compat entrypoint (a strict superset of
/// the core entrypoint, so legacy global-API extensions keep working).
pub const VIRTUAL_MODULE_SPECIFIERS: &[&str] = &[
	"typebox",
	"typebox/compile",
	"typebox/value",
	"@sinclair/typebox",
	"@sinclair/typebox/compile",
	"@sinclair/typebox/value",
	"@earendil-works/pi-agent-core",
	"@earendil-works/pi-tui",
	// Extensions resolve the pi-ai root to the compat entrypoint (a strict
	// superset of the core entrypoint): existing extensions using the old
	// global API keep working at runtime until compat is removed.
	"@earendil-works/pi-ai",
	"@earendil-works/pi-ai/compat",
	"@earendil-works/pi-ai/oauth",
	"@earendil-works/pi-ai/providers/all",
	"@earendil-works/pi-coding-agent",
	"@mariozechner/pi-agent-core",
	"@mariozechner/pi-tui",
	"@mariozechner/pi-ai",
	"@mariozechner/pi-ai/compat",
	"@mariozechner/pi-ai/oauth",
	"@mariozechner/pi-ai/providers/all",
	"@mariozechner/pi-coding-agent",
];

#[cfg(test)]
mod virtual_modules_tests {
	use super::*;

	#[test]
	fn specifier_table_matches_upstream_keys() {
		assert_eq!(VIRTUAL_MODULE_SPECIFIERS.len(), 20);
		assert!(VIRTUAL_MODULE_SPECIFIERS.contains(&"typebox"));
		assert!(VIRTUAL_MODULE_SPECIFIERS.contains(&"@earendil-works/pi-coding-agent"));
		assert!(VIRTUAL_MODULE_SPECIFIERS.contains(&"@mariozechner/pi-ai/providers/all"));
	}
}
